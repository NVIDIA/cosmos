# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
HTTP server that serves ground-truth actions from LIBERO LeRobot datasets.

Same HTTP interface as `cosmos3.scripts.action_policy_server` (the model-backed
server), enabling drop-in replacement for closed-loop evaluation to verify the
action pipeline with known-good GT actions.

Endpoints:
- POST /predict: Return next chunk of GT actions for the given task (matched by prompt)
- GET  /info:    Return dataset info (tasks, episode counts)
- POST /next_episode: Advance to next episode for the task specified in request body
- POST /reset:   Reset all per-task episode/step tracking

Episode advancement:
  The server auto-advances to the next episode when the current episode's actions
  are exhausted.  For early-termination cases (e.g. success before all actions are
  consumed), call POST /next_episode with {"prompt": "<task>"} between episodes.

Example usage:


PYTHONPATH=. python cosmos3/_src/vfm/evaluation/action/libero/dataset_reply_action_server.py \
  --repo_id libero_10 \
  --root /path/to/libero_10_no_noops_1.0.0_lerobot_aligned \
  --action_space frame_wise_relative \
  --rotation_space 6d \
  --pose_coordinate_frame opencv \
  --action_chunk_size 16 \
  --send_video \
  --camera_mode agentview \
  --port 8000

# Multiple datasets:
PYTHONPATH=. python cosmos3/_src/vfm/evaluation/action/libero/dataset_reply_action_server.py \
  --repo_id libero_10,libero_goal \
  --root /path/to/libero_10,/path/to/libero_goal \
  --action_space relative \
  --rotation_space 6d \
  --pose_coordinate_frame opencv \
  --action_chunk_size 16 \
  --port 8000
"""

from __future__ import annotations

import argparse
import base64
import datetime
import io
import json
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from PIL import Image

from cosmos3._src.vfm.datasets.action.libero_pose_utils import (
    libero_rotation_format,
)
from cosmos3._src.vfm.datasets.action.pose_utils import convert_rotation


def _ts() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except Exception:
        return socket.gethostbyname(socket.gethostname())


# ---------------------------------------------------------------------------
# Action processing (mirrors LIBERODataset.__getitem__ logic)
# ---------------------------------------------------------------------------


def _compute_anchored_actions(
    state_raw: torch.Tensor,
    action_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute anchored relative actions, same as LIBERODataset._compute_anchored_actions.

    Actions are expressed in state_raw[0]'s local coordinate frame.

    Args:
        state_raw: (T+1, 8) states [x, y, z, ax, ay, az, grip1, grip2].
        action_raw: (T+1, 7) actions [dx, dy, dz, dax, day, daz, grip].

    Returns:
        anchored_translation (T, 3), anchored_rotation (T, 3, 3), gripper (T, 1).
    """
    p_states = state_raw[:, :3]
    rotvec_states = state_raw[:, 3:6]
    delta_p = action_raw[:-1, :3]
    delta_rotvec = action_raw[:-1, 3:6]
    gripper = action_raw[:-1, 6:7]

    R_states = convert_rotation(rotvec_states, "axisangle", "matrix")
    R_deltas = convert_rotation(delta_rotvec, "axisangle", "matrix")

    p_0 = p_states[0]
    R_0_T = R_states[0].T

    p_t = p_states[:-1]
    R_t = R_states[:-1]

    p_target = p_t + delta_p
    R_target = torch.bmm(R_deltas, R_t)

    anchored_p = (R_0_T @ (p_target - p_0).T).T
    R_0_T_expanded = R_0_T.unsqueeze(0).expand(R_target.shape[0], -1, -1)
    anchored_R = torch.bmm(R_0_T_expanded, R_target)

    return anchored_p, anchored_R, gripper


def _convert_rotation_to_repr(rotation_matrix: torch.Tensor, rotation_space: str) -> torch.Tensor:
    return convert_rotation(rotation_matrix, "matrix", libero_rotation_format(rotation_space))


def _process_action_chunk(
    action_raw: torch.Tensor,
    state_raw: torch.Tensor,
    action_space: str,
    rotation_space: str,
) -> torch.Tensor:
    """Process a chunk of raw actions with the same logic as LIBERODataset.__getitem__.

    Args:
        action_raw: (chunk+1, 7) raw actions covering chunk+1 consecutive frames.
        state_raw:  (chunk+1, 8) raw states  covering chunk+1 consecutive frames.
        action_space: "relative" or "frame_wise_relative".
        rotation_space: "3d", "6d", or "9d".

    Returns:
        Processed actions (chunk, D) where D depends on rotation_space.
    """
    if action_space == "relative":
        translation, rotation_matrix, gripper = _compute_anchored_actions(state_raw, action_raw)
    elif action_space == "frame_wise_relative":
        action = action_raw[:-1].clone()
        translation = action[:, :3]
        rotation_matrix = convert_rotation(action[:, 3:6], "axisangle", "matrix")
        gripper = action[:, 6:]
    else:
        raise ValueError(f"Unsupported action_space: {action_space}")

    rotation = _convert_rotation_to_repr(rotation_matrix, rotation_space)
    return torch.cat([translation, rotation, gripper], dim=-1)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeData:
    action_raw: torch.Tensor  # (N, 7) per-frame raw actions for the full episode
    state_raw: torch.Tensor  # (N, 8) per-frame raw states for the full episode
    task_description: str
    dataset_ref_idx: int  # index into DatasetActionService._hf_datasets
    frame_start: int  # first global frame index in the HF dataset
    frame_end: int  # one-past-last global frame index


@dataclass(frozen=True)
class DatasetServerConfig:
    repo_id: list[str]
    root: list[str | None]
    action_space: str
    rotation_space: str
    pose_coordinate_frame: str
    action_chunk_size: int
    max_action_dim: int
    split: str
    send_video: bool
    camera_mode: str
    image_size: int


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DatasetActionService:
    """Serves GT actions (and optionally GT video) from pre-loaded LIBERO LeRobot episodes."""

    def __init__(self, cfg: DatasetServerConfig) -> None:
        self.cfg = cfg
        self.episodes_by_task: dict[str, list[EpisodeData]] = {}
        self._hf_datasets: list[Any] = []
        self._lerobot_datasets: list[Any] = []
        self._task_state: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

        if cfg.camera_mode in ("concat_view", "both"):
            self._image_keys = ["observation.images.image", "observation.images.wrist_image"]
        elif cfg.camera_mode == "wrist_image":
            self._image_keys = ["observation.images.wrist_image"]
        else:
            self._image_keys = ["observation.images.image"]

        self._load_datasets()

    def _load_datasets(self) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        for repo_id, root in zip(self.cfg.repo_id, self.cfg.root):
            print(f"[{_ts()}] [dataset-server] loading repo_id={repo_id} root={root} ...", flush=True)
            t0 = time.monotonic()

            dataset = LeRobotDataset(repo_id=repo_id, root=root)
            tasks_df = dataset.meta.tasks
            hf = dataset.hf_dataset
            ds_ref_idx = len(self._hf_datasets)
            self._hf_datasets.append(hf)

            if self.cfg.send_video:
                delta_ts: dict[str, list[float]] = {k: [0.0] for k in self._image_keys}
                video_dataset = LeRobotDataset(repo_id=repo_id, root=root, delta_timestamps=delta_ts)
                self._lerobot_datasets.append(video_dataset)
            else:
                self._lerobot_datasets.append(None)

            for ep_meta in dataset.meta.episodes:
                ep_idx = int(ep_meta["episode_index"])  # type: ignore[index]
                start = int(ep_meta["dataset_from_index"])  # type: ignore[index]
                end = int(ep_meta["dataset_to_index"])  # type: ignore[index]

                ep_slice = hf.select(range(start, end))
                actions = torch.tensor(np.array(ep_slice["action"], dtype=np.float32))
                states = torch.tensor(np.array(ep_slice["observation.state"], dtype=np.float32))

                task_idx = int(ep_slice[0]["task_index"])
                matching = tasks_df[tasks_df["task_index"] == task_idx]
                task_desc = str(matching.iloc[0].name) if not matching.empty else f"task_{task_idx}"

                self.episodes_by_task.setdefault(task_desc, []).append(
                    EpisodeData(
                        action_raw=actions,
                        state_raw=states,
                        task_description=task_desc,
                        dataset_ref_idx=ds_ref_idx,
                        frame_start=start,
                        frame_end=end,
                    )
                )

            dt = time.monotonic() - t0
            print(
                f"[{_ts()}] [dataset-server] loaded {repo_id}: {dataset.meta.total_episodes} episodes in {dt:.1f}s",
                flush=True,
            )

        total_tasks = len(self.episodes_by_task)
        total_eps = sum(len(eps) for eps in self.episodes_by_task.values())
        print(
            f"[{_ts()}] [dataset-server] ready: {total_tasks} tasks, {total_eps} episodes "
            f"send_video={self.cfg.send_video} camera_mode={self.cfg.camera_mode}",
            flush=True,
        )

    def _load_video_frames(self, episode: EpisodeData, step: int, num_frames: int) -> list[str]:
        """Load GT video frames from the dataset and encode as base64 PNGs.

        Uses the LeRobotDataset wrapper (not the raw HF dataset) so that video-backed
        datasets are decoded correctly via the configured video backend.

        Args:
            episode: Episode data with dataset reference.
            step: Step offset within the episode (0-based).
            num_frames: Number of frames to load (typically action_chunk_size + 1).

        Returns:
            List of base64-encoded PNG strings.
        """
        lr_dataset = self._lerobot_datasets[episode.dataset_ref_idx]
        if lr_dataset is None:
            return []
        image_size = self.cfg.image_size
        b64_frames: list[str] = []

        for i in range(num_frames):
            global_idx = episode.frame_start + step + i
            if global_idx >= episode.frame_end:
                break

            item = lr_dataset[global_idx]

            pil_images: list[Image.Image] = []
            for key in self._image_keys:
                img_tensor = item[key]
                if isinstance(img_tensor, torch.Tensor):
                    # LeRobot returns (T, C, H, W) with delta_timestamps=[0.0] -> (1, C, H, W)
                    if img_tensor.dim() == 4:
                        img_tensor = img_tensor[0]
                    # (C, H, W) float [0, 1] -> PIL
                    arr = (img_tensor.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).numpy()
                    img = Image.fromarray(arr)
                elif isinstance(img_tensor, Image.Image):
                    img = img_tensor
                else:
                    img = Image.fromarray(np.asarray(img_tensor, dtype=np.uint8))
                img = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
                pil_images.append(img)

            if len(pil_images) > 1:
                total_w = sum(im.width for im in pil_images)
                combined = Image.new("RGB", (total_w, image_size))
                x = 0
                for im in pil_images:
                    combined.paste(im, (x, 0))
                    x += im.width
                frame = combined
            else:
                frame = pil_images[0]

            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            b64_frames.append(base64.b64encode(buf.getvalue()).decode("ascii"))

        return b64_frames

    # -- state management --

    def _get_task_state(self, prompt: str) -> dict[str, int]:
        if prompt not in self._task_state:
            self._task_state[prompt] = {"episode_idx": 0, "step": 0}
        return self._task_state[prompt]

    def _resolve_prompt(self, prompt: str) -> str:
        """Resolve prompt to a known task description (exact or substring match)."""
        if prompt in self.episodes_by_task:
            return prompt
        prompt_lower = prompt.lower().strip()
        for task_desc in self.episodes_by_task:
            if task_desc.lower().strip() == prompt_lower:
                return task_desc
        for task_desc in self.episodes_by_task:
            td_lower = task_desc.lower().strip()
            if prompt_lower in td_lower or td_lower in prompt_lower:
                return task_desc
        raise ValueError(
            f"Task not found for prompt: {prompt!r}. Available tasks: {sorted(self.episodes_by_task.keys())}"
        )

    # -- endpoints --

    def get_info(self) -> dict[str, Any]:
        return {
            "type": "dataset_action_server",
            "action_space": self.cfg.action_space,
            "rotation_space": self.cfg.rotation_space,
            "action_chunk_size": self.cfg.action_chunk_size,
            "tasks": {k: len(v) for k, v in sorted(self.episodes_by_task.items())},
        }

    def predict(self, req: dict[str, Any]) -> dict[str, Any]:
        prompt = req.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")

        resolved_prompt = self._resolve_prompt(prompt)

        with self._lock:
            state = self._get_task_state(resolved_prompt)
            episodes = self.episodes_by_task[resolved_prompt]

            ep_idx = state["episode_idx"] % len(episodes)
            episode = episodes[ep_idx]
            step = state["step"]

            # Number of valid actions = num_frames - 1 (need pairs of consecutive frames)
            max_actions = len(episode.action_raw) - 1

            if step >= max_actions:
                state["episode_idx"] = (ep_idx + 1) % len(episodes)
                state["step"] = 0
                ep_idx = state["episode_idx"]
                episode = episodes[ep_idx]
                step = 0
                max_actions = len(episode.action_raw) - 1

            chunk_size = min(self.cfg.action_chunk_size, max_actions - step)
            # Slice chunk+1 frames for action computation (needs next-frame state)
            raw_slice_end = step + chunk_size + 1
            action_chunk_raw = episode.action_raw[step:raw_slice_end]
            state_chunk_raw = episode.state_raw[step:raw_slice_end]

            processed = _process_action_chunk(
                action_chunk_raw,
                state_chunk_raw,
                self.cfg.action_space,
                self.cfg.rotation_space,
            )

            # Pad to max_action_dim (same as the Action transform pipeline)
            t, d = processed.shape
            if d < self.cfg.max_action_dim:
                processed = torch.cat(
                    [processed, torch.zeros(t, self.cfg.max_action_dim - d)],
                    dim=-1,
                )

            state["step"] += chunk_size

            action_list = processed.float().numpy().tolist()

            video_b64: list[str] = []
            if self.cfg.send_video:
                video_b64 = self._load_video_frames(episode, step, num_frames=chunk_size + 1)

        print(
            f"[{_ts()}] [dataset-server] predict prompt={resolved_prompt!r} "
            f"ep={ep_idx} step={step}..{state['step']} actions={len(action_list)} "
            f"video_frames={len(video_b64)}",
            flush=True,
        )
        return {"action": action_list, "video": video_b64}

    def next_episode(self, prompt: str | None = None) -> dict[str, Any]:
        with self._lock:
            if prompt is not None:
                resolved = self._resolve_prompt(prompt)
                state = self._get_task_state(resolved)
                episodes = self.episodes_by_task[resolved]
                state["episode_idx"] = (state["episode_idx"] + 1) % len(episodes)
                state["step"] = 0
                print(
                    f"[{_ts()}] [dataset-server] next_episode task={resolved!r} -> ep={state['episode_idx']}",
                    flush=True,
                )
                return {"task": resolved, "episode_idx": state["episode_idx"]}

            for task in self._task_state:
                episodes = self.episodes_by_task.get(task, [])
                self._task_state[task]["episode_idx"] = (self._task_state[task]["episode_idx"] + 1) % max(
                    len(episodes), 1
                )
                self._task_state[task]["step"] = 0
            print(f"[{_ts()}] [dataset-server] next_episode (all tasks)", flush=True)
            return {"advanced_all": True}

    def reset(self) -> dict[str, str]:
        with self._lock:
            self._task_state.clear()
        print(f"[{_ts()}] [dataset-server] reset", flush=True)
        return {"status": "reset"}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class _DatasetHandler(BaseHTTPRequestHandler):
    server: ThreadingHTTPServer  # type: ignore[assignment]

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _read_json_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return None
        body = self.rfile.read(max(0, length))
        if not body:
            return {}
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return None
        if not isinstance(req, dict):
            self._send_json(400, {"error": "JSON body must be an object"})
            return None
        return req

    def do_GET(self) -> None:  # noqa: N802
        service: DatasetActionService = getattr(self.server, "service")
        if self.path == "/info":
            self._send_json(200, service.get_info())
        elif self.path == "/":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        service: DatasetActionService = getattr(self.server, "service")

        if self.path in ("/", "/predict"):
            req = self._read_json_body()
            if req is None:
                return
            try:
                out = service.predict(req)
            except Exception as e:
                print(f"[{_ts()}] [dataset-server] predict ERROR: {e}", flush=True)
                self._send_json(400, {"action": [], "error": str(e)})
                return
            self._send_json(200, out)

        elif self.path == "/next_episode":
            req = self._read_json_body()
            prompt = req.get("prompt") if req else None
            try:
                out = service.next_episode(prompt)
            except Exception as e:
                self._send_json(400, {"error": str(e)})
                return
            self._send_json(200, out)

        elif self.path == "/reset":
            out = service.reset()
            self._send_json(200, out)

        else:
            self._send_json(404, {"error": "Not found"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTTP server serving ground-truth actions from LIBERO LeRobot datasets."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Comma-separated LeRobot repo IDs (e.g. libero_10,libero_goal)",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Comma-separated local paths to dataset roots (one per repo_id)",
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="frame_wise_relative",
        choices=["relative", "frame_wise_relative"],
        help="Action space (must match closed-loop eval's --action_space).",
    )
    parser.add_argument(
        "--rotation_space",
        type=str,
        default="6d",
        choices=["3d", "6d", "9d"],
        help="Rotation representation (must match closed-loop eval's action_dim).",
    )
    parser.add_argument(
        "--pose_coordinate_frame",
        type=str,
        default="native",
        choices=["native", "opencv"],
        help="Pose/action coordinate frame. Accepted for compatibility with LIBERO eval launchers.",
    )
    parser.add_argument("--action_chunk_size", type=int, default=16, help="Number of actions per predict call")
    parser.add_argument("--max_action_dim", type=int, default=32, help="Pad actions to this dimension")
    parser.add_argument("--split", type=str, default="full", help="Dataset split (train/val/full)")
    parser.add_argument(
        "--send_video",
        action="store_true",
        help="Include GT video frames (base64 PNGs) in /predict responses, same format as the Action server.",
    )
    parser.add_argument(
        "--camera_mode",
        type=str,
        default="image",
        choices=["agentview", "wrist_image", "concat_view", "both"],
        help="Camera view(s) to include in video frames.",
    )
    parser.add_argument("--image_size", type=int, default=256, help="Resize video frames to this height/width")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    repo_ids = [r.strip() for r in args.repo_id.split(",") if r.strip()]
    roots = [r.strip() for r in args.root.split(",") if r.strip()]
    if len(repo_ids) != len(roots):
        raise ValueError(f"Number of repo_ids ({len(repo_ids)}) must match number of roots ({len(roots)})")

    cfg = DatasetServerConfig(
        repo_id=repo_ids,
        root=roots,
        action_space=args.action_space,
        rotation_space=args.rotation_space,
        pose_coordinate_frame=args.pose_coordinate_frame,
        action_chunk_size=int(args.action_chunk_size),
        max_action_dim=int(args.max_action_dim),
        split=args.split,
        send_video=bool(args.send_video),
        camera_mode=args.camera_mode,
        image_size=int(args.image_size),
    )

    service = DatasetActionService(cfg)
    local_ip = _get_local_ip()

    print(
        f"[{_ts()}] [dataset-server] starting host={args.host} port={args.port} "
        f"action_space={cfg.action_space} rotation_space={cfg.rotation_space} "
        f"action_chunk_size={cfg.action_chunk_size}",
        flush=True,
    )
    print(f"[{_ts()}] [dataset-server] Server accessible at: http://{local_ip}:{args.port}/", flush=True)
    print(f"[{_ts()}] [dataset-server] Endpoints:", flush=True)
    print(f"  - GET  /             : Health check", flush=True)
    print(f"  - GET  /info         : Dataset info (tasks, episode counts)", flush=True)
    print(f"  - POST /predict      : Get next GT action chunk (same interface as Action server)", flush=True)
    print(f"  - POST /next_episode : Advance to next episode for a task", flush=True)
    print(f"  - POST /reset        : Reset all per-task state", flush=True)

    httpd = ThreadingHTTPServer((args.host, int(args.port)), _DatasetHandler)
    setattr(httpd, "service", service)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
