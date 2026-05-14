# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any

import torch

from cosmos.utils import log
from cosmos.data.vfm.action.cosmos3_action_lerobot import (
    BaseActionLeRobotDataset,
)
from cosmos.data.vfm.action.viewpoint_utils import Viewpoint


def _overlay_cross_from_xy_0_512(
    video_tchw: torch.Tensor,
    xy: torch.Tensor,
    *,
    color_rgb: tuple[float, float, float] = (1.0, 0.0, 0.0),
    radius: int = 4,
    thickness: int = 1,
    action_norm_range: float = 512.0,
) -> torch.Tensor:
    """
    Overlay a colored 'X' cross on video frames at (x, y) positions from `xy`.

    Assumptions:
    - `video_tchw` is (T, C, H, W) float in [0, 1] (or at least 3-channel RGB).
    - `xy` is (T_xy, 2+) with raw coords in [0, 512].
    - We draw `xy[t]` directly onto `video_tchw[t]`.
    """
    if video_tchw.dim() != 4:
        raise ValueError(f"Expected video_tchw to be 4D (T,C,H,W), got {tuple(video_tchw.shape)}")
    if video_tchw.shape[1] < 3:
        # Not RGB; nothing sensible to do.
        return video_tchw
    if xy.numel() == 0:
        return video_tchw

    T, _, H, W = video_tchw.shape
    # scale raw [0, 512] to pixel indices [0, W-1]/[0, H-1]
    sx = (W - 1) / action_norm_range
    sy = (H - 1) / action_norm_range

    # Ensure we can index in python loop
    xy_f = xy[:, :2].detach().to(dtype=torch.float32)

    # Draw in-place
    t_max = min(int(xy_f.shape[0]), int(T))
    if t_max <= 0:
        return video_tchw

    r_col, g_col, b_col = (float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))

    for t in range(t_max):
        frame_t = t
        xy_t = xy_f[t]
        if not torch.isfinite(xy_t).all():
            continue
        x = int(torch.round(xy_t[0] * sx).clamp(0, W - 1).item())
        y = int(torch.round(xy_t[1] * sy).clamp(0, H - 1).item())

        # Draw an "X" (two diagonals) centered at (x, y)
        r = int(radius)
        th = int(thickness)
        for d in range(-r, r + 1):
            # Diagonal 1: (x+d, y+d)
            xx1, yy1 = x + d, y + d
            # Diagonal 2: (x+d, y-d)
            xx2, yy2 = x + d, y - d

            for ox in range(-th, th + 1):
                for oy in range(-th, th + 1):
                    # (x+d, y+d)
                    xxx1, yyy1 = xx1 + ox, yy1 + oy
                    if 0 <= xxx1 < W and 0 <= yyy1 < H:
                        video_tchw[frame_t, 0, yyy1, xxx1] = r_col
                        video_tchw[frame_t, 1, yyy1, xxx1] = g_col
                        video_tchw[frame_t, 2, yyy1, xxx1] = b_col
                    # (x+d, y-d)
                    xxx2, yyy2 = xx2 + ox, yy2 + oy
                    if 0 <= xxx2 < W and 0 <= yyy2 < H:
                        video_tchw[frame_t, 0, yyy2, xxx2] = r_col
                        video_tchw[frame_t, 1, yyy2, xxx2] = g_col
                        video_tchw[frame_t, 2, yyy2, xxx2] = b_col

    return video_tchw


class PushTDataset(BaseActionLeRobotDataset):
    """PushT dataset with deferred source registration.

    Sources are registered by ``_register_sources()`` which is called by
    ``ActionUnifiedIterableDataset.assign_worker()`` during training, or
    explicitly for standalone/eval use.
    """

    def __init__(
        self,
        repo_id: str = "lerobot/pusht",
        root: str | None = None,
        chunk_length: int = 16,
        fps: int = 10,
        split: str = "train",
        split_seed: int = 0,
        split_val_ratio: float = 0.05,
        mode: str = "policy",
        tolerance_s: float = 1e-4,
        embodiment_type: str = "pusht",
        force_cache_sync: bool = False,
        action_norm_range: float = 512.0,
        action_space: str = "relative",
        overlay_cross: bool = False,
        overlay_cross_radius: int = 2,
        overlay_cross_thickness: int = 0,
        augment_prompt: bool = False,
        viewpoint: Viewpoint = "third_person_view",
    ) -> None:
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type=embodiment_type,
            viewpoint=viewpoint,
            tolerance_s=tolerance_s,
        )

        self.action_norm_range = action_norm_range
        self.overlay_cross = bool(overlay_cross)
        self.overlay_cross_radius = int(overlay_cross_radius)
        self.overlay_cross_thickness = int(overlay_cross_thickness)
        self.action_space = action_space
        self.augment_prompt = augment_prompt

        self._delta_timestamps = {
            "observation.image": [i * self._dt for i in range(0, chunk_length + 1)],
            "observation.state": [i * self._dt for i in range(0, chunk_length + 1)],
            "action": [i * self._dt for i in range(0, chunk_length + 1)],
        }
        self._repo_id = repo_id
        self._root = root
        self._force_cache_sync = force_cache_sync
        self._all_shard_roots = [root or repo_id]

    def _register_sources(self, indices: list[int] | None = None) -> None:
        if indices is not None and 0 not in indices:
            return
        self._register_source(
            repo_id=self._repo_id,
            root=self._root,
            force_cache_sync=self._force_cache_sync,
            delta_timestamps=self._delta_timestamps,
            tolerance_s=self._tolerance_s,
            dataset_label=self._repo_id,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, _, base_idx, item = self._fetch_sample(idx)

        if self.augment_prompt:
            prefix = "You are given a task to push the green T into the yellow T region."
            prompt = f"Current prediction mode is {mode}."
            post_fix = f"The video is {self._chunk_length / self._fps} seconds long and is of {self._fps} FPS."
            prompt = f"{prefix} {prompt} {post_fix}"
        else:
            prompt = "PushT task"

        # Video: LeRobot returns float32 in [0, 1] with shape (T, C, H, W)
        video_tchw: torch.Tensor = item["observation.image"]

        # Action (raw): typically absolute XY in [0, 512] for PushT.
        action_raw: torch.Tensor = item["action"]
        # State (raw): typically contains current agent state/position, also in [0, 512] for PushT.
        # We use the -1/fps state (first element) as the "current" reference.
        state_raw: torch.Tensor = item["observation.state"]

        # Optionally overlay the raw action XY on the video for debugging/visualization.
        # We draw state/action[t] directly on frame[t] (all are indexed by the same delta timestamps).
        if self.overlay_cross:
            try:
                # State in red
                _overlay_cross_from_xy_0_512(
                    video_tchw,
                    state_raw,  # (T+1, D)
                    color_rgb=(1.0, 0.0, 0.0),
                    radius=self.overlay_cross_radius,
                    thickness=self.overlay_cross_thickness,
                    action_norm_range=self.action_norm_range,
                )
                # Action in purple
                _overlay_cross_from_xy_0_512(
                    video_tchw,
                    action_raw,  # (T+1, D)
                    color_rgb=(1.0, 0.0, 1.0),
                    radius=self.overlay_cross_radius,
                    thickness=self.overlay_cross_thickness,
                    action_norm_range=self.action_norm_range,
                )
            except Exception as e:
                log.warning(f"Failed to overlay action cross for idx={base_idx}: {e}")

        # Keep raw video in LeRobot layout; the base helper does the final conversion.
        video = video_tchw  # [T,C,H,W]

        # Action: (T+1, D) -> (T, D)
        # Compute relative action w.r.t. the "current" state (delta=0).
        # LeRobot returns state/action at deltas [0, 1, ..., chunk_length] (length T+1).
        # We use state[0] as the reference and take actions[0:T] (dropping the last one).
        #
        # This matches: rel_action[t] = action[t] - state_current, for t in [0..T-1].
        action = action_raw[:-1]  # [T,D_action]
        state_current = state_raw[:1]  # [1,D_state]
        if self.action_space == "relative":
            action = action - state_current
        elif self.action_space == "absolute":
            action = action
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}")
        # Normalize action to [-1, 1]
        action = action / self.action_norm_range
        if action.max() > 1.0 or action.min() < -1.0:
            log.warning(f"Action out of range: {action.max()}, {action.min()}")

        key = torch.tensor([base_idx], dtype=torch.long)

        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=prompt,
            action_space=self.action_space,
            state=state_raw,
            __key__=key,
        )
