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

"""HTTP inference server for Action policy mode using OmniMoTModel.

The server exposes two endpoints:

- POST /predict: run policy inference.
  - Input: {"image": "<base64_png>", "prompt": "<task>", "domain_name": "<name>", "image_size": <int>}
  - Output: {"action": [[a0, a1, ...], ...], "video": ["<base64_png>", ...]}
- GET /info: model / runtime info (run_name, checkpoint, sampling params, ...).

Model loading uses ``cosmos3.inference.OmniInference.create`` which auto-dispatches
between two checkpoint flows based on ``--config-file``'s extension:

* ``.py``  — in-tree experiment registry (e.g.
  ``projects/cosmos3/vfm/configs/base/config.py``) plus an ``--experiment-name``.
* ``.yaml`` / ``.yml`` / ``.json`` — frozen Hydra config produced by
  ``cosmos3.scripts.train`` next to the DCP checkpoint.

Example (in-tree experiment):

  PYTHONPATH=. python -m cosmos3.scripts.action_policy_server \
    --experiment-name libero_exp_policy \
    --config-file cosmos3/_src/vfm/configs/base/config.py \
    --checkpoint-dir /path/to/checkpoints/iter_000020000 \
    --port 8000

Example (frozen Hydra YAML produced by cosmos3.scripts.train):

  PYTHONPATH=. python -m cosmos3.scripts.action_policy_server \
    --config-file /path/to/job/config.yaml \
    --checkpoint-dir /path/to/job/checkpoints/iter_000020000 \
    --port 8000
"""

from cosmos3.common.init import init_script, is_rank0  # noqa: F401  (is_rank0 may be useful for logging)

init_script()

import base64
import binascii
import datetime
import io
import json
import os
import socket
import threading
import time
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic
import torch
import tyro
from omegaconf import DictConfig
from PIL import Image
from torch import distributed as dist

from cosmos3.args import OmniSetupArgs, OmniSetupOverrides
from cosmos3.common.args import ConfigFileType, tyro_cli
from cosmos3.common.config import deserialize_config_dict
from cosmos3.common.init import init_output_dir
from cosmos3.inference import OmniInference
from cosmos3._src.imaginaire.lazy_config import instantiate
from cosmos3._src.imaginaire.utils import log

# Action-specific helpers live in the in-tree project tree. Imports stay as
# `projects.cosmos3.vfm.*` and are auto-rewritten to `cosmos3._src.vfm.*` by the
# Cosmos release script.
from cosmos3._src.vfm.datasets.action.domain_utils import get_domain_id
from cosmos3._src.vfm.datasets.action.transforms import (
    build_sequence_plan_from_mode,
    find_closest_target_size,
    reflection_pad_to_target,
    remove_reflection_padding,
)
from cosmos3._src.vfm.utils.data_utils import get_vision_data_resolution

_DEFAULT_ACTION_CHUNK_SIZE = 16
_DEFAULT_FALLBACK_OUTPUT_DIR = Path("/tmp/cosmos3_action_server")
ActionNormalization = Literal["auto", "meanstd", "minmax", "quantile", "quantile_rot"]
ResolvedActionNormalization = Literal["meanstd", "minmax", "quantile", "quantile_rot"]

_DURATION_FPS_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."


# ---------------------------------------------------------------------------
# Pre/post processing helpers (copied verbatim from the previous server, with
# minor adjustments so config-introspection helpers also accept plain dicts).
# ---------------------------------------------------------------------------


def _augment_prompt_with_metadata(
    prompt: str,
    *,
    t_frames: int,
    fps: int,
    height: int,
    width: int,
    append_duration_fps: bool = True,
    append_resolution_info: bool = True,
) -> str:
    """Append duration/FPS and resolution metadata to match training-time augmentation.

    Mirrors ``DurationFPSTextTimeStamps`` and ``ResolutionTextInfo`` augmentors
    from the Action training transform pipeline.  Only appends each piece when the
    corresponding flag is ``True`` (matching the training config).
    """
    if append_duration_fps:
        duration = t_frames / fps
        sep = " " if prompt.rstrip().endswith(".") else ". "
        prompt = prompt + sep + _DURATION_FPS_TEMPLATE.format(duration=duration, fps=fps)
    if append_resolution_info:
        sep = " " if prompt.rstrip().endswith(".") else ". "
        prompt = prompt + sep + _RESOLUTION_TEMPLATE.format(height=height, width=width)
    return prompt


def _extract_bool_from_config(config: Any, key: str, default: bool) -> bool:
    """Recursively search dataloader_train config for a boolean flag."""

    def _search(obj: Any) -> bool | None:
        if isinstance(obj, (DictConfig, dict)):
            if key in obj:
                val = obj[key]
                if isinstance(val, bool):
                    return val
            iterable = obj.values()
            for v in iterable:
                result = _search(v)
                if result is not None:
                    return result
        return None

    try:
        if isinstance(config, dict):
            dl_train = config.get("dataloader_train")
        else:
            dl_train = getattr(config, "dataloader_train", None)
        if dl_train is not None:
            result = _search(dl_train)
            if result is not None:
                return result
    except Exception:
        pass
    return default


def _extract_str_from_config(config: Any, key: str) -> str | None:
    """Recursively search dataloader_train config for a string field."""

    def _search(obj: Any) -> str | None:
        if isinstance(obj, (DictConfig, dict)):
            if key in obj:
                val = obj[key]
                if isinstance(val, str):
                    return val
            iterable = obj.values()
            for v in iterable:
                result = _search(v)
                if result is not None:
                    return result
        return None

    try:
        if isinstance(config, dict):
            dl_train = config.get("dataloader_train")
        else:
            dl_train = getattr(config, "dataloader_train", None)
        if dl_train is not None:
            return _search(dl_train)
    except Exception:
        pass
    return None


def _extract_chunk_length_from_config(config: Any) -> int | None:
    """Try to extract chunk_length from the experiment's dataloader config.

    Recursively searches ``config.dataloader_train`` for ``chunk_length`` or
    ``num_action_per_chunk`` to determine the action chunk size the model was
    trained with.  Returns ``None`` when neither key is found.
    """

    def _search(obj: Any, keys: tuple[str, ...] = ("chunk_length", "num_action_per_chunk")) -> int | None:
        if isinstance(obj, (DictConfig, dict)):
            for key in keys:
                if key in obj:
                    val = obj[key]
                    if isinstance(val, int):
                        return val
            for v in obj.values():
                result = _search(v, keys)
                if result is not None:
                    return result
        return None

    try:
        if isinstance(config, dict):
            dl_train = config.get("dataloader_train")
        else:
            dl_train = getattr(config, "dataloader_train", None)
        if dl_train is not None:
            return _search(dl_train)
    except Exception:
        pass
    return None


def _get_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _get_local_ip() -> str:
    """Get the local IP address of this machine."""
    try:
        # Connect to an external address to determine the local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def _maybe_init_distributed() -> None:
    """
    Initialize a one-rank process group when launched outside torchrun so FSDP
    and distributed utilities have the process-group state they expect.

    ``init_script()`` already inits the PG when ``WORLD_SIZE>1`` (under
    torchrun); this function fills in the single-process case used by the
    standalone server.
    """
    if not dist.is_available() or dist.is_initialized():
        return

    world_size_env = os.getenv("WORLD_SIZE")
    rank_env = os.getenv("RANK")
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if world_size_env is not None and rank_env is not None:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    port = _get_free_local_port()
    dist.init_process_group(backend=backend, init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)


def _strip_data_url_prefix(b64: str) -> str:
    # Accept "data:image/png;base64,...." as well as raw base64.
    if "," in b64 and b64[:64].lower().startswith("data:"):
        return b64.split(",", 1)[1].strip()
    return b64.strip()


def _b64decode_loose(b64: str) -> bytes:
    """
    Decode base64 permissively.

    The simulator/client may include whitespace/newlines, omit padding, or use urlsafe base64.
    """
    s = _strip_data_url_prefix(b64)
    s = "".join(s.split())  # remove whitespace/newlines
    pad = (-len(s)) % 4
    if pad:
        s = s + ("=" * pad)
    try:
        return base64.b64decode(s, validate=False)
    except binascii.Error:
        return base64.urlsafe_b64decode(s)


def _decode_base64_png_to_rgb_uint8(image_b64: str) -> torch.Tensor:
    """
    Returns a tensor with shape (3, H, W), dtype uint8, RGB.
    """
    try:
        raw = _b64decode_loose(image_b64)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64 image: {e}") from e

    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        # Pillow can expose a read-only view; make it writable to avoid PyTorch warnings/UB.
        arr = np.asarray(img, dtype=np.uint8).copy()
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape {arr.shape}")
    chw = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]
    return chw


def _video_tensor_to_pil_images(video_c_t_h_w: torch.Tensor) -> list[Image.Image]:
    """
    Convert (C, T, H, W) float tensor in [-1,1] or [0,1] to a list of PIL RGB frames.
    """
    if video_c_t_h_w.dim() != 4:
        raise ValueError(f"Expected (C,T,H,W), got {tuple(video_c_t_h_w.shape)}")
    if int(video_c_t_h_w.shape[0]) != 3:
        raise ValueError(f"Expected C=3 RGB, got C={int(video_c_t_h_w.shape[0])}")

    images: list[Image.Image] = []
    t = int(video_c_t_h_w.shape[1])
    for ti in range(t):
        frame = video_c_t_h_w[:, ti].detach().cpu().float()
        if frame.min().item() < 0.0:
            frame = (frame + 1.0) / 2.0
        frame = frame.clamp(0.0, 1.0)
        frame_uint8 = (frame * 255.0).round().to(torch.uint8)  # [3,H,W]
        hwc = frame_uint8.permute(1, 2, 0).numpy()  # [H,W,3]
        images.append(Image.fromarray(hwc))
    return images


def _save_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / float(fps))))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    print(f"Saved gif to {path}")


def _save_policy_request_dump(
    *,
    dump_root: Path,
    request_id: int,
    request_json: dict[str, Any],
    obs_chw_uint8: torch.Tensor,
    pred_action: list[list[float]],
    pred_video_c_t_h_w: torch.Tensor | None,
    fps: int,
) -> None:
    """
    Dump input observation, predicted actions, and rollout video for offline debugging.
    Creates:
      - request.json
      - observation.png
      - action_output.json
      - rollout.gif (if pred_video provided)
      - rollout_frames/frame_XXX.png (if pred_video provided)
    """
    ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    out_dir = dump_root / f"{ts}_req{request_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=False)

    # Save request JSON (without the base64 image to save space, image is saved separately)
    request_json_copy = {k: v for k, v in request_json.items() if k != "image"}
    request_json_copy["image"] = "<saved as observation.png>"
    (out_dir / "request.json").write_text(json.dumps(request_json_copy, indent=2), encoding="utf-8")

    # Save observation image
    obs_hwc = obs_chw_uint8.permute(1, 2, 0).cpu().numpy()  # [H,W,3]
    obs_img = Image.fromarray(obs_hwc)
    obs_img.save(out_dir / "observation.png")

    # Save predicted actions
    action_output = {"action": pred_action}
    (out_dir / "action_output.json").write_text(json.dumps(action_output, indent=2), encoding="utf-8")

    if pred_video_c_t_h_w is None:
        return

    # Save rollout video
    frames = _video_tensor_to_pil_images(pred_video_c_t_h_w)
    _save_gif(frames, out_dir / "rollout.gif", fps=fps)

    # Save individual frames
    frames_dir = out_dir / "rollout_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(frames_dir / f"frame_{i:03d}.png")


def _save_failed_request_dump(
    *,
    dump_root: Path,
    request_id: int,
    request_json: dict[str, Any],
    error: str,
) -> None:
    """
    Dump request + error even if inference fails.
    Best-effort: will try to decode and save observation image if present.
    """
    ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    out_dir = dump_root / f"{ts}_req{request_id:06d}_ERROR"
    out_dir.mkdir(parents=True, exist_ok=False)

    # Save request JSON (without base64 image)
    request_json_copy = {k: v for k, v in request_json.items() if k != "image"}
    if "image" in request_json:
        request_json_copy["image"] = "<attempted to save as observation.png>"
    (out_dir / "request.json").write_text(json.dumps(request_json_copy, indent=2), encoding="utf-8")
    (out_dir / "error.txt").write_text(error, encoding="utf-8")

    try:
        image_b64 = request_json.get("image")
        if isinstance(image_b64, str):
            img_chw_uint8 = _decode_base64_png_to_rgb_uint8(image_b64)
            obs_hwc = img_chw_uint8.permute(1, 2, 0).cpu().numpy()  # [H,W,3]
            Image.fromarray(obs_hwc).save(out_dir / "observation.png")
    except Exception:
        # Ignore any dump failures.
        return


def _ts() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _load_introspection_config_dict(setup_args: OmniSetupArgs) -> dict:
    """Load the full experiment config as a plain dict for prompt-augmentation
    introspection.

    For ``MODULE`` (``.py``) configs, ``OmniInference.create`` saves the
    structured config to ``setup_args.output_dir / 'config.yaml'`` via
    ``cosmos3.common.config.save_config`` immediately after model load. For
    ``YAML`` / ``JSON`` configs we just deserialize the source file directly.
    """
    if setup_args.config_file_type == ConfigFileType.MODULE:
        saved = Path(setup_args.output_dir) / "config.yaml"
        if saved.exists():
            return deserialize_config_dict(saved)
        # Fallback: re-parse the .py module without instantiating anything.
        import importlib

        from cosmos3.common.config import unstructure_config
        from cosmos3._src.imaginaire.utils import config_helper

        config_module = importlib.import_module(config_helper.get_config_module(setup_args.config_file))
        config = config_module.make_config()
        config = config_helper.override(
            config, ["--", f"experiment={setup_args.experiment}", *setup_args.experiment_overrides]
        )
        return unstructure_config(config, invalid="ignore")

    return deserialize_config_dict(Path(setup_args.config_file))


class _ActionPolicyServerSetupArgs(OmniSetupArgs):
    """Server-local setup args that avoid instantiating runtime EMA for frozen configs."""

    def load_model_config_dict(self) -> dict:
        model_dict = super().load_model_config_dict()
        model_dict.setdefault("config", {}).setdefault("ema", {})["enabled"] = False
        return model_dict


def _disable_runtime_ema_for_frozen_config(setup_args: OmniSetupArgs) -> OmniSetupArgs:
    """Use server-local setup args to instantiate frozen configs without runtime EMA."""
    if setup_args.config_file_type == ConfigFileType.MODULE:
        return setup_args

    log.info("[action-server] disabled runtime EMA for frozen config model load")
    return _ActionPolicyServerSetupArgs.model_validate(setup_args.model_dump())


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


class ActionServerArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    # We deliberately do NOT expose the full ``OmniSetupOverrides`` (i.e. no
    # ``setup: SetupOverrides`` field). Doing so under ``tyro.conf.OmitArgPrefixes``
    # would surface the same flag names twice (once as ``--config-file`` from
    # ``setup.config_file`` and once as the action-server shortcut) plus expose
    # every ``setup.sample_overrides.*`` knob (``--guidance``, ``--seed``,
    # ``--num-steps``, ...) which collides with our own server-level defaults.
    # Instead we keep a flat arg model and build ``OmniSetupOverrides``
    # programmatically in ``build_setup_overrides``.
    #
    # All flags follow the cosmos3 convention: hyphenated CLI names auto-derived
    # from the underscored field name (``checkpoint_dir`` -> ``--checkpoint-dir``).

    checkpoint_dir: str = ""
    """Path to checkpoint directory (local path or ``s3://`` URI). Maps to
    ``OmniSetupOverrides.checkpoint_path``."""

    config_file: str = "cosmos3/_src/vfm/configs/base/config.py"
    """Hydra config: a ``.py`` config-store module, a ``.yaml`` frozen Hydra
    config (e.g. the one written next to a ``cosmos3.scripts.train`` checkpoint),
    or a Hugging Face ``config.json``. ``OmniInference.create`` auto-dispatches
    on the file extension."""

    experiment_name: str = ""
    """Hydra experiment name (only required for ``.py`` configs; ignored when
    ``--config-file`` is a ``.yaml``/``.json`` frozen config)."""

    experiment_overrides: list[str] = pydantic.Field(default_factory=list)
    """Extra Hydra dotlist overrides to apply to the experiment config."""

    credential_path: str = ""
    """Path to S3 credentials file for remote checkpoint loading."""

    local_cache_dir: Path | None = None
    """Directory for caching S3 checkpoints. Maps to
    ``OmniSetupOverrides.checkpoint_cache_dir``."""

    output_dir: Path | None = None
    """Output directory for ``OmniInference`` (saved config.yaml, benchmarks).
    Defaults to ``--dump-dir`` if set, else ``/tmp/cosmos3_action_server``."""

    # ----- single-rank parallelism / sampler ----------------------------------
    sampler: Literal["unipc", "edm"] = "unipc"
    """Diffusion sampler used by ``OmniInference``."""

    # ----- sampling defaults (per-request, used when client doesn't override) --
    seed: int = 0
    """Random seed for ``model.generate_samples_from_batch``."""
    guidance: float = 1.0
    """Guidance scale for denoising."""
    num_steps: int = 30
    """Number of denoising steps."""
    fps: int = 20
    """Frames per second used for both prompt augmentation and rollout encoding."""

    # ----- action policy parameters -------------------------------------------
    action_chunk_size: int | None = None
    """Number of action steps to predict. Defaults to ``chunk_length`` /
    ``num_action_per_chunk`` from the experiment config (or 16)."""
    max_action_dim: int | None = None
    """Maximum action dimension. Defaults to ``model.config.max_action_dim``
    from the experiment config (or 64)."""
    raw_action_dim: int | None = None
    """Unpadded action dimension used for action-channel masking. Inferred
    from action stats when omitted."""

    # ----- action denormalization ---------------------------------------------
    action_stats_path: Path | None = None
    """Path to action stats JSON for denormalizing predicted actions."""
    action_normalization: ActionNormalization = "auto"
    """Action normalization to invert. ``auto`` reads ``action_normalization``
    from the experiment config (default ``minmax`` if unspecified)."""

    # ----- debug dumps --------------------------------------------------------
    dump_dir: Path | None = None
    """If set, dump observations, predicted actions, and rollout videos under
    this directory for offline debugging."""
    dump_every: int = 1
    """Dump every N-th request (only used when ``--dump-dir`` is set)."""

    # ----- HTTP server --------------------------------------------------------
    host: str = "0.0.0.0"
    """HTTP host to bind."""
    port: int = 8000
    """HTTP port to bind."""
    http_400_on_error: bool = False
    """If set, return HTTP 400 on inference errors. Default is HTTP 200 with an
    empty action list, matching the legacy simulator client expectations."""

    # ----- developer utilities ------------------------------------------------
    run_validation: bool = False
    """If set, run a one-shot validation/training batch through the model on
    startup (developer debugging only)."""

    def build_setup_overrides(self) -> OmniSetupOverrides:
        """Build an ``OmniSetupOverrides`` from the flat CLI fields.

        Required fields (``checkpoint_path``) must be present; optional fields
        keep their ``OmniSetupOverrides`` defaults when not specified by the
        user.
        """
        if not self.checkpoint_dir:
            raise ValueError("--checkpoint-dir is required")

        output_dir = self.output_dir or self.dump_dir or _DEFAULT_FALLBACK_OUTPUT_DIR

        # ``model_construct`` skips validation; we set fields explicitly so
        # downstream ``build_setup`` validators run on a fully-populated object.
        base = OmniSetupOverrides.model_construct()
        base.checkpoint_path = self.checkpoint_dir
        base.config_file = self.config_file
        base.experiment = self.experiment_name
        base.experiment_overrides = list(self.experiment_overrides)
        if self.credential_path:
            base.credential_path = self.credential_path
        if self.local_cache_dir is not None:
            base.checkpoint_cache_dir = self.local_cache_dir
        base.output_dir = output_dir
        base.sampler = self.sampler
        return base


# ---------------------------------------------------------------------------
# Service implementation (predict path is a verbatim port from the previous
# evaluation/action/http_inference_server.py).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionServerConfig:
    """Internal snapshot of the resolved CLI args, threaded through the service.

    Kept as a plain dataclass (vs. carrying ``ActionServerArgs`` directly) so
    request-time code can ``replace(...)`` individual fields after model load.
    """

    seed: int
    guidance: float
    num_steps: int
    fps: int
    action_chunk_size: int
    max_action_dim: int
    raw_action_dim: int | None
    dump_dir: Path | None
    dump_every: int
    http_400_on_error: bool
    action_stats_path: Path | None
    action_normalization: ActionNormalization
    experiment_name: str
    checkpoint_dir: str


class ActionModelService:
    def __init__(self, args: ActionServerArgs) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for OmniMoTModel inference in this repo.")

        # OmniInference internally calls into FSDP / DTensor parallelize utilities
        # that expect a process group; create a single-rank PG when not under
        # torchrun (init_script() only inits the PG when WORLD_SIZE>1).
        _maybe_init_distributed()

        setup_overrides = args.build_setup_overrides()
        setup_args = setup_overrides.build_setup()
        init_output_dir(setup_args.output_dir)
        setup_args = _disable_runtime_ema_for_frozen_config(setup_args)

        # Surface the resolved max_action_dim into the experiment config when
        # the user explicitly overrode it on the CLI; matches the previous
        # ``experiment_opts=[f"model.config.max_action_dim={...}"]`` plumbing.
        if args.max_action_dim is not None:
            setup_args.experiment_overrides = [
                *setup_args.experiment_overrides,
                f"model.config.max_action_dim={int(args.max_action_dim)}",
            ]

        log.info(
            f"[action-server] loading model: config_file='{setup_args.config_file}' "
            f"({setup_args.config_file_type}) experiment='{setup_args.experiment}' "
            f"checkpoint_path='{setup_args.checkpoint_path}'"
        )

        # OmniInference dispatches between MODULE (.py) and YAML/JSON loaders.
        pipe = OmniInference.create(setup_args)
        self.pipe: OmniInference = pipe
        self.model = pipe.model
        self.model.eval()
        # OmniInference always uses OmniSetupArgs at runtime, but the base class
        # types the attribute as the more general SetupArgs.
        assert isinstance(pipe.setup_args, OmniSetupArgs)
        self.setup_args: OmniSetupArgs = pipe.setup_args
        self.experiment_config: dict = _load_introspection_config_dict(self.setup_args)

        # Resolve action_chunk_size: CLI arg > experiment config > default.
        if args.action_chunk_size is not None:
            resolved_chunk_size = int(args.action_chunk_size)
        else:
            config_chunk = _extract_chunk_length_from_config(self.experiment_config)
            if config_chunk is not None:
                resolved_chunk_size = config_chunk
                log.info(
                    f"[action-server] --action-chunk-size not specified, "
                    f"using chunk_length={resolved_chunk_size} from experiment config"
                )
            else:
                resolved_chunk_size = _DEFAULT_ACTION_CHUNK_SIZE
                log.info(
                    f"[action-server] --action-chunk-size not specified and not found in experiment config, "
                    f"using default={resolved_chunk_size}"
                )

        # Resolve max_action_dim: CLI > model config > default (64).
        if args.max_action_dim is not None:
            resolved_max_action_dim = int(args.max_action_dim)
        else:
            model_max_action_dim = getattr(self.model.config, "max_action_dim", None)
            if isinstance(model_max_action_dim, int):
                resolved_max_action_dim = model_max_action_dim
                log.info(
                    f"[action-server] --max-action-dim not specified, "
                    f"using max_action_dim={resolved_max_action_dim} from model config"
                )
            else:
                resolved_max_action_dim = 64
                log.info(
                    f"[action-server] --max-action-dim not specified and not found in model config, "
                    f"using default={resolved_max_action_dim}"
                )

        self.cfg = ActionServerConfig(
            seed=int(args.seed),
            guidance=float(args.guidance),
            num_steps=int(args.num_steps),
            fps=int(args.fps),
            action_chunk_size=resolved_chunk_size,
            max_action_dim=resolved_max_action_dim,
            raw_action_dim=int(args.raw_action_dim) if args.raw_action_dim is not None else None,
            dump_dir=args.dump_dir,
            dump_every=int(args.dump_every),
            http_400_on_error=bool(args.http_400_on_error),
            action_stats_path=args.action_stats_path,
            action_normalization=args.action_normalization,
            experiment_name=setup_args.experiment or "",
            checkpoint_dir=setup_args.checkpoint_path,
        )

        self._lock = threading.Lock()
        self._req_id_lock = threading.Lock()
        self._req_id = 0

        self.append_duration_fps = _extract_bool_from_config(
            self.experiment_config, "append_duration_fps", default=True
        )
        self.append_resolution_info = _extract_bool_from_config(
            self.experiment_config, "append_resolution_info", default=True
        )
        log.info(
            f"[action-server] prompt augmentation: "
            f"append_duration_fps={self.append_duration_fps}, append_resolution_info={self.append_resolution_info}"
        )

        # Action denormalization stats.
        self.action_min: torch.Tensor | None = None
        self.action_range: torch.Tensor | None = None
        self.action_mean: torch.Tensor | None = None
        self.action_std: torch.Tensor | None = None
        self.action_normalization: ResolvedActionNormalization = "minmax"
        self.raw_action_dim: int | None = self.cfg.raw_action_dim
        self._load_action_normalization_stats()
        if self.raw_action_dim is None:
            self.raw_action_dim = 7

        if args.run_validation:
            self._run_developer_validation()

    # ------------------------------------------------------------------
    # Action denormalization
    # ------------------------------------------------------------------

    def _load_action_normalization_stats(self) -> None:
        """Load action denormalization tensors from ``cfg.action_stats_path``.

        Populates ``self.action_normalization``, the relevant tensor attributes
        (``action_mean`` / ``action_std`` for meanstd; ``action_min`` /
        ``action_range`` for minmax / quantile / quantile_rot), and infers
        ``self.raw_action_dim`` when not explicitly set.
        """
        if self.cfg.action_stats_path is None:
            return

        self.action_normalization = self._resolve_action_normalization(self.cfg.action_normalization)
        stats_path = Path(self.cfg.action_stats_path)
        if not stats_path.is_absolute():
            stats_path = Path.cwd() / stats_path
        with open(stats_path) as f:
            raw_stats = json.load(f)
        if not isinstance(raw_stats, dict):
            raise ValueError(f"Action stats file must contain a dict: {stats_path}")
        stats_key = "global_raw" if self.action_normalization == "quantile_rot" else "global"
        stats = raw_stats.get(stats_key, raw_stats)
        if not isinstance(stats, dict):
            raise ValueError(f"Action stats file must contain a dict or {stats_key} stats dict: {stats_path}")
        if self.action_normalization == "meanstd":
            if "mean" not in stats or "std" not in stats:
                raise ValueError(f"Mean/std action normalization requires 'mean' and 'std' in {stats_path}")
            self.action_mean = torch.tensor(stats["mean"], dtype=torch.float32)  # [D]
            action_std = torch.tensor(stats["std"], dtype=torch.float32)  # [D]
            self.action_std = torch.clamp(action_std, min=1e-8)  # [D]
            stats_dim = int(self.action_mean.shape[0])
            stats_summary = f"mean={self.action_mean.tolist()}, std={self.action_std.tolist()}"
        elif self.action_normalization in ("quantile", "quantile_rot"):
            if "q01" not in stats or "q99" not in stats:
                raise ValueError(f"Quantile action normalization requires 'q01' and 'q99' in {stats_path}")
            self.action_min = torch.tensor(stats["q01"], dtype=torch.float32)  # [D]
            action_max = torch.tensor(stats["q99"], dtype=torch.float32)  # [D]
            action_range = action_max - self.action_min  # [D]
            self.action_range = torch.clamp(action_range, min=1e-6)  # [D]
            stats_dim = int(self.action_min.shape[0])
            stats_summary = f"q01={self.action_min.tolist()}, q99={action_max.tolist()}"
        else:
            if "min" not in stats or "max" not in stats:
                raise ValueError(f"Min/max action normalization requires 'min' and 'max' in {stats_path}")
            self.action_min = torch.tensor(stats["min"], dtype=torch.float32)  # [D]
            action_max = torch.tensor(stats["max"], dtype=torch.float32)  # [D]
            action_range = action_max - self.action_min  # [D]
            self.action_range = torch.clamp(action_range, min=1e-6)  # [D]
            stats_dim = int(self.action_min.shape[0])
            stats_summary = f"min={self.action_min.tolist()}, max={action_max.tolist()}"
        if self.raw_action_dim is None:
            self.raw_action_dim = stats_dim
        if stats_dim != self.raw_action_dim:
            raise ValueError(f"Action stats dimension {stats_dim} does not match raw_action_dim={self.raw_action_dim}")
        log.info(
            f"[action-server] Loaded action stats for denormalization from {stats_path}: "
            f"normalization={self.action_normalization}, {stats_summary}"
        )

    def _resolve_action_normalization(
        self, requested_normalization: ActionNormalization
    ) -> ResolvedActionNormalization:
        """Resolve auto action normalization from the loaded experiment config."""
        if requested_normalization != "auto":
            return requested_normalization

        configured_normalization = _extract_str_from_config(self.experiment_config, "action_normalization")
        if configured_normalization is None:
            return "minmax"
        if configured_normalization in ("meanstd", "minmax", "quantile", "quantile_rot"):
            return configured_normalization  # type: ignore[return-value]
        raise ValueError(
            "action_policy_server.py can denormalize action_normalization='minmax', 'meanstd', "
            "'quantile', or 'quantile_rot'; "
            f"loaded experiment config requested {configured_normalization!r}. "
            "Pass --action-normalization explicitly if this checkpoint should use a supported method."
        )

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Invert the configured action normalization."""
        if self.action_normalization == "meanstd":
            if self.action_mean is None or self.action_std is None:
                return action
            action_dim = self.action_mean.shape[0]
            normalized = action[..., :action_dim]  # [...,D]
            action_mean = self.action_mean.to(action.device)  # [D]
            action_std = self.action_std.to(action.device)  # [D]
            return normalized * action_std + action_mean  # [...,D]

        if self.action_min is None or self.action_range is None:
            return action
        action_dim = self.action_min.shape[0]
        normalized = action[..., :action_dim]  # [...,D]
        action_min = self.action_min.to(action.device)  # [D]
        action_range = self.action_range.to(action.device)  # [D]
        return (normalized + 1.0) / 2.0 * action_range + action_min  # [...,D]

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _should_dump(self, request_id: int) -> bool:
        """Decide whether to dump this request.

        We always dump the first request (request_id == 1) as a quick sanity
        check that dumping is wired up, then dump every N-th request controlled
        by ``dump_every``.
        """
        if self.cfg.dump_dir is None:
            return False
        n = int(self.cfg.dump_every)
        if n <= 0:
            return False
        return request_id == 1 or (request_id % n == 0)

    def get_info(self) -> dict[str, Any]:
        """Return model / server info for the /info endpoint.

        Includes all runtime-relevant config so clients can record reproducible
        params.json without needing to know CLI flags.
        """
        return {
            "run_name": self.cfg.experiment_name,
            "checkpoint": self.cfg.checkpoint_dir,
            "config_file": str(self.setup_args.config_file),
            "config_file_type": str(self.setup_args.config_file_type),
            "guidance": self.cfg.guidance,
            "num_steps": self.cfg.num_steps,
            "fps": self.cfg.fps,
            "seed": self.cfg.seed,
            "action_chunk_size": self.cfg.action_chunk_size,
            "max_action_dim": self.cfg.max_action_dim,
            "raw_action_dim": self.cfg.raw_action_dim,
            "action_stats_path": str(self.cfg.action_stats_path) if self.cfg.action_stats_path else None,
        }

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_policy(self, req: dict[str, Any]) -> dict[str, Any]:
        """
        Run policy inference: given an observation image and prompt, predict actions.

        Input request format:
        {
            "image": "<base64_encoded_png>",
            "prompt": "<task_description>",
            "domain_name": "<domain_name>",
            "image_size": <int>
        }

        Output format:
        {
            "action": [[a0_0, a0_1, ...], ..., [aN_0, aN_1, ...]],
            "video": ["<base64_png>", ...]  # List of T base64-encoded PNG frames
        }

        All action dimensions are returned. Video is the decoded predicted rollout as base64 PNGs.
        """
        t0 = time.monotonic()

        # Get or assign request ID
        injected_id = req.get("request_id", None)
        if isinstance(injected_id, int) and injected_id > 0:
            request_id = int(injected_id)
        else:
            with self._req_id_lock:
                self._req_id += 1
                request_id = int(self._req_id)

        # Validate request
        image_b64 = req.get("image")
        if not isinstance(image_b64, str):
            raise ValueError("'image' must be a base64 string")

        prompt = req.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")

        domain_name = req.get("domain_name")
        if not isinstance(domain_name, str):
            raise ValueError("'domain_name' must be a string")

        image_size = req.get("image_size")
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("'image_size' must be a positive integer")

        # Decode image
        t_decode0 = time.monotonic()
        img_chw_uint8 = _decode_base64_png_to_rgb_uint8(image_b64)
        img_h, img_w = img_chw_uint8.shape[-2:]

        # Handle resizing: for multi-view (non-square) images, scale proportionally
        # to maintain aspect ratio while matching height to image_size
        if img_h != image_size:
            # Calculate new width to maintain aspect ratio
            scale = image_size / img_h
            new_w = int(round(img_w * scale))
            hwc = img_chw_uint8.permute(1, 2, 0).cpu().numpy()  # [H,W,3]
            resized = Image.fromarray(hwc).resize((new_w, image_size), resample=Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.uint8).copy()
            img_chw_uint8 = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]  # [3,H,W]
        t_decode1 = time.monotonic()

        # Construct batch in IterativeJointDataLoader format (list-of-lists for multi-item keys)
        t_frames = self.cfg.action_chunk_size + 1
        _, final_h, final_w = img_chw_uint8.shape
        video_c_t_h_w_uint8 = img_chw_uint8.unsqueeze(1).repeat(1, t_frames, 1, 1)  # [3,T,H,W]

        # Apply reflection padding to match closest predefined resolution
        resolution = get_vision_data_resolution((final_h, final_w))
        target_w, target_h = find_closest_target_size(final_h, final_w, resolution)
        pad_dict: dict[str, Any] = {"video": video_c_t_h_w_uint8}
        reflection_pad_to_target(pad_dict, ["video"], True, target_w, target_h)
        video_padded = pad_dict["video"]  # (C, T, target_h, target_w)
        padded_image_size = pad_dict["image_size"]  # (4,)

        # Action: zeros tensor as noise starting point for policy mode
        action_t_d = torch.zeros(
            (self.cfg.action_chunk_size, self.cfg.max_action_dim),
            dtype=torch.float32,
        )  # [T,action_dim]

        input_video_key = getattr(self.model, "input_video_key", None)
        if input_video_key is None:
            input_video_key = getattr(self.model, "config", None).input_video_key  # type: ignore[union-attr]

        sequence_plan = build_sequence_plan_from_mode(
            mode="policy",
            video_length=self.cfg.action_chunk_size + 1,
            action_length=self.cfg.action_chunk_size,
            has_text=True,
        )

        augmented_prompt = _augment_prompt_with_metadata(
            prompt,
            t_frames=t_frames,
            fps=self.cfg.fps,
            height=final_h,
            width=final_w,
            append_duration_fps=self.append_duration_fps,
            append_resolution_info=self.append_resolution_info,
        )

        batch: dict[str, Any] = {
            input_video_key: [[video_padded]],
            "raw_action_dim": [torch.tensor(self.raw_action_dim, dtype=torch.long)],
            "action": [[action_t_d]],
            "mode": ["policy"],
            "ai_caption": [augmented_prompt],
            "prompt": [augmented_prompt],
            "conditioning_fps": [torch.tensor(self.cfg.fps, dtype=torch.long)],
            "image_size": padded_image_size.unsqueeze(0).to(device="cuda"),
            "domain_id": [torch.tensor(get_domain_id(domain_name), dtype=torch.long)],
            "sequence_plan": [sequence_plan],
        }

        if getattr(self.model, "training", False):
            log.warning(f"[action-server] request_id={request_id} WARNING: model.training=True")

        log.info(
            f"[action-server] request_id={request_id} mode=policy "
            f"prompt={augmented_prompt!r} domain_name={domain_name!r} image_size={image_size} "
            f"img={tuple(img_chw_uint8.shape)} steps={self.cfg.num_steps} guidance={self.cfg.guidance}"
        )

        # Run inference
        t_inf0 = time.monotonic()
        with self._lock:
            with torch.inference_mode():
                samples = self.model.generate_samples_from_batch(
                    batch,
                    guidance=self.cfg.guidance,
                    seed=[self.cfg.seed],
                    num_steps=self.cfg.num_steps,
                    has_negative_prompt=False,
                )
                pred_action = samples["action"][0]  # [T,D] or [1,T,D]

                # Decode vision for rollout video (samples["vision"] is a list; take first sample)
                pred_video_c_t_h_w = self.model.decode(samples["vision"][0]).squeeze(0)  # [C,T,H,W]

                # Remove reflection padding so the reported video matches the original resolution
                pred_video_c_t_h_w = remove_reflection_padding(pred_video_c_t_h_w, padded_image_size)
        t_inf1 = time.monotonic()

        # Extract actions: return all dimensions — (T, D) or (1, T, D)
        pred_action = pred_action.float().squeeze(0)  # [T,D]
        pred_action = self._denormalize_action(pred_action)
        pred_action_np = pred_action.detach().cpu().numpy()  # [T,D]
        pred_action_list = pred_action_np.tolist()  # List of [a0, a1, ..., aD]

        # Convert video to base64-encoded PNG frames
        pred_video_frames = _video_tensor_to_pil_images(pred_video_c_t_h_w)
        pred_video_b64: list[str] = []
        for frame in pred_video_frames:
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            pred_video_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

        # Optional offline debug dump
        if self._should_dump(request_id):
            dump_dir = self.cfg.dump_dir
            assert dump_dir is not None
            dump_root = Path(dump_dir)
            dump_root.mkdir(parents=True, exist_ok=True)
            try:
                log.info(f"[action-server] request_id={request_id} dumping to {str(dump_root)}")
                _save_policy_request_dump(
                    dump_root=dump_root,
                    request_id=request_id,
                    request_json=req,
                    obs_chw_uint8=img_chw_uint8,
                    pred_action=pred_action_list,
                    pred_video_c_t_h_w=pred_video_c_t_h_w,
                    fps=int(self.cfg.fps),
                )
            except Exception as e:
                # Never fail serving a request due to dump failures
                log.error(f"[action-server] dump failed for request_id={request_id}: {e}")

        dt_total_ms = (time.monotonic() - t0) * 1000.0
        dt_decode_ms = (t_decode1 - t_decode0) * 1000.0
        dt_inf_ms = (t_inf1 - t_inf0) * 1000.0
        log.info(
            f"[action-server] request_id={request_id} done action_steps={len(pred_action_list)} "
            f"video_frames={len(pred_video_b64)} "
            f"ms_total={dt_total_ms:.1f} ms_decode={dt_decode_ms:.1f} ms_infer={dt_inf_ms:.1f}"
        )
        return {"action": pred_action_list, "video": pred_video_b64}

    # ------------------------------------------------------------------
    # Developer validation (optional, --run-validation)
    # ------------------------------------------------------------------

    def _run_developer_validation(self) -> None:
        """Run a single validation/training batch through the model on startup."""
        # Re-instantiate config so we can spin up dataloaders without the
        # inference-time freezes applied to the model config.
        if self.setup_args.config_file_type != ConfigFileType.MODULE:
            log.warning(
                "[action-server] --run-validation requires a .py config-file (got "
                f"{self.setup_args.config_file_type}); skipping."
            )
            return

        try:
            config = self.setup_args.load_config()
        except Exception as e:
            log.warning(f"[action-server] --run-validation could not load config: {e}; skipping.")
            return

        try:
            val_dataset = instantiate(config.dataloader_val)
            train_dataset = instantiate(config.dataloader_train)
        except Exception as e:
            log.warning(f"[action-server] --run-validation could not instantiate datasets: {e}; skipping.")
            return

        val_batch = next(iter(val_dataset))  # pyrefly: ignore[no-matching-overload]
        train_batch = next(iter(train_dataset))  # pyrefly: ignore[no-matching-overload]

        with torch.inference_mode():
            self.model.training_step(val_batch, 0)
            sample_num = 1
            val_result = self.model.generate_samples_from_batch(
                val_batch,
                guidance=1.0,
                seed=[0 for _ in range(sample_num)],
                num_steps=8,
                n_sample=sample_num,
            )
            video_mse_list = []
            action_mse_list = []
            for i in range(sample_num):
                val_video = self.model.decode(val_result["vision"][i]).detach().cpu()
                val_video_mse = torch.nn.functional.mse_loss(val_video, val_batch["video"][i].cpu())
                val_action = val_result["action"][i].detach().cpu()
                val_action_mse = torch.nn.functional.mse_loss(val_action[:, :6], val_batch["action"][i][0][:, :6].cpu())
                video_mse_list.append(val_video_mse.item())
                action_mse_list.append(val_action_mse.item())
            log.info(f"Val video MSE: {np.mean(video_mse_list)}")
            log.info(f"Val action MSE: {np.mean(action_mse_list)}")

            self.model.training_step(train_batch, 0)
            train_result = self.model.generate_samples_from_batch(
                train_batch,
                guidance=1.0,
                seed=list(range(20)),
                num_steps=8,
                n_sample=20,
            )
            train_video = self.model.decode(train_result["vision"][0])
            train_action = train_result["action"][0]
            train_action_mse = torch.nn.functional.mse_loss(train_action[:, :6], train_batch["action"][0][:, :6])
            train_video_mse = torch.nn.functional.mse_loss(train_video, train_batch["video"][0])
            log.info(f"Train action MSE: {train_action_mse}; Train video MSE: {train_video_mse}")


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class _ActionHandler(BaseHTTPRequestHandler):
    """
    ThreadingHTTPServer handler.

    The service instance is injected via ``server.service``.
    """

    server: ThreadingHTTPServer  # type: ignore[assignment]

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        # Avoid caches/proxies returning stale results.
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Client closed the connection (often due to request timeout).
            # Avoid noisy tracebacks; nothing to do server-side.
            return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/info":
            service: ActionModelService = getattr(self.server, "service")  # type: ignore[attr-defined]
            self._send_json(200, service.get_info())
        elif self.path == "/":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/", "/predict"):
            self._send_json(404, {"error": "Not found"})
            return

        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._send_json(415, {"error": "Content-Type must be application/json"})
            return

        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return

        body = self.rfile.read(max(0, length))
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return

        if not isinstance(req, dict):
            self._send_json(400, {"error": "JSON body must be an object"})
            return

        service: ActionModelService = getattr(self.server, "service")  # type: ignore[attr-defined]

        # Generate a per-request id at the HTTP layer to correlate logs and dumps.
        req_id_lock: threading.Lock | None = getattr(self.server, "_req_id_lock", None)  # type: ignore[attr-defined]
        if req_id_lock is None:
            req_id_lock = threading.Lock()
            setattr(self.server, "_req_id_lock", req_id_lock)  # type: ignore[attr-defined]
            setattr(self.server, "_req_id", 0)  # type: ignore[attr-defined]
        with req_id_lock:
            next_id = int(getattr(self.server, "_req_id")) + 1  # type: ignore[attr-defined]
            setattr(self.server, "_req_id", next_id)  # type: ignore[attr-defined]
        req["request_id"] = next_id

        log.info(
            f"[action-server] HTTP request_id={next_id} from={self.client_address[0]}:{self.client_address[1]} "
            f"path={self.path} bytes={length}"
        )

        try:
            out = service.predict_policy(req)
        except Exception as e:
            err = str(e)
            traceback.print_exc()

            payload = {"action": [], "error": err, "request_id": req.get("request_id")}
            log.error(f"[action-server] request_id={req.get('request_id')} ERROR: {err}")

            # Dump failed request for offline debugging if enabled.
            if service.cfg.dump_dir is not None:
                try:
                    dump_root = Path(service.cfg.dump_dir)
                    dump_root.mkdir(parents=True, exist_ok=True)
                    _save_failed_request_dump(
                        dump_root=dump_root,
                        request_id=int(req.get("request_id") or 0),
                        request_json=req,
                        error=err,
                    )
                except Exception:
                    pass

            status = 400 if service.cfg.http_400_on_error else 200
            self._send_json(status, payload)
            return

        self._send_json(200, out)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Silence default request logging (the simulator can be chatty).
        return


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def serve(args: ActionServerArgs) -> None:
    if args.dump_dir is not None:
        # Create dump dir up front so it's obvious where outputs will go.
        dump_root = Path(args.dump_dir).resolve()
        dump_root.mkdir(parents=True, exist_ok=True)
        log.info(f"[action-server] dump_root={str(dump_root)} dump_every={args.dump_every}")

    service = ActionModelService(args)

    local_ip = _get_local_ip()
    log.info(
        f"[action-server] starting host={args.host} port={int(args.port)} "
        f"experiment_name={service.cfg.experiment_name!r} "
        f"steps={service.cfg.num_steps} guidance={service.cfg.guidance} fps={service.cfg.fps} "
        f"action_chunk_size={service.cfg.action_chunk_size} max_action_dim={service.cfg.max_action_dim} "
        f"raw_action_dim={service.cfg.raw_action_dim} "
        f"dump_dir={service.cfg.dump_dir} dump_every={service.cfg.dump_every} "
        f"http_400_on_error={service.cfg.http_400_on_error}"
    )
    log.info(f"[action-server] Server accessible at: http://{local_ip}:{int(args.port)}/")
    log.info("[action-server] Endpoints:")
    log.info("  - GET  /       : Health check")
    log.info("  - GET  /info   : Model info (run_name, checkpoint, sampling params)")
    log.info("  - POST /predict: Policy inference (image + prompt + domain_name + image_size -> action)")

    httpd: ThreadingHTTPServer = ThreadingHTTPServer((args.host, int(args.port)), _ActionHandler)
    setattr(httpd, "service", service)
    httpd.serve_forever()


def main() -> None:
    args = tyro_cli(
        ActionServerArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    serve(args)


if __name__ == "__main__":
    main()
