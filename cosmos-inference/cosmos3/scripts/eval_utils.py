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

"""Helpers for `eval.py`: per-sample metric computation and aggregation."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos3.common.args import SampleOutputs
from cosmos3.vision import read_media_frames
from cosmos3._src.vfm.datasets.action.transforms import remove_reflection_padding
from cosmos3._src.vfm.evaluation.action.metrics import compute_action_mse, compute_psnr

VIDEO_MODES = {"forward_dynamics"}
ACTION_MODES = {"inverse_dynamics"}
BOTH_MODES = {"policy"}
ALL_MODES = VIDEO_MODES | ACTION_MODES | BOTH_MODES


def extract_gt_video(data_batch: dict) -> torch.Tensor | None:
    """Snapshot the GT video as (C, T, H, W) uint8, trimmed to its content region if padded.

    Must be called BEFORE the inference pipeline runs — the model normalizes
    `data_batch["video"]` in place from uint8 [0, 255] to float [-1, 1].
    """
    video = data_batch.get("video")
    if video is None:
        return None
    gt_video = video[0].detach().clone()
    image_size = data_batch.get("image_size")
    if image_size is not None:
        gt_video = remove_reflection_padding(gt_video, image_size[0])
    return gt_video


def extract_gt_action(data_batch: dict) -> torch.Tensor | None:
    """Snapshot the GT action as a (T, D) float32 tensor, or None when absent."""
    action = data_batch.get("action", [None])[0]
    if action is None:
        return None

    raw_action_dim = data_batch.get("raw_action_dim", [None])[0]
    if raw_action_dim is not None:
        # If raw_action_dim is provided, it indicates that the GT action has been padded to a larger size.
        # We trim the action to its original dimension before returning it.
        raw_action_dim = int(raw_action_dim.item())  # remove batch dim and convert to int
        assert action.shape[-1] >= raw_action_dim, (
            f"invalid raw_action_dim={raw_action_dim} for action with shape {action.shape}"
        )
        action = action[..., :raw_action_dim]

    return action.detach().clone().float()


def _parse_mode_from_name(name: str) -> str:
    parts = name.split("/")
    if len(parts) < 2:
        raise ValueError(f"unexpected sample name: {name!r}")
    mode = parts[-2]
    if mode not in ALL_MODES:
        raise ValueError(f"unexpected mode {mode!r} in sample name {name!r}; expected one of {sorted(ALL_MODES)}")
    return mode


def _compute_video_metrics(gt_video_cthw_uint8: torch.Tensor, pred_path: Path) -> dict[str, float]:
    # +1 so an over-long prediction surfaces as a shape mismatch instead of silent truncation.
    pred, _ = read_media_frames(pred_path, max_frames=gt_video_cthw_uint8.shape[1] + 1)
    # Match GT's spatial dims (top-left crop, mirroring remove_reflection_padding's convention)
    # so a reflection-padded GT trimmed to its content region can be compared against the
    # padded mp4 saved to disk.
    pred = pred[..., : gt_video_cthw_uint8.shape[-2], : gt_video_cthw_uint8.shape[-1]]
    if pred.shape != gt_video_cthw_uint8.shape:
        raise ValueError(
            f"video shape mismatch: gt {tuple(gt_video_cthw_uint8.shape)} vs pred {tuple(pred.shape)} ({pred_path})"
        )
    return {"psnr": compute_psnr(gt_video_cthw_uint8, pred)}


def _compute_action_metrics(gt_action_td: torch.Tensor, pred_action_list: list) -> dict[str, Any]:
    pred = torch.tensor(pred_action_list, dtype=torch.float32)
    if pred.shape != gt_action_td.shape:
        raise ValueError(f"action shape mismatch: gt {tuple(gt_action_td.shape)} vs pred {tuple(pred.shape)}")
    return {"action_mse": compute_action_mse(gt_action_td, pred)}


def compute_sample_metrics(
    name: str,
    gt_video_cthw: torch.Tensor | None,
    gt_action_td: torch.Tensor | None,
    sample_output: SampleOutputs,
    sample_dir: Path,
    vision_extension: str,
) -> dict[str, Any]:
    """Compute metrics for a single sample, dispatched by the mode parsed from `name`."""
    mode = _parse_mode_from_name(name)
    out: dict[str, Any] = {"mode": mode, "name": sample_dir.name}
    if mode in VIDEO_MODES | BOTH_MODES:
        if gt_video_cthw is None:
            raise ValueError(f"mode={mode!r} requires GT video but data_batch had none")
        out.update(_compute_video_metrics(gt_video_cthw, sample_dir / f"vision{vision_extension}"))
    if mode in ACTION_MODES | BOTH_MODES:
        pred_action = sample_output.outputs[0].content.get("action") if sample_output.outputs else None
        if pred_action is None:
            raise ValueError(f"mode={mode!r} requires predicted action but content has none")
        if gt_action_td is None:
            raise ValueError(f"mode={mode!r} requires GT action but data_batch had none")
        out.update(_compute_action_metrics(gt_action_td, pred_action))
    return out


def aggregate_metrics(output_dir: Path) -> dict[str, Any]:
    """Walk `output_dir` for per-sample `metrics.json` files and average per-mode/metric."""
    totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in output_dir.rglob("metrics.json"):
        m = json.loads(f.read_text())
        mode = m.pop("mode", None)
        m.pop("name", None)
        if mode is None:
            continue
        for k, v in m.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    totals[mode][f"{k}/{sub_k}"].append(float(sub_v))
            else:
                totals[mode][k].append(float(v))
    return {
        mode: {metric: {"mean": float(np.mean(vals)), "count": len(vals)} for metric, vals in metrics.items()}
        for mode, metrics in totals.items()
    }
