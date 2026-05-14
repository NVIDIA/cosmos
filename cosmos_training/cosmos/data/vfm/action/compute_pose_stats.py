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

"""Compute raw action translation and rotation statistics for camera and AV pose datasets.

Both camera and AV datasets produce actions via ``pose_abs_to_rel`` with
layout ``[translation(3), rotation(...)]``.

This script iterates dataset samples with ``translation_scale=1.0`` and
``rotation_scale=1.0`` to collect **raw** translation and rotation
values, then reports their distribution (mean, std, min, max,
percentiles) both globally and per-timestep for each block.  The
reported per-dim std is what directly matches MSE loss scale, so ratios
like ``std_translation / std_rotation`` are what you want to use to
pick ``translation_scale`` and ``rotation_scale`` so the two loss blocks
contribute comparably.

Usage:
    # Camera – backward_framewise (used in inverse_dynamics / policy)
    PYTHONPATH=. python cosmos/data/vfm/action/compute_pose_stats.py \
        --dataset camera --split train --pose-convention backward_framewise --max-samples 1000 --max-frames 17 --rotation-format axisangle

    PYTHONPATH=. python cosmos/data/vfm/action/compute_pose_stats.py \
        --dataset camera --split train --pose-convention backward_framewise --max-samples 1000 --max-frames 61 --rotation-format axisangle

    # AV – backward_framewise
    PYTHONPATH=. python cosmos/data/vfm/action/compute_pose_stats.py \
        --dataset av --split train --pose-convention backward_framewise --max-samples 1000 --max-frames 17 --rotation-format axisangle

    PYTHONPATH=. python cosmos/data/vfm/action/compute_pose_stats.py \
        --dataset av --split train --pose-convention backward_framewise --max-samples 1000 --max-frames 61 --rotation-format axisangle

    # Clip per-frame outliers automatically at P99 of each block's L2 norm.
    PYTHONPATH=. python cosmos/data/vfm/action/compute_pose_stats.py \
        --dataset camera --split train --pose-convention backward_framewise --max-samples 1000 --max-frames 61 \
        --rotation-format axisangle --max-trans-norm 10

    PYTHONPATH=. python cosmos/data/vfm/action/compute_pose_stats.py \
        --dataset av --split train --pose-convention backward_framewise --max-samples 1000 --max-frames 61 \
        --rotation-format axisangle --max-trans-norm 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from cosmos.data.vfm.action.pose_utils import RotationConvention

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "stats"

TRANSLATION_DIM = 3
PoseConvention = Literal[
    "backward_framewise",
    "backward_anchored",
]


# ---------------------------------------------------------------------------
# Welford accumulator
# ---------------------------------------------------------------------------
class WelfordAccumulator:
    """Welford's online algorithm for numerically stable mean/variance."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.min_val = np.full(dim, np.inf, dtype=np.float64)
        self.max_val = np.full(dim, -np.inf, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Update with a single sample (D,) or a batch (N, D)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        for sample in x:
            self.count += 1
            delta = sample - self.mean
            self.mean += delta / self.count
            delta2 = sample - self.mean
            self.m2 += delta * delta2
            self.min_val = np.minimum(self.min_val, sample)
            self.max_val = np.maximum(self.max_val, sample)

    def get_std(self) -> np.ndarray:
        if self.count < 2:
            return np.zeros(self.dim, dtype=np.float64)
        return np.sqrt(self.m2 / (self.count - 1))

    def as_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.get_std().tolist(),
            "min": self.min_val.tolist(),
            "max": self.max_val.tolist(),
            "count": self.count,
        }


# ---------------------------------------------------------------------------
# Dataset creation helpers
# ---------------------------------------------------------------------------
def _create_camera_dataset(
    split: str,
    rotation_format: RotationConvention,
    pose_convention: PoseConvention,
    credential_path: str,
    wdinfo_names: list[str] | None,
):
    from cosmos.data.vfm.action.camera_dataset_sharded import (
        CAMERA_WDINFOS,
        CameraDatasetSharded,
    )

    if wdinfo_names:
        wdinfo_paths = [CAMERA_WDINFOS[n] for n in wdinfo_names]
    else:
        wdinfo_paths = [CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"]]

    ds = CameraDatasetSharded(
        wdinfo_paths=wdinfo_paths,
        split=split,
        shuffle=False,
        fix_caption=True,
        mode="forward_dynamics",
        rotation_format=rotation_format,
        pose_convention=pose_convention,
        credential_path=credential_path,
        translation_scale=1.0,
        rotation_scale=1.0,
    )
    print(f"CameraDatasetSharded  wdinfos={wdinfo_names or ['pretrained_clips_260307_100k_filtered']}")
    return ds


def _create_av_dataset(
    split: str,
    rotation_format: RotationConvention,
    pose_convention: PoseConvention,
    credential_path: str,
    av_root: str,
    av_history_len: float,
    av_future_len: float,
    av_fps: int,
):
    from cosmos.data.vfm.action.av_dataset import AVDataset

    ds = AVDataset(
        root=av_root,
        split=split,
        fps=av_fps,
        mode="policy",
        history_len=av_history_len,
        future_len=av_future_len,
        rotation_format=rotation_format,
        pose_convention=pose_convention,
        credential_path=credential_path,
        shuffle=False,
        include_route_in_prompt=False,
        translation_scale=1.0,
        rotation_scale=1.0,
    )
    print(f"AVDataset  root={av_root}")
    return ds


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def _summarize_block(
    kept_values: list[np.ndarray],
    per_timestep_values: list[list[np.ndarray]],
    chunk_length: int,
    block_label: str,
) -> tuple[dict, dict]:
    """Build the per-dim and L2-norm summary dicts for one (already-filtered) action block."""
    percentiles = [5, 10, 25, 50, 75, 90, 95]

    concat = np.concatenate(kept_values, axis=0) if kept_values else np.zeros((0, 0), dtype=np.float64)
    if concat.size == 0:
        raise RuntimeError(f"No {block_label} frames left after filtering")

    dim = concat.shape[1]
    count = int(concat.shape[0])

    global_mean = concat.mean(axis=0).tolist()
    global_median = np.median(concat, axis=0).tolist()
    global_std = (concat.std(axis=0, ddof=1) if count > 1 else np.zeros(dim)).tolist()
    global_min = concat.min(axis=0).tolist()
    global_max = concat.max(axis=0).tolist()

    # Single-scalar summary from the flattened pool. This is the RMS per element
    # (about the pool mean) — the right quantity for picking one global scale
    # factor that preserves the vector's internal geometry.
    flat = concat.reshape(-1)
    flat_mean = float(flat.mean())
    flat_std = float(flat.std(ddof=1)) if flat.size > 1 else 0.0

    zero_dim = [0.0] * dim
    per_timestep_means: list[list[float]] = []
    per_timestep_medians: list[list[float]] = []
    per_timestep_stds: list[list[float]] = []
    per_timestep_mins: list[list[float]] = []
    per_timestep_maxs: list[list[float]] = []
    for t in range(chunk_length):
        vals = per_timestep_values[t]
        if not vals:
            per_timestep_means.append(zero_dim)
            per_timestep_medians.append(zero_dim)
            per_timestep_stds.append(zero_dim)
            per_timestep_mins.append(zero_dim)
            per_timestep_maxs.append(zero_dim)
            continue
        arr = np.stack(vals)
        per_timestep_means.append(arr.mean(axis=0).tolist())
        per_timestep_medians.append(np.median(arr, axis=0).tolist())
        per_timestep_stds.append((arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(dim)).tolist())
        per_timestep_mins.append(arr.min(axis=0).tolist())
        per_timestep_maxs.append(arr.max(axis=0).tolist())

    l2_norms = np.linalg.norm(concat, axis=-1)  # [N_kept]

    global_dict = {
        "mean": global_mean,
        "std": global_std,
        "min": global_min,
        "max": global_max,
        "count": count,
        "median": global_median,
        "flat_mean": flat_mean,
        "flat_std": flat_std,
    }

    scale_name = "translation_scale" if block_label == "translation" else "rotation_scale"
    raw_stats = {
        "description": (
            f"Per-dim statistics on raw action {block_label} block (translation_scale=1.0, rotation_scale=1.0). "
            f"Use flat_std (single scalar across all dims) to choose {scale_name} when you want a uniform "
            f"scale that preserves the block's internal geometry; use per-dim std when per-dim rescaling is acceptable."
        ),
        "global": global_dict,
        "per_timestep": {
            "mean": per_timestep_means,
            "median": per_timestep_medians,
            "std": per_timestep_stds,
            "min": per_timestep_mins,
            "max": per_timestep_maxs,
        },
    }
    l2_stats = {
        "description": f"L2 norm of raw {block_label} vectors across all frames.",
        "median": float(np.median(l2_norms)),
        "mean": float(np.mean(l2_norms)),
        "std": float(np.std(l2_norms, ddof=1)) if len(l2_norms) > 1 else 0.0,
        "min": float(np.min(l2_norms)),
        "max": float(np.max(l2_norms)),
        "percentiles": {str(p): float(np.percentile(l2_norms, p)) for p in percentiles},
    }
    return raw_stats, l2_stats


def compute_action_stats(
    dataset,
    max_samples: int | None = None,
    max_frames: int | None = None,
    max_trans_norm: float | None = None,
    max_rot_norm: float | None = None,
    percentile_clip: float | None = None,
) -> dict:
    """Iterate over *dataset* and collect raw action translation and rotation statistics.

    The dataset must be created with ``translation_scale=1.0`` and
    ``rotation_scale=1.0`` so that the returned actions contain unmodified
    translation and rotation values.

    If *max_frames* is given, each sample's action tensor is truncated to
    the first *max_frames* frames before statistics are accumulated.

    Outlier filtering (applied per-frame, not per-sample):
      * ``max_trans_norm``: drop frames whose translation L2 norm exceeds this value.
      * ``max_rot_norm``:   drop frames whose rotation    L2 norm exceeds this value.
      * ``percentile_clip``: if set (e.g. 99), the thresholds default to the P{n}
        of the corresponding L2-norm distribution from the data actually seen.
        Explicit ``max_*_norm`` arguments take precedence over this.

    Returns a dict ready for JSON serialisation.
    """
    all_translations: list[np.ndarray] = []
    all_rotations: list[np.ndarray] = []
    chunk_length: int | None = None
    rotation_dim: int | None = None
    sample_count = 0
    start = time.time()

    pbar = tqdm(desc="Reading action tensors", unit="samples")
    for sample in dataset:
        action = sample["action"]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if max_frames is not None:
            action = action[:max_frames]

        trans = action[:, :TRANSLATION_DIM].astype(np.float64)  # [T, 3]
        rot = action[:, TRANSLATION_DIM:].astype(np.float64)  # [T, D_rot]
        all_translations.append(trans)
        all_rotations.append(rot)

        if rotation_dim is None:
            rotation_dim = rot.shape[1]
        if chunk_length is None:
            chunk_length = trans.shape[0]
            print(
                f"  action shape : {action.shape}  (first {TRANSLATION_DIM} dims = translation, next {rotation_dim} = rotation)"
            )
            print(f"  chunk_length : {chunk_length}")
            print(f"  rotation_dim : {rotation_dim}")

        sample_count += 1
        pbar.update(1)
        if sample_count % 1000 == 0:
            elapsed = time.time() - start
            pbar.set_postfix(rate=f"{sample_count / elapsed:.1f} s/s")

        if max_samples is not None and sample_count >= max_samples:
            print(f"\nReached max_samples={max_samples}")
            break

    pbar.close()
    elapsed = time.time() - start

    if not all_translations:
        raise RuntimeError("No samples processed – dataset is empty or no actions found")
    assert chunk_length is not None and rotation_dim is not None

    print(f"\nProcessed {sample_count} samples in {elapsed:.1f}s ({sample_count / elapsed:.1f} samples/s)")

    # Per-frame L2 norms used for filtering and percentile-based threshold derivation.
    trans_norms_per_sample = [np.linalg.norm(t, axis=-1) for t in all_translations]
    rot_norms_per_sample = [np.linalg.norm(r, axis=-1) for r in all_rotations]
    all_trans_norms = np.concatenate(trans_norms_per_sample)
    all_rot_norms = np.concatenate(rot_norms_per_sample)
    total_frames = int(all_trans_norms.shape[0])

    if percentile_clip is not None:
        if not (0.0 < percentile_clip <= 100.0):
            raise ValueError(f"percentile_clip must be in (0, 100], got {percentile_clip}")
        if max_trans_norm is None:
            max_trans_norm = float(np.percentile(all_trans_norms, percentile_clip))
            print(f"  Auto-derived max_trans_norm at P{percentile_clip:g} = {max_trans_norm:.6f}")
        if max_rot_norm is None:
            max_rot_norm = float(np.percentile(all_rot_norms, percentile_clip))
            print(f"  Auto-derived max_rot_norm   at P{percentile_clip:g} = {max_rot_norm:.6f}")

    kept_trans: list[np.ndarray] = []
    kept_rotations: list[np.ndarray] = []
    per_timestep_trans: list[list[np.ndarray]] = [[] for _ in range(chunk_length)]
    per_timestep_rot: list[list[np.ndarray]] = [[] for _ in range(chunk_length)]
    kept_frame_count = 0
    for trans, rot, tn, rn in zip(all_translations, all_rotations, trans_norms_per_sample, rot_norms_per_sample):
        mask = np.ones(trans.shape[0], dtype=bool)
        if max_trans_norm is not None:
            mask &= tn <= max_trans_norm
        if max_rot_norm is not None:
            mask &= rn <= max_rot_norm
        if mask.any():
            kept_trans.append(trans[mask])
            kept_rotations.append(rot[mask])
        kept_frame_count += int(mask.sum())
        upper = min(chunk_length, trans.shape[0])
        for t in range(upper):
            if mask[t]:
                per_timestep_trans[t].append(trans[t])
                per_timestep_rot[t].append(rot[t])

    dropped = total_frames - kept_frame_count
    filter_active = max_trans_norm is not None or max_rot_norm is not None
    if filter_active:
        pct = (100.0 * dropped / total_frames) if total_frames else 0.0
        print(f"\nOutlier filter: kept {kept_frame_count} / {total_frames} frames (dropped {dropped}, {pct:.2f}%)")
        print(f"  thresholds: max_trans_norm={max_trans_norm}, max_rot_norm={max_rot_norm}")

    raw_trans_stats, trans_l2_stats = _summarize_block(kept_trans, per_timestep_trans, chunk_length, "translation")
    raw_rot_stats, rot_l2_stats = _summarize_block(kept_rotations, per_timestep_rot, chunk_length, "rotation")

    return {
        "metadata": {
            "translation_dim": TRANSLATION_DIM,
            "rotation_dim": rotation_dim,
            "chunk_length": chunk_length,
            "num_samples": sample_count,
            "total_frames": total_frames,
            "kept_frames": kept_frame_count,
            "dropped_frames": dropped,
            "max_trans_norm": max_trans_norm,
            "max_rot_norm": max_rot_norm,
            "percentile_clip": percentile_clip,
            "processing_time_s": round(elapsed, 2),
        },
        "raw_translation_stats": raw_trans_stats,
        "translation_l2_norm": trans_l2_stats,
        "raw_rotation_stats": raw_rot_stats,
        "rotation_l2_norm": rot_l2_stats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute raw action translation statistics for camera / AV pose datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=["camera", "av"])
    p.add_argument("--split", default="train", choices=["train", "val", "full"])
    p.add_argument("--output", default=None, help="Output JSON path (auto-generated if omitted)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Use only the first N frames per sample (e.g. 17). All frames used if omitted.",
    )

    # Pose options
    p.add_argument(
        "--rotation-format", default="rot6d", choices=["rot9d", "rot6d", "quat_xyzw", "euler_xyz", "axisangle"]
    )
    p.add_argument(
        "--pose-convention",
        default="backward_framewise",
        choices=["backward_anchored", "backward_framewise"],
    )
    p.add_argument("--credential-path", default="credentials/gcp_training.secret")

    # AV-specific
    p.add_argument("--av-root", default="s3://nv-00-10206-robot/cosmos3_action_data/av_v2_02182026_wdinfo/")
    p.add_argument("--av-history-len", type=float, default=0.1)
    p.add_argument("--av-future-len", type=float, default=6.0)
    p.add_argument("--av-fps", type=int, default=10)

    # Camera-specific
    p.add_argument(
        "--camera-wdinfos",
        nargs="*",
        default=None,
        help="Camera wdinfo keys (default: pretrained_clips_260307_100k). "
        "See CAMERA_WDINFOS in camera_dataset_sharded.py for available keys.",
    )

    # Outlier filtering (applied per-frame, not per-sample).
    p.add_argument(
        "--max-trans-norm",
        type=float,
        default=None,
        help="Drop frames whose translation L2 norm exceeds this value. Takes precedence over --percentile-clip.",
    )
    p.add_argument(
        "--max-rot-norm",
        type=float,
        default=None,
        help="Drop frames whose rotation L2 norm exceeds this value. Takes precedence over --percentile-clip.",
    )
    p.add_argument(
        "--percentile-clip",
        type=float,
        default=None,
        help="Auto-derive max_trans_norm / max_rot_norm from this percentile "
        "(e.g. 99 or 99.5) of the observed L2-norm distributions. "
        "Ignored for a block if --max-{trans,rot}-norm is given explicitly.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    tag = f"{args.dataset}_{args.rotation_format}_{args.pose_convention}"
    if args.output:
        output_path = Path(args.output)
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"pose_stats_{tag}_{args.split}.json"

    print(f"Dataset        : {args.dataset}")
    print(f"Split          : {args.split}")
    print(f"Rotation format: {args.rotation_format}")
    print(f"Rel pose format: {args.pose_convention}")
    print(f"Max samples    : {args.max_samples}")
    print(f"Max frames     : {args.max_frames}")
    print(f"Normalization  : none (raw values, translation_scale=1.0, rotation_scale=1.0)")
    filter_desc_parts: list[str] = []
    if args.max_trans_norm is not None:
        filter_desc_parts.append(f"max_trans_norm={args.max_trans_norm}")
    if args.max_rot_norm is not None:
        filter_desc_parts.append(f"max_rot_norm={args.max_rot_norm}")
    if args.percentile_clip is not None:
        filter_desc_parts.append(f"percentile_clip=P{args.percentile_clip:g}")
    print(f"Outlier filter : {', '.join(filter_desc_parts) if filter_desc_parts else 'off'}")
    print()

    rotation_fmt: RotationConvention = args.rotation_format  # type: ignore[assignment]
    pose_conv: PoseConvention = args.pose_convention  # type: ignore[assignment]

    if args.dataset == "camera":
        dataset = _create_camera_dataset(
            split=args.split,
            rotation_format=rotation_fmt,
            pose_convention=pose_conv,
            credential_path=args.credential_path,
            wdinfo_names=args.camera_wdinfos,
        )
    else:
        dataset = _create_av_dataset(
            split=args.split,
            rotation_format=rotation_fmt,
            pose_convention=pose_conv,
            credential_path=args.credential_path,
            av_root=args.av_root,
            av_history_len=args.av_history_len,
            av_future_len=args.av_future_len,
            av_fps=args.av_fps,
        )

    print(f"Dataset length : {len(dataset)}")
    print()

    results = compute_action_stats(
        dataset,
        max_samples=args.max_samples,
        max_frames=args.max_frames,
        max_trans_norm=args.max_trans_norm,
        max_rot_norm=args.max_rot_norm,
        percentile_clip=args.percentile_clip,
    )

    results["metadata"]["dataset"] = args.dataset
    results["metadata"]["split"] = args.split
    results["metadata"]["max_frames"] = args.max_frames
    results["metadata"]["rotation_format"] = args.rotation_format
    results["metadata"]["pose_convention"] = args.pose_convention
    if args.dataset == "av":
        results["metadata"]["av_root"] = args.av_root
        results["metadata"]["av_history_len"] = args.av_history_len
        results["metadata"]["av_future_len"] = args.av_future_len
        results["metadata"]["av_fps"] = args.av_fps
    else:
        results["metadata"]["camera_wdinfos"] = args.camera_wdinfos or ["pretrained_clips_260307_100k"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    meta = results["metadata"]
    if meta["dropped_frames"] > 0:
        pct = 100.0 * meta["dropped_frames"] / meta["total_frames"] if meta["total_frames"] else 0.0
        print(
            f"Filtered frames: kept {meta['kept_frames']} / {meta['total_frames']} "
            f"(dropped {meta['dropped_frames']}, {pct:.2f}%) — "
            f"thresholds max_trans_norm={meta['max_trans_norm']}, max_rot_norm={meta['max_rot_norm']}"
        )

    def _print_block(label: str, raw_key: str, l2_key: str) -> None:
        g = results[raw_key]["global"]
        n = results[l2_key]
        print(f"\nRaw {label} per-dim statistics — translation_scale=1.0, rotation_scale=1.0:")
        print(f"  Mean       : {g['mean']}")
        print(f"  Median     : {g['median']}")
        print(f"  Std        : {g['std']}")
        print(f"  Min        : {g['min']}")
        print(f"  Max        : {g['max']}")
        print(f"  Flat mean  : {g['flat_mean']:.6f}   (pooled across all dims)")
        print(f"  Flat std   : {g['flat_std']:.6f}   (single global scalar, preserves geometry)")
        print(f"  Count      : {g['count']} frames from {results['metadata']['num_samples']} samples")
        print(f"\n{label.capitalize()} L2 norm:")
        print(f"  Median : {n['median']:.6f}")
        print(f"  Mean   : {n['mean']:.6f}")
        print(f"  Std    : {n['std']:.6f}")
        print(f"  Min    : {n['min']:.6f}")
        print(f"  Max    : {n['max']:.6f}")
        for pct, val in n["percentiles"].items():
            print(f"  P{pct:<5}: {val:.6f}")

    _print_block("translation (tx, ty, tz)", "raw_translation_stats", "translation_l2_norm")
    _print_block(
        f"rotation ({args.rotation_format}, {results['metadata']['rotation_dim']} dims)",
        "raw_rotation_stats",
        "rotation_l2_norm",
    )

    trans_flat_std = results["raw_translation_stats"]["global"]["flat_std"]
    rot_flat_std = results["raw_rotation_stats"]["global"]["flat_std"]
    print("\nSuggested uniform scales (matches MSE-loss magnitude per dim, preserves block geometry):")
    print(
        f"  translation_scale   = 1 / trans_flat_std = {1.0 / trans_flat_std:.6f}"
        if trans_flat_std > 0
        else "  translation_scale   : undefined (trans_flat_std=0)"
    )
    print(
        f"  rotation_scale = 1 / rot_flat_std   = {1.0 / rot_flat_std:.6f}"
        if rot_flat_std > 0
        else "  rotation_scale : undefined (rot_flat_std=0)"
    )
    if trans_flat_std > 0 and rot_flat_std > 0:
        print(
            f"  (equivalently, with translation_scale=1: rotation_scale = trans_flat_std / rot_flat_std = {trans_flat_std / rot_flat_std:.6f})"
        )


if __name__ == "__main__":
    main()
