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

"""Script to compute mean and std for action datasets.

This script iterates through a dataset and computes action statistics
using Welford's online algorithm for numerical stability.

Usage:
    # Compute stats for LIBERO training dataset
    PYTHONPATH=. python cosmos/data/vfm/action/compute_action_stats.py \
        --config configs/base/config.py \
        --split train \
        --max-samples 10000 \
        --output cosmos/data/vfm/action/libero_action_stats_10k.json \
        -- experiment=libero_exp

    # Compute stats for PushT
    PYTHONPATH=. python cosmos/data/vfm/action/compute_action_stats.py \
        --config configs/base/config.py \
        --split train \
        --output cosmos/data/vfm/action/pusht_action_stats.json \
        -- experiment=pusht_exp

    # Quick test with limited samples
    PYTHONPATH=. python cosmos/data/vfm/action/compute_action_stats.py \
        --config configs/base/config.py \
        --split train \
        --max-samples 1000 \
        --output test_stats.json \
        -- experiment=libero_exp

    # Override dataset parameters
    PYTHONPATH=. python cosmos/data/vfm/action/compute_action_stats.py \
        --config configs/base/config.py \
        --split train \
        --output stats.json \
        -- experiment=libero_exp \
           dataloader_train.dataloaders.libero_data.dataloader.dataset.use_rotation_9d=False
"""

import argparse
import copy
import importlib
import inspect
import json
import os as _os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import open_dict
from torch.utils.data import ConcatDataset, DataLoader, IterableDataset, Sampler
from tqdm import tqdm

from cosmos.utils.config import load_config
from cosmos.utils.lazy_config import instantiate
from cosmos.utils.context_managers import data_loader_init

# cosmos/data/vfm/action/ => 5 levels up to repo root
IMAGINAIRE4_ROOT = Path(__file__).resolve().parents[5]
# Bundled normalizers directory.  When ``--output`` is not given, stats are
# written to ``<DEFAULT_OUTPUT_DIR>/<embodiment>_<pose>_<rot>.json`` so that
# loaders (e.g. ``BaseActionLeRobotDataset._load_norm_stats``) can resolve
# them from repo paths without hitting lustre.
DEFAULT_OUTPUT_DIR = IMAGINAIRE4_ROOT / "projects" / "cosmos3" / "vfm" / "datasets" / "action" / "normalizers"


# ---------------------------------------------------------------------------
# JSON pretty-printer
# ---------------------------------------------------------------------------
# ``json.dump(indent=2)`` puts every array element on its own line, which for
# stats payloads (action_dim vectors, 50-element quantile arrays) explodes
# what should be a ~17-line file into hundreds of lines. Keep dicts multi-line
# (2-space indent) but collapse numeric/scalar arrays to one line, and
# format floats at 6-decimal fixed precision (no scientific notation).
_FLOAT_DECIMALS = 6


def _fmt_primitive(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{_FLOAT_DECIMALS}f}"
    return json.dumps(value, ensure_ascii=False)


def _fmt_aligned(value: Any) -> str:
    """Format a primitive with leading-space padding for non-negative numerics.

    Used inside numeric arrays so that negative (``-0.000094``) and non-negative
    (`` 0.001623``) values line up column-wise. Still emits valid JSON — the
    extra leading whitespace is insignificant per RFC 8259.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value: .{_FLOAT_DECIMALS}f}"
    if isinstance(value, int):
        return f" {value}" if value >= 0 else f"{value}"
    return json.dumps(value, ensure_ascii=False)


def _is_numeric(x: Any) -> bool:
    """Real numeric (excludes bool, which is a subclass of int in Python)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _serialize(value: Any, indent: int = 0) -> str:
    pad = "  " * indent
    inner = "  " * (indent + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = list(value.items())
        # Align value columns when any sibling value is a flat list (e.g., the
        # ``global`` / ``global_raw`` blocks have mean/std/min/max... lists of
        # different-length keys). Pad each key's trailing whitespace so all
        # opening brackets land in the same column.
        key_strs = [json.dumps(k, ensure_ascii=False) for k in value]
        align = any(isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v) for v in value.values())
        max_key_len = max(len(s) for s in key_strs) if align else 0
        body = []
        for i, ((k, v), ks) in enumerate(zip(items, key_strs)):
            gap = " " * (max_key_len - len(ks) + 1) if align else " "
            sep = "," if i < len(items) - 1 else ""
            body.append(f"{inner}{ks}:{gap}{_serialize(v, indent + 1)}{sep}")
        return "{\n" + "\n".join(body) + f"\n{pad}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(not isinstance(x, (dict, list)) for x in value):
            # Numeric-array path: pad non-negatives with a leading space so
            # ``-X.YYY`` and `` X.YYY`` columns line up vertically.
            if any(isinstance(x, float) for x in value) and all(_is_numeric(x) for x in value):
                return "[" + ", ".join(_fmt_aligned(x) for x in value) + "]"
            return "[" + ", ".join(_fmt_primitive(x) for x in value) + "]"
        body = [f"{inner}{_serialize(v, indent + 1)}{',' if i < len(value) - 1 else ''}" for i, v in enumerate(value)]
        return "[\n" + "\n".join(body) + f"\n{pad}]"
    return _fmt_primitive(value)


def _pretty_dump(obj: Any) -> str:
    return _serialize(obj) + "\n"


class WelfordAccumulator:
    """Welford's online algorithm for computing mean and variance.

    This is numerically stable for large datasets and works with streaming data.

    Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)  # Sum of squared differences
        self.min_val = np.full(dim, np.inf, dtype=np.float64)
        self.max_val = np.full(dim, -np.inf, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a new sample or batch of samples.

        Uses Chan et al.'s parallel algorithm to merge a new batch in O(D)
        numpy ops (no Python per-sample loop).

        Args:
            x: Array of shape (D,) for single sample or (N, D) for batch.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x = x.astype(np.float64, copy=False)

        n_b = x.shape[0]
        if n_b == 0:
            return

        mean_b = x.mean(axis=0)
        # sum of squared diffs within batch
        m2_b = ((x - mean_b) ** 2).sum(axis=0)

        n_a = self.count
        n_new = n_a + n_b
        delta = mean_b - self.mean

        # Merge
        self.mean = self.mean + delta * (n_b / n_new)
        self.m2 = self.m2 + m2_b + (delta * delta) * (n_a * n_b / n_new)
        self.count = n_new

        # Track per-dim min/max
        np.minimum(self.min_val, x.min(axis=0), out=self.min_val)
        np.maximum(self.max_val, x.max(axis=0), out=self.max_val)

    def get_mean(self) -> np.ndarray:
        """Return the current mean."""
        return self.mean.copy()

    def get_variance(self) -> np.ndarray:
        """Return the current sample variance."""
        if self.count < 2:
            return np.zeros(self.dim, dtype=np.float64)
        return self.m2 / (self.count - 1)

    def get_std(self) -> np.ndarray:
        """Return the current standard deviation."""
        return np.sqrt(self.get_variance())

    def get_min(self) -> np.ndarray:
        """Return the minimum values seen."""
        return self.min_val.copy()

    def get_max(self) -> np.ndarray:
        """Return the maximum values seen."""
        return self.max_val.copy()


class ReservoirAccumulator:
    """Reservoir sampling (Algorithm R) for streaming quantile estimation.

    Maintains a fixed-size random sample of all observations seen so far.
    After the stream finishes, exact quantiles are computed on the reservoir.
    """

    def __init__(self, dim: int, max_size: int = 50_000):
        self.dim = dim
        self.max_size = max_size
        self.buffer = np.empty((max_size, dim), dtype=np.float32)
        self.count = 0
        self._rng = np.random.RandomState(42)

    def update(self, x: np.ndarray) -> None:
        """Update reservoir with a new sample or batch of samples."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]

        if self.count < self.max_size:
            space = self.max_size - self.count
            fill_n = min(n, space)
            self.buffer[self.count : self.count + fill_n] = x[:fill_n]
            self.count += fill_n
            x = x[fill_n:]
            n = x.shape[0]
            if n == 0:
                return

        stream_indices = np.arange(self.count, self.count + n, dtype=np.int64)
        j = (self._rng.random(n) * (stream_indices + 1)).astype(np.int64)
        accept = j < self.max_size
        if accept.any():
            self.buffer[j[accept]] = x[accept]
        self.count += n

    def quantile(self, q: float) -> np.ndarray:
        """Return the q-th quantile from the reservoir."""
        n = min(self.count, self.max_size)
        if n == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return np.quantile(self.buffer[:n], q, axis=0).astype(np.float32)


class SortedSubsetSampler(Sampler[int]):
    """Yield a fixed list of indices in ascending order (no shuffling).

    Used to combine *unbiased* random subset sampling with *sequential* access
    order: we draw a random subset once (seeded) and then iterate it sorted,
    so that datasets with internal LRU caches (e.g. ``BaseActionLeRobotDataset``
    with 100+ inner LeRobotDatasets) don't thrash their cache on every batch.

    Unlike :class:`torch.utils.data.SubsetRandomSampler`, indices are **not**
    reshuffled per iteration.
    """

    def __init__(self, indices: list[int]) -> None:
        self._indices = sorted(indices)

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


ROTATION_FORMAT_DIM: dict[str, int] = {"rot9d": 9, "rot6d": 6, "euler_xyz": 3}


def get_rotation_dims(action_dim: int, rotation_format: str = "rot6d") -> list[int]:
    """Infer rotation dim indices from ``action_dim`` and ``rotation_format``.

    Assumes the canonical layouts used in ``projects/cosmos3/vfm/datasets/action/``:

    - 9D  (camera/AV):      ``[pos(3) + rot(R)]``
    - 10D (single arm):     ``[pos(3) + rot(R) + grip(1)]``
    - 2*(3+R+1) (dual arm): ``[L_pos + L_rot + L_grip | R_pos + R_rot + R_grip]``
    - 3*(3+R)+2 (AgiBot):   ``[head_pos + head_rot | R_pos + R_rot + R_grip |
        L_pos + L_rot + L_grip]``
    - 57D (HandPose rot6d, wrist+fingertips):
        ``[cam(9) | R_wrist(9) + R_fingers(15) | L_wrist(9) + L_fingers(15)]``

    For other configurations (e.g. rot9d HandPose or all-finger variants),
    specify ``--skip-rotation-dims`` manually.
    """
    if rotation_format not in ROTATION_FORMAT_DIM:
        raise ValueError(f"rotation_format must be one of {list(ROTATION_FORMAT_DIM)}, got {rotation_format!r}")
    rot_dim = ROTATION_FORMAT_DIM[rotation_format]
    pos_dim = 3
    slot = pos_dim + rot_dim  # 9 for rot6d

    # 9D: camera / AV — [pos + rot]
    if action_dim == slot:
        return list(range(pos_dim, slot))

    # 10D: single arm + gripper — [pos + rot + grip]
    if action_dim == slot + 1:
        return list(range(pos_dim, slot))

    # 20D: dual arm — [L_pos + L_rot + L_grip | R_pos + R_rot + R_grip]
    if action_dim == 2 * (slot + 1):
        right_rot_start = (slot + 1) + pos_dim
        return list(range(pos_dim, slot)) + list(range(right_rot_start, right_rot_start + rot_dim))

    # 29D: Embodiment C/Beta gripper FK-pose layout:
    # [head_pos + head_rot | right_pos + right_rot + right_grip | left_pos + left_rot + left_grip]
    if action_dim == 3 * slot + 2:
        head_rot = list(range(pos_dim, slot))
        right_start = slot
        right_rot = list(range(right_start + pos_dim, right_start + slot))
        left_start = slot + slot + 1
        left_rot = list(range(left_start + pos_dim, left_start + slot))
        return head_rot + right_rot + left_rot

    # 57D (rot6d) HandPose wrist+fingertips:
    # [cam(3+R) | R_wrist(3+R) | R_fingers(5*3) | L_wrist(3+R) | L_fingers(5*3)]
    if action_dim == slot + 2 * (slot + 5 * 3):
        cam_rot = list(range(pos_dim, slot))
        r_wrist_rot = list(range(slot + pos_dim, slot + slot))
        l_wrist_rot = list(range(slot + slot + 15 + pos_dim, slot + slot + 15 + slot))
        return cam_rot + r_wrist_rot + l_wrist_rot

    # Unknown layout — most likely joint-space action (e.g. Embodiment_b 30D =
    # arm(14) + end(14) + effector(2), or robomind joint-space). No SE(3)
    # rotation dims to skip. Warn and return empty list rather than raising,
    # so the stats get saved and the user can inspect the json.
    print(
        f"  [warning] Cannot auto-detect SE(3) rotation dims for "
        f"action_dim={action_dim}, rotation_format={rotation_format!r}. "
        f"Assuming joint-space or custom layout — no dims will be skipped."
    )
    return []


def parse_dim_spec(spec: str | None) -> list[int]:
    """Parse a dim index spec like ``"3-8,12-17,36-41"`` into a sorted unique list.

    Supports comma-separated tokens; each token is either a single integer or a
    closed range ``a-b`` (both endpoints included).
    """
    if not spec:
        return []
    result: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            result.update(range(lo, hi + 1))
        else:
            result.add(int(token))
    return sorted(result)


def _apply_skip_rotation(stats: dict[str, Any], skip_dims: list[int], action_dim: int) -> None:
    """In-place overwrite stats at ``skip_dims`` with identity values.

    After this, downstream ``normalize_action`` (both ``meanstd`` and ``quantile``)
    is the identity for those dims. Useful for rot6d/rot9d rotation dims, which
    must not be per-dim normalized.
    """
    if not skip_dims:
        return
    valid = [d for d in skip_dims if 0 <= d < action_dim]
    if not valid:
        return

    for key, identity in (
        ("mean", 0.0),
        ("std", 1.0),
        ("min", -1.0),
        ("max", 1.0),
        ("q01", -1.0),
        ("q99", 1.0),
    ):
        arr = stats["global"][key]
        for d in valid:
            arr[d] = identity
        stats["global"][key] = arr


def compute_action_stats(
    dataloader: Any,
    action_key: str = "action",
    max_samples: int | None = None,
    action_dim: int | None = None,
    reservoir_size: int = 50_000,
    return_accumulators: bool = False,
) -> dict[str, Any]:
    """Compute mean and std for actions in a dataloader.

    Args:
        dataloader: DataLoader that yields batches with action tensors.
        action_key: Key for action tensor in batch output.
        max_samples: If set, limit the number of samples to process.
        action_dim: If set, only compute stats for first action_dim dimensions.
                    If None, auto-detect from first sample.
        reservoir_size: Number of action frames to keep for quantile estimation.

    Returns:
        Dictionary containing mean, std, quantiles, and metadata.
    """
    print(f"Computing action statistics...")
    print(f"  action_key: {action_key}")
    print(f"  max_samples: {max_samples}")

    # Initialize accumulators (will be created after seeing first sample)
    global_accumulator: WelfordAccumulator | None = None
    reservoir: ReservoirAccumulator | None = None
    detected_action_dim: int | None = None
    chunk_length: int | None = None

    sample_count = 0
    start_time = time.time()

    data_iter = iter(dataloader)
    pbar = tqdm(desc="Computing action stats", unit="samples")

    while True:
        try:
            batch = next(data_iter)
        except StopIteration:
            print("End of dataset reached.")
            break

        if isinstance(batch, torch.Tensor):
            batch = batch.numpy()

        if batch.ndim == 2:
            batch = batch[:, None, :]

        B, chunk_len_batch, dim = batch.shape

        # Initialize accumulators on first batch
        if global_accumulator is None:
            chunk_length = chunk_len_batch
            detected_action_dim = action_dim if action_dim is not None else dim
            print(f"  Detected action shape: {batch.shape[1:]}, using dim={detected_action_dim}")
            global_accumulator = WelfordAccumulator(detected_action_dim)
            reservoir = ReservoirAccumulator(detected_action_dim, max_size=reservoir_size)

        action = batch[:, :, :detected_action_dim].astype(np.float64)

        # Update global accumulator with all timesteps
        flat_actions = action.reshape(-1, detected_action_dim)
        global_accumulator.update(flat_actions)
        reservoir.update(flat_actions.astype(np.float32))

        sample_count += B
        pbar.update(B)

        if sample_count % 1000 == 0:
            elapsed = time.time() - start_time
            rate = sample_count / elapsed
            pbar.set_postfix({"rate": f"{rate:.1f} samples/s"})

        if max_samples is not None and sample_count >= max_samples:
            print(f"\nReached max_samples limit: {max_samples}")
            break
    pbar.close()
    elapsed_time = time.time() - start_time

    if global_accumulator is None:
        raise RuntimeError("No samples processed - dataset is empty or no actions found")

    print(f"\nProcessed {sample_count} samples in {elapsed_time:.1f}s ({sample_count / elapsed_time:.1f} samples/s)")

    # Compile results.  Metadata is kept minimal; callers in main() attach
    # dataset-descriptor fields after instantiation.
    results: dict[str, Any] = {
        "metadata": {
            "action_dim": detected_action_dim,
            "chunk_length": chunk_length,
            "num_samples_stats": sample_count,
        },
        "global": {
            "mean": global_accumulator.get_mean().tolist(),
            "std": global_accumulator.get_std().tolist(),
            "min": global_accumulator.get_min().tolist(),
            "max": global_accumulator.get_max().tolist(),
            "q01": reservoir.quantile(0.01).tolist(),
            "q99": reservoir.quantile(0.99).tolist(),
        },
    }

    if return_accumulators:
        # Multi-rank callers merge the raw accumulators via
        # ``_merge_welford`` / ``_merge_reservoirs`` on rank 0 before
        # rebuilding the final ``global`` block.  Include the live
        # reservoir buffer (only the populated prefix) so no samples are
        # wasted in the merge.
        results["_accumulators"] = {
            "count": int(global_accumulator.count),
            "mean": global_accumulator.mean.copy(),
            "m2": global_accumulator.m2.copy(),
            "min": global_accumulator.min_val.copy(),
            "max": global_accumulator.max_val.copy(),
            "reservoir_buffer": reservoir.buffer[: min(reservoir.count, reservoir.max_size)].copy(),
            "reservoir_count": int(reservoir.count),
            "reservoir_max_size": int(reservoir.max_size),
        }

    return results


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute action statistics for a dataset")
    parser.add_argument("--config", help="Path to the config file", required=True)
    parser.add_argument("--split", help="Dataset split (train/val)", default="train")
    parser.add_argument("--output", help="Output JSON file path", default=None)

    # Processing options
    parser.add_argument(
        "--action-key",
        type=str,
        default="action",
        help="Key for action tensor in dataset output (default: 'action')",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help="Number of action dimensions to use (default: auto-detect)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "If set and smaller than the full dataset size, draw a random "
            "subset of this size (without replacement, deterministic via "
            "--sampling-seed) before iterating. Gives unbiased stats on "
            "datasets too large to fully iterate. Default: None (full data)."
        ),
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=42,
        help=(
            "Seed for the random subset sampler used when --max-samples "
            "truncates the dataset. Default: 42 (reproducible)."
        ),
    )
    parser.add_argument(
        "--reservoir-size",
        type=int,
        default=50_000,
        help="Reservoir size for quantile estimation (default: 50000)",
    )
    parser.add_argument(
        "--skip-rotation-dims",
        type=str,
        default="auto",
        help=(
            "Force identity stats (mean=0, std=1, min=-1, max=1, q01=-1, q99=1) "
            "on given dim indices so downstream normalization is a pass-through. "
            'Default "auto" infers from action_dim + --rotation-format (supports '
            "9D camera, 10D single-arm, 20D dual-arm, 29D AgiBot FK-pose, "
            "57D HandPose rot6d). "
            'Pass explicit indices/ranges to override, e.g. "3-8" or "3-8,12-17,36-41". '
            'Pass "none" (or an empty string) to disable skipping.'
        ),
    )
    parser.add_argument(
        "--rotation-format",
        type=str,
        default="rot6d",
        choices=list(ROTATION_FORMAT_DIM),
        help="Rotation format used by the dataset (for --skip-rotation-dims auto).",
    )

    # DataLoader tuning
    parser.add_argument("--batch-size", type=int, default=256, help="DataLoader batch size")
    parser.add_argument("--num-workers", type=int, default=48, help="DataLoader num_workers")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch_factor")
    parser.add_argument(
        "--enable-fast-init",
        action="store_true",
        help=(
            "Enable BaseActionLeRobotDataset fast shard metadata initialization when "
            "the dataset supports it. Useful for multi-shard AgiBot/RoboMIND/HandPose stats."
        ),
    )
    parser.add_argument(
        "--fast-init-max-workers",
        type=int,
        default=64,
        help="Max metadata prefetch workers used with --enable-fast-init (default: 64)",
    )

    # Sample stride override.  Training uses stride=1 (overlapping chunks);
    # stats converge equally well on non-overlapping chunks and run ~stride×
    # faster.  Applied to each ``BaseActionLeRobotDataset`` instance after
    # construction but before episode indices are built.  Datasets that don't
    # expose ``_sample_stride`` are left untouched.
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=16,
        help=(
            "Sample stride to use for stats computation (default: 16 = "
            "non-overlapping chunks of ``chunk_length=16``).  Passed to each "
            "LeRobot-backed dataset's ``_sample_stride`` attribute before the "
            "episode index is built.  Use ``--sample-stride 1`` to match "
            "training semantics exactly (slower but identical numerics)."
        ),
    )

    # Dataset filter (for experiments that bundle multiple datasets, e.g.
    # ``action_midtrain_exp004_onlyActionData`` — pick one entry at a time).
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            "If set, keep only the entry in ``list_of_datasets`` whose ``name`` "
            "starts with this string (case-sensitive prefix match). Used when "
            "the experiment bundles several datasets (e.g. "
            "``action_midtrain_exp004_onlyActionData``) and you want stats for "
            "just one. Default: None (keep all entries)."
        ),
    )

    # Hydra-style config overrides
    parser.add_argument(
        "opts",
        help="Modify config options (e.g., experiment=libero_exp)",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Distributed (torchrun) helpers
# ---------------------------------------------------------------------------
# The heavy cost on large multi-shard datasets (e.g. ``HandPoseDataset`` with
# ~900 shards) is *index construction* inside ``_register_sources`` — each
# shard opens ``episodes.parquet`` / ``subtasks.parquet`` and builds a
# per-episode span table.  All shards are processed sequentially in a single
# process, so the 3k-second ``init_data_loader`` time dominates the ~3-minute
# actual stats compute.
#
# The shard registration API already supports slicing (``_register_sources(
# indices)``) so each rank can register only its share of shards.  We exploit
# this here: when launched via ``torchrun``, each rank registers
# ``shards[rank::world_size]``, computes partial Welford + reservoir on its
# slice, then ``all_gather_object`` merges into a global set of stats on
# rank 0 (which writes the file).  Other ranks exit.


def _init_distributed() -> tuple[int, int, bool]:
    """Initialize the default process group from ``torchrun`` env vars.

    Uses the ``gloo`` backend (pure CPU — this script has no GPU work).
    Returns ``(rank, world_size, is_distributed)``; when ``WORLD_SIZE`` is 1
    or env vars are missing, returns ``(0, 1, False)`` and no process group
    is created.
    """
    world_size = int(_os.environ.get("WORLD_SIZE", "1"))
    rank = int(_os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return rank, world_size, world_size > 1


def _register_rank_shards(unified_dataset: Any, rank: int, world_size: int) -> None:
    """Register each inner dataset's shards using a rank-sliced index set.

    Replaces the default ``_ensure_sources_registered()`` path with round-robin
    shard assignment: each rank only calls ``_register_sources(shards[rank::
    world_size])`` on datasets that expose ``_all_shard_roots``.  Datasets
    without sharded roots (single-shard LeRobot, plain ``IterableDataset``)
    fall back to a full registration on every rank — acceptable because
    those are cheap.
    """
    from cosmos.data.vfm.action.unified_dataset import MapToIterableAdapter

    for entry in unified_dataset._datasets:
        ds = entry["dataset"]
        if isinstance(ds, MapToIterableAdapter):
            ds = ds.dataset
        if not hasattr(ds, "_register_sources"):
            continue
        shard_roots = getattr(ds, "_all_shard_roots", [])
        if not shard_roots:
            ds._register_sources()
            continue
        n = len(shard_roots)
        my_indices = list(range(rank, n, world_size))
        print(
            f"  [rank {rank}/{world_size}] {ds.__class__.__name__}: "
            f"registering {len(my_indices)}/{n} shards (round-robin)"
        )
        ds._register_sources(my_indices)
    unified_dataset._sources_initialized = True


def _merge_welford(
    counts: list[int],
    means: list[np.ndarray],
    m2s: list[np.ndarray],
    mins: list[np.ndarray],
    maxs: list[np.ndarray],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise-merge Welford stats via Chan et al.'s parallel algorithm.

    All per-rank arrays must share the same ``dim``.  ``min``/``max`` are
    merged element-wise.  Ranks with ``count == 0`` are skipped.
    """
    total_count = 0
    mean = None
    m2 = None
    min_val = None
    max_val = None
    for c, mu, v, mn, mx in zip(counts, means, m2s, mins, maxs):
        if c == 0:
            continue
        if total_count == 0:
            total_count, mean, m2, min_val, max_val = c, mu.copy(), v.copy(), mn.copy(), mx.copy()
            continue
        n_a, n_b = total_count, c
        n_new = n_a + n_b
        delta = mu - mean
        mean = mean + delta * (n_b / n_new)
        m2 = m2 + v + (delta * delta) * (n_a * n_b / n_new)
        total_count = n_new
        np.minimum(min_val, mn, out=min_val)
        np.maximum(max_val, mx, out=max_val)
    return total_count, mean, m2, min_val, max_val


def _merge_reservoirs(
    buffers: list[np.ndarray],
    max_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Merge per-rank reservoir buffers into a single reservoir.

    Each rank's buffer is assumed to be an unbiased uniform sample of its
    local stream.  We concatenate all buffers then uniformly sub-sample
    ``max_size`` rows.  This is strictly unbiased only when every rank
    processed the same number of samples (the common case here, since we
    split ``max_samples`` evenly across ranks) — accurate enough for
    quantile estimation in all practical regimes.
    """
    non_empty = [b for b in buffers if b.size > 0]
    if not non_empty:
        return np.empty((0, 0), dtype=np.float32)
    concat = np.concatenate(non_empty, axis=0)
    if concat.shape[0] <= max_size:
        return concat
    rng = np.random.RandomState(seed)
    keep = rng.choice(concat.shape[0], size=max_size, replace=False)
    keep.sort()
    return concat[keep]


def _build_action_dataloader(
    dl_config: Any,
    action_key: str,
    batch_size: int = 64,
    num_workers: int = 16,
    prefetch_factor: int = 2,
    max_samples: int | None = None,
    sampling_seed: int = 42,
    dataset_name: str | None = None,
    sample_stride: int = 1,
    enable_fast_init: bool = False,
    fast_init_max_workers: int = 64,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, dict[str, Any]]:
    """Instantiate the action dataset from config and wrap it in a DataLoader.

    Must be called inside a ``data_loader_init()`` context.

    When ``max_samples`` is set and smaller than the concatenated dataset size,
    a random subset of size ``max_samples`` is drawn (without replacement,
    seeded by ``sampling_seed``) and iterated in sorted order via
    :class:`SortedSubsetSampler`. Random subset + sorted iteration gives
    unbiased stats + cache-friendly access (important for datasets with many
    inner LeRobotDatasets behind an LRU cache). Otherwise the full dataset is
    iterated in natural order.

    When ``dataset_name`` is set, ``list_of_datasets`` is filtered in-place
    to keep only the entry whose ``name`` field starts with that prefix. This
    is useful when running against an experiment that bundles multiple
    datasets (e.g. ``action_midtrain_exp004_onlyActionData``) — pick one at
    a time.

    Returns:
        (dataloader, dataset_params) where ``dataset_params`` describes each
        inner dataset (class, pose_convention, rotation_format, mode, ...)
        and is written into the stats JSON metadata. ``dataset_params`` also
        carries ``total_len`` and, when sampling is enabled,
        ``sampled_len`` and ``sampling_seed``.
    """
    from cosmos.data.vfm.action.unified_dataset import MapToIterableAdapter

    ds_config = dl_config.dataloaders.action_data.dataloader.dataset
    ds_config.tokenizer_config = None

    # Optional filter: keep only the entry whose ``name`` starts with ``dataset_name``.
    if dataset_name is not None:
        all_names = [entry.get("name", "<unnamed>") for entry in ds_config.list_of_datasets]
        matched = [entry for entry in ds_config.list_of_datasets if str(entry.get("name", "")).startswith(dataset_name)]
        if not matched:
            raise ValueError(
                f"--dataset-name={dataset_name!r} did not match any entry. "
                f"Available names in list_of_datasets: {all_names}"
            )
        if len(matched) > 1:
            matched_names = [entry.get("name") for entry in matched]
            print(
                f"  WARNING: --dataset-name={dataset_name!r} matched {len(matched)} entries "
                f"({matched_names}); keeping only the first."
            )
            matched = matched[:1]
        with open_dict(ds_config):
            ds_config.list_of_datasets = matched
        print(
            f"  Filtered list_of_datasets to 1 entry (name={matched[0].get('name')!r}) "
            f"from {len(all_names)} candidates."
        )

    def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
        """Safely read ``key`` and unwrap OmegaConf containers to plain Python
        types, so downstream metadata never leaks ``ListConfig`` / ``DictConfig``
        into the JSON pretty-printer or ``json.dumps`` fallback.
        """
        try:
            if key not in cfg:
                return default
            val = cfg[key]
        except Exception:
            return default
        try:
            from omegaconf import DictConfig, ListConfig, OmegaConf

            if isinstance(val, (DictConfig, ListConfig)):
                return OmegaConf.to_container(val, resolve=True)
        except ImportError:
            pass
        return val

    def _resolve_target(target: Any) -> Callable | None:
        """Resolve a ``_target_`` field (string or class) to a callable; None on failure."""
        if callable(target):
            return target
        if isinstance(target, str):
            module_path, _, attr = target.rpartition(".")
            if not module_path:
                return None
            try:
                mod = importlib.import_module(module_path)
                return getattr(mod, attr, None)
            except Exception:
                return None
        return None

    def _callable_accepts_kwarg(target: Any, kwarg: str) -> bool:
        """Check whether ``target.__init__`` (or factory) accepts ``kwarg``.

        Returns True only if the signature **explicitly** lists ``kwarg`` as a
        named parameter. A bare ``**kwargs`` is intentionally NOT treated as
        acceptance, because factories like ``get_umi_dataset(..., **kwargs)``
        forward extras into a struct-mode OmegaConf config via
        ``cfg.update(kwargs)``, which raises ``ConfigKeyError`` on unknown
        keys. For those we rely on the post-instantiation
        ``ds._skip_video_loading = True`` fallback below.

        Returns False on any introspection failure (safer to under-inject
        than blow up at instantiation time).
        """
        fn = _resolve_target(target)
        if fn is None:
            return False
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        for p in sig.parameters.values():
            if p.name == kwarg and p.kind is not inspect.Parameter.VAR_KEYWORD:
                return True
        return False

    dataset_params: dict[str, Any] = {"datasets": []}
    for entry in ds_config.list_of_datasets:
        ds_dict = entry.dataset
        # Only disable normalization if the dataset supports that kwarg. Some
        # datasets (e.g. EmbodimentCGripperDataset) don't take ``action_normalization``.
        if "action_normalization" in ds_dict:
            with open_dict(ds_dict):
                ds_dict.action_normalization = None
        # Inject ``skip_video_loading=True`` only when the target class/factory
        # explicitly lists it as a named kwarg. Everything else — subclasses
        # that don't forward ``**kwargs`` to super (BridgeOrigLeRobotDataset,
        # DROIDLeRobotDataset, RoboMINDFrankaDataset, ...) AND factories that
        # forward ``**kwargs`` into a struct-mode OmegaConf config
        # (``get_umi_dataset`` does ``cfg.update(kwargs)``) — falls back to
        # the post-instantiation ``ds._skip_video_loading = True`` below,
        # which the base class honors via attribute lookup.
        target_cfg = _cfg_get(ds_dict, "_target_")
        if _callable_accepts_kwarg(target_cfg, "skip_video_loading"):
            with open_dict(ds_dict):
                ds_dict.skip_video_loading = True
        if enable_fast_init and _callable_accepts_kwarg(target_cfg, "enable_fast_init"):
            with open_dict(ds_dict):
                ds_dict.enable_fast_init = True
        if enable_fast_init and _callable_accepts_kwarg(target_cfg, "fast_init_max_workers"):
            with open_dict(ds_dict):
                ds_dict.fast_init_max_workers = fast_init_max_workers
        # Extract descriptive parameters into metadata (for naming stats files).
        # ``embodiment_type`` / ``root`` / ``sample_stride`` may be absent from
        # the config when the dataset class hard-codes them in ``__init__``
        # (e.g. Bridge / DROID / Fractal); we fill those in after instantiation
        # from the instance attributes below.
        target = _cfg_get(ds_dict, "_target_", "unknown")
        dataset_params["datasets"].append(
            {
                "name": _cfg_get(entry, "name"),
                # ``target`` is typically a class object whose ``repr`` is
                # ``"<class '<module>.<ClassName>'>"``.  Prefer ``__name__``
                # when available; otherwise strip the ``<class '...'>`` wrapper.
                "class": getattr(target, "__name__", None)
                or (str(target).rsplit(".", 1)[-1].rstrip("'>") if target else "unknown"),
                "embodiment_type": _cfg_get(ds_dict, "embodiment_type"),
                "pose_convention": _cfg_get(ds_dict, "pose_convention"),
                "rotation_format": _cfg_get(ds_dict, "rotation_format"),
                "mode": _cfg_get(ds_dict, "mode"),
                "chunk_length": _cfg_get(ds_dict, "chunk_length"),
                "root": _cfg_get(ds_dict, "root"),
                "sample_stride": _cfg_get(ds_dict, "sample_stride"),
            }
        )

    unified_dataset = instantiate(ds_config)
    unified_dataset._transform = lambda data_dict, resolution=None: data_dict

    # Override ``_sample_stride`` on each inner BaseActionLeRobotDataset *before*
    # ``_ensure_sources_registered()`` builds the episode index.  This lets the
    # stats run use non-overlapping chunks (stride=chunk_length) for speed
    # without touching training-side defaults.  Datasets that don't expose the
    # attribute (e.g. UMI BaseDataset, IterableDatasets) are skipped silently.
    if sample_stride != 1:
        for entry in unified_dataset._datasets:
            ds = entry["dataset"]
            if isinstance(ds, MapToIterableAdapter):
                ds = ds.dataset
            if hasattr(ds, "_sample_stride"):
                ds._sample_stride = sample_stride
    if enable_fast_init:
        for entry in unified_dataset._datasets:
            ds = entry["dataset"]
            if isinstance(ds, MapToIterableAdapter):
                ds = ds.dataset
            if hasattr(ds, "_enable_fast_init"):
                ds._enable_fast_init = True
            if hasattr(ds, "_fast_init_max_workers"):
                ds._fast_init_max_workers = fast_init_max_workers

    if world_size > 1:
        _register_rank_shards(unified_dataset, rank=rank, world_size=world_size)
    else:
        unified_dataset._ensure_sources_registered()

    inner_datasets: list[Any] = []
    total_len = 0
    for idx, entry in enumerate(unified_dataset._datasets):
        ds = entry["dataset"]
        if isinstance(ds, MapToIterableAdapter):
            ds = ds.dataset
        ds._skip_video_loading = True
        inner_datasets.append(ds)
        ds_len = len(ds)
        total_len += ds_len
        print(f"  Inner dataset: {ds.__class__.__name__}, len={ds_len}")

        # Backfill fields that the dataset class hard-codes in ``__init__``
        # (not visible in the config).  Falls through silently if the instance
        # doesn't expose a matching attribute — we just keep the None from
        # the pre-instantiation record.
        if idx < len(dataset_params["datasets"]):
            dp_entry = dataset_params["datasets"][idx]
            if not dp_entry.get("embodiment_type"):
                dp_entry["embodiment_type"] = getattr(ds, "_embodiment_type", None)
            if not dp_entry.get("pose_convention"):
                dp_entry["pose_convention"] = getattr(ds, "_pose_convention", None)
            if not dp_entry.get("rotation_format"):
                dp_entry["rotation_format"] = getattr(ds, "_rotation_format", None)
            if not dp_entry.get("sample_stride"):
                dp_entry["sample_stride"] = getattr(ds, "_sample_stride", None)
            if not dp_entry.get("root"):
                # Base class exposes ``_all_shard_roots`` (list of shard roots);
                # for single-shard datasets use it verbatim, else the common
                # parent directory across all shard roots.
                shard_roots = getattr(ds, "_all_shard_roots", None)
                if shard_roots:
                    if len(shard_roots) == 1:
                        dp_entry["root"] = shard_roots[0]
                    else:
                        dp_entry["root"] = _os.path.commonpath(shard_roots)
    if total_len == 0:
        raise RuntimeError(
            "All inner datasets are empty (total len=0). Most likely the data "
            "files are missing or inaccessible on this machine. Check the "
            "dataset root paths and warnings printed above."
        )

    combined_ds = ConcatDataset(inner_datasets) if len(inner_datasets) > 1 else inner_datasets[0]

    def collate_actions(batch: list[dict]) -> torch.Tensor | np.ndarray:
        actions = [sample[action_key] for sample in batch]
        is_tensor = isinstance(actions[0], torch.Tensor)
        # Fast path: uniform chunk length (the common case). Preserves the
        # (B, T, D) shape so ``chunk_length`` metadata reflects the true
        # sampling window.
        shapes = {tuple(a.shape) for a in actions}
        if len(shapes) == 1:
            return torch.stack(actions) if is_tensor else np.stack(actions)
        # Fallback: variable-length chunks (e.g. HandPoseDataset with
        # ``snap_to_subtask``). Per-frame stats are order-/shape-invariant,
        # so we flatten each chunk and concatenate along the time axis.
        # Downstream compute_action_stats reshapes 2D input to (N, 1, D) and
        # computes stats over all rows — numerics unchanged. The reported
        # ``chunk_length`` metadata will be 1 in this mode; the true
        # config-side chunk_length is still captured in dataset_params.
        if is_tensor:
            dim = actions[0].shape[-1]
            return torch.cat([a.reshape(-1, dim) for a in actions], dim=0)
        dim = actions[0].shape[-1]
        return np.concatenate([np.asarray(a).reshape(-1, dim) for a in actions], axis=0)

    dataset_params["total_len"] = total_len


    # then yield them **sorted ascending** via SortedSubsetSampler. Random
    # iteration order (as SubsetRandomSampler would give) causes catastrophic
    # LRU cache misses on datasets like RoboMINDFrankaDataset that hold
    # O(100) inner LeRobotDatasets behind a size-32 LRU — observed ~170x
    # slowdown (10K samples/s sorted vs ~60 samples/s random). Stats only
    # depend on the *set* of samples, not on visit order, so sorting is free.
    sampler: Sampler[int] | None = None
    is_iterable = isinstance(combined_ds, IterableDataset) or any(
        isinstance(ds, IterableDataset) for ds in inner_datasets
    )
    # With multi-rank launches each rank only holds ``shards[rank::world_size]``,
    # so ``total_len`` above is already this rank's slice.  Split the global
    # ``max_samples`` evenly across ranks; combined across all ranks we still
    # process ~``max_samples`` rows total.  Use a per-rank seed offset so
    # different ranks draw different (non-overlapping) subsets from their
    # disjoint shard slices.
    if max_samples is not None and max_samples > 0 and world_size > 1:
        per_rank_max = (max_samples + world_size - 1) // world_size
        per_rank_seed = sampling_seed + rank
    else:
        per_rank_max = max_samples
        per_rank_seed = sampling_seed

    if is_iterable and per_rank_max is not None and per_rank_max > 0:
        # IterableDataset doesn't support DataLoader samplers. Rely on the
        # ``max_samples`` hard-break inside ``collect_stats_loop`` instead.
        # Note: stats may be biased toward whatever order the underlying
        # iterable yields (e.g. early shards) — acceptable for stats-only use.
        dataset_params["sampled_len"] = per_rank_max
        print(
            f"  IterableDataset detected — skipping subset sampler. "
            f"Will stop after {per_rank_max:,} samples in natural iteration "
            "order (may be biased toward early shards)."
        )
    elif per_rank_max is not None and per_rank_max > 0 and per_rank_max < total_len:
        g = torch.Generator().manual_seed(per_rank_seed)
        indices = torch.randperm(total_len, generator=g)[:per_rank_max].tolist()
        sampler = SortedSubsetSampler(indices)
        dataset_params["sampled_len"] = per_rank_max
        dataset_params["sampling_seed"] = per_rank_seed
        print(
            f"  [rank {rank}/{world_size}] Random subset sampling: "
            f"{per_rank_max:,} / {total_len:,} "
            f"(seed={per_rank_seed}, iterated in sorted order for cache locality)"
        )
    elif per_rank_max is not None and per_rank_max > 0:
        print(
            f"  [rank {rank}/{world_size}] per_rank_max={per_rank_max:,} >= "
            f"total_len={total_len:,}; iterating this rank's full slice in natural order."
        )

    dataloader = DataLoader(
        combined_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_actions,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )
    return dataloader, dataset_params


def _resolve_default_output_path(
    dataset_params: dict[str, Any],
    split: str,
    exp_name: str,
    cli_rotation_format: str,
) -> Path:
    """Compute the default output path under ``DEFAULT_OUTPUT_DIR``.

    Matches ``BaseActionLeRobotDataset._normalizer_filename`` exactly so the
    file the loader will look up is the one we write here:

    - SE(3) pose datasets (``embodiment`` + ``pose`` + ``rot``):
      ``normalizers/<embodiment>_<pose>_<rot>.json``
    - joint-space datasets (``pose`` / ``rot`` both ``None``):
      ``normalizers/<embodiment>.json``
    - ``_<split>`` suffix is appended for non-train splits.

    Falls back to ``action_stats_<exp>_<split>.json`` only when even
    ``embodiment`` is unknown.
    """
    datasets = dataset_params.get("datasets") or []
    first = datasets[0] if datasets else {}
    embodiment = first.get("embodiment_type")
    pose = first.get("pose_convention")
    # Datasets that don't expose ``rotation_format`` as a kwarg (e.g. DROID,
    # Fractal, RoboMIND) still produce rot6d actions internally; fall back
    # to the CLI ``--rotation-format`` which defaults to rot6d.  For
    # joint-space datasets (e.g. Embodiment_b) have ``pose`` set to ``None`` — keep
    # ``rot`` ``None`` too so the filename drops both suffixes instead of
    # synthesizing a bogus ``_rot6d``. FK-pose datasets such as Embodiment C
    # and AgiBotWorld-Beta expose both ``pose`` and ``rotation_format``.
    rot = first.get("rotation_format")
    if rot is None and pose is not None:
        rot = cli_rotation_format

    if embodiment and pose and rot:
        stem = f"{embodiment}_{pose}_{rot}"
    elif embodiment and pose is None and rot is None:
        stem = str(embodiment)
    else:
        return DEFAULT_OUTPUT_DIR / f"action_stats_{exp_name}_{split}.json"

    if split != "train":
        stem = f"{stem}_{split}"
    return DEFAULT_OUTPUT_DIR / f"{stem}.json"


def main() -> None:
    args = get_args()

    # Initialize distributed (torchrun) early so all the ``[rank R/W]`` log
    # prefixes below have the right values.  ``WORLD_SIZE == 1`` (plain
    # ``python`` invocation) is a no-op — the code paths below are identical
    # to the single-process pre-change behavior.
    rank, world_size, is_distributed = _init_distributed()
    if is_distributed:
        print(f"[dist] rank={rank} world_size={world_size} backend=gloo")

    # Infer experiment name for default output path
    exp_name = "default"
    if args.opts:
        for opt in args.opts:
            if "experiment=" in opt:
                exp_name = opt.split("=")[1]
                exp_name = exp_name.strip("'\"")
                break

    # Load config
    if rank == 0:
        print(f"Loading config: {args.config}")
        print(f"Config overrides: {args.opts}")
    try:
        config = load_config(args.config, args.opts, enable_one_logger=False)
    except Exception as e:
        print(f"Failed to load config: {e}")
        raise

    # Instantiate dataloader
    with data_loader_init():
        if args.split == "train":
            dl_config = config.dataloader_train
        else:
            dl_config = config.dataloader_val

        if rank == 0:
            print(f"Instantiating {args.split} dataloader...")
        try:
            dataloader, dataset_params = _build_action_dataloader(
                dl_config,
                args.action_key,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                max_samples=args.max_samples,
                sampling_seed=args.sampling_seed,
                dataset_name=args.dataset_name,
                sample_stride=args.sample_stride,
                enable_fast_init=args.enable_fast_init,
                fast_init_max_workers=args.fast_init_max_workers,
                rank=rank,
                world_size=world_size,
            )
        except Exception as e:
            print(f"Failed to instantiate dataloader: {e}")
            raise

    # Resolve output path now that we know embodiment / pose / rotation.
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _resolve_default_output_path(
            dataset_params=dataset_params,
            split=args.split,
            exp_name=exp_name,
            cli_rotation_format=args.rotation_format,
        )
        if rank == 0:
            print(f"  Default output path: {output_path}")

    # Compute statistics.  In multi-rank mode each rank iterates its own
    # shard slice and returns raw accumulators; we merge them below before
    # re-deriving the final ``global`` block.  The global cap (``max_samples``)
    # stays as-is for single-process mode, but is split evenly across ranks
    # inside ``_build_action_dataloader`` when ``world_size > 1``.
    per_rank_cap = args.max_samples
    if is_distributed and per_rank_cap is not None:
        per_rank_cap = (per_rank_cap + world_size - 1) // world_size
    results = compute_action_stats(
        dataloader=dataloader,
        action_key=args.action_key,
        max_samples=per_rank_cap,
        action_dim=args.action_dim,
        reservoir_size=args.reservoir_size,
        return_accumulators=is_distributed,
    )

    if is_distributed:
        # Gather every rank's raw accumulators on rank 0, merge via Chan's
        # parallel Welford algorithm + reservoir union, then rebuild the
        # ``global`` block.  ``all_gather_object`` is symmetric (all ranks
        # receive the list) so we only have one collective call; the
        # non-zero ranks simply discard their copy.
        local_accum = results.pop("_accumulators")
        gathered: list[dict[str, Any]] = [None] * world_size  # type: ignore[list-item]
        dist.all_gather_object(gathered, local_accum)

        # Sum the raw (pre-cap) sample counts for metadata reporting.
        total_samples = int(sum(g["count"] for g in gathered))

        if rank == 0:
            total_count, mean, m2, min_val, max_val = _merge_welford(
                counts=[g["count"] for g in gathered],
                means=[g["mean"] for g in gathered],
                m2s=[g["m2"] for g in gathered],
                mins=[g["min"] for g in gathered],
                maxs=[g["max"] for g in gathered],
            )
            if total_count == 0:
                raise RuntimeError("Distributed merge saw zero total samples across all ranks.")

            merged_reservoir = _merge_reservoirs(
                buffers=[g["reservoir_buffer"] for g in gathered],
                max_size=args.reservoir_size,
                seed=args.sampling_seed,
            )
            # ``std`` from Chan-merged ``m2``: sample variance = m2 / (N-1).
            std = np.sqrt(m2 / max(total_count - 1, 1))
            q01 = np.quantile(merged_reservoir, 0.01, axis=0) if merged_reservoir.size else np.zeros_like(mean)
            q99 = np.quantile(merged_reservoir, 0.99, axis=0) if merged_reservoir.size else np.zeros_like(mean)

            results["global"] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "min": min_val.tolist(),
                "max": max_val.tolist(),
                "q01": q01.astype(np.float32).tolist(),
                "q99": q99.astype(np.float32).tolist(),
            }
            results["metadata"]["num_samples_stats"] = total_samples
            results["metadata"]["world_size"] = world_size
            print(
                f"\n[dist] merged across world_size={world_size}: total_samples={total_samples:,}, "
                f"reservoir_size={merged_reservoir.shape[0]:,}"
            )
        # Non-rank-0 processes have nothing more to do.
        dist.barrier()
        if rank != 0:
            dist.destroy_process_group()
            return

    # Zero-out rotation dims so normalization is a pass-through for them.
    # Keep this *before* the metadata re-assembly below so skip_rotation_dims
    # is ready when we compose the final payload.
    act_dim = results["metadata"]["action_dim"]
    spec = (args.skip_rotation_dims or "").strip().lower()
    if spec in ("", "none"):
        skip_dims: list[int] = []
        print("\nSkip rotation: disabled (stats reflect raw data on all dims).")
    elif spec == "auto":
        skip_dims = get_rotation_dims(act_dim, args.rotation_format)
        print(
            f"\nAuto-detected rotation dims (action_dim={act_dim}, "
            f"rotation_format={args.rotation_format}): {skip_dims or '[] (none)'}"
        )
    else:
        skip_dims = parse_dim_spec(args.skip_rotation_dims)
        if skip_dims:
            print(f"\nForcing identity stats for rotation dims: {skip_dims} (action_dim={act_dim})")

    if skip_dims:
        # Preserve the raw (pre-replacement) stats under ``global_raw`` so
        # consumers can inspect the data's true distribution on rotation dims
        # even after ``global`` has been overwritten with identity values.
        results["global_raw"] = copy.deepcopy(results["global"])
        _apply_skip_rotation(results, skip_dims, act_dim)

    # Assemble the final flat metadata block.  Identity fields first (who/what
    # these stats describe), then stats-run configuration (how they were
    # computed).  Source priority for ``rotation_format``: dataset config >
    # CLI ``--rotation-format``.  Fields with no source stay ``None`` rather
    # than being silently dropped, so gaps are visible.
    first = (dataset_params.get("datasets") or [{}])[0]
    meta = results["metadata"]
    ordered: dict[str, Any] = {
        # Identity
        "embodiment_type": first.get("embodiment_type"),
        "pose_convention": first.get("pose_convention"),
        # Leave ``rotation_format`` as ``None`` for joint-space datasets; do
        # NOT fall back to the CLI default, which would silently claim
        # "rot6d" on a dataset with no rotation.
        "rotation_format": first.get("rotation_format"),
        "action_dim": meta["action_dim"],
        "skip_rotation_dims": skip_dims,
        "chunk_length": meta["chunk_length"],
        "sample_stride": first.get("sample_stride"),
        # Dataset provenance
        "dataset_name": first.get("name"),
        "dataset_class": first.get("class"),
        "dataset_root": first.get("root"),
        "split": args.split,
        # Stats-run configuration (how these numbers were computed)
        "num_samples_stats": meta["num_samples_stats"],
        "reservoir_size": args.reservoir_size,
    }
    if args.max_samples is not None:
        ordered["max_samples"] = args.max_samples
        ordered["sampling_seed"] = args.sampling_seed

    results["metadata"] = ordered

    # Save results. Use a compact pretty-printer so numeric arrays stay on
    # one line (``json.dump(indent=2)`` puts every number on its own line,
    # blowing a 17-line payload into ~900 lines).
    # Serialize fully *before* touching disk: opening with ``"w"`` truncates
    # the target, so a formatting crash would otherwise wipe any previously
    # good file on disk. Stage to a sibling ``.tmp`` and atomically rename
    # only after successful formatting.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = _pretty_dump(results)
    except Exception:
        fallback = output_path.with_suffix(output_path.suffix + ".raw.json")
        with open(fallback, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[warning] pretty-print failed; raw results saved to: {fallback}")
        raise
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        f.write(payload)
    _os.replace(tmp_path, output_path)

    print(f"\nResults saved to: {output_path}")
    print("\nGlobal statistics:")
    print(f"  Mean: {results['global']['mean']}")
    print(f"  Std:  {results['global']['std']}")
    print(f"  Min:  {results['global']['min']}")
    print(f"  Max:  {results['global']['max']}")
    print(f"  q01:  {results['global']['q01']}")
    print(f"  q99:  {results['global']['q99']}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
