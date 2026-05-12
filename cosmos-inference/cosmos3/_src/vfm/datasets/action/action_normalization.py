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

"""Action normalization helpers."""

import json
from pathlib import Path

import numpy as np
import torch

from cosmos3._src.imaginaire.utils import log


def load_action_stats(stats_path: str, stats_key: str = "global") -> dict[str, np.ndarray]:
    """Load pre-computed action normalization stats from a JSON file."""
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Action normalization stats not found at {stats_path}.")
    log.info(f"Loading action normalization stats from {stats_path}")
    with path.open("r") as f:
        raw = json.load(f)
    if stats_key in raw:
        raw = raw[stats_key]
        if not isinstance(raw, dict):
            raise TypeError(f"Action normalization stats block {stats_key!r} in {stats_path} must be a dict.")
    elif stats_key != "global":
        raise KeyError(f"Action normalization stats block {stats_key!r} not found in {stats_path}.")
    stat_keys = {"mean", "std", "min", "max", "q01", "q99"}
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items() if k in stat_keys}


def normalize_action(
    action: torch.Tensor,
    method: str,
    stats: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Normalize action tensor (all dimensions including gripper)."""
    if method == "quantile":
        q01, q99 = stats["q01"], stats["q99"]
        denom = (q99 - q01).clamp(min=1e-8)
        return (2.0 * (action - q01) / denom - 1.0).clamp(-1.0, 1.0)
    if method == "meanstd":
        return (action - stats["mean"]) / stats["std"].clamp(min=1e-8)
    if method == "minmax":
        lo, hi = stats["min"], stats["max"]
        denom = (hi - lo).clamp(min=1e-8)
        return (2.0 * (action - lo) / denom - 1.0).clamp(-1.0, 1.0)
    raise ValueError(f"Unknown normalization method: {method!r}")
