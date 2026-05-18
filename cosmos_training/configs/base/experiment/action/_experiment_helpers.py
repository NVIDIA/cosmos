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

import copy
import os
from typing import Any

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils import log
from cosmos.data.vfm.action.libero_dataset import LIBERO_ROOTS, LIBERODataset
from cosmos.data.vfm.action.unified_dataset import dataset_entry

MODES = [
    ("fd", "forward_dynamics"),
    ("id", "inverse_dynamics"),
    ("policy", "policy"),
    ("i2v", "image2video"),
    ("joint", "joint"),
]


def register_modes(cs: ConfigStore, parent_name: str, base_config: dict, *, dataloader_key: str = "dataloader") -> None:
    """Register mode variants (fd, id, policy, i2v, joint) for a parent experiment.

    Args:
        dataloader_key: Key inside ``dataloaders`` dict.  Legacy configs use
            ``"dataloader"``; new ``make_2b/8b_experiment`` configs use ``"action_data"``.
    """
    for suffix, mode in MODES:
        name = f"{parent_name}_{suffix}"
        list_of_datasets = copy.deepcopy(
            base_config["dataloader_train"]["dataloaders"][dataloader_key]["dataloader"]["dataset"]["list_of_datasets"]
        )
        list_of_datasets[0]["dataset"]["mode"] = mode
        node = dict(
            defaults=[f"/experiment/{parent_name}", "_self_"],
            dataloader_train=dict(
                dataloaders={dataloader_key: dict(dataloader=dict(dataset=dict(list_of_datasets=list_of_datasets)))}
            ),
            job=dict(name=f"${{now:%Y-%m-%d_%H-%M-%S}}_{name}"),
        )
        cs.store(name, node, group="experiment", package="_global_")


def register_embodiment_type(
    cs: ConfigStore,
    parent_name: str,
    suffix: str,
    embodiment_type: str,
    base_config: dict,
    *,
    dataloader_key: str = "dataloader",
    dataset_target: Any | None = None,
) -> dict:
    """Register an embodiment-type variant for a parent experiment.

    Args:
        dataloader_key: Key inside ``dataloaders`` dict.  Legacy configs use
            ``"dataloader"``; new ``make_2b/8b_experiment`` configs use ``"action_data"``.
    """
    name = f"{parent_name}_{suffix}"
    list_of_datasets = copy.deepcopy(
        base_config["dataloader_train"]["dataloaders"][dataloader_key]["dataloader"]["dataset"]["list_of_datasets"]
    )
    list_of_datasets[0]["dataset"]["embodiment_type"] = embodiment_type
    if dataset_target is not None:
        list_of_datasets[0]["dataset"]["_target_"] = dataset_target
    node = dict(
        defaults=[f"/experiment/{parent_name}", "_self_"],
        dataloader_train=dict(
            dataloaders={dataloader_key: dict(dataloader=dict(dataset=dict(list_of_datasets=list_of_datasets)))}
        ),
    )
    cs.store(name, node, group="experiment", package="_global_")
    return node


# ---------------------------------------------------------------------------
# LIBERO ``list_of_datasets`` helper
# ---------------------------------------------------------------------------
# The default ``root`` for each LIBERO suite resolves at import time:
#   1. If ``$LIBERO_LOCAL_DATA_ROOT`` is set, use it as the base directory and
#      look up each suite's local ``_20260124``-style subfolder.
#   2. Otherwise fall back to ``LIBERO_ROOTS`` from ``libero_dataset.py`` (the
#      cluster lustre layout).

LIBERO_LOCAL_ROOT_ENV = "LIBERO_LOCAL_DATA_ROOT"

# Shared training-loop baseline used by libero_exp and all libero_*_experiment
# variants (fd, joint, policy). Individual experiments may still override these
LIBERO_BASELINE_BATCH_SIZE: int = 4
LIBERO_BASELINE_NUM_WORKERS: int = 16
LIBERO_BASELINE_TRAINING_ITERATIONS: int = 4_000

LIBERO_REPO_IDS: list[str] = [
    "libero_10",
    "libero_90",
    "libero_object",
    "libero_spatial",
    "libero_goal",
]

# Subfolder name (under ${LIBERO_LOCAL_DATA_ROOT}) for each repo_id. Order matches
# LIBERO_REPO_IDS. These mirror the gs://nv-00-10206-robot/lerobot_v30/ layout.
LIBERO_LOCAL_SUITE_DIRS: list[str] = [
    "libero_10_no_noops_1.0.0_lerobot_aligned_20260124",
    "libero_90_no_noops_lerobot_shuffled_20260124",
    "libero_object_no_noops_1.0.0_lerobot_aligned_20260124",
    "libero_spatial_no_noops_1.0.0_lerobot_20260124",
    "libero_goal_no_noops_1.0.0_lerobot_20260124",
]


def _resolve_libero_default_roots() -> list[str]:
    """Resolve the default ``root`` list for ``LIBERODataset``.

    Reads ``$LIBERO_LOCAL_DATA_ROOT`` and joins each suite subdir. Falls back to
    ``LIBERO_ROOTS`` (cluster lustre layout) when the env var is unset.
    """
    base = os.environ.get(LIBERO_LOCAL_ROOT_ENV)
    if not base:
        return list(LIBERO_ROOTS)

    candidates = [os.path.join(base, suite) for suite in LIBERO_LOCAL_SUITE_DIRS]
    missing = [p for p in candidates if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(
            f"${LIBERO_LOCAL_ROOT_ENV}={base} is set but the following LIBERO suite directories are missing: {missing}"
        )
    log.info(f"[libero] using local LIBERO mount at {base} (via ${LIBERO_LOCAL_ROOT_ENV})")
    return candidates


LIBERO_DEFAULT_KWARGS: dict = dict(
    repo_id=list(LIBERO_REPO_IDS),
    root=_resolve_libero_default_roots(),
    split="train",
    camera_mode="image",
)


def make_libero_dataset(**libero_dataset_kwargs):
    """Build a single-entry ``list_of_datasets`` for a libero experiment.

    Returns a one-element list containing one ``L(dataset_entry)(...)`` LazyCall
    whose inner ``LIBERODataset`` kwargs are ``LIBERO_DEFAULT_KWARGS`` layered
    with the caller-provided overrides. Pass the returned value as the
    ``datasets=`` argument to ``make_2b_experiment``.
    """
    kwargs = {**LIBERO_DEFAULT_KWARGS, **libero_dataset_kwargs}
    return [
        L(dataset_entry)(
            name="libero",
            dataset=L(LIBERODataset)(**kwargs),
            ratio=1.0,
        ),
    ]
