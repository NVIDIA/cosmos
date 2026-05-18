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

# LIBERO experiment — Cosmos3 2B pretrained base

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action._experiment_helpers import (
    LIBERO_BASELINE_BATCH_SIZE,
    LIBERO_BASELINE_NUM_WORKERS,
    LIBERO_BASELINE_TRAINING_ITERATIONS,
)
from configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment
from cosmos.data.vfm.action.libero_dataset import LIBERODataset
from cosmos.data.vfm.action.unified_dataset import dataset_entry

cs = ConfigStore.instance()

DATALOADER_SEED = 0

LIBERO_REPO_IDS = ["libero_10", "libero_90", "libero_object", "libero_spatial", "libero_goal"]

# ---------------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------------
LIBERO_TRAIN_DATASET = [
    L(dataset_entry)(
        name="libero",
        dataset=L(LIBERODataset)(
            repo_id=LIBERO_REPO_IDS,
            split="train",
            camera_mode="image",
        ),
        ratio=1.0,
    ),
]

# ---------------------------------------------------------------------------
# Base experiment — 2B, 4k iters
# ---------------------------------------------------------------------------
libero_exp = make_2b_experiment(
    exp_name="libero_exp",
    datasets=LIBERO_TRAIN_DATASET,
    batch_size=LIBERO_BASELINE_BATCH_SIZE,
    num_workers=LIBERO_BASELINE_NUM_WORKERS,
    training_iterations=LIBERO_BASELINE_TRAINING_ITERATIONS,
)

# --- Experiment-specific overrides ---
libero_exp["job"]["group"] = "debugging"

# Replace the callbacks entry in defaults
for i, d in enumerate(libero_exp["defaults"]):
    if isinstance(d, dict) and "override /callbacks" in d:
        libero_exp["defaults"][i] = {
            "override /callbacks": [
                "basic",
                "optimization",
                "job_monitor",
                "training_stats",
            ]
        }
        break

# Scheduler: LIBERO uses f_max=1.0
libero_exp["scheduler"]["f_max"] = [1.0]
libero_exp["scheduler"]["warm_up_steps"] = [LIBERO_BASELINE_TRAINING_ITERATIONS // 20]

cs.store(group="experiment", package="_global_", name="libero_exp", node=libero_exp)

# Alias for backward compatibility
cs.store(group="experiment", package="_global_", name="libero_exp_streaming", node=libero_exp)
