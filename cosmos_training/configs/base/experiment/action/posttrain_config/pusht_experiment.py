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

# PushT experiment — Cosmos3 2B pretrained base
#
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 \
#     --master_port=12341 scripts/train.py \
#     --config=configs/base/config.py \
#     -- experiment=pusht_exp

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action._experiment_helpers import register_modes
from configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment
from cosmos.data.vfm.action.pusht_dataset import PushTDataset
from cosmos.data.vfm.action.unified_dataset import dataset_entry

cs = ConfigStore.instance()

TRAINING_ITERATIONS = 4_000
DATALOADER_SEED = 0

# ---------------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------------
PUSHT_TRAIN_DATASET = [
    L(dataset_entry)(
        name="pusht",
        dataset=L(PushTDataset)(
            repo_id="lerobot/pusht_image",
            split="train",
            split_seed=DATALOADER_SEED,
            split_val_ratio=0.05,
        ),
        ratio=1.0,
    ),
]

# ---------------------------------------------------------------------------
# Base experiment — 2B, 4k iters
# ---------------------------------------------------------------------------
pusht_exp = make_2b_experiment(
    exp_name="pusht_exp",
    datasets=PUSHT_TRAIN_DATASET,
    training_iterations=TRAINING_ITERATIONS,
)

# Checkpoint save interval
pusht_exp["checkpoint"]["save_iter"] = 1000

cs.store(
    group="experiment",
    package="_global_",
    name="pusht_exp",
    node=pusht_exp,
)
register_modes(cs, "pusht_exp", pusht_exp, dataloader_key="action_data")
