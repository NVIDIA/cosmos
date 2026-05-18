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

# Dummy dataset experiment — Cosmos3 2B pretrained base (for debugging)
#
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 \
#     --master_port=12341 scripts/train.py \
#     --config=configs/base/config.py \
#     -- experiment=action_dummy_dataset_exp

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment
from cosmos.data.vfm.action.dummy_dataset import DummyDataset
from cosmos.data.vfm.action.unified_dataset import dataset_entry

cs = ConfigStore.instance()

# ---------------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------------
DUMMY_TRAIN_DATASET = [
    L(dataset_entry)(
        name="default",
        dataset=L(DummyDataset)(length=1e6),
        ratio=1.0,
    ),
]

# ---------------------------------------------------------------------------
# Base experiment — 2B, long run for debugging
# ---------------------------------------------------------------------------
action_dummy_dataset_exp = make_2b_experiment(
    exp_name="dummy_dataset_exp",
    datasets=DUMMY_TRAIN_DATASET,
    batch_size=1,
    num_workers=2,
    training_iterations=1_000_000,
)

# --- Experiment-specific overrides ---
action_dummy_dataset_exp["job"]["group"] = "debugging"
action_dummy_dataset_exp["checkpoint"]["save_iter"] = 100_000_000

cs.store(group="experiment", package="_global_", name="action_dummy_dataset_exp", node=action_dummy_dataset_exp)
