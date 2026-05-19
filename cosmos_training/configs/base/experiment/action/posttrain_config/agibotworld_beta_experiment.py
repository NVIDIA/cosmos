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

# AgiBotWorld Beta experiment — Cosmos3 2B pretrained base
#
# Base experiment (policy mode) + mode variants (fd, id, policy, i2v, joint).
# Plus keep-aspect-ratio variant.

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action._experiment_helpers import register_modes
from configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment
from cosmos.data.vfm.action.agibotworld_beta_dataset import AgiBotWorldBetaDataset
from cosmos.data.vfm.action.unified_dataset import dataset_entry

cs = ConfigStore.instance()

# ---------------------------------------------------------------------------
# Dataset definition (shared across all modes)
# ---------------------------------------------------------------------------
AGIBOTWORLD_BETA_DATASET = [
    L(dataset_entry)(
        name="agibotworld_beta",
        dataset=L(AgiBotWorldBetaDataset)(chunk_length=16, split="train"),
        ratio=1.0,
    ),
]

# ---------------------------------------------------------------------------
# Base experiment — 2B, 4k iters (policy mode default)
# ---------------------------------------------------------------------------
agibotworld_beta = make_2b_experiment(
    exp_name="agibotworld_beta",
    datasets=AGIBOTWORLD_BETA_DATASET,
    training_iterations=4_000,
)
agibotworld_beta["job"]["group"] = "agibotworld_beta"

cs.store("agibotworld_beta", agibotworld_beta, group="experiment", package="_global_")
register_modes(cs, "agibotworld_beta", agibotworld_beta, dataloader_key="action_data")


# ---------------------------------------------------------------------------
# Keep aspect ratio.
# ---------------------------------------------------------------------------
agibotworld_beta_kar = dict(
    defaults=["/experiment/agibotworld_beta", "_self_"],
    dataloader_train=dict(dataloaders=dict(action_data=dict(dataloader=dict(dataset=dict(keep_aspect_ratio=True))))),
)
cs.store("agibotworld_beta_kar", agibotworld_beta_kar, group="experiment", package="_global_")
register_modes(cs, "agibotworld_beta_kar", agibotworld_beta, dataloader_key="action_data")
