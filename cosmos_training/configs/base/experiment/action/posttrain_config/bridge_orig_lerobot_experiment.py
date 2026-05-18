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

# Bridge Original LeRobot experiment — Cosmos3 2B & 8B pretrained base
#
# Base experiment (policy mode) + mode variants (fd, id, policy, i2v, joint).
# Historical 8B variant configs (pre302003, pre302000v7, mid003001) are rebuilt
# on the centralized 8B pretrained base with custom checkpoint paths.

import copy

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action._experiment_helpers import register_modes
from configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment
from configs.base.experiment.action.pretrained_config.cosmos3_8b import make_8b_experiment
from cosmos.data.vfm.action.bridge_orig_lerobot_dataset import BridgeOrigLeRobotDataset
from cosmos.data.vfm.action.unified_dataset import dataset_entry

cs = ConfigStore.instance()

# ---------------------------------------------------------------------------
# Dataset definition (shared across all modes)
# ---------------------------------------------------------------------------
BRIDGE_DATASET = [
    L(dataset_entry)(
        name="bridge",
        dataset=L(BridgeOrigLeRobotDataset)(chunk_length=16, split="train"),
        ratio=1.0,
    ),
]

# ---------------------------------------------------------------------------
# Base experiment — 2B, 4k iters (policy mode default)
# ---------------------------------------------------------------------------
bridge_orig_lerobot = make_2b_experiment(
    exp_name="bridge_orig_lerobot",
    datasets=BRIDGE_DATASET,
    training_iterations=4_000,
)
bridge_orig_lerobot["job"]["group"] = "bridge_orig_lerobot"

cs.store("bridge_orig_lerobot", bridge_orig_lerobot, group="experiment", package="_global_")
register_modes(cs, "bridge_orig_lerobot", bridge_orig_lerobot, dataloader_key="action_data")


# ---------------------------------------------------------------------------
# Training iteration: 20000.
# ---------------------------------------------------------------------------
bridge_orig_lerobot_iter2e4 = dict(
    defaults=["/experiment/bridge_orig_lerobot", "_self_"],
    scheduler=dict(cycle_lengths=[20000]),
    trainer=dict(max_iter=20000),
)
cs.store("bridge_orig_lerobot_iter2e4", bridge_orig_lerobot_iter2e4, group="experiment", package="_global_")
register_modes(cs, "bridge_orig_lerobot_iter2e4", bridge_orig_lerobot, dataloader_key="action_data")


# ---------------------------------------------------------------------------
# 8B variants — built on cosmos3_8b pretrained base
# ---------------------------------------------------------------------------

# pre 302_003
bridge_orig_lerobot_pre302003 = make_8b_experiment(
    exp_name="bridge_orig_lerobot_pre302003",
    datasets=copy.deepcopy(BRIDGE_DATASET),
    training_iterations=4_000,
)
bridge_orig_lerobot_pre302003["job"]["group"] = "bridge_orig_lerobot"
bridge_orig_lerobot_pre302003["model"]["config"]["rectified_flow_training_config"]["shift"] = 3
bridge_orig_lerobot_pre302003["model"]["config"]["rectified_flow_training_config"]["train_time_video_distribution"] = (
    "waver"
)
bridge_orig_lerobot_pre302003["model"]["config"]["diffusion_expert_config"][
    "unified_3d_mrope_temporal_modality_margin"
] = 15000
bridge_orig_lerobot_pre302003["checkpoint"]["load_path"] = (
    "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/"
    "t2w_mot_exp302_003_qwen3_vl_8b_multires_modality_offset/checkpoints/iter_000006750"
)

cs.store("bridge_orig_lerobot_pre302003", bridge_orig_lerobot_pre302003, group="experiment", package="_global_")
register_modes(cs, "bridge_orig_lerobot_pre302003", bridge_orig_lerobot_pre302003, dataloader_key="action_data")


# pre 302_000_v7
bridge_orig_lerobot_pre302000v7 = make_8b_experiment(
    exp_name="bridge_orig_lerobot_pre302000v7",
    datasets=copy.deepcopy(BRIDGE_DATASET),
    training_iterations=4_000,
)
bridge_orig_lerobot_pre302000v7["job"]["group"] = "bridge_orig_lerobot"
bridge_orig_lerobot_pre302000v7["model"]["config"]["rectified_flow_training_config"]["shift"] = 3
bridge_orig_lerobot_pre302000v7["model"]["config"]["rectified_flow_training_config"][
    "train_time_video_distribution"
] = "waver"
bridge_orig_lerobot_pre302000v7["model"]["config"]["diffusion_expert_config"][
    "unified_3d_mrope_temporal_modality_margin"
] = 15000
bridge_orig_lerobot_pre302000v7["checkpoint"]["load_path"] = (
    "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/t2w_mot_exp302_000_qwen3_vl_8b_multires_recipe_v7/checkpoints/iter_000043500"
)

cs.store("bridge_orig_lerobot_pre302000v7", bridge_orig_lerobot_pre302000v7, group="experiment", package="_global_")
register_modes(cs, "bridge_orig_lerobot_pre302000v7", bridge_orig_lerobot_pre302000v7, dataloader_key="action_data")


# mid 003_001 (base — no specific checkpoint, used by iteration variants below)
bridge_orig_lerobot_mid003001 = make_8b_experiment(
    exp_name="bridge_orig_lerobot_mid003001",
    datasets=copy.deepcopy(BRIDGE_DATASET),
    training_iterations=4_000,
)
bridge_orig_lerobot_mid003001["job"]["group"] = "bridge_orig_lerobot"
bridge_orig_lerobot_mid003001["checkpoint"]["keys_to_skip_loading"] = []
bridge_orig_lerobot_mid003001["model"]["config"]["rectified_flow_training_config"]["shift"] = 3
bridge_orig_lerobot_mid003001["model"]["config"]["rectified_flow_training_config"]["train_time_video_distribution"] = (
    "waver"
)
bridge_orig_lerobot_mid003001["model"]["config"]["diffusion_expert_config"][
    "unified_3d_mrope_temporal_modality_margin"
] = 15000

cs.store("bridge_orig_lerobot_mid003001", bridge_orig_lerobot_mid003001, group="experiment", package="_global_")

for i in (16250, 25000, 50000):
    bridge_orig_lerobot_mid003001_iter = dict(
        defaults=["/experiment/bridge_orig_lerobot_mid003001", "_self_"],
        checkpoint=dict(load_path=f"cosmos3_vfm/video_uva_joint/exp003_001/checkpoints/iter_{i:09d}"),
    )
    cs.store(
        f"bridge_orig_lerobot_mid003001_{i}",
        bridge_orig_lerobot_mid003001_iter,
        group="experiment",
        package="_global_",
    )
    register_modes(
        cs, f"bridge_orig_lerobot_mid003001_{i}", bridge_orig_lerobot_mid003001, dataloader_key="action_data"
    )
