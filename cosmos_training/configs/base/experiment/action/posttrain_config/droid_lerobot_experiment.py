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

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action._experiment_helpers import register_modes
from configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment
from configs.base.experiment.action.pretrained_config.cosmos3_8b import make_8b_experiment
from configs.base.experiment.action.pretrained_config.cosmos3_32b import make_32b_experiment
from cosmos.data.vfm.action.droid_lerobot_dataset import DROIDLeRobotDataset
from cosmos.data.vfm.action.unified_dataset import dataset_entry

cs = ConfigStore.instance()

DROID_DATASET = [
    L(dataset_entry)(
        name="droid",
        dataset=L(DROIDLeRobotDataset)(
            root="/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_action_datasets/droid_plus_lerobot_320x180_20260406_sharded",
            viewpoint="wrist_view",
            use_success_only=True,
            video_mode="wrist",
            action_space="ee_pose_delta",
            use_filter_dict=True,
        ),
    )
]


droid_lerobot_2b = make_2b_experiment(
    exp_name="${now:%Y-%m-%d_%H-%M-%S}",
    datasets=DROID_DATASET,
    batch_size=19,
    num_workers=39,
    training_iterations=4_000,
    use_deterministic_seed=True,
)
for i, d in enumerate(droid_lerobot_2b["defaults"]):
    if isinstance(d, dict) and "override /callbacks" in d:
        droid_lerobot_2b["defaults"][i] = {
            "override /callbacks": ["basic", "optimization", "job_monitor", "training_stats"]
        }
        break
droid_lerobot_2b["model"]["config"]["tokenizer"]["encode_exact_durations"] = [17]
droid_lerobot_2b["model"]["config"]["vlm_config"]["pretrained_weights"]["enabled"] = False
droid_lerobot_2b["model"]["config"]["max_num_tokens_after_packing"] = -1
droid_lerobot_2b["scheduler"]["warm_up_steps"] = [0]
droid_lerobot_2b["dataloader_train"]["dataloaders"]["action_data"]["dataloader"]["dataset"]["list_of_datasets"][0][
    "resolution"
] = "256"
droid_lerobot_2b["dataloader_train"]["max_sequence_length"] = None
droid_lerobot_2b["dataloader_train"]["max_samples_per_batch"] = 256
droid_lerobot_2b["job"]["group"] = "cosmos3_action_2b_posttrain"
droid_lerobot_2b["trainer"]["callbacks"]["compile_tokenizer"]["warmup_resolutions"] = ["256"]
droid_lerobot_2b["checkpoint"]["load_from_object_store"] = {"bucket": "nv-00-10206-checkpoint"}

cs.store("droid_lerobot_2b", droid_lerobot_2b, group="experiment", package="_global_")
register_modes(cs, "droid_lerobot_2b", droid_lerobot_2b, dataloader_key="action_data")


droid_lerobot_8b = make_8b_experiment(
    exp_name="${now:%Y-%m-%d_%H-%M-%S}",
    datasets=DROID_DATASET,
    batch_size=19,
    num_workers=39,
    training_iterations=4_000,
    use_deterministic_seed=True,
)
droid_lerobot_8b["model"]["config"]["tokenizer"]["encode_exact_durations"] = [17]
droid_lerobot_8b["model"]["config"]["vlm_config"]["pretrained_weights"]["enabled"] = False
droid_lerobot_8b["model"]["config"]["max_num_tokens_after_packing"] = -1
droid_lerobot_8b["dataloader_train"]["dataloaders"]["action_data"]["dataloader"]["dataset"]["list_of_datasets"][0][
    "resolution"
] = "256"
droid_lerobot_8b["dataloader_train"]["max_sequence_length"] = None
droid_lerobot_8b["dataloader_train"]["max_samples_per_batch"] = 256
droid_lerobot_8b["job"]["group"] = "cosmos3_action_8b_posttrain"
droid_lerobot_8b["trainer"]["callbacks"]["compile_tokenizer"]["warmup_resolutions"] = ["256"]
droid_lerobot_8b["checkpoint"]["load_from_object_store"] = {"bucket": "nv-00-10206-checkpoint"}

cs.store("droid_lerobot_8b", droid_lerobot_8b, group="experiment", package="_global_")
register_modes(cs, "droid_lerobot_8b", droid_lerobot_8b, dataloader_key="action_data")


droid_lerobot_32b = make_32b_experiment(
    exp_name="${now:%Y-%m-%d_%H-%M-%S}",
    datasets=DROID_DATASET,
    batch_size=19,
    num_workers=39,
    training_iterations=4_000,
    use_deterministic_seed=True,
)
droid_lerobot_32b["model"]["config"]["tokenizer"]["encode_exact_durations"] = [17]
droid_lerobot_32b["model"]["config"]["vlm_config"]["load_pretrained"] = False
droid_lerobot_32b["model"]["config"]["max_num_tokens_after_packing"] = -1
droid_lerobot_32b["dataloader_train"]["dataloaders"]["action_data"]["dataloader"]["dataset"]["list_of_datasets"][0][
    "resolution"
] = "256"
droid_lerobot_32b["dataloader_train"]["max_sequence_length"] = None
droid_lerobot_32b["dataloader_train"]["max_samples_per_batch"] = 256
droid_lerobot_32b["job"]["group"] = "cosmos3_action_32b_posttrain"
droid_lerobot_32b["trainer"]["callbacks"]["compile_tokenizer"]["warmup_resolutions"] = ["256"]
droid_lerobot_32b["checkpoint"]["load_from_object_store"] = {"bucket": "nv-00-10206-checkpoint"}

cs.store("droid_lerobot_32b", droid_lerobot_32b, group="experiment", package="_global_")
register_modes(cs, "droid_lerobot_32b", droid_lerobot_32b, dataloader_key="action_data")
