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

# Embodiment C Gripper experiments — 2B and 8B pretrained bases
#
# 2B: make_2b_experiment() (480p mrope, exp202_001 2B checkpoint)
# 8B: make_8b_experiment() (480p mrope, exp202_001 8B checkpoint)
# Both include train dataloaders and mode variants (fd, id, policy, video, joint).

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment
from configs.base.experiment.action.pretrained_config.cosmos3_8b import make_8b_experiment
from cosmos.data.vfm.action.embodiment_c_dataset import EmbodimentCGripperDataset
from cosmos.data.vfm.action.unified_dataset import dataset_entry

cs = ConfigStore.instance()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAINING_ITERATIONS_2B = 4_000
TRAINING_ITERATIONS_8B = 50_000

MODES = [
    ("exp_fd", "forward_dynamics"),
    ("exp_id", "inverse_dynamics"),
    ("exp_policy", "policy"),
    ("exp_video", "image2video"),
    ("exp_joint", "joint"),
]

# ---------------------------------------------------------------------------
# 2B base experiment
# ---------------------------------------------------------------------------
GRIPPER_DATASET = [
    L(dataset_entry)(
        name="embodiment_c_gripper",
        dataset=L(EmbodimentCGripperDataset)(split="train"),
        ratio=1.0,
    ),
]

embodiment_c_gripper = make_2b_experiment(
    exp_name="embodiment_c_gripper",
    datasets=GRIPPER_DATASET,
    training_iterations=TRAINING_ITERATIONS_2B,
)
embodiment_c_gripper["job"]["group"] = "embodiment_c_gripper"
embodiment_c_gripper["checkpoint"]["save_iter"] = 500

cs.store("embodiment_c_gripper", embodiment_c_gripper, group="experiment", package="_global_")

# 2B mode variants
for suffix, mode in MODES:
    name = f"embodiment_c_gripper_{suffix}"
    node = dict(
        defaults=["/experiment/embodiment_c_gripper", "_self_"],
        dataloader_train=dict(
            dataloaders=dict(action_data=dict(dataloader=dict(dataset=dict(list_of_datasets=dict(mode=mode))))),
        ),
        job=dict(name=f"{name}_${{now:%Y%m%d_%H%M%S}}"),
    )
    cs.store(name, node, group="experiment", package="_global_")

# ---------------------------------------------------------------------------
# 8B base experiment
# ---------------------------------------------------------------------------
embodiment_c_gripper_8b = make_8b_experiment(
    exp_name="embodiment_c_gripper_8b",
    datasets=GRIPPER_DATASET,
    training_iterations=TRAINING_ITERATIONS_8B,
)
embodiment_c_gripper_8b["job"]["group"] = "embodiment_c_gripper"
embodiment_c_gripper_8b["checkpoint"]["save_iter"] = 2000

cs.store("embodiment_c_gripper_8b", embodiment_c_gripper_8b, group="experiment", package="_global_")

# 8B mode variants
for suffix, mode in MODES:
    name = f"embodiment_c_gripper_8b_{suffix}"
    node = dict(
        defaults=["/experiment/embodiment_c_gripper_8b", "_self_"],
        job=dict(name=f"{name}_${{now:%Y%m%d_%H%M%S}}"),
    )
    cs.store(name, node, group="experiment", package="_global_")
