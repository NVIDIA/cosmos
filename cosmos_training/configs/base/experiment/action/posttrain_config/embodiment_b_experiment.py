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

# Embodiment_b experiment — Cosmos3 2B pretrained base
#
# Base experiment (policy mode) + mode variants (fd, id, policy, i2v, joint).
# Plus wrist camera and keep-aspect-ratio variants.


from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action._experiment_helpers import register_modes
from configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment
from cosmos.data.vfm.action.embodiment_b_dataset import Embodiment_bDataset
from cosmos.data.vfm.action.unified_dataset import dataset_entry

cs = ConfigStore.instance()

# ---------------------------------------------------------------------------
# Dataset definition (shared across all modes)
# ---------------------------------------------------------------------------
EMBODIMENT_B_DATASET = [
    L(dataset_entry)(
        name="embodiment_b",
        dataset=L(Embodiment_bDataset)(chunk_length=16, split="train"),
        ratio=1.0,
    ),
]

# ---------------------------------------------------------------------------
# Base experiment — 2B, 4k iters (policy mode default)
# ---------------------------------------------------------------------------
embodiment_b = make_2b_experiment(
    exp_name="embodiment_b",
    datasets=EMBODIMENT_B_DATASET,
    training_iterations=4_000,
)
embodiment_b["job"]["group"] = "embodiment_b"

cs.store("embodiment_b", embodiment_b, group="experiment", package="_global_")
register_modes(cs, "embodiment_b", embodiment_b, dataloader_key="action_data")


# ---------------------------------------------------------------------------
# Training iteration: 20000.
# ---------------------------------------------------------------------------
embodiment_b_iter2e4 = dict(
    defaults=["/experiment/embodiment_b", "_self_"],
    scheduler=dict(cycle_lengths=[20000]),
    trainer=dict(max_iter=20000),
)
cs.store("embodiment_b_iter2e4", embodiment_b_iter2e4, group="experiment", package="_global_")
register_modes(cs, "embodiment_b_iter2e4", embodiment_b, dataloader_key="action_data")


# ---------------------------------------------------------------------------
# Left wrist camera.
# ---------------------------------------------------------------------------
embodiment_b_wrist_left = dict(
    defaults=["/experiment/embodiment_b", "_self_"],
    dataloader_train=dict(
        dataloaders=dict(
            action_data=dict(
                dataloader=dict(dataset=dict(list_of_datasets=dict(video_key="observation.images.camera_wrist_left")))
            )
        )
    ),
    job=dict(name="${now:%Y-%m-%d_%H-%M-%S}_embodiment_b_wrist_left"),
)
cs.store("embodiment_b_wrist_left", embodiment_b_wrist_left, group="experiment", package="_global_")
register_modes(cs, "embodiment_b_wrist_left", embodiment_b, dataloader_key="action_data")


# ---------------------------------------------------------------------------
# Right wrist camera.
# ---------------------------------------------------------------------------
embodiment_b_wrist_right = dict(
    defaults=["/experiment/embodiment_b", "_self_"],
    dataloader_train=dict(
        dataloaders=dict(
            action_data=dict(
                dataloader=dict(dataset=dict(list_of_datasets=dict(video_key="observation.images.camera_wrist_right")))
            )
        )
    ),
    job=dict(name="${now:%Y-%m-%d_%H-%M-%S}_embodiment_b_wrist_right"),
)
cs.store("embodiment_b_wrist_right", embodiment_b_wrist_right, group="experiment", package="_global_")
register_modes(cs, "embodiment_b_wrist_right", embodiment_b, dataloader_key="action_data")


# ---------------------------------------------------------------------------
# Keep aspect ratio.
# ---------------------------------------------------------------------------
embodiment_b_kar = dict(
    defaults=["/experiment/embodiment_b", "_self_"],
    dataloader_train=dict(dataloaders=dict(action_data=dict(dataloader=dict(dataset=dict(keep_aspect_ratio=True))))),
)
cs.store("embodiment_b_kar", embodiment_b_kar, group="experiment", package="_global_")
register_modes(cs, "embodiment_b_kar", embodiment_b, dataloader_key="action_data")
