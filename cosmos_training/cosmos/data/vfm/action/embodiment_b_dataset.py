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

from typing import Any

import torch

from cosmos.data.vfm.action.cosmos3_action_lerobot import (
    ActionNormalization,
    BaseActionLeRobotDataset,
)
from cosmos.data.vfm.action.embodiment_b_dataset_config import LEROBOT_ROOTS
from cosmos.data.vfm.action.viewpoint_utils import Viewpoint


class Embodiment_bDataset(BaseActionLeRobotDataset):
    """ """

    def __init__(
        self,
        fps: float = 10.0,
        chunk_length: int = 16,
        split_seed: int = 42,
        split_val_ratio: float = 0.05,
        split: str = "train",
        video_key: str = "observation.images.camera_top_left",
        mode: str = "policy",
        action_normalization: ActionNormalization | None = None,
        viewpoint: Viewpoint = "ego_view",
    ) -> None:
        """ """
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type="embodiment_b",
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            tolerance_s=1e-4,
        )
        self._video_key = video_key

        self._all_shard_roots = [
            f"/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/ychao/datasets/lerobot_v30/embodiment_b/{x}"
            for x in LEROBOT_ROOTS
        ]

        self._delta_timestamps = {
            "observation.images.camera_top_left": [i * self._dt for i in range(0, self._chunk_length + 1)],
            "observation.images.camera_wrist_left": [i * self._dt for i in range(0, self._chunk_length + 1)],
            "observation.images.camera_wrist_right": [i * self._dt for i in range(0, self._chunk_length + 1)],
            "action.arm.position": [i * self._dt for i in range(0, self._chunk_length)],
            "action.end.position": [i * self._dt for i in range(0, self._chunk_length)],
            "action.effector.position": [i * self._dt for i in range(0, self._chunk_length)],
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """ """
        mode, _, _, sample = self._fetch_sample(idx)

        ai_caption = sample["task"]

        video = sample[self._video_key]  # [T,C,H,W]
        arm_position = sample["action.arm.position"]
        end_position = sample["action.end.position"]
        effector_position = sample["action.effector.position"]
        action = torch.cat([arm_position, end_position, effector_position], dim=-1)  # [T,30]

        return self._build_result(mode=mode, video=video, action=action, ai_caption=ai_caption)

    @property
    def action_dim(self) -> int:
        """ """
        return 30
