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

import random
from typing import Any

import torch
from torch.utils.data import Dataset

from cosmos.utils import log


class DummyDataset(Dataset):
    """
    A dummy dataset that generates random images/videos, camera poses, and actions
    mimicking the structure of UMI/iPhUMI datasets.
    """

    def __init__(
        self,
        length: int = 100,
        image_size: int = 256,
        chunk_length: int = 16,  # must be divisible by 4
        camera_pose_dim: int = 9,  # 3 pos + 6 rot
        fps: int = 16,
        mode: str = "joint",
        randomize_resolution: bool = True,
    ):
        self.length = int(length)
        self.image_size = image_size
        self.chunk_length = chunk_length
        assert self.chunk_length % 4 == 0, "chunk_length must be divisible by 4"
        self.camera_pose_dim = camera_pose_dim
        self.fps = fps
        self.randomize_resolution = randomize_resolution

        assert mode in ["joint", "forward_dynamics", "inverse_dynamics", "policy", "image2video"], (
            "mode must be either joint, forward_dynamics, inverse_dynamics, policy, or image2video"
        )
        self.mode = mode

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.mode == "joint":
            mode = random.choice(["forward_dynamics", "inverse_dynamics", "policy", "image2video"])
        else:
            mode = self.mode

        # Video: (C, T, H, W) uint8
        video_length = self.chunk_length + 1

        if self.randomize_resolution:
            # Randomize one dimension downward so the aspect ratio varies, while
            # keeping both dimensions <= image_size.  This ensures auto-detected
            # resolution stays in the expected tier and exercises the padding path
            # (the transforms pipeline will pad back up to the predefined target).
            h = self.image_size - random.randint(32, 32)
            w = self.image_size - random.randint(32, 32)
            log.debug(f"DummyDataset[{idx}]: before padding resolution = ({h}, {w})")
        else:
            h = self.image_size
            w = self.image_size

        video = torch.randint(0, 256, (3, video_length, h, w), dtype=torch.uint8)  # [3,T+1,H,W]

        # Camera poses: (T, 9) float32
        # 3 position + 6 rotation (first two rows of rotation matrix flattened)
        camera_poses = torch.randn(self.chunk_length, self.camera_pose_dim, dtype=torch.float32)  # [T,camera_pose_dim]

        # EEF commands (actions): (T, 1) float32
        eef_commands = torch.randn(self.chunk_length, 1, dtype=torch.float32)  # [T,1]

        # FPS: scalar (0-D tensor so batching produces [B] not [B, 1])
        fps = torch.tensor(self.fps, dtype=torch.long)  # scalar

        # Index
        key = torch.tensor([idx], dtype=torch.long)  # [1]


        # chunk_length is L, given L + 1 video frames and optionally L relative action
        # video: predicting L video frames given 1 frame
        # forward_dynamics: predicting L video frames given 1 frame and L action
        # inverse_dynamics: predicting L action given L+1 frames (TODO: do we need a state too?)
        # policy: predicting L action and L frames given 1 frame

        # Combine camera poses and eef commands into raw action tensor
        action_tensor = torch.cat([camera_poses, eef_commands], dim=1)  # [T,camera_pose_dim+1]

        return {
            "video": video,
            "action": action_tensor,
            "conditioning_fps": fps,
            "ai_caption": "A dummy video for testing.",
            "mode": mode,
            "__key__": key,
            "domain_id": torch.tensor(0, dtype=torch.long),
            "viewpoint": "ego_view",
        }
