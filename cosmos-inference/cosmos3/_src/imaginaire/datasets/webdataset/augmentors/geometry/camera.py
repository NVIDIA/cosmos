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

"""Camera parameter augmentors for webdataset."""

from typing import Optional

import torch

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos3._src.imaginaire.modules.camera import Camera


class CameraParamDecoder(Augmentor):
    """Decodes camera parameters from text files.

    The text file format is: fx fy cx cy qx qy qz qw tx ty tz
    where:
        - fx, fy: focal lengths
        - cx, cy: principal points
        - qx, qy, qz, qw: quaternion rotation (world to camera)
        - tx, ty, tz: translation vector (world to camera)
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        """Initialize the camera parameter decoder.

        Args:
            input_keys: List of input keys (typically ['camera'])
            output_keys: List of output keys (typically ['intrinsics', 'world_to_cam'])
            args: Additional arguments (not used)
        """
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        """Decode camera parameters from text data.

        Args:
            data_dict: Input data dictionary containing camera text data

        Returns:
            data_dict: Output data dictionary with decoded camera parameters
        """
        # Get the camera text data
        camera_text = data_dict[self.input_keys[0]]

        # Convert text to string if it's bytes
        if isinstance(camera_text, bytes):
            camera_text = camera_text.decode("utf-8")

        # Parse the camera parameters
        parts = list(map(float, camera_text.strip().split()))
        if len(parts) != 11:
            raise ValueError(f"Invalid camera parameter format. Expected 11 values, got {len(parts)}")

        # Extract parameters
        fx, fy, cx, cy = parts[0:4]  # focal lengths and principal points
        quat = parts[4:8]  # qx, qy, qz, qw
        trans = parts[8:11]  # tx, ty, tz

        # Convert intrinsics to 3x3 matrix via helper
        intrinsics = Camera.intrinsic_params_to_matrices(torch.tensor([fx, fy, cx, cy], dtype=torch.float32))

        # Convert quaternion + translation to 4x4 World->Cam matrix via helper
        qxyzw_t = torch.tensor([*quat, *trans], dtype=torch.float32)
        w2c_3x4 = Camera.extrinsic_params_to_matrices(qxyzw_t)
        world_to_cam = torch.eye(4, dtype=torch.float32)
        world_to_cam[:3, :] = w2c_3x4

        # Convert to torch tensors
        intrinsics = intrinsics.float()
        world_to_cam = world_to_cam.float()

        # Store in output dictionary
        data_dict[self.output_keys[0]] = intrinsics
        data_dict[self.output_keys[1]] = world_to_cam

        # Remove the original camera text data
        data_dict.pop(self.input_keys[0])

        return data_dict


class CameraParamListDecoder(Augmentor):
    """Decodes a list of camera parameters from text files.

    The text file format is multiple lines, where each line contains:
    fx fy cx cy qx qy qz qw tx ty tz
    where:
        - fx, fy: focal lengths
        - cx, cy: principal points
        - qx, qy, qz, qw: quaternion rotation (world to camera)
        - tx, ty, tz: translation vector (world to camera)

    Each line corresponds to one frame's camera parameters.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        """Initialize the camera parameter list decoder.

        Args:
            input_keys: List of input keys (typically ['camera'])
            output_keys: List of output keys (typically ['intrinsics', 'world_to_cam'])
            args: Additional arguments (not used)
        """
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        """Decode a list of camera parameters from text data.

        Args:
            data_dict: Input data dictionary containing camera text data

        Returns:
            data_dict: Output data dictionary with decoded camera parameters as lists
        """
        # Get the camera text data
        camera_text = data_dict[self.input_keys[0]]

        # Convert text to string if it's bytes
        if isinstance(camera_text, bytes):
            camera_text = camera_text.decode("utf-8")

        # Split into lines and parse each line
        lines = camera_text.strip().split("\n")
        num_frames = len(lines)

        if num_frames == 0:
            raise ValueError("Empty camera parameter file")

        # Initialize lists to store camera parameters
        intrinsics_list = []
        world_to_cam_list = []

        # Parse each line
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            parts = list(map(float, line.split()))
            if len(parts) != 11:
                raise ValueError(
                    f"Invalid camera parameter format at line {i + 1}. Expected 11 values, got {len(parts)}"
                )

            # Extract parameters
            fx, fy, cx, cy = parts[0:4]  # focal lengths and principal points
            quat = parts[4:8]  # qx, qy, qz, qw
            trans = parts[8:11]  # tx, ty, tz

            # Convert intrinsics and extrinsics via helpers
            intrinsics = Camera.intrinsic_params_to_matrices(torch.tensor([fx, fy, cx, cy], dtype=torch.float32))
            qxyzw_t = torch.tensor([*quat, *trans], dtype=torch.float32)
            w2c_3x4 = Camera.extrinsic_params_to_matrices(qxyzw_t)
            world_to_cam = torch.eye(4, dtype=torch.float32)
            world_to_cam[:3, :] = w2c_3x4

            intrinsics_list.append(intrinsics)
            world_to_cam_list.append(world_to_cam)

        # Convert lists to torch tensors with batch dimension
        intrinsics_tensor = torch.stack(intrinsics_list).float()  # T x 3 x 3
        world_to_cam_tensor = torch.stack(world_to_cam_list).float()  # T x 4 x 4

        # Store in output dictionary
        data_dict[self.output_keys[0]] = intrinsics_tensor
        data_dict[self.output_keys[1]] = world_to_cam_tensor

        # Remove the original camera text data
        data_dict.pop(self.input_keys[0])
        return data_dict
