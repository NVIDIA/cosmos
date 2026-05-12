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

"""Depth augmentors for webdataset."""

from typing import Optional

import torch

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor


class DepthMask(Augmentor):
    """Generates a binary mask for valid depth values.

    This augmentor takes a depth image and generates a binary mask indicating
    which pixels have valid depth values. A pixel is considered valid if:
    1. Its depth value is greater than min_depth
    2. Its depth value is less than max_depth
    3. Its depth value is not NaN or infinite
    4. Its depth value is not larger than median_multiplier times the median depth

    Args:
        min_depth (float): Minimum valid depth value
        max_depth (float): Maximum valid depth value
        median_multiplier (float): Maximum allowed depth as a multiple of median depth
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        """Initialize the depth mask generator.

        Args:
            input_keys: List of input keys (typically ['depth'])
            output_keys: List of output keys (typically ['depth_mask'])
            args: Additional arguments including:
                - min_depth (float): Minimum valid depth value
                - max_depth (float): Maximum valid depth value
                - median_multiplier (float): Maximum allowed depth as a multiple of median depth
        """
        super().__init__(input_keys, output_keys, args)
        self.min_depth = args.get("min_depth", 0.1) if args else 0.1
        self.max_depth = args.get("max_depth", 100.0) if args else 100.0
        self.median_multiplier = args.get("median_multiplier", 10) if args else 10

    def __call__(self, data_dict: dict) -> dict:
        """Generate depth mask.

        Args:
            data_dict: Input data dictionary containing depth image

        Returns:
            data_dict: Output data dictionary with depth mask
        """
        # Get depth image
        depth = data_dict[self.input_keys[0]]  # H x W

        # Create mask for valid depth values
        mask = torch.ones_like(depth, dtype=torch.bool)

        # Check for minimum depth
        mask = mask & (depth > self.min_depth)

        # Check for maximum depth
        mask = mask & (depth < self.max_depth)

        # Check for NaN and infinite values
        mask = mask & torch.isfinite(depth) & (~torch.isnan(depth))

        # Compute median depth from currently valid depths
        if mask.any():
            valid_depths = depth[mask]
            median_depth = torch.median(valid_depths)

            # Filter out depths larger than median_multiplier times the median
            max_allowed_depth = self.median_multiplier * median_depth
            mask = mask & (depth <= max_allowed_depth)

        # Store in output dictionary
        data_dict[self.output_keys[0]] = mask
        data_dict[self.input_keys[0]][~mask] = self.max_depth
        return data_dict


class ConsecutiveFrameSampler(Augmentor):
    """Randomly samples N consecutive frames from a video sequence.

    This augmentor takes a video sequence and randomly samples N consecutive frames
    starting from a random position within the valid range.

    Args:
        num_frames (int): Number of consecutive frames to sample
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = None,
        random_sample: bool = True,
        args: Optional[dict] = None,
    ) -> None:
        """Initialize the consecutive frame sampler.

        Args:
            input_keys: List of input keys (typically ['depth', 'points', etc.])
            output_keys: List of output keys (same as input_keys)
            args: Additional arguments including:
                - num_frames (int): Number of consecutive frames to sample
        """
        super().__init__(input_keys, output_keys, args)
        self.num_frames = args.get("num_frames", 25) if args else 25
        self.random_sample = random_sample

    def __call__(self, data_dict: dict) -> dict:
        """Sample consecutive frames from video sequences.

        Args:
            data_dict: Input data dictionary containing video sequences

        Returns:
            data_dict: Output data dictionary with sampled frames
        """

        # Get the first input key to determine the temporal dimension
        first_key = self.input_keys[0]
        video_tensor = data_dict[first_key]

        if video_tensor.dim() == 4:  # CxTxHxW
            total_frames = video_tensor.shape[1]
        elif video_tensor.dim() == 3:  # TxHxW
            total_frames = video_tensor.shape[0]
        else:
            raise ValueError(f"Expected 3D (TxHxW) or 4D (CxTxHxW) tensor, got {video_tensor.dim()}D")

        # Calculate valid start indices
        max_start_idx = max(0, total_frames - self.num_frames)
        if self.num_frames > total_frames:
            return None

        if max_start_idx == 0:
            # If video is shorter than requested frames, use all available frames
            start_idx = 0
            actual_num_frames = total_frames
        else:
            if self.random_sample:
                # Randomly sample start index
                start_idx = torch.randint(0, max_start_idx + 1, size=(1,)).item()
            else:
                start_idx = 0
            actual_num_frames = self.num_frames

        # Sample frames for all input keys
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            tensor = data_dict[input_key]

            if tensor.dim() == 4:  # CxTxHxW
                sampled_tensor = tensor[:, start_idx : start_idx + actual_num_frames, :, :]
                assert sampled_tensor.shape[1] == actual_num_frames, (
                    f"Sampled tensor {input_key} has {sampled_tensor.shape[1]} frames, expected {actual_num_frames}"
                )
            elif tensor.dim() == 3:  # TxHxW
                sampled_tensor = tensor[start_idx : start_idx + actual_num_frames, :, :]
                assert sampled_tensor.shape[0] == actual_num_frames, (
                    f"Sampled tensor {input_key} has {sampled_tensor.shape[0]} frames, expected {actual_num_frames}"
                )
            else:
                raise ValueError(f"Expected 3D (TxHxW) or 4D (CxTxHxW) tensor for {input_key}, got {tensor.dim()}D")

            data_dict[output_key] = sampled_tensor
        data_dict["frame_start"] = start_idx
        data_dict["frame_end"] = start_idx + actual_num_frames

        return data_dict
