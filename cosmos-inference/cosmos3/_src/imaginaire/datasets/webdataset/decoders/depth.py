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

"""Depth decoder for EXR files."""

import re
from io import BytesIO

import numpy as np
import torch

_EXR_EXTENSIONS = "exr"
MAX_DEPTH = 100000
_NPZ_EXTENSIONS = "npz"


def exr_loader(key, data):
    """Load depth data from EXR file.

    Args:
        key (str): Key of the data
        data (bytes): Raw EXR file data

    Returns:
        torch.Tensor: Depth map as tensor
    """
    # pyrefly: ignore  # import-error
    import OpenEXR

    extension = re.sub(r".*[.]", "", key)
    if extension.lower() not in _EXR_EXTENSIONS:
        return None

    # Convert bytes to BytesIO for OpenEXR
    exr_file = OpenEXR.InputFile(BytesIO(data))

    # Get the header information
    header = exr_file.header()
    dw = header["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    # Read the depth data from 'R' channel
    depth = np.frombuffer(exr_file.channel("R"), dtype=np.float32).reshape((h, w))
    mask = depth == np.nan
    depth = depth.copy()
    depth[mask] = MAX_DEPTH

    # Convert to tensor and normalize to [0, 1]
    depth = torch.from_numpy(depth).float()

    depth = depth.unsqueeze(0)
    return depth


def npz_loader(key, data):
    """Load depth data from NPZ file."""

    extension = re.sub(r".*[.]", "", key)
    if extension.lower() not in _NPZ_EXTENSIONS:
        return None

    # Convert bytes to BytesIO for np.load
    npz_file = BytesIO(data)

    # Load the NPZ file
    with np.load(npz_file) as npz_data:
        # Assuming the depth data is stored in the first array
        # You may need to adjust this based on your specific NPZ file structure
        depth_array = npz_data[list(npz_data.keys())[0]]
    # Convert to tensor and normalize to [0, 1] if needed
    depth = torch.from_numpy(depth_array).float()

    return depth


def construct_videodepth_decoder():
    """Construct videodepth decoder with frame count filtering.

    Args:
        min_frames (int): Minimum number of frames required. Samples with fewer frames will be skipped.

    Returns:
        callable: Videodepth decoder function that filters by frame count
    """

    def videodepth_decoder(key, data):
        """Decode depth video data from NPZ file and filter by frame count.

        Args:
            key (str): Key of the data
            data (bytes): Raw NPZ file data

        Returns:
            torch.Tensor: Depth video tensor if it has enough frames, None otherwise (to skip)
        """
        # Load the depth data using npz_loader
        depth = npz_loader(key, data)
        if depth is None:
            return None

        # Check frame count - determine temporal dimension
        if depth.dim() == 4:  # CxTxHxW
            total_frames = depth.shape[1]
        elif depth.dim() == 3:  # TxHxW
            total_frames = depth.shape[0]
        else:
            # For 2D depth maps (single frame), skip filtering
            return depth

        return depth

    return videodepth_decoder


def construct_depth_decoder(sequence_length: int = 0):
    """Construct depth decoder.

    Args:
        sequence_length (int): Number of frames to decode. Set to 0 for single frame.

    Returns:
        callable: Depth decoder function
    """

    def depth_decoder(key, sample):
        """Decode depth data from sample.

        Args:
            key (str): Key of the data
            sample (dict): Sample dictionary containing depth data

        Returns:
            dict: Sample dictionary with decoded depth data
        """
        depth = exr_loader(key, sample)
        if depth is None:
            return None
        return depth

    return depth_decoder
