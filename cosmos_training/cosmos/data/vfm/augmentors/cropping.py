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

from typing import Optional

import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor


class CropToMultiple(Augmentor):
    """Crops images/videos to the nearest multiple of a specified value using center crop.

    This augmentor crops the height and width of images/videos to be divisible by
    a given multiple (default 16). The crop is centered, removing equal amounts
    from opposite edges.

    Supports:
        - PIL Images (for image data)
        - Torch tensors with shape (C, H, W) or (C, T, H, W) (for video data)

    Example:
        Input: 209x187 with multiple=16
        Output: 208x176 (center cropped to nearest lower multiple of 16)
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.multiple = 16
        if self.args is not None and "multiple" in self.args:
            self.multiple = self.args["multiple"]

    def __call__(self, data_dict: dict) -> dict:
        """Center crops images/videos to the nearest multiple of the specified value.

        Args:
            data_dict (dict): Input data dict containing images/videos to crop.

        Returns:
            data_dict (dict): Output dict with center cropped images/videos.
        """
        for key in self.input_keys:
            if key not in data_dict:
                continue

            data = data_dict[key]

            # Get dimensions based on data type
            if isinstance(data, Image.Image):
                # PIL Image: size returns (width, height)
                w, h = data.size
            elif isinstance(data, torch.Tensor):
                # Torch tensor: (C, H, W) or (C, T, H, W)
                if data.ndim == 3:
                    _, h, w = data.shape
                elif data.ndim == 4:
                    _, _, h, w = data.shape
                else:
                    raise ValueError(f"Unexpected tensor dimensions: {data.ndim}, expected 3 or 4")
            else:
                raise ValueError(f"Unexpected data type: {type(data)}, expected PIL Image or torch Tensor")

            # Calculate new dimensions (nearest lower multiple)
            new_h = (h // self.multiple) * self.multiple
            new_w = (w // self.multiple) * self.multiple

            # Center crop: calculate offsets to center the crop
            if new_h != h or new_w != w:
                top = (h - new_h) // 2
                left = (w - new_w) // 2
                # log.info(f"Data cropped from ({h}, {w}) to ({new_h}, {new_w})")
                data_dict[key] = transforms_F.crop(data, top=top, left=left, height=new_h, width=new_w)

            # Store final dimensions for downstream use (e.g., resolution text info)
            data_dict["final_height"] = new_h
            data_dict["final_width"] = new_w

        return data_dict
