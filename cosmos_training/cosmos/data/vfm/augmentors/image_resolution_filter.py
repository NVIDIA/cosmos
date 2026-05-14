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

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.data.vfm.utils import IMAGE_RES_SIZE_INFO

# Map dataset_resolution_type to resolution tier key in IMAGE_RES_SIZE_INFO
_DATASET_RESOLUTION_TIER: dict[str, str] = {"gt480p": "480", "gt720p": "720", "gt1080p": "1080"}


class ImageResolutionFilter(Augmentor):
    """
    Filters out image samples whose (width, height) are below the minimum for
    the sample's aspect ratio when dataset_resolution_type is not "all".
    Mirrors the resolution check used in video_parsing.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.image_key = args.get("image_key", "images") if args else "images"
        self.dataset_resolution_type = args.get("dataset_resolution_type", "all") if args else "all"
        self.resolution_tier = _DATASET_RESOLUTION_TIER.get(self.dataset_resolution_type)

    def __call__(self, data_dict: dict) -> dict | None:
        image = data_dict.get(self.image_key)
        if image is None:
            return data_dict

        # PIL Image has .size as (width, height)
        width, height = image.size

        aspect_ratio: str | None = None
        if "__url__" in data_dict:
            aspect_ratio = data_dict["__url__"].meta.opts["aspect_ratio"]

        # If the resolution of the image is smaller than the minimum resolution for the aspect ratio, skip the sample. This will ensure that we do not upsample any image.
        if self.resolution_tier is not None:
            min_w, min_h = IMAGE_RES_SIZE_INFO[self.resolution_tier][aspect_ratio]
            if width < min_w and height < min_h:
                return None

        return data_dict
