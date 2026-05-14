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

"""Augmentations to remove keys from the output data_dict"""

from typing import Dict, List, Optional

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.utils import log


class FilterOutputKey(Augmentor):
    """
    Keep a subset of keys in the output data_dict
    """

    def __init__(
        self,
        input_keys: List = [],
        output_keys: Optional[list] = [
            "__key__",
            "__url__",
            "dialog_str",
            "input_ids",
            "token_mask",
            "attention_mask",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
            "raw_video",  # for debugging
            "pixel_values",
            "image_grid_thw",
            "raw_image",  # for debugging
            # For collate_fn
            "pad_token_id",
            "ignore_index",
            "labels",
        ],
        text_only: bool = False,
        args: Optional[dict] = None,
    ) -> None:
        self.output_keys = output_keys
        self.text_only = text_only

    def __call__(self, data_dict: Dict) -> Dict:
        data_dict = {k: data_dict[k] for k in self.output_keys if k in data_dict}

        has_media = "pixel_values" in data_dict or "pixel_values_videos" in data_dict
        has_text = "input_ids" in data_dict and "labels" in data_dict
        is_valid_data = has_media or has_text
        if not self.text_only and not is_valid_data:
            log.critical(
                f"No media input in data_dict: {data_dict.keys()} | __url__: {data_dict['__url__']} | __key__: {data_dict['__key__']} | dialog_str: {data_dict.get('dialog_str', '')} | does not contain pixel_values or pixel_values_videos",
                rank0_only=False,
            )
            return None

        return data_dict
