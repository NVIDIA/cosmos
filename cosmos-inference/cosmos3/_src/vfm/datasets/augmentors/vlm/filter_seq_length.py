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

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos3._src.imaginaire.utils import log
from projects.cosmos3.vlm.processors.qwen3vl_processor import Qwen3VLProcessor


class FilterSeqLength(Augmentor):
    """
    Check the sequence length of the input data_dict and filter out the samples that are too long (TODO: Instead of removing them, we can truncate the input ids, but need to make sure the image tokens are not truncated)
    """

    def __init__(
        self,
        input_keys: List = ["input_ids"],
        output_keys: Optional[list] = ["input_ids"],
        max_token_length: int = 24000,
        processor: Qwen3VLProcessor = None,
    ) -> None:
        self.max_token_length = max_token_length
        self.processor = processor

    def __call__(self, data_dict: Dict) -> Dict:
        input_ids = data_dict["input_ids"]
        if input_ids.shape[-1] > self.max_token_length:
            # check if there is pixel values or pixel value videos in the remaining tokens, if not truncate the input ids
            input_ids_extra = input_ids[self.max_token_length :]
            has_video_tokens = sum(input_ids_extra == self.processor.video_token_id) > 0
            has_image_tokens = sum(input_ids_extra == self.processor.image_token_id) > 0
            if not has_video_tokens and not has_image_tokens:
                log.debug(
                    f"Truncating input_ids from {input_ids.shape[-1]} to {self.max_token_length} because there are no video or image tokens in the remaining tokens | __url__: path={data_dict['__url__'].path} root={data_dict['__url__'].root} | __key__: {data_dict['__key__']} | dialog_str: {data_dict.get('dialog_str', '')}"
                )
                data_dict["input_ids"] = data_dict["input_ids"][: self.max_token_length]
                data_dict["token_mask"] = data_dict["token_mask"][: self.max_token_length]
                data_dict["attention_mask"] = data_dict["attention_mask"][: self.max_token_length]
                data_dict["labels"] = data_dict["labels"][: self.max_token_length]
                return data_dict

        if input_ids.shape[-1] > self.max_token_length:
            msg = f"Input ids length {input_ids.shape[-1]} is greater than max token length {self.max_token_length} | __url__: path={data_dict['__url__'].path} root={data_dict['__url__'].root} | __key__: {data_dict['__key__']} | dialog_str: {data_dict.get('dialog_str', '')}"
            if "pixel_values" in data_dict:
                msg += f" | pixel_values: {data_dict['pixel_values'].shape}"
            if "pixel_values_videos" in data_dict:
                msg += f" | pixel_values_videos: {data_dict['pixel_values_videos'].shape}"
            log.critical(msg, rank0_only=False)
            return None
        return data_dict
