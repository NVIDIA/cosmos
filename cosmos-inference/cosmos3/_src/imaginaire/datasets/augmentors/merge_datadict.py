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

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos3._src.imaginaire.utils import log


class DataDictMerger(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Merge the dictionary associated with the input keys into data_dict. Only keys in output_keys are merged.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with dictionary associated with the input keys merged.
        """
        for key in self.input_keys:
            if key not in data_dict:
                log.warning(
                    f"DataDictMerger dataloader error: missing {key}, {data_dict['__url__']}, {data_dict['__key__']}",
                    rank0_only=False,
                )
                return None
            key_dict = data_dict.pop(key)
            if key == "depth" and "depth" in self.output_keys:
                data_dict["depth"] = key_dict
            if key == "human_annotation" and "human_annotation" in self.output_keys:
                data_dict["human_annotation"] = key_dict
            elif key == "segmentation" and "segmentation" in self.output_keys:
                data_dict["segmentation"] = key_dict
            elif key == "canny" and "canny" in self.output_keys:
                data_dict["canny"] = key_dict
            for sub_key in key_dict:
                if sub_key in self.output_keys and sub_key not in data_dict:
                    data_dict[sub_key] = key_dict[sub_key]
            del key_dict
        return data_dict
