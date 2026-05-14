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

"""
Augmentations to randomly swap media/text order in user prompts if there is only one video/image and one text messeage.
Default swap probability is 1%.
"""

import random
from typing import Dict, Optional

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.utils import log


class ShuffleTextMediaOrder(Augmentor):
    def __init__(
        self,
        shuffle_ratio: float = 0.01,
    ) -> None:
        """
        Args:
            input_keys (list): List of input keys.
        """
        self.shuffle_ratio = shuffle_ratio

    def __call__(self, data_dict: Dict) -> Optional[Dict]:
        url = data_dict["__url__"]
        try:
            # process conversation
            conversation = data_dict["conversation"]
            for item in conversation:
                if item["role"] == "user":
                    if (
                        len(item["content"]) == 2
                        and item["content"][0]["type"] in ["video", "image"]
                        and item["content"][1]["type"] == "text"
                    ):
                        # random.shuffle(item["content"]) # randomly shuffle media and text
                        if random.random() < self.shuffle_ratio:
                            item["content"] = item["content"][::-1]
            data_dict["conversation"] = conversation
            return data_dict
        except Exception as e:
            log.warning(
                f"Error replacing invalid characters in RFT: {e}. Skipping this sample {url.root} {data_dict['__key__']}."
            )
            return None
