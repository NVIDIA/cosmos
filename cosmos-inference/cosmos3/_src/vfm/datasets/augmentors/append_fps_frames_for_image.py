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


class AppendFPSFramesForImage(Augmentor):
    def __init__(
        self, input_keys: Optional[list] = None, output_keys: Optional[list] = None, args: Optional[dict] = None
    ) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Remove the input keys from the data dict.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with keys removed.
        """
        data_dict["fps"] = 30.0  # set image model fps = 30, which is the most common fps we used to train video.
        data_dict["num_frames"] = 1
        return data_dict
