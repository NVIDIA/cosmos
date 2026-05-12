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
import os
from typing import Iterator

import attrs

from cosmos3._src.imaginaire.datasets.webdataset.config.schema import SampleInfo


def repeat_list(x: list, n: int) -> list:
    r"""Function to repeat the list to a fixed shape.
    n is the desired length of the extended list.
    Args:
        x (list): Input list
        n (int): Desired length
    Returns:
        Extended list
    """
    if n == 0:
        return []
    assert len(x) > 0

    x_extended = []
    while len(x_extended) < n:
        x_extended = x_extended + x
    x_extended = x_extended[0:n]

    return x_extended


def remove_extensions_from_keys(data: Iterator[dict]) -> Iterator[dict]:
    r"""Function to remove extension from keys
    Args:
        data (dict): Input data dict
    Returns:
        data dict with keys removed
    """

    for data_dict in data:
        data_dict_remapped = dict()

        for key in data_dict:
            key_split = key.split(".")
            if len(key_split) > 1:
                key_new = ".".join(key_split[:-1])
            else:
                key_new = key
            data_dict_remapped[key_new] = data_dict[key]

        yield data_dict_remapped


def update_url(data: Iterator[dict]) -> Iterator[dict]:
    r"""Function to update the URLs so that the TarSample is removed from data.
    Instead, we replace the URL with a string.
    Args:
        data (dict): Input data dict
    Returns:
        data dict with URL replaced with a string
    """
    for data_dict in data:
        # unpack meta information from TarSample
        url = data_dict["__url__"]
        sample_meta = url.sample_meta
        if sample_meta is not None:
            assert isinstance(sample_meta, SampleInfo)
            data_dict.update(attrs.asdict(sample_meta))

        # unpack url
        data_dict["__url__"] = os.path.join(url.root, url.path)
        yield data_dict


def skip_keys(data: Iterator[dict]) -> Iterator[dict]:
    r"""
    Function to skip keys
    Args:
        data (dict): Input data dict
    Returns:
        data_dict with keys skipped
    """

    for data_dict in data:
        if ("keys_to_skip" in data_dict) and (int(data_dict["keys_to_skip"]) == 1):
            # Skip this key if data_dict["skip_key"] is True
            continue
        else:
            yield data_dict
