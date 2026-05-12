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

import webdataset

import cosmos3._src.imaginaire.datasets.webdataset.webdataset
from cosmos3._src.imaginaire.utils.distributed import get_world_size


class Sampler:
    r"""
    A sampler function for setting the epoch number and iteration number.
    In webdataset, information is propagated using environment flags.
    In our case,
        WDS_EPOCH_NUM: Epoch number
        WDS_START_INDEX: Start index in this epoch.
    """

    def __init__(self, mode: str):
        self.mode = mode
        assert self.mode in ["train", "val"]

    def set_epoch(self, epoch: int):
        if self.mode == "train":
            os.environ["WDS_EPOCH_NUM"] = str(epoch)
        else:
            pass

    def set_iteration(self, start_index: int):
        # start_index should be iters * batch_size
        # It is the number of samples that have been seen by one GPU
        if self.mode == "train":
            os.environ["WDS_START_INDEX"] = str(start_index)
        else:
            pass


class DataLoader(webdataset.WebLoader):
    r"""
    This class is a wrapper on webloader class with a len attribute.
    len function is needed in Imaginaire dataloaders.
    """

    def __init__(self, dataset: cosmos3._src.imaginaire.datasets.webdataset.webdataset.Dataset, batch_size: int = 1, *args, **kw):  # type: ignore
        # Setting data length. Webdataset is an iterable dataset, so it does not have data_len attr.
        # So, we compute it from dataset and set it.
        dataset_obj = dataset.build_dataset()
        world_size = get_world_size()
        if dataset_obj.total_images < world_size * batch_size:  # type: ignore
            data_length = 1
        else:
            data_length = dataset_obj.total_images // (world_size * batch_size)  # type: ignore
        self.data_len = data_length

        super().__init__(dataset_obj, batch_size, *args, **kw)

    def __len__(self) -> int:
        return self.data_len
