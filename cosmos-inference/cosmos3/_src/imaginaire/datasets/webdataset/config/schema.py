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

from typing import Optional, Type

import attrs
from torch.utils.data import IterableDataset

from cosmos3._src.imaginaire import config
from cosmos3._src.imaginaire.config import make_freezable
from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor


@make_freezable
@attrs.define(slots=False)
class DatasetInfo:
    object_store_config: config.ObjectStoreConfig  # Object strore config
    wdinfo: list[str]  # List of wdinfo files
    opts: dict = attrs.Factory(dict)  # Additional dataset info args
    per_dataset_keys: list[str] = attrs.Factory(list)  # List of keys per dataset
    source: str = ""  # data source


@make_freezable
@attrs.define(slots=False)
class SampleInfo:
    dataset_name: str
    sample_rank: int = 0
    sample_worker_id: int = 0
    sample_epoch: int = 0
    sample_index: int = 0


@make_freezable
@attrs.define(slots=False)
class TarSample:
    path: str  # Path to the sample
    root: str  # Root folder
    keys: list  # List of keys to be loaded from the webdataset
    meta: DatasetInfo  # Metadata
    dset_id: str  # Dataset id
    num_samples: int = 0  # Number of samples in this tar file (data_list_key_count from wdinfo)
    sample_keys_full_list: str = None  # Path to the file containing full sample keys for the tar file
    sample_meta: SampleInfo = None


@make_freezable
@attrs.define(slots=False)
class Wdinfo:
    tar_files: list[TarSample]  # List of all tar samples
    total_key_count: int  # Total number of elements present in the dataset
    chunk_size: int  # Number of elements present in each tar


@make_freezable
@attrs.define(slots=False)
class AugmentorConfig:
    # Type of augmentor
    type: Type[Augmentor]
    # Input keys used by the augmentor
    input_keys: list[str]
    # Output keys returned by the augmentor
    output_keys: Optional[list[str]] = None
    # Additional arguments used by the augmentor
    args: Optional[dict] = None

    def make_instance(self) -> Augmentor:
        return self.type(input_keys=self.input_keys, output_keys=self.output_keys, args=self.args)


@make_freezable
@attrs.define(slots=False)
class DatasetConfig:
    keys: list[str]  # List of keys used
    buffer_size: int  # Buffer size used by each worker
    dataset_info: list[DatasetInfo]  # List of dataset info files, one for each dataset
    distributor: IterableDataset  # Iterator for returning list of tar files
    decoders: list  # List of decoder functions for decoding bytestream
    augmentation: dict[str, AugmentorConfig]  # Dictionary containing all augmentations
    streaming_download: bool = True  # Whether to use streaming loader
    remove_extension_from_keys: bool = True  # True: objects will have a key of data_type; False: data_type.extension
    sample_keys_full_list_path: Optional[str] = (
        None  # Path to the file containing all keys present in the dataset, e.g., "index"
    )
