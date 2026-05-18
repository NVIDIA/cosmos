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

from typing import Callable, Optional

import omegaconf
import webdataset as wds
from webdataset import filters
from webdataset.handlers import reraise_exception

from imaginaire.datasets.webdataset.config.schema import DatasetConfig
from imaginaire.datasets.webdataset.utils.iterators import WebDataset
from imaginaire.datasets.webdataset.utils.misc import remove_extensions_from_keys, skip_keys, update_url
from imaginaire.datasets.webdataset.webdataset import Dataset as BaseDataset
from imaginaire.datasets.webdataset.webdataset import _sample_timer
from cosmos.utils import log


class Dataset(BaseDataset):
    def __init__(
        self,
        config: DatasetConfig,
        handler: Callable = reraise_exception,
        decoder_handler: Optional[Callable] = None,
        detshuffle: bool = False,
    ):
        r"""Webdataloader class

        Args:
            config: Dataset config
            handler (Callable): Error handler for webdataset class
            decoder_handler (Callable): Error handler during decoding
        """
        super().__init__(config=config, handler=handler)
        self.decoder_handler = decoder_handler
        self.detshuffle = detshuffle

    def build_dataset(self, **kwargs) -> WebDataset:
        r"""
        Build the dataset object.
        The function only diffs from BaseDataset.build_dataset by only adding the decoder_handler to the WebDataset object.
        """
        tar_list = self.wdinfo.tar_files
        num_tars = len(tar_list)
        assert num_tars > 0, "Did not find any data."

        shuffle_buffer_size = getattr(self.config, "buffer_size", self.wdinfo.chunk_size)

        # update distributor urls and chunk size
        distributor_fn = self.config.distributor

        distributor_fn.set_urls(tar_list)
        distributor_fn.set_chunk_size(self.wdinfo.chunk_size)

        dataset = WebDataset(
            distributor_fn,
            load_from_object_store=self.use_object_store,
            s3_client=self.s3_client,
            s3_bucket_name=self.bucket,
            streaming_download=self.streaming_download,
            handler=self.handler,
        )

        # Creating a shuffle buffer
        if self.detshuffle:
            dataset.append(filters.detshuffle(shuffle_buffer_size))
        else:
            dataset.append(wds.shuffle(shuffle_buffer_size))

        # Adding decoders
        # Decoders are functions that decode the input IO stream
        decoder_list = getattr(self.config, "decoders", [])
        decoder_functions = []
        for decoder in decoder_list:
            # If the specified decoder is a string, use the webdataset decoder
            # If its a callable function, use the defined function to decode data
            assert isinstance(decoder, str) or callable(decoder), "Decoder should either be callable or a str"
            decoder_functions.append(decoder)
        dataset.append(wds.decode(*decoder_functions, handler=self.decoder_handler))

        # After the decoders are added, remove extension from the keys
        # Extensions in the data keys are needed for auto-detection of decoders in webdataset.
        if self.config.remove_extension_from_keys:
            dataset.append(remove_extensions_from_keys)

        # Function to skip keys
        dataset.append(skip_keys)
        # Building augmentors
        augmentor_cfg = getattr(self.config, "augmentation", None)
        assert isinstance(augmentor_cfg, (dict, omegaconf.dictconfig.DictConfig)), (
            f"getting type: {type(augmentor_cfg)}"
        )
        augmentation_fn = self.build_data_augmentor(augmentor_cfg)
        dataset.append(augmentation_fn)

        # Updates URL names so that the collate function can handle
        dataset.append(update_url)

        dataset.append(_sample_timer)

        dataset.total_images = self.wdinfo.total_key_count  # type: ignore
        log.info("Total number of training shards: %d" % num_tars)
        log.info("Total training key count: %d" % dataset.total_images)  # type: ignore

        return dataset
