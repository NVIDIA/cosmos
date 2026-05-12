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
import random
import time

from webdataset.pytorch import IterableDataset
from webdataset.utils import pytorch_worker_info

from cosmos3._src.imaginaire.datasets.webdataset.config.schema import TarSample
from cosmos3._src.imaginaire.datasets.webdataset.utils.misc import repeat_list
from cosmos3._src.imaginaire.utils import log


class ShardlistBasic(IterableDataset):
    r"""
    An iterable dataset that parses and yields tar files.
    The dataset restored from an iteration number and index number.
    """

    def __init__(
        self,
        shuffle: bool = True,
        split_by_node: bool = True,
        split_by_worker: bool = True,
        resume_flag: bool = True,
        verbose: bool = False,
        is_infinite_loader: bool = False,
        max_epochs: int = 100000,
        repeat_url: bool = True,
    ):
        r"""Create a ShardList.
        Args:
            shuffle (bool): shuffle samples before iterating.
            split_by_node (bool): split shards by node if True
            split_by_worker (bool): split shards by worker if True
            resume_flag (bool): If enabled, resumes from a specific iteration and epoch number.
            verbose (bool): Prints some logs if true
            is_infinite_loader (bool): If true, creates an infinite dataloader.
                So, the dataset will be only one epoch and will not terminate.
            max_epochs (int): Infinite dataloader is created with max_epochs number of epochs.
                Should be a very large number.
            repeat_url (bool): If true, each worker will receive the same number of batches by repeating urls.
        """
        super().__init__()

        self.verbose = verbose
        if self.verbose:
            log.info("ShardListWithResumes init")
        self.epoch = 0
        self.start_index = 0
        self.shuffle = shuffle
        self.split_by_node = split_by_node
        self.split_by_worker = split_by_worker
        self.resume_flag = resume_flag
        self.is_infinite_loader = is_infinite_loader
        self.max_epochs = max_epochs
        self.repeat_url = repeat_url

    def set_urls(self, urls: list[TarSample]):
        """Set urls

        Args:
            urls (list[TarSample]): a list of tar files along with their metadata
        """
        self.urls = urls

    def set_chunk_size(self, chunk_size: int):
        """Set chunk size

        Args:
            chunk_size (int): chunk size used in webdataset creation
        """
        self.chunk_size = chunk_size

    def set_epoch(self, epoch: int, start_index: int):
        r"""Set the current epoch. Used for per-node shuffling.
        Args:
            epoch (int): Epoch number
            start_index (int): iteraton number
        """
        self.epoch = epoch
        self.start_index = start_index

    def obtain_url_list(self):
        r"""Return an iterator over the shards."""

        rank, world_size, worker_id, num_workers = pytorch_worker_info()

        # Setting epoch and start index
        if self.resume_flag:
            self.epoch = int(os.environ.get("WDS_EPOCH_NUM", 0))
            # This tells us number of chunks that have been seen by one GPU
            self.start_index = int(os.environ.get("WDS_START_INDEX", 0)) // self.chunk_size

        urls = self.urls
        num_urls = len(urls)

        if self.repeat_url:
            # Extending urls so that each workers receive the same number of batches.
            # This serves the job of ddp_equalize.
            nworkers_all = world_size * num_workers
            num_urls_per_process = (num_urls + nworkers_all - 1) // nworkers_all
            extended_url_list_size = num_urls_per_process * nworkers_all
            urls = repeat_list(urls, extended_url_list_size)

        # Splits the urls by node and worker id. This ensures each worker sees different urls.
        if self.split_by_node:
            urls = urls[rank::world_size]
        if self.split_by_worker:
            urls = urls[worker_id::num_workers]

        if self.verbose:
            log.info("List of urls (before shuffle)")
            log.info(urls[0:10])

        if self.shuffle:
            # Shuffle based on the world worker id.
            random.Random(rank * num_workers + worker_id).shuffle(urls)

        # This tells us the number of chunks seen by one worker.
        # Do not iterate over the seen chunks.
        start_index_per_worker = self.start_index // num_workers
        if not self.is_infinite_loader:
            urls = urls[start_index_per_worker:]

        if self.verbose:
            log.info("List of urls (after shuffle)")
            log.info(urls[0:10])
            log.info(f"PytorchShardList got {len(urls)} urls")

        return urls

    def __iter__(self):
        url_list = self.obtain_url_list()

        if self.is_infinite_loader:
            for _ in range(self.max_epochs):
                cur_time = int(time.time())
                random.Random(cur_time).shuffle(url_list)
                for url in url_list:
                    yield dict(url=url)
        else:
            for url in url_list:
                yield dict(url=url)
