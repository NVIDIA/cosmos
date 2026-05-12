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

# This script contains the code for multi-aspect ratio shard iterator

import math
import os
import random
import time
from collections import defaultdict
from copy import deepcopy

from webdataset.pytorch import IterableDataset
from webdataset.utils import pytorch_worker_info

from cosmos3._src.imaginaire.datasets.webdataset.config.schema import TarSample
from cosmos3._src.imaginaire.datasets.webdataset.utils.misc import repeat_list
from cosmos3._src.imaginaire.utils import log


class ShardlistMultiAspectRatio(IterableDataset):
    r"""
    An iterable dataset that parses and yields tar files.
    This distributor handles the multi-aspect ratio case. For the dataloader to be successful,
    each worker should load only one aspect ratio. Else, there can be a batch where two
    aspect ratios would be present which would raise an error in collate function.
    So, we design data distribution strategy so that each worker sees only one aspect ratio.
    """

    def __init__(
        self,
        shuffle: bool = True,
        split_by_node: bool = True,
        split_by_worker: bool = True,
        chunk_size: int = 1,
        resume_flag: bool = True,
        verbose: bool = False,
        is_infinite_loader: bool = False,
    ):
        r"""Create a multi-aspect ratio ShardList.
        Args:
            urls (list[TarSample]): a list of tar files along with their metadata
            epoch_shuffle (bool): Shuffles the whole epoch. If disabled, each node will see the same set of urls.
            shuffle (bool): shuffle samples before iterating.
            split_by_node (bool): split shards by node if True
            split_by_worker (bool): split shards by worker if True
            chunk_size (int): chunk size used in webdataset creation
            resume_flag (bool): If enabled, resumes from a specific iteration and epoch number.
            verbose (bool): Prints some logs if true
            is_infinite_loader (bool): If true, creates an infinite dataloader.
                So, the dataset will be only one epoch and will not terminate.
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
        self.chunk_size = chunk_size
        self.resume_flag = resume_flag
        self.is_infinite_loader = is_infinite_loader

    def set_urls(self, urls: list[TarSample]):
        self.urls = urls
        self._split_urls_by_aspect_ratio()

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

    def _split_urls_by_aspect_ratio(self):
        r"""Function for splitting urls by aspect ratio.
        We assume that urls are grouped by dataset_id. That is, data belonging to
        one dataset_id should have all data in the same aspect ratio.
        """

        url_aspect_split = defaultdict(list)

        for url in self.urls:
            dset_info = url.meta
            if "aspect_ratio" not in dset_info.opts:
                raise ValueError("aspect_ratio should be specified in dataset_info when using multi aspect distributor")
            aspect_ratio = dset_info.opts["aspect_ratio"]
            url_aspect_split[aspect_ratio].append(url)

        aspect_ratio_with_most_elems = -1
        aspect_ratio_with_least_elems = -1
        max_aspect_ratio_count = -1
        min_aspect_ratio_count = 1000000000

        for aspect_ratio in url_aspect_split:
            # Sort the url list
            url_aspect_split[aspect_ratio] = sorted(
                url_aspect_split[aspect_ratio], key=lambda tar: (tar.path, tar.root)
            )

            # Finding max and min tar counts per aspect ratio
            if len(url_aspect_split[aspect_ratio]) > max_aspect_ratio_count:
                aspect_ratio_with_most_elems = aspect_ratio
                max_aspect_ratio_count = len(url_aspect_split[aspect_ratio])
            if len(url_aspect_split[aspect_ratio]) < min_aspect_ratio_count:
                aspect_ratio_with_least_elems = aspect_ratio
                min_aspect_ratio_count = len(url_aspect_split[aspect_ratio])

        self.url_aspect_split = url_aspect_split
        self.aspect_ratio_with_most_elems = aspect_ratio_with_most_elems
        self.aspect_ratio_with_least_elems = aspect_ratio_with_least_elems

    def _ddp_equalize(
        self, url_aspect_split: dict[str, list[TarSample]], nworkers_all: int
    ) -> tuple[dict[str, list[TarSample]], int]:
        r"""This function performs tar file equalization. That is, we repeat the number of tars in each aspect
        ratio so that when the tars are split across workers, each worker recieves the same number of tars.
        This function is important for ddp to terminate well at the end of each epoch.

        Args:
            url_aspect_split (dict[list[TarSample]]): TarSample split by aspect ratio
            nworkers_all (int): Total number of dataloader workers

        Returns:
            url_aspect_split (dict[list[TarSample]]): TarSample split after DDP equalization
            num_urls_per_worker (int): Number of tars in each worker
        """
        betas = []
        n_total = sum([len(url_aspect_split[aspect_ratio]) for aspect_ratio in url_aspect_split])

        # Initial assignment
        aspect_ind_with_most_elems = 0
        for i, aspect_ratio in enumerate(url_aspect_split):
            betas.append(math.ceil((len(url_aspect_split[aspect_ratio]) / n_total) * nworkers_all))
            if aspect_ratio == self.aspect_ratio_with_most_elems:
                aspect_ind_with_most_elems = i

        # Constraint that total number of workers is fixed
        betas[aspect_ind_with_most_elems] += nworkers_all - sum(betas)

        # Rebalance the number of urls
        num_urls_per_worker = math.ceil(n_total / sum(betas))
        for i, aspect_ratio in enumerate(url_aspect_split):
            url_aspect_split[aspect_ratio] = repeat_list(url_aspect_split[aspect_ratio], betas[i] * num_urls_per_worker)

        return url_aspect_split, num_urls_per_worker

    def _obtain_node_worker_url_mapping(
        self,
        url_aspect_split: dict[str, list[TarSample]],
        num_urls_per_worker: int,
        rank: int,
        world_size: int,
        worker_id: int,
        num_workers: int,
    ):
        r"""This function obtains the worker-URL mapping. It assigns the tar list seen by
        each workers.

        Args:
            url_aspect_split (dict[list[TarSample]]: TarSample split by aspect ratio
            num_urls_per_worker (int): Number of tar files seen by each worker
            rank (int): Rank of the current GPU
            world_size (int): Total number of GPUs
            worker_id (int): ID for the current worker in the dataloader
            num_workers (int): Total number of workers in the dataloader

        Returns:
            URL list for the current worker
        """
        assert self.split_by_node is True and self.split_by_worker is True

        # First chunk the tars
        chunk_mappings = []
        for aspect_ratio in url_aspect_split:
            samples_asp = url_aspect_split[aspect_ratio]
            nchunks_asp = int(len(samples_asp) / num_urls_per_worker)
            for chunk_id in range(nchunks_asp):
                chunk_mappings.append((aspect_ratio, samples_asp[chunk_id::nchunks_asp]))

        # Split by rank and workers
        chunk_mappings = chunk_mappings[rank::world_size]
        chunk_mappings = chunk_mappings[worker_id::num_workers]

        assert len(chunk_mappings) == 1
        return chunk_mappings[0][1]

    def obtain_url_list(self):
        r"""Return an iterator over the shards."""

        rank, world_size, worker_id, num_workers = pytorch_worker_info()

        # Setting epoch and start index
        if self.resume_flag:
            self.epoch = int(os.environ.get("WDS_EPOCH_NUM", 0))

            # This tells us number of chunks that have been seen by one GPU
            self.start_index = int(os.environ.get("WDS_START_INDEX", 0)) // self.chunk_size

        urls = deepcopy(self.urls)
        url_aspect_split = deepcopy(self.url_aspect_split)

        # Splitting the shards by worker and node
        if self.verbose:
            log.info(f"PytorchShardList rank {rank} of {world_size}")
            log.info(f"PytorchShardList worker {worker_id} of {num_workers}")

        nworkers_all = world_size * num_workers

        # Perform DDP equalization
        url_aspect_split, num_urls_per_worker = self._ddp_equalize(url_aspect_split, nworkers_all)

        # Form a mapping of url_aspect_split to node and workers
        urls = self._obtain_node_worker_url_mapping(
            url_aspect_split, num_urls_per_worker, rank, world_size, worker_id, num_workers
        )

        if self.verbose:
            log.info("List of urls (before shuffle)")
            log.info(urls[0:10])

        if self.shuffle:
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
            while True:
                cur_time = time.time_ns()
                random.Random(cur_time).shuffle(url_list)
                for url in url_list:
                    yield dict(url=url)
        else:
            for url in url_list:
                yield dict(url=url)
