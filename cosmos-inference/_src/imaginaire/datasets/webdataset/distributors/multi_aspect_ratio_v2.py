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

import random
import time
from collections import defaultdict

import numpy as np
from webdataset.pytorch import IterableDataset
from webdataset.utils import pytorch_worker_info

from cosmos3._src.imaginaire.datasets.webdataset.config.schema import TarSample
from cosmos3._src.imaginaire.utils import log


class ShardlistMultiAspectRatioInfinite(IterableDataset):
    r"""
    An iterable dataset that parses and yields tar files.
    This distributor handles the multi-aspect ratio case. For the dataloader to be successful,
    each worker should load only one aspect ratio. Else, there can be a batch where two
    aspect ratios would be present which would raise an error in collate function.
    So, we design data distribution strategy so that each worker sees only one aspect ratio.

    This version only supports infinite loader mode. This enables a simpler code that is faster to initialize
    and produces samples better matching the dataset distribution.
    """

    def __init__(
        self,
        shuffle: bool = True,
        split_by_node: bool = True,
        split_by_worker: bool = True,
        chunk_size: int = 1,
        resume_flag: bool = True,
        verbose: bool = False,
        is_infinite_loader: bool = True,
    ):
        r"""Create a multi-aspect ratio ShardList.
        Args:
            urls (list[TarSample]): a list of tar files along with their metadata
            epoch_shuffle (bool): Shuffles the whole epoch. If disabled, each node will see the same set of urls.
            shuffle (bool): shuffle samples before iterating.
            split_by_node (bool): split shards by node if True
            split_by_worker (bool): split shards by worker if True
            chunk_size (int): Ignored
            resume_flag (bool): Ignored
            verbose (bool): Prints some logs if true
            is_infinite_loader (bool): If true, creates an infinite dataloader.
                So, the dataset will be only one epoch and will not terminate.
        """
        super().__init__()

        self.verbose = verbose
        if self.verbose:
            log.info("ShardlistMultiAspectRatioInfinite init")
        self.shuffle = shuffle
        self.split_by_node = split_by_node
        self.split_by_worker = split_by_worker
        self.chunk_size = chunk_size  # Ignored
        self.resume_flag = resume_flag  # Ignored
        assert is_infinite_loader is True

    def set_urls(self, urls: list[TarSample]):
        self.url_aspect_split = self._split_urls_by_aspect_ratio(urls)

    def set_chunk_size(self, chunk_size: int):
        """Set chunk size
        For backward compatibility. Ignored.

        Args:
            chunk_size (int): chunk size used in webdataset creation
        """
        self.chunk_size = chunk_size

    def set_epoch(self, epoch: int, start_index: int):
        r"""Set the current epoch. Used for per-node shuffling.
        For backward compatibility. Ignored.

        Args:
            epoch (int): Epoch number
            start_index (int): iteraton number
        """
        self.epoch = epoch
        self.start_index = start_index

    def _split_urls_by_aspect_ratio(self, urls):
        r"""Function for splitting urls by aspect ratio.
        We assume that urls are grouped by dataset_id. That is, data belonging to
        one dataset_id should have all data in the same aspect ratio.
        """

        url_aspect_split = defaultdict(list)

        for url in urls:
            dset_info = url.meta
            if "aspect_ratio" not in dset_info.opts:
                raise ValueError("aspect_ratio should be specified in dataset_info when using multi aspect distributor")
            aspect_ratio = dset_info.opts["aspect_ratio"]
            url_aspect_split[aspect_ratio].append(url)

        for aspect_ratio in url_aspect_split:
            # Sort the url list
            url_aspect_split[aspect_ratio] = sorted(
                url_aspect_split[aspect_ratio], key=lambda tar: (tar.path, tar.root)
            )

        return url_aspect_split

    def _allocate_workers_to_aspects(
        self, url_aspect_split: dict[str, list[TarSample]], num_workers_all: int
    ) -> list[tuple[str, int]]:
        r"""Allocate workers to each aspect ratio so that:
        1. Each aspect ratio has at least one worker
        2. All the workers have jobs to do

        Args:
            url_aspect_split (dict[list[TarSample]]): TarSample split by aspect ratio
            num_workers_all (int): Total number of dataloader workers

        Returns:
            aspect_worker_allocation (list): List of tuple containing (aspect_key, num_workers)
        """
        if self.verbose:
            log.info(
                f"#URLs for each aspect ratio: {[len(url_aspect_split[aspect_ratio]) for aspect_ratio in url_aspect_split]}"
            )

        # Must have more global workers than the number of aspect ratios, as each global worker can only load a single
        # aspect ratio.
        num_aspects = len(url_aspect_split)
        assert num_workers_all >= num_aspects

        aspect_keys = list(url_aspect_split.keys())
        # Allocate at least one worker per aspect ratios
        target_ratio = np.array([len(url_aspect_split[key]) for key in aspect_keys])
        target_ratio = target_ratio / target_ratio.sum()
        aspect_worker_allocation = np.ones([num_aspects], dtype=np.int64)
        for _i in range(num_workers_all - num_aspects):
            current_ratio = aspect_worker_allocation / aspect_worker_allocation.sum()
            aspect_worker_allocation[np.argmin(current_ratio - target_ratio)] += 1

        if self.verbose:
            log.info(f"Aspects: {aspect_keys}")
            log.info(f"Target ratio: {target_ratio}")
            log.info(f"Worker allocation: {aspect_worker_allocation}")
            log.info(f"Discrepancy: {aspect_worker_allocation / aspect_worker_allocation.sum() / target_ratio}")
        return [(k, v) for k, v in zip(aspect_keys, aspect_worker_allocation.tolist())]

    def _obtain_node_worker_url_mapping(
        self,
        url_aspect_split: dict[str, list[TarSample]],
        aspect_worker_allocation: list[tuple[str, int]],
        rank: int,
        world_size: int,
        worker_id: int,
        num_workers: int,
    ):
        r"""This function obtains the worker-URL mapping. It assigns the tar list seen by
        each workers.

        Args:
            url_aspect_split (dict[list[TarSample]]: TarSample split by aspect ratio
            aspect_worker_allocation (dict): Number of workers allocated to each aspect ratio
            rank (int): Rank of the current GPU
            world_size (int): Total number of GPUs
            worker_id (int): ID for the current worker in the dataloader
            num_workers (int): Total number of workers in the dataloader

        Returns:
            URL list for the current worker
        """
        assert self.split_by_node is True and self.split_by_worker is True

        # First determine the aspect ratio for the current worker
        global_worker_id = rank * num_workers + worker_id

        cumulative = 0
        for aspect_key, worker_count in aspect_worker_allocation:
            cumulative += worker_count
            if global_worker_id < cumulative:
                chunk_id = global_worker_id - cumulative + worker_count
                break

        if self.verbose:
            log.info(f"GID={global_worker_id}, aspect_key={aspect_key}, chunk_id={chunk_id}")
        # chunk the urls for the target aspect ratio
        urls_asp = url_aspect_split[aspect_key]
        if len(urls_asp) >= worker_count:
            url_chunk = urls_asp[chunk_id::worker_count]
        else:
            url_chunk = urls_asp[chunk_id % len(urls_asp) : chunk_id % len(urls_asp) + 1]

        return url_chunk

    def obtain_url_list(self):
        r"""Return an iterator over the shards."""

        rank, world_size, worker_id, num_workers = pytorch_worker_info()

        # Splitting the shards by worker and node
        if self.verbose:
            log.info(f"PytorchShardList rank {rank} of {world_size}")
            log.info(f"PytorchShardList worker {worker_id} of {num_workers}")

        nworkers_all = world_size * num_workers

        # Assigning workers to process each aspect ratio
        aspect_worker_allocation = self._allocate_workers_to_aspects(self.url_aspect_split, nworkers_all)

        # Form a mapping of url_aspect_split to node and workers
        urls = self._obtain_node_worker_url_mapping(
            self.url_aspect_split, aspect_worker_allocation, rank, world_size, worker_id, num_workers
        )

        if self.verbose:
            log.info("List of urls (before shuffle)")
            log.info(urls[0:10])

        if self.shuffle:
            global_worker_id = rank * num_workers + worker_id
            random.Random(global_worker_id).shuffle(urls)

        if self.verbose:
            log.info("List of urls (after shuffle)")
            log.info(urls[0:10])
            log.info(f"PytorchShardList got {len(urls)} urls")

        return urls

    def __iter__(self):
        url_list = self.obtain_url_list()
        while True:
            if self.shuffle:
                cur_time = time.time_ns()
                random.Random(cur_time).shuffle(url_list)
            assert len(url_list) > 0, "No urls found"
            for url in url_list:
                yield dict(url=url)
