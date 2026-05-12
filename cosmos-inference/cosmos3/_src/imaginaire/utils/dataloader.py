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

from typing import Iterable, Iterator

import torch
import torch.distributed as dist
import torch.utils.data


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """A dataloader that relentlessly samples from the dataset.

    This eliminates the overhead of prefetching data before each epoch.
    Ref: https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/loader.py
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)  # type: ignore
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)  # type: ignore
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)  # type: ignore

    def __iter__(self) -> Iterable:
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """A sampler wrapper that repeats data sampling forever.

    Args:
        sampler (Sampler): Data sampler object.
    """

    def __init__(self, sampler: torch.utils.data.Sampler):
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        while True:
            yield from iter(self.sampler)


class DistributedEvalSampler(torch.utils.data.Sampler):
    """Distributed data sampler for evaluation.

    Ref: https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py (by snah)
    DistributedEvalSampler is different from DistributedSampler in that it does not pad extra samples to make it
    evenly divisible. It should not be used for training, or the distributed processes could hang forever.
    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, shuffle: bool = False, seed: int = 0):
        """Constructor of DistributedEvalSampler,

        Args:
            dataset (torch.utils.data.Dataset): Dataset used for sampling.
            shuffle (bool): Whether to shuffle the indices (default: False).
            seed (int): Random seed for shuffling if enabled (default: 0).
        """
        self.dataset = dataset
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        self.dataset_size = len(self.dataset)  # type: ignore
        indices = list(range(self.dataset_size))
        indices = indices[self.rank : self.dataset_size : self.num_replicas]
        self.num_samples = len(indices)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed.
            gen = torch.Generator()
            gen.manual_seed(self.seed)
            indices = torch.randperm(self.dataset_size, generator=gen).tolist()
        else:
            indices = list(range(self.dataset_size))
        # Subsample.
        indices = indices[self.rank : self.dataset_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
