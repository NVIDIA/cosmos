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

from __future__ import annotations

import functools
import random
import time
from typing import Callable, Iterator

import numpy as np
import torch
import torch.utils.data

from cosmos.utils import distributed
from cosmos.data.vfm.action.unified_dataset import ActionUnifiedIterableDataset
from cosmos.data.vfm.joint_dataloader import custom_collate_fn


def _action_worker_init_fn(
    worker_id: int, seed: int = 42, use_deterministic_seed: bool = True, rank: int = 0, world_size: int = 1
) -> None:
    if use_deterministic_seed:
        worker_seed = seed + rank * 9999 + worker_id
    else:
        worker_seed = int(time.time() * 1000) % (2**32) + rank * 9999 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)

    info = torch.utils.data.get_worker_info()
    assert info is not None
    dataset = info.dataset
    if isinstance(dataset, ActionUnifiedIterableDataset):
        dataset.assign_worker(worker_id, info.num_workers, rank, world_size)


def create_action_worker_init_fn(seed: int = 42, use_deterministic_seed: bool = True) -> Callable[[int], None]:
    """Create a worker_init_fn for Action training with ``ActionUnifiedIterableDataset``.

    Seeds RNGs first, then calls ``dataset.assign_worker()`` to set up
    rank-level dataset assignment and worker-level shard distribution.

    Passed to ``DataLoader`` (or ``InfiniteDataLoader``) as the
    ``worker_init_fn`` parameter.  Only called when ``num_workers > 0``.

    Args:
        seed: Base seed for deterministic worker seeding.  Ignored when
            ``use_deterministic_seed=False`` (time-based seed used instead).
        use_deterministic_seed: If True, use the provided seed for reproducible
            RNG initialization. If False, derive a time-based seed so that
            each resume sees different data. This is preferred for large-scale
            runs that resume frequently, and when ``in_order=False`` already
            makes iteration order non-deterministic.

    Returns:
        A ``worker_init_fn`` suitable for ``torch.utils.data.DataLoader``.
    """
    try:
        rank = distributed.get_rank()
        world_size = distributed.get_world_size()
    except RuntimeError:
        rank = 0
        world_size = 1

    return functools.partial(
        _action_worker_init_fn,
        seed=seed,
        use_deterministic_seed=use_deterministic_seed,
        rank=rank,
        world_size=world_size,
    )


class InfiniteDataLoader(torch.utils.data.DataLoader):
    """A dataloader that yields forever with proper seeding for reproducibility.

    All Action datasets are ``IterableDataset`` instances (map-style datasets
    are automatically wrapped by :class:`~.transforms.MapToIterableAdapter`).
    The loader catches ``StopIteration`` and restarts the iterator so that
    iteration never ends.
    """

    def __init__(
        self,
        *args,
        seed: int = 42,
        use_deterministic_seed: bool = True,
        **kwargs,
    ) -> None:
        """Initialize InfiniteDataLoader.

        Args:
            *args: Positional arguments passed to parent DataLoader.
            seed: Random seed for reproducible worker initialization.
                  Default is 42 for reproducibility.
            use_deterministic_seed: If True, use the provided seed for reproducible
                  RNG initialization. If False, derive a time-based seed so that
                  each resume sees different data. This is preferred for large-scale
                  runs that resume frequently, and when ``in_order=False`` already
                  makes iteration order non-deterministic.
            **kwargs: Keyword arguments passed to parent DataLoader.
        """
        kwargs.pop("shuffle", None)
        kwargs["shuffle"] = False

        # Default to ``custom_collate_fn`` so that variable-length per-sample
        # tensors (e.g. ``text_token_ids``) and multi-item keys (``video``,
        # ``action``, ...) are returned as lists rather than stacked by
        # PyTorch's ``default_collate``.
        if kwargs.get("collate_fn") is None:
            kwargs["collate_fn"] = custom_collate_fn

        if "worker_init_fn" not in kwargs or kwargs["worker_init_fn"] is None:
            kwargs["worker_init_fn"] = create_action_worker_init_fn(seed, use_deterministic_seed=use_deterministic_seed)

        num_workers = kwargs.get("num_workers", 0)
        if num_workers == 0:
            try:
                rank = distributed.get_rank()
            except RuntimeError:
                rank = 0
            if use_deterministic_seed:
                rank_seed = seed + rank * 9999
            else:
                rank_seed = int(time.time() * 1000) % (2**32) + rank * 9999
            random.seed(rank_seed)
            np.random.seed(rank_seed % (2**32))
            torch.manual_seed(rank_seed)

        super().__init__(*args, **kwargs)
        self._stream_iterator: Iterator | None = None

    def __len__(self) -> int:
        # Delegate to DataLoader which calls len(self.dataset).
        # Raises TypeError if the underlying dataset has no __len__.
        return super().__len__()

    def __iter__(self) -> Iterator:
        """Yield batches forever."""
        while True:
            if self._stream_iterator is None:
                self._stream_iterator = super().__iter__()
            try:
                yield next(self._stream_iterator)  # type: ignore[arg-type]
            except StopIteration:
                self._stream_iterator = super().__iter__()
                yield next(self._stream_iterator)  # type: ignore[arg-type]
