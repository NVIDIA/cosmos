# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, without express written consent of
# NVIDIA is strictly prohibited.
# -----------------------------------------------------------------------------

"""
OSS-facing dataloader that wires any Python iterable + DataPacker into the
shared PackingIterableDataset engine.

Follows the same two-layer pattern as the internal path:
  private  _DataPackerIterableDataset   ↔  private  _JointIterableDataset
  public   DataPackerDataLoader         ↔  public   JointDatasetDynamicBatchingWebLoader

Data-parallel sharding
----------------------
When ``torch.distributed`` is initialized, ``DataPackerDataLoader`` automatically
shards ``data_source`` across ranks **and** DataLoader workers using round-robin
filtering — the same pattern as ``SFTDataset`` in
``projects/cosmos3/vfm/datasets/local_datasets/sft_dataset.py``.

Each ``(dp_rank, worker_id)`` pair sees every
``dp_world_size × num_workers``-th item, giving disjoint coverage.

Usage
-----
Pass a pre-built iterable directly::

    loader = DataPackerDataLoader(
        data_source=my_dataset,           # any Python iterable
        data_packer=MyDataPacker(...),
        max_tokens=16000,
        num_workers=4,
    )

Or load a HuggingFace / local dataset via ``load_data_source`` — compatible
with Hydra ``LazyCall`` so CLI overrides work without editing Python files::

    from cosmos.utils.lazy_config import LazyCall as L
    from cosmos.data.vfm.data_packer_dataloader import (
        DataPackerDataLoader,
        load_data_source,
    )

    dataloader_train = L(DataPackerDataLoader)(
        data_source=L(load_data_source)(
            name="liuhaotian/LLaVA-Instruct-150K",
            split=["train"],
        ),
        data_packer=L(MyDataPacker)(...),
        max_tokens=16000,
    )

    # CLI override (no Python file edit needed):
    # dataloader_train.data_source.name=my-org/my-dataset
    # dataloader_train.data_source.split=[train,validation]

    # FSDP + TP/PP (pass parallel_dims for correct DP rank):
    loader = DataPackerDataLoader(
        data_source=...,
        data_packer=...,
        max_tokens=16000,
        parallel_dims=parallel_dims,  # uses parallel_dims.dp_coord
    )
"""

from __future__ import annotations

from typing import Any

import torch
import torch.utils.data

from cosmos.utils import log
from cosmos.data.vfm.data_packer import DataPacker
from cosmos.data.vfm.packing_iterable_dataset import PackingIterableDataset


def load_data_source(
    name: str,
    split: str | list[str] = "train",
    subset: str | None = None,
    revision: str | None = None,
) -> Any:
    """Load a HuggingFace or local dataset for use as ``data_source``.

    Designed to be used as a ``LazyCall`` in Hydra experiment configs so that
    dataset name and split can be overridden from the CLI without editing Python
    files (see module docstring for an example).

    Parameters
    ----------
    name:
        HuggingFace dataset name (e.g. ``"liuhaotian/LLaVA-Instruct-150K"``) or
        a local directory path to a dataset saved with ``dataset.save_to_disk()``.
        Local paths are detected via ``os.path.isdir`` and loaded with
        ``load_from_disk``; all other values go through ``load_dataset``.
    split:
        Split name or list of split names to load.  When a list is given the
        splits are concatenated into a single dataset.
    subset:
        HuggingFace dataset subset / config name (optional).
    revision:
        Git revision / commit hash of the dataset (optional).

    Returns
    -------
    datasets.Dataset
        A concatenated ``datasets.Dataset`` ready to be passed to
        ``DataPackerDataLoader`` as ``data_source``.

    Raises
    ------
    ImportError
        If the ``datasets`` package is not installed.
    """
    try:
        from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required by load_data_source. Install it with: pip install datasets"
        ) from exc

    import os

    if os.path.isdir(name):
        # Dataset saved with dataset.save_to_disk() — use load_from_disk.
        raw = load_from_disk(name)
    else:
        # HuggingFace Hub name or other format supported by load_dataset.
        raw = load_dataset(name, subset, revision=revision)

    if isinstance(raw, Dataset):
        # load_from_disk on a single Dataset (not DatasetDict) — return as-is.
        return raw

    # DatasetDict: select and concatenate requested splits.
    splits = [split] if isinstance(split, str) else split
    return concatenate_datasets([raw[s] for s in splits])


class _IterableWrapper(torch.utils.data.IterableDataset):
    """Wraps any Python iterable as a ``torch.utils.data.IterableDataset``
    with built-in data-parallel + multi-worker sharding.

    Sharding follows the same ``(dp_rank × num_workers)`` formula as
    ``SFTDataset`` — each ``(dp_rank, worker_id)`` pair receives every
    ``dp_world_size × num_workers``-th item starting at
    ``dp_rank * num_workers + worker_id``.

    .. warning::
        For ``num_workers=0``, worker-level sharding is skipped automatically.
    """

    def __init__(self, iterable: Any, dp_rank: int = 0, dp_world_size: int = 1):
        super().__init__()
        self._iterable = iterable
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers, worker_id = 1, 0

        # Total independent streams = dp_world_size × num_workers.
        # Each (rank, worker) pair owns stream = rank * num_workers + worker_id.
        total_streams = self._dp_world_size * num_workers
        my_stream = self._dp_rank * num_workers + worker_id

        for i, item in enumerate(self._iterable):
            if i % total_streams == my_stream:
                yield item


class _DataPackerIterableDataset(PackingIterableDataset):
    """Private: injects a DataPacker into the shared packing engine.

    Not registered in Hydra directly.  Use ``DataPackerDataLoader`` instead.
    """

    def __init__(
        self,
        data_source: Any,
        data_packer: DataPacker,
        max_tokens: int,
        pool_size: int,
        max_batch_size: int,
        long_threshold: int,
        batching_strategy: str,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ):
        # Always wrap through _IterableWrapper so sharding applies uniformly.
        # If data_source is already an IterableDataset it is still wrapped —
        # _IterableWrapper just adds the rank×worker filter on top.
        data_source = _IterableWrapper(data_source, dp_rank=dp_rank, dp_world_size=dp_world_size)
        datasets_cfg = {"default": {"dataset": data_source, "ratio": 1.0}}
        super().__init__(
            datasets_cfg=datasets_cfg,
            max_tokens=max_tokens,
            pool_size=pool_size,
            max_batch_size=max_batch_size,
            long_threshold=long_threshold,
            batching_strategy=batching_strategy,
        )
        self._data_packer = data_packer

    def _get_next_sample(self) -> dict:
        raw_item = super()._get_next_sample()
        return self._data_packer.sft_process_sample(raw_item)

    def compute_sample_tokens(self, sample: dict) -> int:
        return self._data_packer.compute_num_tokens(sample)

    def collate_batch(self, samples: list) -> dict:
        max_len = max(self.compute_sample_tokens(s) for s in samples)
        return self._data_packer.sft_collate_fn(samples, max_len)


class DataPackerDataLoader(torch.utils.data.DataLoader):
    """Public OSS entry point for bringing any dataset into i4 training.

    Wraps ``_DataPackerIterableDataset`` in a standard
    ``torch.utils.data.DataLoader`` — no WebDataset dependency required.
    OSS users' data can be HuggingFace datasets, local files, generators,
    or any Python iterable.

    Data-parallel sharding is automatic when ``torch.distributed`` is
    initialized.  Each ``(dp_rank, worker_id)`` pair receives a disjoint
    subset of ``data_source``.

    Parameters
    ----------
    data_source:
        Any Python iterable — ``torch.utils.data.IterableDataset``,
        HuggingFace ``Dataset``, a generator, or plain list.
    data_packer:
        A ``DataPacker`` subclass instance.  Provides sample-level transform
        (``sft_process_sample``), token counting (``compute_num_tokens``), and
        batch collation (``sft_collate_fn``).
    max_tokens:
        Token budget per batch.
    pool_size:
        Samples to buffer before bin-packing.
    max_batch_size:
        Hard cap on items per batch.
    long_threshold:
        Samples with token count >= this are emitted as singleton batches.
    batching_strategy:
        ``"prefer_closest"`` (default) or ``"prefer_first"``.
    num_workers, prefetch_factor, persistent_workers, pin_memory:
        Forwarded to ``torch.utils.data.DataLoader``.
    parallel_dims:
        Optional ``ParallelDims`` instance (from cosmos-rl).  When provided,
        ``parallel_dims.dp_coord`` supplies the data-parallel rank and world
        size, which is correct for FSDP+TP/PP where the DP degree differs from
        the global world size.  When ``None`` (default), rank info is read from
        ``torch.distributed`` if initialized, else defaults to ``(0, 1)``.
    """

    def __init__(
        self,
        data_source: Any,
        data_packer: DataPacker,
        max_tokens: int,
        pool_size: int = 16,
        max_batch_size: int = 1,
        long_threshold: int = 6400,
        batching_strategy: str = "prefer_closest",
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        parallel_dims=None,
    ):
        # Resolve data-parallel rank and world-size.
        # Priority: explicit parallel_dims > torch.distributed > single-GPU default.
        if parallel_dims is not None:
            dp_rank, dp_world_size = parallel_dims.dp_coord
        elif torch.distributed.is_initialized():
            dp_rank = torch.distributed.get_rank()
            dp_world_size = torch.distributed.get_world_size()

            # rank/world_size differ from the data-parallel rank/world_size.
            # Pass `parallel_dims` to use the correct DP coordinates; otherwise
            # data sharding will be incorrect (each logical DP group sees the
            # same shard as another group).
            if dp_world_size > 1:
                log.info(
                    "DataPackerDataLoader: using global rank for DP sharding. "
                    "For FSDP+TP/PP setups pass parallel_dims= to use the correct "
                    "DP rank/world_size.",
                    rank0_only=True,
                )
        else:
            dp_rank, dp_world_size = 0, 1

        dataset = _DataPackerIterableDataset(
            data_source=data_source,
            data_packer=data_packer,
            max_tokens=max_tokens,
            pool_size=pool_size,
            max_batch_size=max_batch_size,
            long_threshold=long_threshold,
            batching_strategy=batching_strategy,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
        loader_kwargs: dict = dict(
            num_workers=num_workers,
            persistent_workers=persistent_workers and num_workers > 0,
            pin_memory=pin_memory,
        )
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        # batch_size=None disables PyTorch's automatic batching/collation.
        # _DataPackerIterableDataset.__iter__ already yields fully-collated batch dicts;
        # letting the DataLoader re-collate them adds spurious batch dimensions.
        super().__init__(dataset, batch_size=None, **loader_kwargs)
