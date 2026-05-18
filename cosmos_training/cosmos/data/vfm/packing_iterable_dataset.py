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
Abstract base class for pool-based token-budget bin-packing over multiple datasets.

Extracted from ``projects.cosmos3.vfm.datasets.vlm.joint_dataset_dynamic_batch_webloader``
so that both the VLM and VFM internal dataloaders can share a single packing implementation.

Usage
-----
Subclass and implement ``compute_sample_tokens(sample) -> int``.
Optionally override ``collate_batch(samples) -> Any`` for custom collation.

    class MyPacker(PackingIterableDataset):
        def compute_sample_tokens(self, sample):
            return len(sample["input_ids"])
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import Any, Union

import torch

from cosmos.utils.lazy_config import instantiate
from cosmos.utils import log


class Modality(Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


class PackingIterableDataset(torch.utils.data.IterableDataset, ABC):
    """Pool-based greedy bin-packing IterableDataset.

    Maintains a pool of ``pool_size`` samples and assembles batches by
    greedily selecting candidates that fit within the token budget
    ``max_tokens``.  Subclasses supply two hooks:

    * ``compute_sample_tokens(sample)`` — token cost of one sample (abstract).
    * ``collate_batch(samples)`` — assemble a packed list into a batch
      (default: identity, returns the list unchanged).

    Parameters
    ----------
    datasets_cfg:
        Mapping ``{name: {"dataset": <iterable>, "ratio": <float>}}``.
        The *dataset* value may be a Hydra lazy config, an already-constructed
        ``IterableDataset``, or a plain ``DataLoader`` (its ``.dataset`` is
        unwrapped automatically).
    max_tokens:
        Token budget per batch (padded cost = ``cur_max_len * batch_size``).
    pool_size:
        Number of samples to buffer before selecting a batch.
    max_batch_size:
        Hard cap on items per batch (0 or None = no cap).
    long_threshold:
        Samples with token count ``>= long_threshold`` are emitted as
        singletons regardless of budget.
    batching_strategy:
        ``"prefer_closest"`` (default) or ``"prefer_first"``.
    """

    def __init__(
        self,
        datasets_cfg: dict[str, dict[str, Union[int, object]]],
        max_tokens: int,
        pool_size: int,
        max_batch_size: int,
        long_threshold: int,
        batching_strategy: str,
    ):
        super().__init__()

        assert batching_strategy in ("prefer_first", "prefer_closest"), (
            f"batching_strategy must be 'prefer_first' or 'prefer_closest', got {batching_strategy!r}"
        )

        self.max_tokens = max_tokens
        self.pool_size = pool_size
        self.long_threshold = long_threshold
        self.max_batch_size = max_batch_size
        self.batching_strategy = batching_strategy

        self._pool: deque[dict] = deque()
        self._dataset_names: list[str] = []
        self._ratios: list[float] = []
        self._datasets: list[torch.utils.data.IterableDataset] = []

        for name, cfg in datasets_cfg.items():
            assert {"ratio", "dataset"} <= cfg.keys(), (
                f"Each entry must have 'dataset' and 'ratio' keys: {name} -> {cfg.keys()}"
            )
            ratio = cfg["ratio"]
            if ratio == 0:
                log.info(f"Skipping dataset {name} with ratio {ratio}")
                continue
            dataset_cfg = cfg["dataset"]

            ds = (
                instantiate(dataset_cfg)
                if not isinstance(dataset_cfg, (torch.utils.data.IterableDataset, torch.utils.data.DataLoader))
                else dataset_cfg
            )
            if isinstance(ds, torch.utils.data.DataLoader):
                ds = ds.dataset
            if hasattr(ds, "build_dataset") and callable(getattr(ds, "build_dataset")):
                ds = ds.build_dataset()

            assert isinstance(ds, torch.utils.data.IterableDataset), (
                f"Expected an IterableDataset, got {type(ds)} for {name}"
            )

            self._dataset_names.append(name)
            self._ratios.append(float(ratio))
            self._datasets.append(ds)
            log.info(f"Added dataset {name} with ratio {ratio}")

        log.info(f"added data: {list(datasets_cfg.keys())}")
        assert len(self._datasets) > 0, "No datasets added"
        self._data_len: int = sum(int(getattr(ds, "total_images", 0)) for ds in self._datasets)
        if self._data_len == 0:
            self._data_len = 10**12
        self.iterators = [iter(ds) for ds in self._datasets]

    # ------------------------------------------------------------------
    # Abstract / overridable hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_sample_tokens(self, sample: dict) -> int:
        """Return the token cost of one sample for packing budget accounting."""

    def collate_batch(self, samples: list[dict]) -> Any:
        """Assemble a packed list of samples into one batch.

        Default implementation returns the list unchanged (identity).
        Override to pad, stack, or transform samples into tensors.
        """
        return samples

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._data_len

    def __iter__(self):
        while True:
            batch = self._best_fit_batch()
            yield self.collate_batch(batch)

    # ------------------------------------------------------------------
    # Internal packing helpers (moved verbatim from _JointIterableDataset)
    # ------------------------------------------------------------------

    def _max_tokens(self, cur_max: int) -> int:
        if cur_max < 1000:
            return self.max_tokens
        return self.max_tokens // 2

    def _get_next_sample(self) -> dict:
        index_id = random.choices(range(len(self.iterators)), weights=self._ratios, k=1)[0]
        curr_dataset = self.iterators[index_id]
        try:
            output = next(curr_dataset)
        except StopIteration:
            log.critical(f"dataset {self._dataset_names[index_id]} exhausted")
            self.iterators[index_id] = iter(self._datasets[index_id])
            output = next(self.iterators[index_id])
        return output

    def _fill_pool(self):
        while len(self._pool) < self.pool_size:
            self._pool.append(self._get_next_sample())

    def _padded_cost(self, cur_max: int, k: int) -> int:
        return cur_max * k

    def _get_modality(self, sample: dict) -> Modality:
        if "pixel_values" in sample:
            return Modality.IMAGE
        elif "pixel_values_videos" in sample:
            return Modality.VIDEO
        return Modality.TEXT

    def _best_fit_batch(self) -> list[dict]:
        """Build one batch using the configured token-budget strategy."""
        self._fill_pool()
        seed = self._pool.popleft()
        seed_modality = self._get_modality(seed)
        L0 = self.compute_sample_tokens(seed)

        if L0 >= self.long_threshold or L0 >= self._max_tokens(L0):
            return [seed]

        chosen = [seed]
        cur_max = L0

        while self._pool:
            if self.max_batch_size and len(chosen) >= self.max_batch_size:
                break
            best_idx = self._find_best_candidate(cur_max, len(chosen), seed_modality)
            if best_idx is None:
                break
            cand = self._remove_from_pool(best_idx)
            chosen.append(cand)
            cur_max = max(cur_max, self.compute_sample_tokens(cand))

        return chosen

    def _find_best_candidate(self, cur_max: int, num_chosen: int, seed_modality: Modality) -> int | None:
        if self.batching_strategy == "prefer_first":
            return self._find_best_candidate_prefer_first(cur_max, num_chosen, seed_modality)
        return self._find_best_candidate_prefer_closest(cur_max, num_chosen, seed_modality)

    def _find_best_candidate_prefer_first(self, cur_max: int, num_chosen: int, seed_modality: Modality) -> int | None:
        best_idx = None
        best_new_tokens = None
        for idx, cand in enumerate(self._pool):
            if self._get_modality(cand) != seed_modality:
                continue
            L = self.compute_sample_tokens(cand)
            new_max = max(cur_max, L)
            new_tokens = self._padded_cost(new_max, num_chosen + 1)
            if new_tokens <= self._max_tokens(cur_max):
                if best_new_tokens is None or new_tokens < best_new_tokens:
                    best_new_tokens = new_tokens
                    best_idx = idx
        return best_idx

    def _find_best_candidate_prefer_closest(self, cur_max: int, num_chosen: int, seed_modality: Modality) -> int | None:
        best_idx = None
        best_new_tokens = None
        smallest_length_diff = None
        for idx, cand in enumerate(self._pool):
            if self._get_modality(cand) != seed_modality:
                continue
            L = self.compute_sample_tokens(cand)
            new_max = max(cur_max, L)
            new_tokens = self._padded_cost(new_max, num_chosen + 1)
            if new_tokens <= self._max_tokens(cur_max):
                length_diff = abs(L - cur_max)
                if (
                    best_new_tokens is None
                    or new_tokens < best_new_tokens
                    or (new_tokens == best_new_tokens and length_diff < smallest_length_diff)
                ):
                    best_new_tokens = new_tokens
                    best_idx = idx
                    smallest_length_diff = length_diff
        return best_idx

    def _remove_from_pool(self, idx: int) -> dict:
        if idx == 0:
            return self._pool.popleft()
        elif idx == len(self._pool) - 1:
            return self._pool.pop()
        else:
            self._pool.rotate(-idx)
            item = self._pool.popleft()
            self._pool.rotate(idx)
            return item
