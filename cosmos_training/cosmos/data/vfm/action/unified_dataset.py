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

"""Unified iterable dataset for Action multi-embodiment robot data.

``ActionUnifiedIterableDataset`` is the Layer 2 component of the Action data loading
pipeline.  It wraps *all* Action datasets into a single ``IterableDataset`` and
handles:

- **Rank-level dataset assignment** (Hare-Niemeyer proportional allocation)
- **Worker-level shard distribution** (round-robin within a dataset family)
- **Per-sample transforms** via :class:`~.transforms.ActionTransformPipeline`
- **Weighted random fallback** when worker assignment is not active

See ``docs/dataloader.md`` for the full design document.
"""

from __future__ import annotations

import gc
import random
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from torch.utils.data import Dataset, IterableDataset

from cosmos.utils import log
from cosmos.data.vfm.action.transforms import ActionTransformPipeline
from cosmos.data.vfm.utils import VIDEO_RES_SIZE_INFO

_iterable_dataset_len_warning_suppressed = False


def _suppress_iterable_dataset_len_warning() -> None:
    """Register a one-time filter for PyTorch's IterableDataset len() warning.

    The inner datasets may not implement ``__len__``, so the wrapper reports
    ``len()=0``.  PyTorch's iterator then warns on every ``__next__`` when
    samples are fetched.  This filter suppresses that warning.
    """
    global _iterable_dataset_len_warning_suppressed
    if _iterable_dataset_len_warning_suppressed:
        return
    _iterable_dataset_len_warning_suppressed = True
    warnings.filterwarnings(
        "ignore",
        message="Length of IterableDataset.*was reported to be 0",
        category=UserWarning,
        module="torch.utils.data.dataloader",
    )


# ---------------------------------------------------------------------------
# Worker-side periodic garbage collection
# ---------------------------------------------------------------------------
# DataLoader workers running IterableDatasets are long-lived forked processes.
# Complex sample dictionaries (nested dicts, tensors, Arrow references) can
# create circular-reference chains that Python's reference counting alone
# cannot free.  The generational GC *does* collect them eventually, but its
# default thresholds are too conservative for high-throughput data loading,
# causing RSS to grow monotonically until the node OOMs.
#
# Calling ``gc.collect()`` periodically inside the worker iteration loop
# eliminates the leak with negligible overhead (<1 ms per call vs ~6 s
# iteration time).
#
_GC_INTERVAL: int = 10


def _maybe_gc(interval: int, count: int) -> int:
    """Increment *count* and run ``gc.collect()`` every *interval* samples."""
    if interval <= 0:
        return count
    count += 1
    if count % interval == 0:
        gc.collect()
    return count


class ActionUnifiedIterableDataset(IterableDataset):
    """Single IterableDataset wrapping all Action datasets.

    Handles worker-to-dataset assignment, shard distribution, and transforms.

    Args:
        datasets: List of dicts, each with keys ``"name"`` (str identifier),
            ``"dataset"`` (the dataset instance), and ``"ratio"`` (float
            sampling weight).
        transform: Transform pipeline applied to every yielded sample.
        shard_across_workers: When ``True``, ranks are assigned to
            dataset families via Hare-Niemeyer and workers get round-robin
            shards.  When ``False`` (default), every worker loads all
            datasets and iterates with weighted random selection.
    """

    def __init__(
        self,
        datasets: list[dict[str, Any]],
        transform: ActionTransformPipeline,
        shard_across_workers: bool = False,
    ) -> None:
        super().__init__()
        self._datasets = datasets
        self._transform = transform
        self._shard_across_workers = shard_across_workers

        # Set per-worker by assign_worker; None means not yet assigned.
        self._dataset: Any | None = None
        self._resolution: str | None = None  # resolution for single-dataset path
        self._sources_initialized = False

        # Backward compat: expose ``self.dataset`` pointing to the first
        # inner dataset and ``self.transform`` exposing the pipeline
        # (mirrors old TransformedIterableDataset interface).
        self.dataset = datasets[0]["dataset"] if datasets else None
        self.transform = transform

    # -- source initialization ------------------------------------------------

    def _ensure_sources_registered(self) -> None:
        if self._sources_initialized:
            return
        self._sources_initialized = True
        for entry in self._datasets:
            ds = entry["dataset"]
            shard_roots = getattr(ds, "_all_shard_roots", [])
            if shard_roots and hasattr(ds, "_register_sources"):
                ds._register_sources()

    # -- backward-compat helpers -----------------------------------------------

    def __len__(self) -> int:  # type: ignore[override]
        total = 0
        for entry in self._datasets:
            ds = entry["dataset"]
            try:
                total += len(ds)  # type: ignore[arg-type]
            except TypeError:
                pass
        return total

    def __getattr__(self, name: str) -> Any:
        """Forward attribute lookups to the first inner dataset."""
        if name.startswith("_") or not self._datasets:
            raise AttributeError(name)
        return getattr(self._datasets[0]["dataset"], name)

    # -- Hare-Niemeyer rank allocation -----------------------------------------

    @staticmethod
    def _compute_rank_ranges(
        datasets: list[dict[str, Any]],
        world_size: int,
    ) -> list[tuple[int, int]]:
        """Hare-Niemeyer allocation of ranks to datasets.

        Guarantees at least 1 rank per dataset, distributes the rest
        proportionally.  Returns a list of ``(start_rank, end_rank)`` ranges.

        Raises:
            ValueError: If ``world_size < len(datasets)``.
        """
        n_ds = len(datasets)
        if world_size < n_ds:
            raise ValueError(f"world_size ({world_size}) must be >= number of datasets ({n_ds})")
        ratios = [d["ratio"] for d in datasets]
        total = sum(ratios)

        # Hare-Niemeyer (largest-remainder) method:
        # 1. Give every dataset a guaranteed minimum of 1 rank.
        # 2. Distribute the leftover ranks proportionally to each dataset's
        #    ratio.  Take the floor of each fractional allocation, then award
        #    the still-unassigned ranks one-by-one to datasets with the
        #    largest fractional remainders.
        # Example: world_size=8, ratios=[3, 1] (2 datasets)
        #   remaining = 8 - 2 = 6
        #   fractional = [6*3/4, 6*1/4] = [4.5, 1.5]
        #   floors     = [4, 1], remainders = [0.5, 0.5], leftover = 1
        #   award 1 extra to first dataset -> floors = [5, 1]
        #   counts = [1+5, 1+1] = [6, 2]
        counts = [1] * n_ds
        remaining = world_size - n_ds
        if remaining > 0:
            fractional = [remaining * r / total for r in ratios]
            floors = [int(f) for f in fractional]
            remainders = [f - fl for f, fl in zip(fractional, floors)]
            leftover = remaining - sum(floors)
            for idx in sorted(range(n_ds), key=lambda j: -remainders[j])[:leftover]:
                floors[idx] += 1
            counts = [1 + f for f in floors]

        # Convert per-dataset counts into contiguous rank intervals.
        # Example continued: counts=[6, 2] -> ranges=[(0,6), (6,8)]
        #   ranks 0..5 serve dataset 0, ranks 6..7 serve dataset 1.
        ranges: list[tuple[int, int]] = []
        cursor = 0
        for c in counts:
            ranges.append((cursor, cursor + c))
            cursor += c
        return ranges

    # -- worker assignment -----------------------------------------------------

    def assign_worker(
        self,
        worker_id: int,
        num_workers: int,
        rank: int,
        world_size: int,
    ) -> None:
        """Assign this worker to a dataset family and distribute shards.

        Called by the DataLoader's ``worker_init_fn`` (via
        :func:`~.dataloaders.create_action_worker_init_fn`) -- not by the
        dataset itself.

        Two-level assignment:

        1. **Rank -> dataset family** (Hare-Niemeyer over *world_size*
           ranks).  Every rank is fully dedicated to one family.
        2. **Workers -> shards** (round-robin within the family's worker
           pool).  ``family_worker_id = rank_within_family * num_workers
           + worker_id``.

        When ``shard_across_workers=False``: no assignment is performed.
        Every worker loads all datasets and ``__iter__`` uses weighted
        random selection.
        """
        self._sources_initialized = True
        if not self._shard_across_workers:
            for entry in self._datasets:
                ds = entry["dataset"]
                shard_roots = getattr(ds, "_all_shard_roots", [])
                if shard_roots and hasattr(ds, "_register_sources"):
                    ds._register_sources()
            return

        rank_ranges = self._compute_rank_ranges(self._datasets, world_size)

        # Step 1: which dataset family does this rank belong to?
        # ``rank_ranges`` is a list of (start_rank, end_rank) intervals -- one
        # per dataset family -- produced by ``_compute_rank_ranges()`` above
        # using Hare-Niemeyer allocation.  The intervals are contiguous and
        # non-overlapping, covering [0, world_size), so every rank belongs to
        # exactly one family.
        #
        # We scan through the intervals to find the one containing this rank,
        # then derive two values:
        #   - rank_within_family: this rank's 0-based position inside its
        #     family (used in Step 2 to build a globally unique worker id).
        #   - num_family_ranks: total number of ranks assigned to this family
        #     (used in Step 2 to compute the family's worker pool size).
        #
        # ``self._dataset`` is set to the matched family's dataset object so
        # that ``__iter__`` only yields samples from this one dataset.
        #
        # Example with world_size=8, ratios=[3,1] -> ranges=[(0,6), (6,8)]:
        #   rank 3 -> family 0, rank_within_family=3, num_family_ranks=6
        #   rank 6 -> family 1, rank_within_family=0, num_family_ranks=2
        num_family_ranks = 1
        rank_within_family = 0
        for i, (start_rank, end_rank) in enumerate(rank_ranges):
            if start_rank <= rank < end_rank:
                entry = self._datasets[i]
                self._dataset = entry["dataset"]
                self._resolution = entry["resolution"]
                rank_within_family = rank - start_rank
                num_family_ranks = end_rank - start_rank
                break

        # Step 2: distribute shards across workers within the family.
        # Each rank spawns ``num_workers`` DataLoader workers (set by the
        # DataLoader's ``num_workers`` arg).  So the family's total worker
        # pool is ``num_family_ranks * num_workers``.
        #
        # We flatten the 2D index (rank_within_family, worker_id) into a
        # single linear ``family_worker_id`` so every worker in the family
        # gets a globally unique id within that family:
        #   family_worker_id = rank_within_family * num_workers + worker_id
        #
        # Example: family has 3 ranks, each rank spawns 2 workers -> 6 total:
        #   rank_within_family=0: worker_id 0 -> fwid 0, worker_id 1 -> fwid 1
        #   rank_within_family=1: worker_id 0 -> fwid 2, worker_id 1 -> fwid 3
        #   rank_within_family=2: worker_id 0 -> fwid 4, worker_id 1 -> fwid 5
        #
        # This linear id is then used for round-robin shard assignment below.
        family_total_workers = num_family_ranks * num_workers
        family_worker_id = rank_within_family * num_workers + worker_id

        # Round-robin assignment: worker k gets shards k, k+stride, k+2*stride, ...
        # This ensures shards are evenly spread across the family's workers.
        #
        # When family_total_workers > num_shards, some workers get an empty
        # list from range() (any worker with family_worker_id >= num_shards,
        # since start >= stop).  The ``if not my_shards`` guard catches this
        # and falls back to ``family_worker_id % num_shards``, wrapping the
        # worker around to an existing shard so it shares rather than idles.
        #
        # Example: AgiBotWorld with 190 shards and 256 family workers:
        #   Workers 0-189  -> each gets 1 unique shard via range()
        #   Workers 190-255 -> empty range, fallback to family_worker_id % 190,
        #                      sharing a shard with an earlier worker.
        #
        # Multiple workers reading the same shard is fine because each worker
        # has a different RNG seed (``seed + rank * 9999 + worker_id``), so
        # they produce different sample orderings from the same underlying data.
        shard_roots = getattr(self._dataset, "_all_shard_roots", [])
        if shard_roots and hasattr(self._dataset, "_register_sources"):
            num_shards = len(shard_roots)
            my_shards = list(range(family_worker_id, num_shards, family_total_workers))
            if not my_shards:
                my_shards = [family_worker_id % num_shards]
            self._dataset._register_sources(my_shards)

    # -- iteration -------------------------------------------------------------

    def _iter_all_datasets_weighted(self) -> Iterator[dict[str, Any]]:
        """Iterate all datasets with weighted random selection.

        Used when ``shard_across_workers=False`` (every worker sees all
        datasets) or as the ``num_workers=0`` fallback.
        """
        iterators = [iter(d["dataset"]) for d in self._datasets]
        ratios = [d["ratio"] for d in self._datasets]
        total = sum(ratios)
        weights = [r / total for r in ratios]

        gc_count = 0

        while True:
            chosen = random.choices(range(len(self._datasets)), weights=weights, k=1)[0]
            resolution = self._datasets[chosen]["resolution"]
            try:
                yield self._transform(next(iterators[chosen]), resolution=resolution)
            except StopIteration:
                iterators[chosen] = iter(self._datasets[chosen]["dataset"])
                try:
                    yield self._transform(next(iterators[chosen]), resolution=resolution)
                except StopIteration:
                    continue
            gc_count = _maybe_gc(_GC_INTERVAL, gc_count)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._dataset is not None:
            gc_count = 0
            for sample in self._dataset:
                yield self._transform(sample, resolution=self._resolution)
                gc_count = _maybe_gc(_GC_INTERVAL, gc_count)
            return

        if not self._shard_across_workers:
            self._ensure_sources_registered()
            yield from self._iter_all_datasets_weighted()
            return

        # num_workers=0 fallback (shard_across_workers=True but no worker
        # processes exist, so assign_worker was never called).
        log.warning(
            "ActionUnifiedIterableDataset: num_workers=0 fallback — "
            "loading ALL datasets in main process. Use only for debugging."
        )
        self._ensure_sources_registered()
        yield from self._iter_all_datasets_weighted()


class MapToIterableAdapter(IterableDataset):
    """Wraps a map-style ``Dataset`` as an ``IterableDataset``.

    Each iteration yields a sample from a uniformly random index, using
    ``random.randint`` for O(1) time and zero extra memory.      The per-worker
    RNG seed (set by :func:`~.dataloaders.create_action_worker_init_fn`) ensures
    different DataLoader workers produce different random sequences.

    Args:
        dataset: A map-style ``Dataset`` with ``__len__`` and ``__getitem__``.
    """

    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.dataset)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator:
        n = len(self.dataset)  # type: ignore[arg-type]
        while True:
            yield self.dataset[random.randint(0, n - 1)]

    def __getattr__(self, name: str) -> Any:
        """Forward attribute lookups to the inner dataset for transparency."""
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


def dataset_entry(
    name: str,
    dataset: Dataset | IterableDataset,
    ratio: float = 1.0,
    resolution: str | None = None,
) -> dict:
    """Factory for a single dataset descriptor used inside ``wrap_dataset``.

    Wrapping each entry with ``LazyCall(dataset_entry)(...)`` gives it a
    ``_target_`` so that ``instantiate`` recurses into the nested dataset
    config automatically.

    Args:
        name: Identifier for the dataset.
        dataset: The dataset instance.
        ratio: Sampling weight. Defaults to 1.0.
        resolution: Optional resolution tier (e.g. ``"256"``, ``"480"``) for
            this dataset. When ``None``, falls back to ``wrap_dataset``'s
            global ``resolution`` (which may be ``None`` for auto-detect).
    """
    return {"name": name, "dataset": dataset, "ratio": ratio, "resolution": resolution}


def wrap_dataset(
    list_of_datasets: Sequence[dict] | list[dict] | Dataset | IterableDataset,
    resolution: str | None = None,
    pad_keys: list[str] | None = None,
    keep_aspect_ratio: bool = True,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.0,
    caption_key: str = "ai_caption",
    text_token_key: str = "text_token_ids",
    video_temporal_downsample: int = 4,
    max_action_dim: int = 32,
    shard_across_workers: bool = False,
    action_channel_masking: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    idle_frames_dropout: float = 0.05,
    format_prompt_as_json: bool = False,
) -> ActionUnifiedIterableDataset:
    """Factory that wraps one or more datasets with the Action transform pipeline.

    ``list_of_datasets`` accepts either:

    * A **list of dicts**, where each dict has the keys:
        - ``name`` (``str``): identifier for the dataset.
        - ``dataset`` (``Dataset | IterableDataset``): the dataset instance.
        - ``ratio`` (``float``, optional): sampling weight. Defaults to ``1``.
        - ``resolution`` (``str | None``, optional): resolution tier for this
            dataset. When missing, falls back to ``wrap_dataset``'s global
            ``resolution`` (which may be ``None`` for auto-detect).
    * A **single** ``Dataset`` or ``IterableDataset`` for backward compatibility
      (auto-wrapped as ``[{"name": "default", "dataset": <ds>, "ratio": 1}]``).

    Map-style datasets are automatically wrapped with
    :class:`MapToIterableAdapter` so the returned dataset is always an
    ``IterableDataset``.  This means callers can mix map-style and
    iterable-style datasets freely.

    Args:
        list_of_datasets: The dataset(s) to wrap.
        resolution: Resolution tier key (e.g. ``"256"``, ``"480"``, ``"720"``).
            Spatial dimensions are resized and reflection-padded to the closest
            predefined target from ``VIDEO_RES_SIZE_INFO``.  When ``None``, the
            tier is auto-detected per sample via ``get_vision_data_resolution``.
            Defaults to ``None``.
        pad_keys: Data-dict keys whose values should be resized and padded. Pass
            an empty list or ``None`` to disable padding. Defaults to ``["video"]``.
        tokenizer_config: A lazy-instantiable config dict for the VLM tokenizer. When
            ``None``, text tokenization is skipped. Defaults to ``None``.
        cfg_dropout_rate: Probability of replacing the caption with an empty string for
            classifier-free guidance. Defaults to ``0.0``.
        caption_key: The data-dict key that contains the input caption string.
            Defaults to ``"ai_caption"``.
        text_token_key: The data-dict key where tokenized text IDs will be stored.
            Defaults to ``"text_token_ids"``.
        video_temporal_downsample: Temporal downsampling factor of the video tokenizer.
            Used when building a ``SequencePlan`` for ``"inverse_dynamics"`` mode.
            Defaults to 4.
        max_action_dim: Target action dimension to pad to.  The ``"action"`` tensor
            in every sample is padded along its last dimension.  Defaults to 32.
        action_channel_masking: When ``True`` (default), stores the original action
            dimension in ``"raw_action_dim"`` so the model masks loss/noise/velocity
            on padded channels.  Set to ``False`` to disable (original behavior).
        shard_across_workers: When ``True``, the returned dataset
            supports rank-level dataset assignment and worker-level shard
            distribution via ``assign_worker()``.  When ``False`` (default),
            every worker iterates all datasets with weighted random selection.
        append_duration_fps_timestamps: Whether to append duration and FPS metadata to the
            caption before tokenization.  Defaults to ``True``.
        append_resolution_info: Whether to append resolution metadata to the
            caption before tokenization.  Defaults to ``True``.
        append_idle_frames: Whether to append the idle-frame count out of the
            total action frames (Pi0.7-style metadata) to the caption before
            tokenization.  The dataset is responsible for populating
            ``data_dict["idle_frames"]``; samples without it are silently
            skipped.  Defaults to ``False`` so existing experiments are
            unaffected.
        idle_frames_dropout: Per-field dropout rate for the idle-frame segment.
            Independent of ``cfg_dropout_rate`` (which empties the whole
            caption). Defaults to 0.05.
        format_prompt_as_json: Whether to replace the plain text prompt with a
            structured JSON-compatible dictionary before tokenization. Defaults
            to ``False``.

    Returns:
        A :class:`ActionUnifiedIterableDataset` wrapping the dataset(s) with the
        configured transforms applied.

    Raises:
        TypeError: If the dataset(s) are not ``Dataset`` or ``IterableDataset``.
        ValueError: If ``list_of_datasets`` is an empty list.
    """
    if pad_keys is None:
        pad_keys = ["video"]

    # ------------------------------------------------------------------
    # Backward compatibility: single dataset -> list-of-dicts
    # ------------------------------------------------------------------
    if isinstance(list_of_datasets, (Dataset, IterableDataset)):
        list_of_datasets = [{"name": "default", "dataset": list_of_datasets, "ratio": 1}]

    if (
        not isinstance(list_of_datasets, Sequence)
        or isinstance(list_of_datasets, (str, bytes))
        or len(list_of_datasets) == 0
    ):
        raise ValueError(
            "list_of_datasets must be a non-empty list/sequence of dicts or a single Dataset/IterableDataset, "
            f"got {type(list_of_datasets).__name__}"
        )

    # ------------------------------------------------------------------
    # Parse list-of-dicts, wrapping map-style datasets with
    # MapToIterableAdapter so every dataset is iterable. Compute effective
    # resolution per entry (per-entry overrides global).
    # ------------------------------------------------------------------
    datasets: list[dict] = []
    for entry in list_of_datasets:
        if not isinstance(entry, Mapping):
            raise TypeError(f"Each entry in list_of_datasets must be a dict/mapping, got {type(entry).__name__}")
        name: str = entry["name"]
        dataset: Dataset | IterableDataset = entry["dataset"]
        ratio: float = float(entry.get("ratio", 1))
        resolution: str | None = entry.get("resolution", None)
        if resolution is not None:
            res_key = str(resolution) if isinstance(resolution, int) else resolution
            if res_key not in VIDEO_RES_SIZE_INFO:
                raise ValueError(
                    f"Resolution '{resolution}' for dataset '{name}' not found in VIDEO_RES_SIZE_INFO. "
                    f"Available: {list(VIDEO_RES_SIZE_INFO.keys())}"
                )
        if not isinstance(dataset, IterableDataset):
            dataset = MapToIterableAdapter(dataset)
        datasets.append({"name": name, "dataset": dataset, "ratio": ratio, "resolution": resolution})

    # ------------------------------------------------------------------
    # Build the transform pipeline (resolution supplied at call time)
    # ------------------------------------------------------------------
    transform = ActionTransformPipeline(
        pad_keys=pad_keys,
        keep_aspect_ratio=keep_aspect_ratio,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        caption_key=caption_key,
        text_token_key=text_token_key,
        video_temporal_downsample=video_temporal_downsample,
        max_action_dim=max_action_dim,
        action_channel_masking=action_channel_masking,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        idle_frames_dropout=idle_frames_dropout,
        format_prompt_as_json=format_prompt_as_json,
    )

    _suppress_iterable_dataset_len_warning()

    return ActionUnifiedIterableDataset(
        datasets=datasets,
        transform=transform,
        shard_across_workers=shard_across_workers,
    )
