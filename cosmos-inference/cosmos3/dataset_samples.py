# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Lazy dataset sample iterators for map-style and iterable-style datasets."""

import itertools
import json
from collections.abc import Iterable, Iterator
from typing import Any, Callable

import torch
from loguru import logger
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataloader import default_collate

from cosmos3.args import OmniSampleOverrides
from cosmos3.scripts.dataset_utils import set_dataset_mode
from cosmos3._src.vfm.utils.data_utils import get_vision_data_resolution

Sample = tuple[OmniSampleOverrides, dict[str, Any]]


def _collate_sample(sample: dict) -> dict:
    """Collate a single sample dict, adding a batch dim to tensors."""
    result: dict[str, Any] = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            result[key] = val.unsqueeze(0)
        else:
            try:
                result[key] = default_collate([val])
            except TypeError:
                result[key] = [val]
    return result


def _normalize_caption(raw_sample: dict) -> str:
    """Normalize ``ai_caption`` to a string in-place and return it.

    JSON-dict captions (from ``ActionPromptJsonFormatter``) are serialized
    so collation, batch merging, and the model input treat them identically
    to plain-text captions, matching the training side's
    ``TextTokenizerTransform``.

    Raises:
        TypeError: If ``ai_caption`` is present and is neither ``str`` nor
            ``dict``.
    """
    caption = raw_sample.get("ai_caption", "")
    if isinstance(caption, dict):
        caption = json.dumps(caption)
        raw_sample["ai_caption"] = caption
    elif not isinstance(caption, str):
        raise TypeError(f"ai_caption must be str or dict, got {type(caption).__name__}")
    return caption


class _BaseSamples(Iterable[Sample]):
    """Base iterator yielding ``(OmniSampleOverrides, data_batch)`` pairs.

    Iterates over every (mode, sample_id) combination, applying an optional
    transform to each raw item. Subclasses implement ``__iter__`` for
    map-style and iterable-style datasets respectively.
    """

    def __init__(
        self,
        dataset: Dataset | IterableDataset,
        modes: list[str],  # model modes to iterate (e.g. ["joint", "forward_dynamics", etc.])
        sample_ids: list[int],  # indices into dataset to yield
        transform: Callable | None,  # UVA transform pipeline applied per item, or None
        resolution: str | None,  # global resolution override; inferred from video shape if None
        dataset_name: str,  # name of the dataset"
        sample_overrides_data: dict[str, Any] | None = None,  # additional overrides to apply to every sample
    ) -> None:
        self._dataset = dataset
        self._modes = modes
        self._sample_ids = sample_ids
        self._transform = transform
        self._resolution = resolution
        self._dataset_name = dataset_name
        self._sample_overrides_data = sample_overrides_data

    def __len__(self) -> int:
        return len(self._modes) * len(self._sample_ids)

    def _make_sample_from_raw(self, raw_sample: Any, sample_idx: int, mode: str) -> Sample:
        """Apply transform, collate, and wrap a raw dataset item into a ``Sample``."""
        resolution = self._resolution
        if resolution is None:
            video = raw_sample.get("video")
            if video is not None:
                resolution = get_vision_data_resolution(video.shape[-2:])

        if self._transform is not None:
            raw_sample = self._transform(raw_sample, resolution=resolution)
        prompt = _normalize_caption(raw_sample)
        sample_data = _collate_sample(raw_sample)

        sample_name = f"{self._dataset_name}/{mode}/{sample_idx}" if self._dataset_name else f"{mode}/{sample_idx}"
        sample_args = OmniSampleOverrides(
            name=sample_name,
            prompt=prompt,
            resolution=resolution,  # type: ignore
            raw_action_dim=sample_data.get("raw_action_dim", [None])[0],
        )
        # Apply any additional sample overrides specified in the setup config (e.g. num_steps, guidance, etc.)
        sample_args = sample_args.model_copy(update=self._sample_overrides_data)
        return sample_args, sample_data


class MapDatasetSamples(_BaseSamples):
    """Iterator for map-style datasets (``Dataset``), accessed via ``__getitem__``.

    Iterates modes in order, indexing each sample directly by its — enabling
    random access.
    """

    def __iter__(self) -> Iterator[Sample]:
        for mode in self._modes:
            set_dataset_mode(self._dataset, mode)
            for sample_idx in self._sample_ids:
                raw_sample = self._dataset[sample_idx]  # type: ignore[index]
                yield self._make_sample_from_raw(raw_sample, sample_idx, mode)


class IterableDatasetSamples(_BaseSamples):
    """Iterator for iterable-style datasets (``IterableDataset``), accessed via ``__iter__``.

    Since random access is unavailable, advances the underlying iterator using
    ``islice`` to reach each target sample index in order — requires
    ``sample_ids`` to be sorted ascending.
    """

    def __iter__(self) -> Iterator[Sample]:
        for mode in self._modes:
            set_dataset_mode(self._dataset, mode)
            dataset = iter(getattr(self._dataset, "dataset", self._dataset))
            cur_ix = 0
            for sample_idx in sorted(self._sample_ids):
                try:
                    raw_sample = next(itertools.islice(dataset, sample_idx - cur_ix, None))
                except StopIteration:
                    # Dataset exhausted early (inaccurate __len__); move on to next mode.
                    logger.warning(
                        f"Dataset {self._dataset_name!r} exhausted early while iterating mode={mode!r}: "
                        f"tried to reach sample_idx={sample_idx}, expected __len__={len(self._dataset)}. "  # type: ignore[arg-type]
                        "Moving on to next mode."
                    )
                    break
                cur_ix = sample_idx + 1
                yield self._make_sample_from_raw(raw_sample, sample_idx, mode)


DatasetSamples = MapDatasetSamples | IterableDatasetSamples
