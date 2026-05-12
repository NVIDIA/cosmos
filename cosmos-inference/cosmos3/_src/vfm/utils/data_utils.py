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

from collections.abc import Iterable
from typing import Any

import torch


def get_vision_data_resolution(spatial_shape: tuple[int, int]) -> str:
    """Determine the resolution string from spatial dimensions.

    Maps the spatial shape (height, width) to a resolution string based on the
    minimum dimension. This is used for resolution-dependent shift lookup when
    using dict-based shift configuration.

    Args:
        spatial_shape: Tuple of (height, width) in pixels.

    Returns:
        Resolution string: "256", "480", or "720" based on the minimum dimension.

    Raises:
        ValueError: If the minimum dimension exceeds 960 pixels (unsupported resolution).

    Note:
        See VIDEO_RES_SIZE_INFO for more details on resolution definitions.
        For the current definition of resolution, these conditions are satisfied.
    """
    min_dim = min(spatial_shape[0], spatial_shape[1])
    if min_dim <= 256:
        return "256"
    elif min_dim <= 640:
        return "480"
    elif min_dim <= 960:
        return "720"
    else:
        raise ValueError(f"Unsupported resolution: {spatial_shape}")


def slice_data_batch(
    data_batch: dict[str, Any],
    start: int,
    limit: int,
    multi_item_fields: Iterable[str] = ("image", "video", "image_size"),
) -> dict[str, Any]:
    """Slice a data batch based on the start and limit indices.

    For most fields, the slice ``[start:limit]`` is applied directly along the
    sample dimension. For fields listed in ``multi_item_fields`` (e.g. ``image``
    and ``video``), each sample may contribute multiple visual items that are
    concatenated in flat order. In that case, when
    ``num_vision_items_per_sample`` is present in ``data_batch``, the slice is
    expanded to cover all visual items belonging to the requested samples.

    Example:
        ``num_vision_items_per_sample = [2, 2]`` and
        ``video = [v1_s1, v2_s1, v1_s2, v2_s2]``. Slicing with
        ``start=0, limit=1`` returns ``video = [v1_s1, v2_s1]``.

    Args:
        data_batch: The data batch to slice.
        start: The start sample index (inclusive).
        limit: The end sample index (exclusive).
        multi_item_fields: Field names whose values store multiple visual
            items per sample concatenated in flat order. Only used when
            ``data_batch`` contains ``num_vision_items_per_sample``.

    Returns:
        The sliced data batch.
    """
    assert start >= 0 and limit > 0, "Start and limit must be positive"
    assert start < limit, "Start must be less than limit"

    num_items = data_batch.get("num_vision_items_per_sample")
    if num_items is not None:
        if isinstance(num_items, torch.Tensor):
            num_items_list = num_items.tolist()
        else:
            num_items_list = list(num_items)
        flat_start = sum(num_items_list[:start])
        flat_limit = sum(num_items_list[:limit])
    else:
        flat_start, flat_limit = start, limit

    multi_item_fields = set(multi_item_fields)

    sliced_batch = {}
    for key, value in data_batch.items():
        if key in multi_item_fields and num_items is not None:
            s, e = flat_start, flat_limit
        else:
            s, e = start, limit
        if isinstance(value, torch.Tensor):
            sliced_batch[key] = value[s:e]
        elif isinstance(value, list):
            sliced_batch[key] = value[s:e]
        else:
            sliced_batch[key] = value
    return sliced_batch
