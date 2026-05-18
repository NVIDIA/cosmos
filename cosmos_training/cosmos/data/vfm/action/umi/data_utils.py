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

from typing import Any, Callable

import numpy as np
import torch


def aggregate_batch(batch: list[Any], aggregate_fn: Callable[[list[Any]], Any], merge_none: bool = True) -> Any:
    """
    Custom collate function to concatenate nested tensors/ndarray/float along a specified axis.
    If merge_none is True, the field that has None values will be merged into a single None value. Otherwise will return a list of None values.
    Popular choices of aggregate_fn:
        - partial(torch.cat, dim=existing_dim), if you want to concatenate along an existing dimension
        - partial(torch.stack, dim=new_dim), if you want to stack to a new dimension

    Args:
        batch (List[Any]): A list of samples from the dataset.
        aggregate_fn (Callable[[list[Any]], Any]): The function to aggregate the tensors/ndarray/float.

    Returns:
        Any: The concatenated batch.
    """
    if len(batch) == 0:
        return batch
    elem = batch[0]
    if isinstance(elem, torch.Tensor) or isinstance(elem, np.ndarray) or isinstance(elem, float):
        return aggregate_fn(batch)
    elif isinstance(elem, dict):
        return {key: aggregate_batch([d[key] for d in batch], aggregate_fn) for key in elem.keys()}
    elif isinstance(elem, list):
        return [aggregate_batch(samples, aggregate_fn) for samples in zip(*batch)]
    elif elem is None:
        if merge_none:
            return None
    else:
        return batch
