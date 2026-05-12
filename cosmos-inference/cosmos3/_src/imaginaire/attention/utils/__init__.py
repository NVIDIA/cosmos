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

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Utilities: compute capability detection, helpers, and more.
"""

from typing import Any

import torch

from cosmos3._src.imaginaire.attention.utils.determinism import torch_deterministic_mode
from cosmos3._src.imaginaire.attention.utils.environment import is_torch_compiling
from cosmos3._src.imaginaire.attention.utils.safe_ops import log


def get_arch_tag(device: torch.device | None = None) -> int:
    """
    Returns the compute capability of a given torch device if it's a CUDA device, otherwise returns 0.

    Args:
        device (torch.device | None): torch device. Uses default device if None.

    Returns:
        device_cc (int): compute capability in the SmXXX format (i.e. 90 for Hopper).
    """
    if torch.cuda.is_available() and torch.version.cuda and (device is None or device.type == "cuda"):
        major, minor = torch.cuda.get_device_capability(device)
        return major * 10 + minor
    return 0


def log_or_raise_error(msg: str, raise_error: bool = False, exception: Any = RuntimeError):
    if raise_error:
        raise exception(msg)
    else:
        log.debug(msg)


def is_full(dtype: torch.dtype) -> bool:
    return dtype == torch.float32


def is_half(dtype: torch.dtype) -> bool:
    return dtype in [torch.float16, torch.bfloat16]


def is_fp8(dtype: torch.dtype) -> bool:
    return dtype in [torch.float8_e5m2, torch.float8_e4m3fn]


def is_hopper(device: torch.device | None = None) -> bool:
    return get_arch_tag(device) == 90


def is_blackwell_dc(device: torch.device | None = None) -> bool:
    return get_arch_tag(device) in [100, 103]


__all__ = [
    "get_arch_tag",
    "log_or_raise_error",
    "is_full",
    "is_half",
    "is_fp8",
    "is_hopper",
    "is_blackwell_dc",
    "is_torch_compiling",
    "torch_deterministic_mode",
]
