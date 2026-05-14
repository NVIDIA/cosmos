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

NATTEN Backend: metadata
Always safe to import (as long as torch is available.)
"""

import torch

from cosmos3._src.imaginaire.attention.utils.safe_ops import log


def get_fwd_dtypes(arch_tag: int) -> list[torch.dtype]:
    """
    Returns data type choices for forward pass according to arch tag (attention.utils.get_arch_tag).

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        data_type_choices (list): a list of PyTorch data types. Empty if device is not supported.

    """

    if arch_tag < 75:
        log.debug("NATTEN is not supported because compute capability is below the minimum (7.5).")
        return []

    if arch_tag in [100, 103]:
        return [torch.float32, torch.float16, torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]

    return [torch.float32, torch.float16, torch.bfloat16]


def get_bwd_dtypes(arch_tag: int) -> list[torch.dtype]:
    """
    Returns data type choices for backward pass according to arch tag (attention.utils.get_arch_tag).

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        data_type_choices (list): a list of PyTorch data types. Empty if device is not supported.

    """

    if arch_tag < 75:
        log.debug("NATTEN is not supported because compute capability is below the minimum (7.5).")
        return []

    return [torch.float32, torch.float16, torch.bfloat16]
