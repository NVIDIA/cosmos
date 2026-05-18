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

Mask utilities
"""

from enum import Enum


class CausalType(Enum):
    """
    Different types of causal masking supported by backends of interest.
    """

    # Top-Left: Simplified: mask if q_idx < kv_idx
    # CUTLASS / NATTEN default
    # Q = 2, KV = 5:
    # O____
    # OO___
    #
    # Q = 5, KV = 2:
    # O_
    # OO
    # OO
    # OO
    # OO
    TopLeft = 0

    # Bottom-right: mask if q_idx + KV - Q < kv_idx
    # Flash Attention default
    # Q = 2, KV = 5:
    # OOOO_
    # OOOOO
    #
    # Q = 5, KV = 2:
    # __
    # __
    # __
    # O_
    # OO
    BottomRight = 1

    # When seqlen_q == seqlen_kv, we don't care about the causal type
    # because top-left and bottom-right are equivalent
    DontCare = 2
