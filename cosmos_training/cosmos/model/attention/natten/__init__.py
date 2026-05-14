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

NATTEN Backend
"""

import torch

from cosmos.model.attention.utils.safe_ops import log
from cosmos.model.attention.utils.version import version_at_least

# 0.21.5.dev1 patches some varlen issues
# 0.21.5.dev2 adds torch compile support
# 0.21.5.dev3 fixes a few compat issues for older torch versions
# 0.21.5.dev6 gqa/mqa support
# 0.21.5.dev9 fixes attention merging
NATTEN_MIN_VERSION = "0.21.5.dev9"

# Hopper FMHA causal and varlen support
NATTEN_HOPPER_CAUSAL_VARLEN_VERSION = "0.21.6.dev3"

# Blackwell-FMHA Deterministic bwd support
NATTEN_BLACKWELL_DETERMINISTIC_VERSION = "0.21.6.dev7"

# Blackwell-FMHA/FNA support extended to head dims meeting alignment constraint and <= 128
NATTEN_BLACKWELL_PARTIAL_HEAD_DIM_VERSION = "0.21.6.dev8"

# 0.21.9.dev0 adds varlen multi-dimensional (sparse) attention
NATTEN_VARLEN_MULTI_DIM_VERSION = "0.21.9.dev0"


def get_natten_version() -> str:
    try:
        import natten
    except (ImportError, Exception):
        return "0.0.0"

    return natten.__version__


def natten_version_satisfies(min_version_str: str) -> bool:
    """
    Check if the installed NATTEN version satisfies a specific minimum version requirement.

    Parameters:
        min_version_str (str): Minimum version string (e.g., "0.21.5" or "0.21.5.dev12").

    Returns:
        bool: True if NATTEN is installed and meets the minimum version requirement.
    """
    return version_at_least(get_natten_version(), min_version_str)


def natten_supported() -> bool:
    """
    Returns whether NATTEN is supported in this environment.
    Requirements are:
        * Presence of CUDA Runtime (via PyTorch)
        * Presence of NATTEN, meeting minimum version requirements

    This check guards imports / dependencies on the NATTEN package.
    """
    if not torch.cuda.is_available():
        log.debug("NATTEN Attention is not supported because PyTorch did not detect CUDA runtime.")
        return False

    try:
        import natten
    except ImportError:
        log.debug("NATTEN Attention is not supported because the Python package was not found.")
        return False
    except Exception as e:
        log.debug(f"NATTEN Attention is not supported because importing the Python package failed: {e}")
        return False

    if not version_at_least(natten.__version__, NATTEN_MIN_VERSION):
        log.debug(
            f"NATTEN Attention is not supported due to insufficient NATTEN version "
            f"{natten.__version__}, expected at least {NATTEN_MIN_VERSION}."
        )
        return False

    return True


NATTEN_SUPPORTED = natten_supported()

if NATTEN_SUPPORTED:
    from cosmos.model.attention.natten.functions import (
        natten_attention,
        natten_multi_dim_attention,
        natten_multi_dim_attention_varlen,
    )

else:
    from cosmos.model.attention.natten.stubs import (
        natten_attention,
        natten_multi_dim_attention,
        natten_multi_dim_attention_varlen,
    )

__all__ = [
    "natten_attention",
    "natten_multi_dim_attention",
    "natten_multi_dim_attention_varlen",
    "NATTEN_SUPPORTED",
    "NATTEN_MIN_VERSION",
    "NATTEN_VARLEN_MULTI_DIM_VERSION",
    "get_natten_version",
    "natten_version_satisfies",
]
