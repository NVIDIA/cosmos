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

Flash Attention v2 (flash2) Backend
"""

import torch

from cosmos3._src.imaginaire.attention.utils.safe_ops import log
from cosmos3._src.imaginaire.attention.utils.version import version_in_range

# We lock to safe releases of Flash 2
# We will have a separate backend identifier for 2025 releases with CuTeDSL
# kernels.
FLASH_ATTENTION_V2_MIN_VERSION = "2.7.0"
FLASH_ATTENTION_V2_MAX_VERSION = "2.7.4.post1"


def flash2_supported() -> bool:
    """
    Returns whether Flash Attention is supported in this environment.
    Requirements are:
        * Presence of CUDA Runtime (via PyTorch)
        * Presence of Flash Attention, meeting minimum version requirements

    This check guards imports / dependencies on the Flash Attention package.
    """
    if not torch.cuda.is_available():
        log.debug("Flash Attention v2 is not supported because PyTorch did not detect CUDA runtime.")
        return False

    try:
        import flash_attn

    except ImportError:
        log.debug("Flash Attention v2 is not supported because the Python package was not found.")
        return False
    except Exception as e:
        log.debug(f"Flash Attention v2 is not supported because importing the Python package failed: {e}")
        return False

    flash2_version_str = None
    if not hasattr(flash_attn, "__version__"):
        from importlib.metadata import version

        flash2_version_str = version("flash_attn")
    else:
        flash2_version_str = flash_attn.__version__

    if not version_in_range(flash2_version_str, FLASH_ATTENTION_V2_MIN_VERSION, FLASH_ATTENTION_V2_MAX_VERSION):
        log.debug(
            "Flash Attention v2 build is not supported; this backend only supports versions "
            f"{FLASH_ATTENTION_V2_MIN_VERSION} through {FLASH_ATTENTION_V2_MAX_VERSION}, got "
            f"{flash2_version_str}."
        )
        return False

    return True


FLASH2_SUPPORTED = flash2_supported()

if FLASH2_SUPPORTED:
    from cosmos3._src.imaginaire.attention.flash2.functions import flash2_attention

else:
    from cosmos3._src.imaginaire.attention.flash2.stubs import flash2_attention

__all__ = ["flash2_attention", "FLASH2_SUPPORTED"]
