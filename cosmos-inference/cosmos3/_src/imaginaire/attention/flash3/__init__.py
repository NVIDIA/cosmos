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

Flash Attention v3 (flash3) Backend
"""

import torch

from cosmos3._src.imaginaire.attention.utils.safe_ops import log
from cosmos3._src.imaginaire.attention.utils.version import version_at_least

FLASH_ATTENTION_V3_MIN_VERSION = "1.0.3"


def flash3_supported() -> bool:
    """
    Returns whether Flash Attention is supported in this environment.
    Requirements are:
        * Presence of CUDA Runtime (via PyTorch)
        * Presence of Flash Attention, meeting minimum version requirements

    This check guards imports / dependencies on the Flash Attention package.
    """
    if not torch.cuda.is_available():
        log.debug("Flash Attention v3 is not supported because PyTorch did not detect CUDA runtime.")
        return False

    try:
        # pyrefly: ignore  # missing-import
        import flash_attn_3_nv

    except ImportError:
        log.debug("Flash Attention v3 is not supported because the Python package ('flash_attn_3_nv'_) was not found.")
        return False
    except Exception as e:
        log.debug(f"Flash Attention v3 is not supported because importing the Python package failed: {e}")
        return False

    if not version_at_least(flash_attn_3_nv.__version__, FLASH_ATTENTION_V3_MIN_VERSION):
        log.debug(
            f"Flash Attention v3 ('flash_attn_3_nv') build is not supported; minimum required version is "
            f"{FLASH_ATTENTION_V3_MIN_VERSION}, got {flash_attn_3_nv.__version__}."
        )
        return False

    return True


FLASH3_SUPPORTED = flash3_supported()


if FLASH3_SUPPORTED:
    from cosmos3._src.imaginaire.attention.flash3.functions import flash3_attention

else:
    from cosmos3._src.imaginaire.attention.flash3.stubs import flash3_attention

__all__ = ["flash3_attention", "FLASH3_SUPPORTED"]
