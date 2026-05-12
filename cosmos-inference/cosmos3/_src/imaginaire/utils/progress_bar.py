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
Progress bar wrapper that gets automatically disabled when in a Timer region, or any other context
where we'd want to disable progress bars, including when TQDM is not present, or when user sets
DISABLE_TQDM=1. We can eventually add a simple ascii progress bar as fallback for missing
dependencies.
"""

import os

from cosmos3._src.imaginaire.utils import distributed
from cosmos3._src.imaginaire.utils.timer import in_timer_region

try:
    import tqdm as _tqdm  # noqa: F401

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
except Exception as e:
    HAS_TQDM = True


def _tqdm_wrapper(*args, **kwargs):
    if HAS_TQDM:
        import tqdm

        return tqdm.tqdm(*args, **kwargs)

    raise ImportError("TQDM is not installed. Please install it and try again.")


def progress_bar(fn, desc=None, total=None, force_display: bool = False):
    """
    Progress bars a great, but they're not for everybody, certainly not for everywhere.
    They must be guarded against:
        * We're benchmarking performance (with Timer)
        * If tqdm / other progress bars aren't available, skip instead of failing.
        * If multi-process / GPU, only one (usually rank 0) must display it, just like prints.
        * If the user just doesn't want progress bars (toggle via environment variables.

    This function consideres all of those cases
    """

    disable_tqdm = os.environ.get("DISABLE_TQDM", "0") == "1"
    is_in_timer_region = in_timer_region()
    is_rank0 = True

    # Wide-scope try/except on determining rank, in case distributed context is uninitialized in a
    # single-process program. If exception occurs, it's better to just assume single-process.
    try:
        is_rank0 = distributed.get_rank() == 0
    except Exception as e:
        pass

    if not force_display and (not is_rank0 or is_in_timer_region or disable_tqdm):
        return fn

    return _tqdm_wrapper(fn, desc=desc, total=total)


__all__ = ["progress_bar"]
