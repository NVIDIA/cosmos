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

Environment-related utilities.
"""

import os

import torch

from cosmos.utils import log


# Controls all regions guarded against torch compile
# Logs, and certain assertions cause graph breaks.
def is_torch_compiling() -> bool:
    try:
        return torch.compiler.is_compiling()
    except Exception as e:
        log.exception(f"Exception occurred checking whether in torch compiled region: {e}")
        # Assume too old to support torch compile
        return False


def parse_backend_filter(env_var_value: str | None, default_backends: list[str]) -> list[str]:
    """
    Parse backend filter from environment variable value.

    Logic:
    - If env_var_value is None or empty: return default_backends (no filtering)
    - If all items start with "-": start with default_backends and remove specified backends (ban-list)
    - Otherwise: only include specified backends (allow-list)
    - Mixing bans and allows raises a ValueError
    - For ban-list: invalid backend names only issue a warning (may not be available on this GPU)
    - For allow-list: invalid backend names raise a ValueError (explicit request for unavailable backend)

    Parameters:
        env_var_value (str | None): The environment variable value (comma-separated list)
        default_backends (list[str]): The default list of backends

    Returns:
        list[str]: Filtered list of backends

    Raises:
        ValueError: If the list mixes bans and allows, or if any backend name in allow-list is invalid
    """
    if not env_var_value:
        return default_backends

    # Parse comma-separated list
    items = [item.strip() for item in env_var_value.split(",") if item.strip()]

    if not items:
        return default_backends

    # Check if items are bans or allows
    bans = [item for item in items if item.startswith("-")]
    allows = [item for item in items if not item.startswith("-")]

    # Validate: cannot mix bans and allows
    if bans and allows:
        raise ValueError(
            f"Cannot mix ban-list (items starting with '-') and allow-list (items without '-') in backend filter. "
            f"Got bans: {bans}, allows: {allows}. "
            f"Either specify all bans (e.g., '-flash2,-cudnn') or all allows (e.g., 'natten,flash2')."
        )

    default_backends_set = set(default_backends)

    if bans:
        # Ban-list mode: start with defaults and remove specified backends
        ban_backends = {item[1:] for item in bans}  # Remove "-" prefix

        # Warn about banned backends that don't exist in default list
        # (they may not be available on this GPU, which is fine)
        invalid_bans = ban_backends - default_backends_set
        if invalid_bans:
            log.warning(
                f"Attempting to ban backend(s) that are not in the available list: {sorted(invalid_bans)}. "
                f"Available backends are: {default_backends}. "
                f"This may be expected if these backends are not supported on this GPU."
            )

        filtered = [b for b in default_backends if b not in ban_backends]
        log.debug(f"Backend filter (ban-list): removing {ban_backends} from {default_backends}, result: {filtered}")
        return filtered
    else:
        # Allow-list mode: only include specified backends
        allow_backends = set(allows)

        # Validate: all allowed backends must exist in default list
        invalid_allows = allow_backends - default_backends_set
        if invalid_allows:
            raise ValueError(
                f"Invalid backend(s) in allow-list: {sorted(invalid_allows)}. "
                f"Available backends are: {default_backends}. "
                f"Check for typos in the backend names."
            )

        # Preserve order from default_backends for consistency
        filtered = [b for b in default_backends if b in allow_backends]
        log.debug(f"Backend filter (allow-list): allowing {allow_backends} from {default_backends}, result: {filtered}")
        return filtered


def filter_attention_backends(default_backends: list[str]) -> list[str]:
    env_var_value = os.environ.get("I4_ATTN_BACKENDS")
    return parse_backend_filter(env_var_value, default_backends)


def filter_multi_dim_attention_backends(default_backends: list[str]) -> list[str]:
    env_var_value = os.environ.get("I4_ATTN_BACKENDS_MULTIDIM")
    return parse_backend_filter(env_var_value, default_backends)


def filter_attention_merge_backends(default_backends: list[str]) -> list[str]:
    env_var_value = os.environ.get("I4_ATTN_BACKENDS_MERGE")
    return parse_backend_filter(env_var_value, default_backends)
