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

Utilities: version checking helpers shared across backends.
"""

from packaging.version import Version


def parse_version(version_str: str) -> Version | None:
    """Parse a version string into a ``packaging.version.Version``, returning ``None`` on failure."""
    try:
        return Version(version_str)
    except Exception:
        return None


def version_at_least(version_str: str, min_version: str) -> bool:
    """Return ``True`` if *version_str* >= *min_version*. Returns ``False`` on parse failure."""
    v = parse_version(version_str)
    m = parse_version(min_version)
    if v is None or m is None:
        return False
    return v >= m


def version_in_range(version_str: str, min_version: str, max_version: str) -> bool:
    """Return ``True`` if *min_version* <= *version_str* <= *max_version*. Returns ``False`` on parse failure."""
    v = parse_version(version_str)
    lo = parse_version(min_version)
    hi = parse_version(max_version)
    if v is None or lo is None or hi is None:
        return False
    return lo <= v <= hi
