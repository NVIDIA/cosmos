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
Unit tests for version parsing and comparison utilities.
Covers version formats used across all attention backends (NATTEN, Flash2, Flash3, cuDNN).
"""

import pytest

from cosmos.model.attention.utils.version import parse_version, version_at_least, version_in_range


@pytest.mark.L0
class TestParseVersion:
    """Test parse_version function."""

    def test_simple_release(self):
        v = parse_version("1.0.0")
        assert v is not None
        assert str(v) == "1.0.0"

    def test_three_component(self):
        for s in ["0.21.5", "2.7.0", "1.0.3", "1.14.0"]:
            assert parse_version(s) is not None

    def test_dev_version(self):
        v = parse_version("0.21.5.dev9")
        assert v is not None
        assert v.dev == 9

    def test_dev_zero(self):
        v = parse_version("0.21.5.dev0")
        assert v is not None
        assert v.dev == 0

    def test_invalid_returns_none(self):
        for s in ["", "not_a_version", "abc.def.ghi"]:
            assert parse_version(s) is None


@pytest.mark.L0
class TestVersionAtLeast:
    """Test version_at_least across version formats from all backends."""

    # --- NATTEN-style versions (with .devN) ---

    def test_natten_dev_ordering(self):
        assert version_at_least("0.21.5.dev12", "0.21.5.dev9")
        assert version_at_least("0.21.5.dev9", "0.21.5.dev9")
        assert not version_at_least("0.21.5.dev8", "0.21.5.dev9")

    def test_natten_release_beats_dev(self):
        assert version_at_least("0.21.5", "0.21.5.dev9")
        assert version_at_least("0.21.5", "0.21.5.dev99")
        assert not version_at_least("0.21.5.dev99", "0.21.5")

    def test_natten_cross_base_dev(self):
        assert version_at_least("0.21.6.dev1", "0.21.5")
        assert version_at_least("0.21.6.dev1", "0.21.5.dev99")
        assert not version_at_least("0.21.5.dev99", "0.21.6.dev1")

    def test_natten_min_version(self):
        assert version_at_least("0.21.5.dev9", "0.21.5.dev9")
        assert version_at_least("0.21.6", "0.21.5.dev9")
        assert not version_at_least("0.21.5.dev8", "0.21.5.dev9")

    # --- Flash3 / cuDNN-style versions (simple M.m.p) ---

    def test_simple_release_ordering(self):
        assert version_at_least("1.0.3", "1.0.3")
        assert version_at_least("1.0.4", "1.0.3")
        assert version_at_least("1.1.0", "1.0.3")
        assert version_at_least("2.0.0", "1.0.3")
        assert not version_at_least("1.0.2", "1.0.3")
        assert not version_at_least("0.99.99", "1.0.3")

    def test_cudnn_frontend_version(self):
        assert version_at_least("1.14.0", "1.14.0")
        assert version_at_least("1.15.0", "1.14.0")
        assert not version_at_least("1.13.9", "1.14.0")

    # --- Edge cases ---

    def test_major_version_jump(self):
        assert version_at_least("1.0.0", "0.21.5")
        assert version_at_least("1.0.0.dev1", "0.99.99")

    def test_minor_version_jump(self):
        assert version_at_least("0.22.0", "0.21.99")

    def test_invalid_returns_false(self):
        assert not version_at_least("invalid", "0.21.5")
        assert not version_at_least("0.21.5", "invalid")
        assert not version_at_least("invalid", "invalid")
        assert not version_at_least("", "1.0.0")
        assert not version_at_least("1.0.0", "")


@pytest.mark.L0
class TestVersionInRange:
    """Test version_in_range, primarily used by the Flash2 backend."""

    # --- Flash2-style range [2.7.0, 2.7.4] ---

    def test_flash2_in_range(self):
        assert version_in_range("2.7.0", "2.7.0", "2.7.4")
        assert version_in_range("2.7.2", "2.7.0", "2.7.4")
        assert version_in_range("2.7.4", "2.7.0", "2.7.4")

    def test_flash2_below_range(self):
        assert not version_in_range("2.6.9", "2.7.0", "2.7.4")
        assert not version_in_range("2.6.99", "2.7.0", "2.7.4")
        assert not version_in_range("1.0.0", "2.7.0", "2.7.4")

    def test_flash2_above_range(self):
        assert not version_in_range("2.7.5", "2.7.0", "2.7.4")
        assert not version_in_range("2.8.0", "2.7.0", "2.7.4")
        assert not version_in_range("3.0.0", "2.7.0", "2.7.4")

    # --- Dev versions in ranges ---

    def test_dev_in_range(self):
        # dev9 < release 0.21.5, so it's in range [0.21.4, 0.21.5]
        assert version_in_range("0.21.5.dev9", "0.21.4", "0.21.5")
        # but dev9 < release 0.21.5, so it's NOT >= 0.21.5 as a lower bound
        assert not version_in_range("0.21.5.dev9", "0.21.5", "0.21.6")

    def test_single_version_range(self):
        assert version_in_range("2.7.0", "2.7.0", "2.7.0")
        assert not version_in_range("2.7.1", "2.7.0", "2.7.0")

    # --- Edge cases ---

    def test_invalid_returns_false(self):
        assert not version_in_range("invalid", "2.7.0", "2.7.4")
        assert not version_in_range("2.7.2", "invalid", "2.7.4")
        assert not version_in_range("2.7.2", "2.7.0", "invalid")
        assert not version_in_range("", "2.7.0", "2.7.4")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
