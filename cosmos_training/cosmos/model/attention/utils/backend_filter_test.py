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
Unit tests for backend filtering via environment variables.
"""

import os
import unittest

import pytest

from cosmos.model.attention.utils.environment import (
    filter_attention_backends,
    filter_attention_merge_backends,
    filter_multi_dim_attention_backends,
    parse_backend_filter,
)


@pytest.mark.L0
class TestParseBackendFilter(unittest.TestCase):
    """Test the parse_backend_filter function."""

    def test_basic_functionality(self):
        """Test basic ban-list and allow-list functionality."""
        default = ["flash2", "natten", "cudnn"]

        # Empty/None returns defaults
        assert parse_backend_filter(None, default) == default
        assert parse_backend_filter("", default) == default
        assert parse_backend_filter("  ", default) == default

        # Ban-list mode
        assert parse_backend_filter("-flash2", default) == ["natten", "cudnn"]
        assert parse_backend_filter("-flash2,-cudnn", default) == ["natten"]
        assert parse_backend_filter("-flash2,-natten", ["flash2", "natten"]) == []

        # Allow-list mode
        assert parse_backend_filter("natten", default) == ["natten"]
        assert parse_backend_filter("flash2,natten", default) == ["flash2", "natten"]

        # Preserves order from defaults
        default_ordered = ["flash3", "flash2", "natten", "cudnn"]
        assert parse_backend_filter("cudnn,flash2", default_ordered) == ["flash2", "cudnn"]

        # Whitespace handling
        assert parse_backend_filter(" flash2 , natten ", default) == ["flash2", "natten"]
        assert parse_backend_filter(" -flash2 , -cudnn ", default) == ["natten"]

        # Empty items ignored
        assert parse_backend_filter("flash2,,natten", ["flash2", "natten"]) == ["flash2", "natten"]

    def test_invalid_ban_backends_warns(self):
        """Test that invalid backends in ban-list only issue warnings (not errors)."""
        default = ["flash2", "natten", "cudnn"]

        # Invalid ban backends should just warn, not error (they may not be available on this GPU)
        result = parse_backend_filter("-flash3", default)
        assert result == default  # Nothing removed since flash3 wasn't in the list

        result = parse_backend_filter("-flash2,-invalid_backend", default)
        assert result == ["natten", "cudnn"]  # flash2 removed, invalid_backend ignored with warning

    @pytest.mark.xfail(raises=ValueError, strict=True)
    def test_error_mixing_bans_and_allows(self):
        """Test that mixing bans and allows raises an error."""
        default = ["flash2", "natten", "cudnn"]
        # This should raise ValueError about mixing ban-list and allow-list
        parse_backend_filter("-flash2,natten", default)

    @pytest.mark.xfail(raises=ValueError, strict=True)
    def test_error_invalid_allow_backends(self):
        """Test that invalid backends in allow-list raise an error."""
        default = ["flash2", "natten", "cudnn"]
        # This should raise ValueError about invalid backend in allow-list
        parse_backend_filter("flash3", default)

    @pytest.mark.xfail(raises=ValueError, strict=True)
    def test_error_case_sensitivity_allow(self):
        """Test that backend names are case-sensitive in allow-list."""
        default = ["flash2", "natten"]
        # This should raise ValueError about invalid backend (case mismatch)
        parse_backend_filter("Flash2", default)


@pytest.mark.L0
class TestFilterAPIs(unittest.TestCase):
    """Test the filter_*_backends functions."""

    def setUp(self):
        """Save and clear environment variables before each test."""
        self.saved_env = {}
        for var in ["I4_ATTN_BACKENDS", "I4_ATTN_BACKENDS_MULTIDIM", "I4_ATTN_BACKENDS_MERGE"]:
            self.saved_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        """Restore environment variables after each test."""
        for var, value in self.saved_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_attention_backends_no_env_var(self):
        """Test filter_attention_backends with no env var returns defaults."""
        default = ["flash2", "natten"]
        result = filter_attention_backends(default)
        assert result == default

    def test_attention_backends_ban_list(self):
        """Test filter_attention_backends with ban-list."""
        os.environ["I4_ATTN_BACKENDS"] = "-flash2"
        default = ["flash2", "natten", "cudnn"]
        result = filter_attention_backends(default)
        assert result == ["natten", "cudnn"]

    def test_attention_backends_allow_list(self):
        """Test filter_attention_backends with allow-list."""
        os.environ["I4_ATTN_BACKENDS"] = "natten"
        default = ["flash2", "natten", "cudnn"]
        result = filter_attention_backends(default)
        assert result == ["natten"]

    def test_multi_dim_backends_no_env_var(self):
        """Test filter_multi_dim_attention_backends with no env var."""
        default = ["natten"]
        result = filter_multi_dim_attention_backends(default)
        assert result == default

    def test_multi_dim_backends_ban_list(self):
        """Test filter_multi_dim_attention_backends with ban-list."""
        os.environ["I4_ATTN_BACKENDS_MULTIDIM"] = "-natten"
        default = ["natten"]
        result = filter_multi_dim_attention_backends(default)
        assert result == []

    def test_merge_backends_no_env_var(self):
        """Test filter_attention_merge_backends with no env var."""
        default = ["natten"]
        result = filter_attention_merge_backends(default)
        assert result == default

    def test_merge_backends_ban_list(self):
        """Test filter_attention_merge_backends with ban-list."""
        os.environ["I4_ATTN_BACKENDS_MERGE"] = "-natten"
        default = ["natten"]
        result = filter_attention_merge_backends(default)
        assert result == []

    def test_multiple_env_vars_independent(self):
        """Test that different filter functions use independent env vars."""
        os.environ["I4_ATTN_BACKENDS"] = "-flash2"
        os.environ["I4_ATTN_BACKENDS_MULTIDIM"] = "natten"

        default_sdpa = ["flash2", "natten"]
        default_multidim = ["natten"]

        result_sdpa = filter_attention_backends(default_sdpa)
        result_multidim = filter_multi_dim_attention_backends(default_multidim)

        assert result_sdpa == ["natten"]
        assert result_multidim == ["natten"]


@pytest.mark.L0
class TestBackendListIntegration(unittest.TestCase):
    """Integration tests for backend list functions with environment variables."""

    def setUp(self):
        """Save and clear environment variables before each test."""
        self.saved_env = {}
        for var in ["I4_ATTN_BACKENDS", "I4_ATTN_BACKENDS_MULTIDIM", "I4_ATTN_BACKENDS_MERGE"]:
            self.saved_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        """Restore environment variables after each test."""
        for var, value in self.saved_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_sdpa_backend_filtering(self):
        """Test that get_backend_list respects backend filtering."""
        from cosmos.model.attention.backends import get_backend_list

        # Test without env var (should return defaults based on arch)
        backends = get_backend_list(90)  # H100
        assert "flash3" in backends or "flash2" in backends or "natten" in backends

        # Test with ban-list
        os.environ["I4_ATTN_BACKENDS"] = "-flash3,-flash2"
        backends = get_backend_list(90)
        assert "flash3" not in backends
        assert "flash2" not in backends

    def test_multidim_backend_filtering(self):
        """Test that get_multi_dim_backend_list respects backend filtering."""
        from cosmos.model.attention.backends import get_multi_dim_backend_list

        # Test without env var
        backends = get_multi_dim_backend_list(90)
        assert backends == ["natten"]

        # Test with ban-list
        os.environ["I4_ATTN_BACKENDS_MULTIDIM"] = "-natten"
        backends = get_multi_dim_backend_list(90)
        assert backends == []


if __name__ == "__main__":
    unittest.main()
