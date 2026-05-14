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
Unit tests for torch.compile-safe lru_cache.
"""

import unittest
from unittest.mock import patch

import pytest

from cosmos.model.attention.utils.safe_ops.functools import lru_cache as safe_lru_cache


@pytest.mark.L0
class TestSafeLruCache(unittest.TestCase):
    def test_caching_behavior(self):
        """Test all normal caching functionality."""
        # Basic caching
        call_count = [0]

        @safe_lru_cache
        def func1(x):
            call_count[0] += 1
            return x * 2

        assert func1(5) == 10
        assert call_count[0] == 1
        assert func1(5) == 10
        assert call_count[0] == 1  # Cached
        assert func1(10) == 20
        assert call_count[0] == 2  # Different arg

        # Multiple arguments
        call_count[0] = 0

        @safe_lru_cache
        def func2(x, y):
            call_count[0] += 1
            return x + y

        assert func2(1, 2) == 3
        assert call_count[0] == 1
        assert func2(1, 2) == 3
        assert call_count[0] == 1  # Cached
        assert func2(2, 1) == 3
        assert call_count[0] == 2  # Different args

        # Kwargs
        call_count[0] = 0

        @safe_lru_cache
        def func3(x, y=10):
            call_count[0] += 1
            return x + y

        assert func3(5) == 15
        assert call_count[0] == 1
        assert func3(5) == 15
        assert call_count[0] == 1  # Cached
        assert func3(5, y=20) == 25
        assert call_count[0] == 2  # Different kwargs

        # Maxsize parameter
        call_count[0] = 0

        @safe_lru_cache(maxsize=2)
        def func4(x):
            call_count[0] += 1
            return x * 2

        func4(1)
        func4(2)
        assert call_count[0] == 2
        func4(1)
        func4(2)
        assert call_count[0] == 2  # Both cached
        func4(3)
        assert call_count[0] == 3  # Evicts oldest (1)
        func4(1)
        assert call_count[0] == 4  # Must recompute

        # Decorator syntax without parens
        call_count[0] = 0

        @safe_lru_cache
        def func5(x):
            call_count[0] += 1
            return x * 2

        func5(5)
        assert call_count[0] == 1
        func5(5)
        assert call_count[0] == 1  # Cached

        # Decorator syntax with parens
        call_count[0] = 0

        @safe_lru_cache()
        def func6(x):
            call_count[0] += 1
            return x * 2

        func6(5)
        assert call_count[0] == 1
        func6(5)
        assert call_count[0] == 1  # Cached

        # cache_clear method
        call_count[0] = 0

        @safe_lru_cache
        def func7(x):
            call_count[0] += 1
            return x * 2

        func7(5)
        assert call_count[0] == 1
        func7(5)
        assert call_count[0] == 1  # Cached
        func7.cache_clear()
        func7(5)
        assert call_count[0] == 2  # Recomputed

        # cache_info method
        @safe_lru_cache
        def func8(x):
            return x * 2

        info = func8.cache_info()
        assert info.hits == 0
        assert info.misses == 0
        func8(5)
        info = func8.cache_info()
        assert info.hits == 0
        assert info.misses == 1
        func8(5)
        info = func8.cache_info()
        assert info.hits == 1
        assert info.misses == 1

    def test_compile_mode_behavior(self):
        """Test torch.compile-aware behavior."""
        call_count = [0]

        @safe_lru_cache
        def func(x):
            call_count[0] += 1
            return x * 2

        # Normal mode: caching enabled
        func(5)
        assert call_count[0] == 1
        func(5)
        assert call_count[0] == 1  # Cached

        # Compile mode: caching disabled
        with patch("cosmos.model.attention.utils.environment.is_torch_compiling", return_value=True):
            from cosmos.model.attention.utils.environment import is_torch_compiling

            assert is_torch_compiling()

            func(5)
            assert call_count[0] == 2  # Not cached
            func(5)
            assert call_count[0] == 3  # Not cached

        # Back to normal mode: caching enabled again
        func(5)
        assert call_count[0] == 3  # Should use old cache


if __name__ == "__main__":
    unittest.main()
