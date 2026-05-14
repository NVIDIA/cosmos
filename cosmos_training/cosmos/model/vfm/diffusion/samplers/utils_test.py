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

import pytest

from cosmos.model.vfm.diffusion.samplers.utils import run_multiseed

# ---------------------------------------------------------------------------
# Single-call (no list kwargs) passthrough
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_no_list_kwargs_passthrough():
    result = run_multiseed(lambda x, y: x + y, x=3, y=7)
    assert result == 10


@pytest.mark.L0
def test_no_list_kwargs_tuple_passthrough():
    result = run_multiseed(lambda x: (x, -x), x=5)
    assert result == (5, -5)


# ---------------------------------------------------------------------------
# All-list kwargs — basic multi-call
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_all_list_kwargs_scalar_return():
    result = run_multiseed(lambda x, y: x * y, x=[1, 2, 3], y=[10, 20, 30])
    assert result == [10, 40, 90]


@pytest.mark.L0
def test_all_list_kwargs_single_element():
    result = run_multiseed(lambda x: x + 1, x=[10])
    assert result == [11]


# ---------------------------------------------------------------------------
# Mixing list and non-list kwargs is forbidden
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_mixed_list_and_non_list_raises():
    with pytest.raises(AssertionError, match="cannot mix"):
        run_multiseed(lambda x, scale: x * scale, x=[1, 2, 3], scale=10)


@pytest.mark.L0
def test_mixed_single_list_single_non_list_raises():
    with pytest.raises(AssertionError, match="cannot mix"):
        run_multiseed(lambda a, b: a + b, a=[1, 2], b=100)


# ---------------------------------------------------------------------------
# Tuple return transposition
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_tuple_return_transposed():
    result = run_multiseed(lambda x: (x, -x), x=[1, 2, 3])
    assert result == ([1, 2, 3], [-1, -2, -3])


@pytest.mark.L0
def test_tuple_return_three_elements():
    result = run_multiseed(lambda x: (x, x * 2, x * 3), x=[10, 20])
    assert result == ([10, 20], [20, 40], [30, 60])


@pytest.mark.L0
def test_tuple_return_with_multiple_list_kwargs():
    result = run_multiseed(lambda x, y: (x + y, x - y), x=[5, 10], y=[3, 3])
    assert result == ([8, 13], [2, 7])


# ---------------------------------------------------------------------------
# fn receives correct arguments per call
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_fn_receives_correct_args_per_call():
    received = []
    run_multiseed(
        lambda a, b, c: received.append((a, b, c)),
        a=["x", "y"],
        b=[1, 2],
        c=["shared", "also_shared"],
    )
    assert received == [("x", 1, "shared"), ("y", 2, "also_shared")]


@pytest.mark.L0
def test_fn_called_correct_number_of_times():
    call_count = [0]

    def fn(x: int) -> int:
        call_count[0] += 1
        return x

    run_multiseed(fn, x=[10, 20, 30, 40])
    assert call_count[0] == 4


# ---------------------------------------------------------------------------
# Validation / error cases
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_list_length_mismatch_raises():
    with pytest.raises(AssertionError, match="same length"):
        run_multiseed(lambda a, b: a + b, a=[1, 2, 3], b=[10, 20])


@pytest.mark.L0
def test_empty_list_consistent_length():
    """Two empty lists have length 0 — no calls are made, returns empty list."""
    result = run_multiseed(lambda a, b: a + b, a=[], b=[])
    assert result == []


@pytest.mark.L0
def test_no_kwargs_passthrough():
    result = run_multiseed(lambda: 42)
    assert result == 42
