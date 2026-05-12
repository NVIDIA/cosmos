# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Callable


def run_multiseed(fn: Callable, **kwargs: list[Any] | Any) -> Any:
    """Run a callable once per seed, indexing all list kwargs in lockstep.

    All keyword arguments must be **either** all lists or all non-lists.
    Mixing is not allowed.

    - **All non-list**: ``fn`` is called once with the kwargs as-is, and its
      return value is passed through directly.
    - **All list**: every list must have the same length *N*.  ``fn`` is called
      *N* times — call *i* receives ``{k: v[i] for k, v in kwargs}``.
      Results are collected into a list.  If ``fn`` returns a tuple, the
      results are transposed into a tuple of lists.

    Args:
        fn: Callable to invoke per seed.
        **kwargs: Keyword arguments for ``fn``.  Must be **all lists** (one
            element per seed, all the same length) or **all non-lists** (single
            call).

    Returns:
        - All non-list kwargs: the raw return value of ``fn``.
        - All list kwargs, ``fn`` returns a tuple: a ``tuple`` of lists,
          transposed across calls.
        - All list kwargs, ``fn`` returns non-tuple: a ``list`` of return
          values.

    Raises:
        AssertionError: If kwargs mix lists and non-lists, or if list kwargs
            have differing lengths.

    Examples:
        Single call (no lists)::

            run_multiseed(lambda x, y: x + y, x=1, y=2)  # returns 3

        Multiple calls with all-list kwargs::

            run_multiseed(lambda x, y: x * y, x=[1, 2, 3], y=[10, 20, 30])
            # returns [10, 40, 90]

        Tuple return transposition::

            run_multiseed(lambda x: (x, -x), x=[1, 2])
            # returns ([1, 2], [-1, -2])
    """
    all_list = all(isinstance(v, list) for v in kwargs.values())
    all_non_list = all(not isinstance(v, list) for v in kwargs.values())
    assert all_list or all_non_list, "All kwargs must be lists or all must be non-lists, cannot mix"

    if all_non_list:
        return fn(**kwargs)

    lengths = {len(v) for v in kwargs.values()}
    assert len(lengths) == 1, f"All list arguments must have the same length, got {lengths}"
    num_calls = lengths.pop()

    results = []
    for i in range(num_calls):
        kwargs_i = {k: v[i] for k, v in kwargs.items()}
        results.append(fn(**kwargs_i))

    if results and isinstance(results[0], tuple):
        return tuple(list(items) for items in zip(*results))
    return results
