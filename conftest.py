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

"""Root pytest configuration.

This is a slimmed-down adaptation of ``cosmos-inference/conftest.py``.
The richer Args / fixture infrastructure (``cosmos3.fixtures.args``,
``cosmos3._src.imaginaire.lazy_config``, ``cosmos3.common.init``) is
not yet ported to the root ``cosmos/`` package. As ``cosmos/`` grows
the equivalent helpers, re-port the additional fixtures (init logging,
seed, lazy_call._CONVERT_TARGET_TO_STRING, etc.) from cosmos-inference.
"""

from __future__ import annotations

import gc
import os
from functools import cache
from pathlib import Path

import pytest

ALL_NUM_GPUS = (0, 1, 2, 4, 8)
ALL_LEVELS = (0, 1, 2)
# Tests at level ``l`` are allowed to request ``ALLOWED_GPUS_BY_LEVEL[l]`` GPUs.
ALLOWED_GPUS_BY_LEVEL: dict[int, tuple[int, ...]] = {
    0: (0, 1),
    1: (0, 1, 2, 4),
    2: ALL_NUM_GPUS,
}


@pytest.fixture(scope="module")
def original_datadir(request: pytest.FixtureRequest) -> Path:
    root_dir = request.config.rootpath
    relative_path = request.path.with_suffix("").relative_to(root_dir)
    return root_dir / "tests/data" / relative_path


@cache
def _get_available_gpus() -> int:
    try:
        import pynvml
    except ImportError:
        return 0
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return device_count
    except pynvml.NVMLError as e:
        print(f"WARNING: Failed to get available GPUs: {e}")
        return 0


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--manual", action="store_true", default=False, help="Run manual tests")
    parser.addoption(
        "--num-gpus",
        default=None,
        type=int,
        choices=ALL_NUM_GPUS,
        help="Run tests with the specified number of GPUs",
    )
    parser.addoption("--levels", default=None, help="Run tests with the specified levels (comma-separated list)")


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    num_gpus: int | None = config.option.num_gpus
    if num_gpus is None:
        return 1
    if num_gpus == 0:
        return None

    available_gpus = _get_available_gpus()
    if available_gpus < num_gpus:
        raise ValueError(f"Not enough GPUs available. Required: {num_gpus}, Available: {available_gpus}")
    return available_gpus // num_gpus


def _parse_levels(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    levels = tuple(int(x) for x in value.split(","))
    for level in levels:
        if level not in ALL_LEVELS:
            raise ValueError(f"Invalid level {level} not in {ALL_LEVELS}")
    return levels


def _get_marker(item: pytest.Item, name: str) -> pytest.Mark | None:
    markers = list(item.iter_markers(name=name))
    if not markers:
        return None
    marker = markers[0]
    for other_marker in markers[1:]:
        if other_marker != marker:
            raise ValueError(f"Multiple different markers found for {name}: {markers}")
    return marker


def _parse_level_marker(mark: pytest.Mark) -> int:
    if len(mark.args) != 1:
        raise ValueError(f"Invalid arguments: {mark.args}")
    if mark.kwargs:
        raise ValueError(f"Invalid keyword arguments: {mark.kwargs}")
    level = mark.args[0]
    if level not in ALL_LEVELS:
        raise ValueError(f"Invalid level {level} not in {ALL_LEVELS}")
    return level


def _parse_gpus_marker(mark: pytest.Mark) -> int:
    if len(mark.args) != 1:
        raise ValueError(f"Invalid arguments: {mark.args}")
    if mark.kwargs:
        raise ValueError(f"Invalid keyword arguments: {mark.kwargs}")
    required_gpus = int(mark.args[0])
    if required_gpus not in ALL_NUM_GPUS:
        raise ValueError(f"Invalid number of GPUs {required_gpus} not in {ALL_NUM_GPUS}")
    return required_gpus


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    enable_manual: bool = config.getoption("--manual")
    num_gpus: int | None = config.option.num_gpus
    levels = _parse_levels(config.getoption("--levels"))

    for item in items:
        manual_mark = _get_marker(item, "manual")
        level_mark = _get_marker(item, "level")
        gpus_mark = _get_marker(item, "gpus")
        try:
            level = _parse_level_marker(level_mark) if level_mark else 0
            gpus = _parse_gpus_marker(gpus_mark) if gpus_mark else 0
        except ValueError as e:
            pytest.fail(f"Invalid marker on test {item.name}: {e}")
            assert False, "unreachable"

        allowed_gpus = ALLOWED_GPUS_BY_LEVEL[level]
        if gpus not in allowed_gpus:
            pytest.fail(f"Level {level} tests must have {allowed_gpus} GPUs, but {item.name} has {gpus} GPUs")

        if not enable_manual and manual_mark is not None:
            item.add_marker(pytest.mark.skip(reason="test requires --manual"))
        if levels is not None and level not in levels:
            item.add_marker(pytest.mark.skip(reason=f"test requires --levels={level}"))
        if num_gpus is not None and gpus != num_gpus:
            item.add_marker(pytest.mark.skip(reason=f"test requires --num-gpus={gpus}"))
        available_gpus = _get_available_gpus()
        if gpus > available_gpus:
            item.add_marker(
                pytest.mark.skip(reason=f"test requires {gpus} GPUs, but only {available_gpus} are available")
            )

    selected_items = []
    deselected_items = []
    for item in items:
        if item.get_closest_marker("skip"):
            deselected_items.append(item)
            continue
        selected_items.append(item)
    items[:] = selected_items
    config.hook.pytest_deselected(items=deselected_items)


@pytest.fixture(autouse=True)
def init_torch_test():
    try:
        import torch
    except ImportError:
        yield
        return
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_WHITELIST_ENV_VARS = {
    "LD_LIBRARY_PATH",
    "QT_QPA_FONTDIR",
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "TORCHINDUCTOR_CACHE_DIR",
}


@pytest.fixture(autouse=True)
def detect_env_modifications():
    original_env = dict(os.environ)

    yield

    new_env = dict(os.environ)

    for env in [original_env, new_env]:
        for k in list(env.keys()):
            if k.startswith("PYTEST_") or k in _WHITELIST_ENV_VARS:
                del env[k]
    if new_env != original_env:
        added, removed, modified = _compare_dict(new_env, original_env)
        os.environ.clear()
        os.environ.update(original_env)
        raise ValueError(
            f"Environment variables modified by test! Use 'monkeypatch.setenv' to temporarily modify environment variables. \n"
            f"Added: {added}\n"
            f"Removed: {removed}\n"
            f"Modified: {modified}"
        )


def _compare_dict(actual: dict[str, str], expected: dict[str, str]) -> tuple[set[str], set[str], set[str]]:
    added = set(actual) - set(expected)
    removed = set(expected) - set(actual)
    modified = {k for k in expected if k in actual and expected[k] != actual[k]}
    return added, removed, modified
