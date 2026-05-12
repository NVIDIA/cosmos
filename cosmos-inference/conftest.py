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

from cosmos3._src.imaginaire.lazy_config import lazy_call


lazy_call._CONVERT_TARGET_TO_STRING = True

import gc
import os
from functools import cache
from pathlib import Path

import pytest

from cosmos3.fixtures.args import ALL_LEVELS, ALL_NUM_GPUS, ALLOWED_GPUS_BY_LEVEL, Args, get_args, init_args


@pytest.fixture(scope="module")
def original_datadir(request: pytest.FixtureRequest) -> Path:
    root_dir = request.config.rootpath
    relative_path = request.path.with_suffix("").relative_to(root_dir)
    return root_dir / "tests/data" / relative_path


@cache
def _get_available_gpus() -> int:
    import pynvml

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


def pytest_configure(config: pytest.Config):
    args = Args.from_config(config)
    init_args(args)

    if (
        args.num_gpus is not None
        and args.levels is not None
        and all(args.num_gpus not in ALLOWED_GPUS_BY_LEVEL[level] for level in args.levels)
    ):
        pytest.exit(f"No tests for {args.num_gpus} GPUs and levels {args.levels}.", returncode=0)

    if args.worker_id == "master":
        return

    if args.worker_index > 1:
        if args.num_gpus is None:
            raise NotImplementedError(f"Running parallel tests requires --num-gpus to be set.")

    # Check if there are enough GPUs available.
    if args.num_gpus is not None and args.num_gpus > 0:
        required_gpus = args.num_gpus * (args.worker_index + 1)
    else:
        required_gpus = 1
    available_gpus = _get_available_gpus()
    if available_gpus < required_gpus:
        raise ValueError(f"Not enough GPUs available. Required: {required_gpus}, Available: {available_gpus}")

    # Limit threading to reduce contention
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


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
    args = get_args()

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

        # Check if the test should be skipped
        if not args.enable_manual and manual_mark is not None:
            item.add_marker(pytest.mark.skip(reason="test requires --manual"))
        if args.levels is not None and level not in args.levels:
            item.add_marker(pytest.mark.skip(reason=f"test requires --levels={level}"))
        if args.num_gpus is not None and gpus != args.num_gpus:
            item.add_marker(pytest.mark.skip(reason=f"test requires --num-gpus={gpus}"))
        available_gpus = _get_available_gpus()
        if gpus > available_gpus:
            item.add_marker(
                pytest.mark.skip(reason=f"test requires {gpus} GPUs, but only {available_gpus} are available")
            )

    # Exclude skipped tests
    selected_items = []
    deselected_items = []
    for item in items:
        if item.get_closest_marker("skip"):
            deselected_items.append(item)
            continue
        selected_items.append(item)
    items[:] = selected_items
    config.hook.pytest_deselected(items=deselected_items)


def pytest_runtest_setup(item: pytest.Item):
    import torch

    args = get_args()

    gpus_mark = item.get_closest_marker(name="gpus")
    try:
        gpus = _parse_gpus_marker(gpus_mark) if gpus_mark else 0
    except ValueError as e:
        pytest.fail(f"Invalid marker on test {item.name}: {e}")
        assert False, "unreachable"

    # Limit the number of GPUs used by the test
    if gpus > 0:
        device_start = args.worker_index * gpus
        device_end = device_start + gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(device_start, device_end)))
        os.environ["NUM_GPUS"] = str(gpus)
    else:
        device = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        os.environ["NUM_GPUS"] = "1"

        test_max_processes = int(os.environ.get("TEST_MAX_PROCESSES", "8"))
        device_memory_fraction = 1 / max(args.worker_count, test_max_processes)
        os.environ["DEVICE_MEMORY_FRACTION"] = str(device_memory_fraction)
        torch.cuda.set_per_process_memory_fraction(device_memory_fraction)


@pytest.fixture(autouse=True)
def init_cosmos_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from cosmos3.common.init import _init_log_console, _init_log_files

    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path / "imaginaire4-output"))

    _init_log_console()
    _init_log_files(tmp_path)

    yield


@pytest.fixture(autouse=True)
def init_torch_test():
    import torch

    from cosmos3.common.init import set_seed

    # Reproducibility
    set_seed(0)

    yield

    # Cleanup memory
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
