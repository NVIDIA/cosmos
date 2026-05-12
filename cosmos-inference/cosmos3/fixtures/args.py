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

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pytest
from typing_extensions import Self

MAX_GPUS = int(os.environ.get("TEST_MAX_GPUS", "8"))
"""Maximum number of GPUs."""
ALL_LEVELS = (0, 1, 2)
ALL_NUM_GPUS = (0, 1, MAX_GPUS)
ALLOWED_GPUS_BY_LEVEL: dict[int, tuple[int, ...]] = {
    0: (0, 1),
    1: (0, 1, MAX_GPUS),
    2: (0, 1, MAX_GPUS),
}
"""Allowed number of GPUs by level."""

if TYPE_CHECKING:
    Level = int
    NumGpus = int
else:
    Level = Literal[tuple(ALL_LEVELS)]
    NumGpus = Literal[tuple(ALL_NUM_GPUS)]


@dataclass(frozen=True)
class Args:
    worker_id: str
    worker_index: int
    worker_count: int

    enable_manual: bool
    num_gpus: int | None
    levels: set[int] | None

    @classmethod
    def from_config(cls, config: pytest.Config) -> Self:
        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
        if worker_id == "master":
            worker_index = 0
        else:
            worker_index = int(worker_id.removeprefix("gw"))
        worker_count = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))

        if config.option.levels is not None:
            levels = set(map(int, config.option.levels.split(",")))
            if levels.difference(ALL_LEVELS):
                raise ValueError(f"Invalid levels: {levels}")
        else:
            levels = None

        return cls(
            worker_id=worker_id,
            worker_index=worker_index,
            worker_count=worker_count,
            enable_manual=config.option.manual,
            num_gpus=config.option.num_gpus,
            levels=levels,
        )


_ARGS: Args | None = None


def init_args(args: Args) -> None:
    global _ARGS
    if _ARGS is not None:
        raise ValueError("Args already initialized")
    _ARGS = args


def get_args() -> Args:
    global _ARGS
    if _ARGS is None:
        raise ValueError("Args not initialized")
    return _ARGS
