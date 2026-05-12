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

"""Feature flags."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Final


class StrEnum(str, Enum):
    """Backport of StrEnum from Python 3.11."""

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
        return name.lower()


def _parse_bool(value: str) -> bool:
    """Parse string to a boolean."""
    return value.lower() in ["true", "1", "yes", "y"]


def _get_bool(name: str, default: bool) -> bool:
    """Get a boolean flag from the environment."""
    value = os.environ.get(name, "")
    if not value:
        return default
    return _parse_bool(value)


TRAINING: Final[bool] = _get_bool("COSMOS_TRAINING", False)
"""Whether to enable training features.

This is used to make training dependencies optional.
"""

INTERNAL: Final[bool] = _get_bool("COSMOS_INTERNAL", False)
"""Whether to use internal (nvidia-only) resources (e.g. S3)."""

SMOKE: Final[bool] = _get_bool("COSMOS_SMOKE", False)
"""Whether to enable smoke test.

Sets parameters to minimum values (e.g. num_steps=1, num_layers=2).
"""


class Device(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"
    META = "meta"


DEVICE: Final[Device] = Device(os.environ.get("COSMOS_DEVICE", "cuda").lower())
"""Torch device to use.

Used for checkpoint conversion and smoke tests.
"""

VERBOSE: Final[bool] = _get_bool("COSMOS_VERBOSE", INTERNAL)
"""Whether to enable verbose console output."""

EXPERIMENTAL_CHECKPOINTS: Final[bool] = _get_bool("COSMOS_EXPERIMENTAL_CHECKPOINTS", INTERNAL)
"""Whether to enable experimental checkpoints."""


if INTERNAL:
    TRAINING = True


@dataclass
class Flags:
    internal: bool = INTERNAL
    training: bool = TRAINING
    smoke: bool = SMOKE
    device: Device = DEVICE
    verbose: bool = VERBOSE
    experimental_checkpoints: bool = EXPERIMENTAL_CHECKPOINTS


FLAGS = Flags()
"""Convenience object for accessing flags."""
