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

torch.compile-safe log wrappers.
"""

from cosmos.model.attention.utils.environment import is_torch_compiling
from cosmos.utils import log


def trace(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.trace(message=message, rank0_only=rank0_only)


def debug(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.debug(message=message, rank0_only=rank0_only)


def info(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.info(message=message, rank0_only=rank0_only)


def success(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.success(message=message, rank0_only=rank0_only)


def warning(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.warning(message=message, rank0_only=rank0_only)


def error(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.critical(message=message, rank0_only=rank0_only)


def critical(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.critical(message=message, rank0_only=rank0_only)


def exception(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.exception(message=message, rank0_only=rank0_only)
