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

from contextlib import contextmanager
from typing import Generator

from cosmos.utils.log import logger
from cosmos.utils.one_logger.one_logger_utils import get_one_logger, one_logger_is_initialized


@contextmanager
def data_loader_init() -> Generator[None, None, None]:
    """
    Wrap the execution data loader initialization by invoking the one logger callbacks.
    """
    try:
        one_logger = get_one_logger()
        if one_logger_is_initialized():
            one_logger.on_dataloader_init_start()

        yield

    finally:
        try:
            if one_logger_is_initialized():
                one_logger.on_dataloader_init_end()
        except Exception as exc:  # noqa: BLE001
            logger.warning("one_logger.on_dataloader_init_end() failed (non-fatal): %s", exc)


@contextmanager
def model_init(set_barrier: bool = False) -> Generator[None, None, None]:
    """
    Wrap the instantiation of the model by invoking the one logger callbacks.
    """
    try:
        one_logger = get_one_logger()
        if one_logger_is_initialized():
            one_logger.on_model_init_start(set_barrier=set_barrier)

        yield

    finally:
        try:
            if one_logger_is_initialized():
                one_logger.on_model_init_end(set_barrier=set_barrier)
        except Exception as exc:  # noqa: BLE001
            logger.warning("one_logger.on_model_init_end() failed (non-fatal): %s", exc)
