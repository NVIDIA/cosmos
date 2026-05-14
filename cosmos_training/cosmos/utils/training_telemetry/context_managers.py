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
from typing import Callable, Generator

from cosmos.utils.training_telemetry.utils import get_telemetry_recorder, import_training_telemetry


@contextmanager
def get_context(create_span_name: Callable) -> Generator[None, None, None]:
    """
    Wrap the execution of some code in a telemetry context.
    """
    try:
        recorder = get_telemetry_recorder()
        if recorder:
            training_telemetry = import_training_telemetry()
            span_name = create_span_name(training_telemetry)
            span = recorder.start(span_name)

        yield

    finally:
        if recorder:
            recorder.stop(span)


@contextmanager
def data_loader_init() -> Generator[None, None, None]:
    with get_context(lambda training_telemetry: training_telemetry.SpanName.DATA_LOADER_INIT):
        yield


@contextmanager
def model_init() -> Generator[None, None, None]:
    with get_context(lambda training_telemetry: training_telemetry.SpanName.MODEL_INIT):
        yield


@contextmanager
def distributed_init() -> Generator[None, None, None]:
    with get_context(lambda training_telemetry: training_telemetry.SpanName.DIST_INIT):
        yield
