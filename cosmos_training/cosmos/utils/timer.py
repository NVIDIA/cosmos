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
Timer: helps measure CPU and CUDA times easily and reliably.
"""

import time
from contextlib import ContextDecorator
from contextvars import ContextVar
from functools import wraps
from typing import Callable

import torch

from cosmos.utils import log

_timer_active = ContextVar("_timer_active", default=False)


def in_timer_region() -> bool:
    return _timer_active.get()


def _autoformat_time_us(time_us: float) -> str:
    """
    Automatically format time in nanoseconds.
    """
    if time_us >= 1e6:
        time_s = time_us * 1e-6
        return f"{time_s:.2f} s"

    if time_us >= 1e3:
        time_ms = time_us * 1e-3
        return f"{time_ms:.2f} ms"

    return f"{time_us:.2f} us"


def format_time_str(time_us: float, unit: str | None = None) -> str:
    """
    Automatically format time in nanoseconds either automatically or based on
    desired unit.
    """
    if unit is None:
        return _autoformat_time_us(time_us)

    if unit == "us":
        return f"{time_us:.2f} us"

    if unit == "ms":
        return f"{time_us * 1e-3:.2f} ms"

    if unit == "s":
        return f"{time_us * 1e-6:.2f} s"

    raise NotImplementedError(f"Time unit {unit} is not supported.")


def format_time(time_us: float, unit: str) -> float:
    """
    Format time in nanoseconds based on desired unit.
    """

    if unit == "us":
        return time_us

    if unit == "ms":
        return time_us * 1e-3

    if unit == "s":
        return time_us * 1e-6

    raise NotImplementedError(f"Time unit {unit} is not supported.")


class Timer(ContextDecorator):
    """
    Reliable CPU and CUDA Timer.

    Args:
        tag (str | None): Optional tag used in logs/prints.

        measure_cpu (bool): Whether to measure CPU time (using `time`). Default: `True`.

        measure_cuda (bool): Whether to measure CUDA time (using CUDA events). Default: `True`.

        unit (str | None): Optional time unit. Must be either "s" (seconds), "ms" (microseconds),
            "us" (nanoseconds), or None (format automatically based on value).

        debug (bool): Whether to log results in debug mode instead of info. Default is False.

    Examples:
        ```python
        with Timer(measure_cpu=True, measure_cuda=True, unit="ms"):
            model(x)
        ```

        ```python
        @Timer(measure_cpu=True, measure_cuda=True, unit="ms")
        def func(x):
            return model(x)
        ```
    """

    def __init__(
        self,
        tag: str | None = None,
        measure_cpu: bool = True,
        measure_cuda: bool = True,
        unit: str | None = None,
        debug: bool = False,
    ):
        self.measure_cpu = measure_cpu
        self.measure_cuda = measure_cuda

        self.measured = False
        self.cpu_time_us = 0
        self.cuda_time_us = 0

        self.busy = False
        self.cpu_time_start = None
        self.cuda_start_event = None
        self.cuda_end_event = None
        self.cuda_stream = None

        self.tag = "unknown" if tag is None else tag
        self.unit = unit
        if self.unit is not None and self.unit not in ["s", "ms", "us"]:
            raise NotImplementedError(f"Time unit {self.unit} is not supported.")

        self.debug = debug

    def _log(self, msg: str):
        if self.debug:
            log.debug(msg)
        else:
            log.info(msg)

    def __enter__(self):
        self.token = _timer_active.set(True)
        self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.end()
        self.report()
        _timer_active.reset(self.token)

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN202
            self.start()
            result = func(*args, **kwargs)
            self.end()
            self.report()
            return result

        return wrapper  # type: ignore

    def report(self):
        """
        Reports measurements.
        """
        if self.measure_cpu and self.measure_cuda:
            self._log(f"Time spent on {self.tag}: CPU: {self.get_cpu_time_str()}, CUDA: {self.get_cuda_time_str()}")
        elif self.measure_cpu:
            self._log(f"Time spent on {self.tag}: {self.get_cpu_time_str()}")
        elif self.measure_cuda:
            self._log(f"CUDA time spent on {self.tag}: {self.get_cuda_time_str()}")
        else:
            raise NotImplementedError()

    def get_cpu_time(self) -> float:
        """
        Returns CPU time measurement.
        """
        if not self.measure_cpu:
            raise RuntimeError(f"CPU timer is disabled ({self.measure_cpu=}).")

        if not self.measured:
            raise RuntimeError("No measurements were made yet!")

        if self.unit is None:
            raise RuntimeError("No unit was specified. Please use get_cpu_time_str() instead.")

        assert self.unit is not None
        return format_time(self.cpu_time_us, unit=self.unit)

    def get_cuda_time(self) -> float:
        """
        Returns CUDA time measurement.
        """
        if not self.measure_cuda:
            raise RuntimeError(f"CUDA timer is disabled ({self.measure_cuda=}).")

        if not self.measured:
            raise RuntimeError("No measurements were made yet!")

        if self.unit is None:
            raise RuntimeError("No unit was specified. Please use get_cuda_time_str() instead.")

        assert self.unit is not None
        return format_time(self.cuda_time_us, unit=self.unit)

    def get_cpu_time_str(self) -> str:
        """
        Returns CPU time measurement in string format.
        """
        if not self.measure_cpu:
            raise RuntimeError(f"CPU timer is disabled ({self.measure_cpu=}).")

        if not self.measured:
            raise RuntimeError("No measurements were made yet!")

        return format_time_str(self.cpu_time_us, unit=self.unit)

    def get_cuda_time_str(self) -> str:
        """
        Returns CUDA time measurement in string format.
        """
        if not self.measure_cuda:
            raise RuntimeError(f"CUDA timer is disabled ({self.measure_cuda=}).")

        if not self.measured:
            raise RuntimeError("No measurements were made yet!")

        return format_time_str(self.cuda_time_us, unit=self.unit)

    def reset(self):
        """
        Resets recorded measurements
        """
        self.measured = False
        self.cpu_time_us = 0
        self.cuda_time_us = 0

    def start(self, cuda_device: torch.device | None = None, cuda_stream: torch.cuda.Stream | None = None):
        """
        Start time measurements.

        Args:
            cuda_device (torch.device | None): CUDA device. Will use default CUDA device if not indicated.

            cuda_stream (torch.cuda.Stream | None): CUDA stream to use for CUDA time measurement.
                Will use default stream for current CUDA device if not indicated.
        """
        if self.busy:
            raise RuntimeError("Already called Timer.start() once!")

        self.busy = True

        if self.measure_cuda:
            self.cuda_stream = cuda_stream if cuda_stream is not None else torch.cuda.current_stream(cuda_device)
            self.cuda_stream.synchronize()

        if self.measure_cpu:
            self.cpu_time_start = time.time()

        if self.measure_cuda:
            self.cuda_start_event = torch.cuda.Event(enable_timing=True)
            self.cuda_end_event = torch.cuda.Event(enable_timing=True)
            self.cuda_stream.record_event(self.cuda_start_event)

    def end(self):
        """
        Ends time measurements.

        NOTE: must be done on the same CUDA device and stream as start().
        """
        if not self.busy:
            raise RuntimeError("Timer.start() must be called exactly once before end()!")

        if self.measure_cuda:
            self.cuda_stream.record_event(self.cuda_end_event)
            self.cuda_end_event.synchronize()

        if self.measure_cpu:
            self.cpu_time_end = time.time()
            self.cpu_time_us = (self.cpu_time_end - self.cpu_time_start) * 1e6

        if self.measure_cuda:
            self.cuda_time_us = self.cuda_start_event.elapsed_time(self.cuda_end_event) * 1e3

        self.busy = False
        self.measured = True
