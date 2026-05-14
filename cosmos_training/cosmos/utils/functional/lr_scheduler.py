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

from typing import Optional

import numpy as np

from cosmos.utils import distributed, log


class TeroPolyScheduler:
    def __init__(
        self,
        total_Mimg: int,
        batch_size: int,
        ref_Mimg: Optional[int] = None,
        ref_batches: float = 70e3 / 1024,
        max_lr_ratio: Optional[float] = 1.0,
        min_lr_ratio: Optional[float] = None,
        rampup_Mimg: float = 0,
        rampdown_Mimg: int = 0,
        verbosity_interval: int = 0,
        formula: str = "poly",
        poly_exp: float = 0.5,
    ):
        self.total_Mimg = total_Mimg
        self.batch_size = batch_size * distributed.get_world_size()
        self.ref_Mimg = ref_Mimg or ref_batches * batch_size / 1e6
        self.ref_batches = ref_batches
        self.max_lr_ratio = max_lr_ratio
        self.min_lr_ratio = min_lr_ratio
        self.rampup_Mimg = rampup_Mimg
        self.rampdown_Mimg = rampdown_Mimg
        self.verbosity_interval = verbosity_interval
        self.formula = formula
        self.poly_exp = poly_exp

        self._model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def schedule(self, n, **kwargs):
        cur_Mimg = getattr(self.model, "sample_counter", 0) / 1e6

        if self.formula == "constant":
            lr = 1.0
        elif self.formula == "poly":
            lr = max(cur_Mimg / self.ref_Mimg, 1e-8) ** -self.poly_exp
        else:
            raise ValueError(f'Invalid learning rate formula "{self.formula}"')

        if self.max_lr_ratio is not None:
            lr = min(lr, self.max_lr_ratio)
        if self.min_lr_ratio is not None:
            lr = max(lr, self.min_lr_ratio)

        if self.rampup_Mimg > 0 and cur_Mimg < self.rampup_Mimg:
            lr *= cur_Mimg / self.rampup_Mimg
        if self.rampdown_Mimg > 0 and cur_Mimg > self.total_Mimg - self.rampdown_Mimg:
            lr *= (self.total_Mimg - cur_Mimg) / self.rampdown_Mimg

        return lr

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaWarmUpCosineScheduler:
    """
    A learning rate scheduler that combines warm-up with a cosine decay schedule for multiple cycles.
    It supports different configurations for each cycle, including the number of warm-up steps, minimum
    and maximum scaling factors for the learning rate.

    The scheduler is intended to be used with a base learning rate of 1.0, where the actual learning
    rate at any step is the base learning rate multiplied by the scaling factor computed by the scheduler.

    Parameters:
        warm_up_steps (list[int]): List of integers where each element represents the number of warm-up
                                   steps for the corresponding cycle.
        f_min (list[float]): List of the minimum scaling factors for each cycle after warm-up.
        f_max (list[float]): List of the maximum scaling factors at the start and end of each cosine cycle.
        f_start (list[float]): List of starting scaling factors for each warm-up phase.
        cycle_lengths (list[int]): List of the total lengths of each cycle, including warm-up steps.
        verbosity_interval (int, optional): Interval of training steps at which to print current step and
                                            scaling factor information. Set to 0 by default to disable verbosity.

    Examples:
        >>> scheduler = LambdaWarmUpCosineScheduler2(
                warm_up_steps=[10, 10],
                f_min=[0.1, 0.1],
                f_max=[1.0, 1.0],
                f_start=[0.01, 0.01],
                cycle_lengths=[50, 50],
                verbosity_interval=10)
        >>> for step in range(100):
        >>>     lr_multiplier = scheduler(step)
        >>>     print(f"Step {step}: LR Multiplier = {lr_multiplier}")
    """

    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.0
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                log.info(f"current step: {n}, recent lr-multiplier: {self.last_f}, current cycle {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler):
    """
    Linear instead of cosine decay for the main part of the cycle.
    """

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                log.info(f"current step: {n}, recent lr-multiplier: {self.last_f}, current cycle {cycle}")

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / (
                self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle]
            )
            self.last_f = f
            return f
