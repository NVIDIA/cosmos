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

from abc import abstractmethod
from typing import Optional

import torch

from cosmos.model._base import ImaginaireModel
from cosmos.trainer import ImaginaireTrainer
from cosmos.utils import distributed, log
from cosmos.utils.callback import Callback


class EveryN(Callback):
    def __init__(
        self,
        every_n: Optional[int] = None,
        step_size: int = 1,
        barrier_after_run: bool = True,
        run_at_start: bool = False,
    ) -> None:
        """Constructor for `EveryN`.

        Args:
            every_n (int): Frequency with which callback is run during training.
            step_size (int): Size of iteration step count. Default 1.
            barrier_after_run (bool): Whether to have a distributed barrier after each execution. Default True, to avoid timeouts.
            run_at_start (bool): Whether to run at the beginning of training. Default False.
        """
        self.every_n = every_n
        if self.every_n == 0:
            log.warning(
                f"every_n is set to 0. Callback {self.__class__.__name__} will be invoked only once in the beginning of the training. Calls happens on_training_step_end will be skipped."
            )

        self.step_size = step_size
        self.barrier_after_run = barrier_after_run
        self.run_at_start = run_at_start

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # every_n = 0 is a special case which means every_n_impl will be called only once in the beginning of the training
        if self.every_n != 0:
            trainer = self.trainer
            global_step = iteration // self.step_size
            should_run = (iteration == 1 and self.run_at_start) or (
                global_step % self.every_n == 0
            )  # (self.every_n - 1)
            if should_run:
                log.debug(f"Callback {self.__class__.__name__} fired on train_batch_end step {global_step}")
                self.every_n_impl(trainer, model, data_batch, output_batch, loss, iteration)
                log.debug(f"Callback {self.__class__.__name__} finished on train_batch_end step {global_step}")
                # add necessary barrier to avoid timeout
                if self.barrier_after_run:
                    distributed.barrier()

    @abstractmethod
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None: ...
