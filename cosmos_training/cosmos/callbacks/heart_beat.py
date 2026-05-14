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

import time
from datetime import datetime

import pytz
import torch

from cosmos.callbacks.every_n import EveryN
from cosmos.model._base import ImaginaireModel
from cosmos.trainer import ImaginaireTrainer
from cosmos.utils import distributed
from cosmos.utils.easy_io import easy_io


class HeartBeat(EveryN):
    """
    A callback that logs a heartbeat message at regular intervals to indicate that the training process is still running.

    Args:
        every_n (int): The frequency at which the callback is invoked.
        step_size (int, optional): The step size for the callback. Defaults to 1.
        update_interval_in_minute (int, optional): The interval in minutes for logging the heartbeat. Defaults to 20 minutes.
        save_s3 (bool, optional): Whether to save the heartbeat information to S3. Defaults to False.
    """

    def __init__(self, every_n: int, step_size: int = 1, update_interval_in_minute: int = 20, save_s3: bool = False):
        super().__init__(every_n=every_n, step_size=step_size)
        self.name = self.__class__.__name__
        self.update_interval_in_minute = update_interval_in_minute
        self.save_s3 = save_s3
        self.pst = pytz.timezone("America/Los_Angeles")
        self.is_hitted = False

    @distributed.rank0_only
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        self.time = time.time()
        if self.save_s3:
            current_time_pst = datetime.now(self.pst).strftime("%Y_%m_%d-%H_%M_%S")
            info = {
                "iteration": iteration,
                "time": current_time_pst,
            }
            easy_io.dump(info, f"s3://rundir/{self.name}_start.yaml")
            easy_io.dump(info, f"s3://timestamps_rundir/{self.name}_start.yaml")

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if not self.is_hitted:
            self.is_hitted = True
            if distributed.get_rank() == 0:
                self.report(iteration)
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    @distributed.rank0_only
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        if time.time() - self.time > 60 * self.update_interval_in_minute:
            self.report(iteration)

    def report(self, iteration: int = 0):
        self.time = time.time()
        if self.save_s3:
            current_time_pst = datetime.now(self.pst).strftime("%Y_%m_%d-%H_%M_%S")
            info = {
                "iteration": iteration,
                "time": current_time_pst,
            }
            easy_io.dump(info, f"s3://rundir/{self.name}.yaml")

    @distributed.rank0_only
    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if self.save_s3:
            current_time_pst = datetime.now(self.pst).strftime("%Y_%m_%d-%H_%M_%S")
            info = {
                "iteration": iteration,
                "time": current_time_pst,
            }
            easy_io.dump(info, f"s3://rundir/{self.name}_end.yaml")
            easy_io.dump(info, f"s3://timestamps_rundir/{self.name}_end.yaml")
