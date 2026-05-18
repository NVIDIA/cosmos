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

import torch
import wandb

from cosmos.utils.callback import Callback


class LearningRateLogger(Callback):
    """Logs per-model-part learning rate every ``every_n × logging_iter`` steps.

    Designed for VLM training where the optimizer is an
    ``OptimizersContainer`` exposing ``.optimizers`` (list of single-element
    optimizer lists) paired with ``.model_part_names``. Silently no-ops when
    those attributes are absent so it can be registered alongside plain
    ``torch.optim.Optimizer`` setups without harm.
    """

    def __init__(self, every_n: int = 10):
        self.every_n = every_n

    def on_before_optimizer_step(
        self,
        model_ddp: torch.nn.Module | list[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del model_ddp, scheduler, grad_scaler
        gate = self.config.trainer.logging_iter * self.every_n
        if not (iteration == 1 or (gate > 0 and iteration % gate == 0)):
            return
        if not wandb.run:
            return
        if not (hasattr(optimizer, "optimizers") and hasattr(optimizer, "model_part_names")):
            return
        unique_lr: dict[str, float] = {}
        for optim_per_model, name in zip(optimizer.optimizers, optimizer.model_part_names):
            if not optim_per_model:
                continue
            for pg in optim_per_model[0].param_groups:
                unique_lr[f"optim/lr_{name}"] = pg["lr"]
        if not unique_lr:
            return
        wandb.log(unique_lr, step=iteration)
