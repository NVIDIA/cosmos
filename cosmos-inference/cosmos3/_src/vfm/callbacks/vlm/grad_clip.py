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

from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import wandb

from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.callback import Callback
from cosmos3._src.vfm.utils.vlm.optimizer import OptimizersContainer
from projects.cosmos3.vlm.utils.distributed import gradient_norm_clipping


@torch.jit.script
def _fused_nan_to_num(params: List[torch.Tensor]):
    # Note this will raise error:
    #     for param in params:
    #         torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0, out=param)                                                                                                                                                                         ~~~~~~~~~~~~~~~~ <--- HERE
    # RuntimeError: nan_to_num(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.
    for param in params:
        torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0, out=param)


@dataclass
class _MagnitudeRecord:
    state: float = 0
    iter_count: int = 0

    def reset(self) -> None:
        self.state = 0
        self.iter_count = 0

    def update(self, cur_state: torch.Tensor) -> None:
        self.state += cur_state
        self.iter_count += 1

    def get_stat(self) -> Tuple[float, float]:
        if self.iter_count > 0:
            avg_state = self.state / self.iter_count
            avg_state = avg_state.item()
        else:
            avg_state = 0
        self.reset()
        return avg_state


class GradClip(Callback):
    def __init__(self, clip_norm=1.0, force_finite: bool = False):
        self.clip_norm = clip_norm
        self.force_finite = force_finite
        # Single global-norm tracker. Earlier versions partitioned by DeviceMesh
        # and clipped each bucket independently — that changed the relative
        # rescale between dense and EP-split MoE params, distorting the update
        # direction. gradient_norm_clipping() now combines per-mesh norms into
        # one scalar and applies a uniform rescale across every bucket.
        self._global_norm_log = _MagnitudeRecord()

    def on_before_optimizer_step(
        self,
        model_ddp: Union[torch.nn.Module, list[torch.nn.Module]],
        optimizer: OptimizersContainer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del scheduler
        model_parts = model_ddp
        if not isinstance(model_parts, list):
            model_parts = [model_parts]

        all_params: List[torch.Tensor] = []
        for model_part in model_parts:
            for param in model_part.parameters():
                if param.grad is not None:
                    all_params.append(param)

        if self.force_finite:
            _fused_nan_to_num([p.grad for p in all_params])

        total_norm = gradient_norm_clipping(
            all_params,
            self.clip_norm,
            foreach=True,
        )
        self._global_norm_log.update(total_norm)

        if iteration % self.config.trainer.logging_iter == 0 and wandb.run:
            avg_norm = self._global_norm_log.get_stat()
            log.info(f"clip_grad_norm/global: {avg_norm}")
            wandb.log({"iteration": iteration, "clip_grad_norm/global": avg_norm}, step=iteration)

        if (iteration == 1 or iteration % (self.config.trainer.logging_iter * 10) == 0) and wandb.run:
            # Log learning rate
            unique_lr = {}
            for optim_per_model, model_part_name in zip(optimizer.optimizers, optimizer.model_part_names):
                for param_group in optim_per_model[0].param_groups:
                    unique_lr[f"optim/lr_{model_part_name}"] = param_group["lr"]

            log.info(f"learning_rate: {unique_lr}")
            wandb.log(unique_lr, step=iteration)
