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

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist
import torch.utils.data
import wandb

from cosmos.model._base import ImaginaireModel
from cosmos.utils import distributed, log
from cosmos.utils.callback import Callback
from cosmos.utils.easy_io import easy_io


@dataclass
class _LossRecord:
    loss: float = 0
    iter_count: int = 0
    name: str = None

    def reset(self) -> None:
        self.loss = 0
        self.iter_count = 0

    def get_stat(self, return_valid_mask_sum: bool = False) -> Tuple[float, float]:
        if self.iter_count == 0:
            self.loss = torch.tensor([float("nan")], device="cuda")  # [1]
            self.iter_count = 1
        msg_str = f"{self.name}: sum_loss={self.loss.item()}/iter_count={self.iter_count}="
        avg_loss_tensor = self.loss / self.iter_count
        # Create a mask (1 if valid, 0 if NaN or Inf)
        valid_mask = torch.tensor([torch.isfinite(avg_loss_tensor).float()], device="cuda")  # [1]
        msg_str += f"avg_loss={avg_loss_tensor.item()}, valid_mask={valid_mask.item()}, "

        # Replace NaN/Inf with 0 to avoid affecting sum
        avg_loss_tensor = torch.where(
            torch.isfinite(avg_loss_tensor),
            avg_loss_tensor,
            torch.tensor([0.0], device="cuda"),  # [1]
        )

        # Reduce across all ranks
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)  # Sum of valid losses
        dist.all_reduce(valid_mask, op=dist.ReduceOp.SUM)  # Count of valid losses
        msg_str += f" | all_reduce: avg_loss={avg_loss_tensor.item()}, valid_mask={valid_mask.item()}"
        # Compute final average, avoiding division by zero
        if valid_mask.item() > 0:
            final_avg_loss = (avg_loss_tensor / valid_mask).item()
            valid_mask_sum = valid_mask.item()
        else:
            final_avg_loss = 0.0  # Default to zero if all values were invalid
            valid_mask_sum = 0

        avg_loss = final_avg_loss
        msg_str += f" | final: avg_loss={final_avg_loss}"
        if self.name is not None:
            log.debug(msg_str, rank0_only=False)
        self.reset()
        if return_valid_mask_sum:
            return avg_loss, valid_mask_sum
        else:
            return avg_loss


class WandbCallback(Callback):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.final_loss_log = _LossRecord()
        self.final_all_loss_log = {}
        self.logging_iter_multipler = logging_iter_multipler
        self.save_logging_iter_multipler = save_logging_iter_multipler
        assert self.logging_iter_multipler > 0, "logging_iter_multipler should be greater than 0"
        self.save_s3 = save_s3
        self.wandb_extra_tag = f"@{logging_iter_multipler}" if logging_iter_multipler > 1 else ""
        self.name = "wandb_loss_log" + self.wandb_extra_tag
        self.unstable_count = torch.zeros(1, device="cuda")  # [1]

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if torch.isnan(loss) or torch.isinf(loss):
            log.critical(
                f"Unstable loss {loss} at iteration {iteration}",
                rank0_only=False,
            )
            self.unstable_count += 1

        self.final_loss_log.loss += loss.detach().float()
        self.final_loss_log.iter_count += 1

        for key in output_batch.keys():
            if "loss" in key:
                if key not in self.final_all_loss_log:
                    self.final_all_loss_log[key] = _LossRecord()
                self.final_all_loss_log[key].loss += output_batch[key].detach().float()
                self.final_all_loss_log[key].iter_count += 1

        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0:
            avg_final_loss = self.final_loss_log.get_stat()

            avg_final_all_loss = {}
            for key in self.final_all_loss_log.keys():
                avg_final_all_loss[key] = self.final_all_loss_log[key].get_stat()

            dist.all_reduce(self.unstable_count, op=dist.ReduceOp.SUM)

            if distributed.is_rank0() and wandb.run is not None:
                info = {}
                info.update(
                    {
                        f"train{self.wandb_extra_tag}/loss": avg_final_loss,
                        f"train{self.wandb_extra_tag}/unstable_count": self.unstable_count.item(),
                        "iteration": iteration,
                    }
                )
                for key, loss in avg_final_all_loss.items():
                    info.update(
                        {
                            f"train{self.wandb_extra_tag}_detail/{key}": loss,
                        }
                    )
                if self.save_s3:
                    if (
                        iteration
                        % (
                            self.config.trainer.logging_iter
                            * self.logging_iter_multipler
                            * self.save_logging_iter_multipler
                        )
                        == 0
                    ):
                        easy_io.dump(
                            info,
                            f"s3://rundir/{self.name}/Train_Iter{iteration:09d}.json",
                        )

                if wandb:
                    wandb.log(info, step=iteration)
            # reset unstable count
            self.unstable_count.zero_()
