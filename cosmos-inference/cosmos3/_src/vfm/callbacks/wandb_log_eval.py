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

from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.utils import distributed, log
from cosmos3._src.imaginaire.utils.callback import Callback
from cosmos3._src.imaginaire.utils.easy_io import easy_io


@dataclass
class _LossRecord:
    loss: float = 0
    iter_count: int = 0

    def reset(self) -> None:
        self.loss = 0
        self.iter_count = 0

    def get_stat(self) -> Tuple[float, float]:
        if self.iter_count > 0:
            avg_loss_tensor = self.loss / self.iter_count
            # Create a mask (1 if valid, 0 if NaN or Inf)
            valid_mask = torch.tensor([torch.isfinite(avg_loss_tensor).float()], device="cuda")

            # Replace NaN/Inf with 0 to avoid affecting sum
            avg_loss_tensor = torch.where(
                torch.isfinite(avg_loss_tensor), avg_loss_tensor, torch.tensor([0.0], device="cuda")
            )

            # Reduce across all ranks
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)  # Sum of valid losses
            dist.all_reduce(valid_mask, op=dist.ReduceOp.SUM)  # Count of valid losses

            # Compute final average, avoiding division by zero
            if valid_mask.item() > 0:
                final_avg_loss = (avg_loss_tensor / valid_mask).item()
            else:
                final_avg_loss = 0.0  # Default to zero if all values were invalid

            avg_loss = final_avg_loss
        else:
            avg_loss = 0
        self.reset()
        return avg_loss


class WandbCallback(Callback):
    def __init__(
        self,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.final_loss_log = _LossRecord()
        self.final_loss_log_per_dataset = {}

        self.save_s3 = save_s3
        self.wandb_extra_tag = ""
        self.name = "wandb_loss_val_log"
        self.unstable_count = torch.zeros(1, device="cuda")
        self.url_key_list = []

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if torch.isnan(loss) or torch.isinf(loss):
            log.critical(
                f"Unstable val loss {loss} at iteration {iteration}",
                rank0_only=False,
            )
            self.unstable_count += 1

        dataset_name = data_batch.get("dataset_name", "default")

        # Handle case where dataset_name gets batched into a list
        if isinstance(dataset_name, list):

            assert len(dataset_name) == 1, "dataset_name should be a list of 1"
            dataset_name = dataset_name[0]

        if dataset_name not in self.final_loss_log_per_dataset:
            self.final_loss_log_per_dataset[dataset_name] = _LossRecord()

        self.final_loss_log_per_dataset[dataset_name].loss += loss.detach().float()
        self.final_loss_log_per_dataset[dataset_name].iter_count += 1
        self.final_loss_log.loss += loss.detach().float()
        self.final_loss_log.iter_count += 1

        self.url_key_list.append(f"{data_batch.get('__url__', [''])[0]}, {data_batch.get('__key__', [''])[0]}")

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        avg_final_loss = self.final_loss_log.get_stat()

        log.info(f"avg_final_loss: {avg_final_loss}")
        dist.all_reduce(self.unstable_count, op=dist.ReduceOp.SUM)
        # gather url and key list from all ranks
        url_key_list = [None] * dist.get_world_size()
        dist.all_gather_object(url_key_list, self.url_key_list)
        url_key_list = [item for sublist in url_key_list for item in sublist]

        unique_url_key_list = list(set(url_key_list))
        if distributed.is_rank0():
            info = {}
            log.info(
                f"[val] number of unique url and key: {len(unique_url_key_list)} / {len(url_key_list)}; avg_final_loss: {avg_final_loss}"
            )
            info.update(
                {
                    f"val{self.wandb_extra_tag}/loss": avg_final_loss,
                    f"val{self.wandb_extra_tag}/unstable_count": self.unstable_count.item(),
                    "iteration": iteration,
                    f"val{self.wandb_extra_tag}/num_unique_url_key": len(unique_url_key_list),
                    f"val{self.wandb_extra_tag}/total_url_key": len(url_key_list),
                }
            )
            if self.save_s3:
                easy_io.dump(
                    info,
                    f"s3://rundir/{self.name}/Val_Iter{iteration:09d}.json",
                )

            if wandb.run is not None:
                wandb.log(info, step=iteration)

        # reset unstable count
        self.unstable_count.zero_()
        self.url_key_list = []
