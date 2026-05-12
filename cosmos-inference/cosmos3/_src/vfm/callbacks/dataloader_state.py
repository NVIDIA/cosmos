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

import os
from dataclasses import dataclass
from typing import Any

import torch

from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.callback import Callback


@dataclass
class NoReplaceShardlistState:
    epoch: int = 0
    index: int = 0


class DataLoaderStateCallback(Callback):
    checkpoint_component: str = "dataloader"

    def __init__(
        self,
        distributor_type: str | None = None,
    ) -> None:
        super().__init__()
        self.distributor_type = distributor_type
        self.config: Any = None
        self.state: dict[int, NoReplaceShardlistState] = {}
        self.verbose = True

    def _update_state_from_batch(self, data_batch: dict[str, torch.Tensor]) -> None:
        worker_ids = data_batch["sample_worker_id"].tolist()  # [B]
        epochs = data_batch["sample_epoch"].tolist()  # [B]
        indices = data_batch["sample_index"].tolist()  # [B]
        for worker_id, epoch, index in zip(worker_ids, epochs, indices, strict=True):
            if worker_id not in self.state:
                self.state[worker_id] = NoReplaceShardlistState(epoch=epoch, index=index)

            elif self.state[worker_id].epoch < epoch or (
                self.state[worker_id].index < index and self.state[worker_id].epoch == epoch
            ):
                self.state[worker_id] = NoReplaceShardlistState(epoch=epoch, index=index)

    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.distributor_type == "no_replace":
            self._update_state_from_batch(data_batch)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.distributor_type == "no_replace":
            if self.verbose:
                if iteration % self.config.trainer.logging_iter == 0:
                    msg = "\n"
                    for wid, state in self.state.items():
                        msg += f"worker {wid}: epoch={state.epoch}, index={state.index}\n"
                    log.info(msg)

    def has_checkpoint_state(self) -> bool:
        return self.distributor_type == "no_replace"

    def state_dict(self) -> dict[int, dict[str, int]]:
        if self.distributor_type != "no_replace":
            return {}

        state_dict: dict[int, dict[str, int]] = {}
        for worker_id, per_worker_state in self.state.items():
            state_dict[worker_id] = {"epoch": per_worker_state.epoch, "index": per_worker_state.index}
            log.info(
                f"Saved dataloader state for worker {worker_id}: "
                f"epoch={per_worker_state.epoch}, index={per_worker_state.index}"
            )
        return state_dict

    def load_state_dict(self, state_dict: dict[int, dict[str, int]]) -> None:
        if self.distributor_type != "no_replace":
            return

        if not state_dict:
            log.info("No dataloader state found in checkpoint")
            return

        self.state = {}
        for worker_id, per_worker_state in state_dict.items():
            epoch = per_worker_state["epoch"]
            index = per_worker_state["index"]
            self.state[worker_id] = NoReplaceShardlistState(epoch=epoch, index=index)
            os.environ[f"NSL_STATE_WORKER_{worker_id}_EPOCH"] = str(epoch)
            os.environ[f"NSL_STATE_WORKER_{worker_id}_INDEX"] = str(index)
            log.info(f"Loaded no replace dataloader state for worker {worker_id}: epoch={epoch}, index={index}")

    def on_save_checkpoint(self, model: ImaginaireModel, state_dict: dict[str, Any]) -> None:
        if self.distributor_type == "no_replace" and "dataloader" not in state_dict:
            state_dict["dataloader"] = self.state_dict()

    def on_load_checkpoint(self, model: ImaginaireModel, state_dict: dict[str, Any]) -> None:
        if "dataloader" in state_dict:
            self.load_state_dict(state_dict["dataloader"])
