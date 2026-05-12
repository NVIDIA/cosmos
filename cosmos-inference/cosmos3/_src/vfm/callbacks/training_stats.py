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

import torch
import torch.distributed as dist
import wandb

from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.utils import distributed
from cosmos3._src.imaginaire.utils.callback import Callback
from cosmos3._src.vfm.callbacks.wandb_log import _LossRecord
from cosmos3._src.vfm.datasets.action.domain_utils import EMBODIMENT_TO_DOMAIN_ID

# Build inverse mapping: domain_id -> embodiment_type. First occurrence wins when multiple embodiment names share the
# same domain id.
DOMAIN_ID_TO_EMBODIMENT: dict[int, str] = {}
for _k, _v in EMBODIMENT_TO_DOMAIN_ID.items():
    DOMAIN_ID_TO_EMBODIMENT.setdefault(_v, _k)


class TrainingStatsCallback(Callback):
    """Callback for tracking and logging training mode and embodiment statistics to wandb."""

    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.log_freq = log_freq
        self._mode_counts: dict[str, int] = {}
        self._mode_total_count: int = 0
        self._embodiment_counts: dict[str, int] = {}
        self._embodiment_total_count: int = 0
        self._per_embodiment_loss: dict[str, _LossRecord] = {}
        self._per_embodiment_sub_loss: dict[str, dict[str, _LossRecord]] = {}

    def _accumulate_mode_counts(self, data_batch: dict[str, torch.Tensor]) -> None:
        modes = data_batch.get("mode", None)
        if modes is None:
            return

        if isinstance(modes, str):
            modes_list = [modes]
        elif isinstance(modes, (list, tuple)):
            modes_list = [str(m) for m in modes]
        elif isinstance(modes, torch.Tensor):
            # Defensive: support cases where mode might be encoded numerically.
            modes_list = [str(m) for m in modes.detach().cpu().tolist()]
        else:
            modes_list = [str(modes)]

        for mode in modes_list:
            self._mode_total_count += 1
            self._mode_counts[mode] = self._mode_counts.get(mode, 0) + 1

    def _accumulate_embodiment_counts(self, data_batch: dict[str, torch.Tensor]) -> None:
        domain_ids = data_batch.get("domain_id", None)
        if domain_ids is None:
            return

        if isinstance(domain_ids, int):
            domain_id_list = [domain_ids]
        elif isinstance(domain_ids, (list, tuple)):
            domain_id_list = [int(d) for d in domain_ids if d is not None]
        elif isinstance(domain_ids, torch.Tensor):
            # Flatten to handle any shape (scalar, 1D, or 2D with trailing dim)
            domain_id_list = [int(d) for d in domain_ids.detach().cpu().flatten().tolist()]
        else:
            domain_id_list = [int(domain_ids)]

        for domain_id in domain_id_list:
            embodiment = DOMAIN_ID_TO_EMBODIMENT.get(domain_id, f"unknown_{domain_id}")
            self._embodiment_total_count += 1
            self._embodiment_counts[embodiment] = self._embodiment_counts.get(embodiment, 0) + 1

    def _gather_global_mode_counts(self) -> tuple[int, dict[str, int]]:
        """
        Returns (global_total, global_mode_counts) aggregated across all ranks.
        """
        local: dict[str, int] = dict(self._mode_counts)
        local["__total__"] = int(self._mode_total_count)

        if dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())
            gathered: list[dict[str, int] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]

        global_total = 0
        global_counts: dict[str, int] = {}
        for item in gathered:
            if not item:
                continue
            global_total += int(item.get("__total__", 0))
            for k, v in item.items():
                if k == "__total__":
                    continue
                global_counts[k] = global_counts.get(k, 0) + int(v)
        return global_total, global_counts

    def _gather_global_embodiment_counts(self) -> tuple[int, dict[str, int]]:
        """
        Returns (global_total, global_embodiment_counts) aggregated across all ranks.
        """
        local: dict[str, int] = dict(self._embodiment_counts)
        local["__total__"] = int(self._embodiment_total_count)

        if dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())
            gathered: list[dict[str, int] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]

        global_total = 0
        global_counts: dict[str, int] = {}
        for item in gathered:
            if not item:
                continue
            global_total += int(item.get("__total__", 0))
            for k, v in item.items():
                if k == "__total__":
                    continue
                global_counts[k] = global_counts.get(k, 0) + int(v)
        return global_total, global_counts

    def _build_mode_log_dict(
        self, *, log_prefix: str, global_total: int, global_counts: dict[str, int]
    ) -> dict[str, float]:
        info: dict[str, float] = {}

        denom = float(global_total) if global_total > 0 else 0.0
        for mode in sorted(global_counts.keys()):
            count = float(global_counts.get(mode, 0))
            pct = (100.0 * count / denom) if denom > 0 else 0.0
            info[f"{log_prefix}_stats_mode/{mode}"] = pct

        return info

    def _build_embodiment_log_dict(
        self, *, log_prefix: str, global_total: int, global_counts: dict[str, int]
    ) -> dict[str, float]:
        info: dict[str, float] = {}

        denom = float(global_total) if global_total > 0 else 0.0
        for embodiment in sorted(global_counts.keys()):
            count = float(global_counts.get(embodiment, 0))
            pct = (100.0 * count / denom) if denom > 0 else 0.0
            info[f"{log_prefix}_stats_embodiment/{embodiment}"] = pct

        return info

    def _get_batch_embodiment(self, data_batch: dict[str, torch.Tensor]) -> str | None:
        """Extract the embodiment name from the first non-None sample's domain_id."""
        domain_ids = data_batch.get("domain_id", None)
        if domain_ids is None:
            return None
        if isinstance(domain_ids, torch.Tensor):
            if domain_ids.numel() == 0:
                return None
            domain_id = int(domain_ids.flatten()[0].item())
        elif isinstance(domain_ids, (list, tuple)):
            first = next((d for d in domain_ids if d is not None), None)
            if first is None:
                return None
            domain_id = int(first)
        else:
            domain_id = int(domain_ids)
        return DOMAIN_ID_TO_EMBODIMENT.get(domain_id, f"unknown_{domain_id}")

    def _accumulate_per_embodiment_loss(
        self,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
    ) -> None:
        embodiment = self._get_batch_embodiment(data_batch)
        if embodiment is None:
            return

        if embodiment not in self._per_embodiment_loss:
            self._per_embodiment_loss[embodiment] = _LossRecord()
        self._per_embodiment_loss[embodiment].loss += loss.detach().float()
        self._per_embodiment_loss[embodiment].iter_count += 1

        if embodiment not in self._per_embodiment_sub_loss:
            self._per_embodiment_sub_loss[embodiment] = {}
        for key in output_batch:
            if "loss" in key and "per_instance" not in key:
                if key not in self._per_embodiment_sub_loss[embodiment]:
                    self._per_embodiment_sub_loss[embodiment][key] = _LossRecord()
                self._per_embodiment_sub_loss[embodiment][key].loss += output_batch[key].detach().float()
                self._per_embodiment_sub_loss[embodiment][key].iter_count += 1

    def _compute_per_embodiment_loss_stats(self, log_prefix: str) -> dict[str, float]:
        """Compute per-embodiment loss averages across all ranks.

        All ranks must call this method (contains collective operations).
        Returns the log dict (only meaningful on rank 0).
        """
        dist_available = dist.is_available() and dist.is_initialized()
        world_size = int(dist.get_world_size()) if dist_available else 1

        # Step 1: gather union of embodiment names across ranks
        local_embodiments = sorted(self._per_embodiment_loss.keys())
        if dist_available:
            all_embodiments: list[list[str] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(all_embodiments, local_embodiments)
        else:
            all_embodiments = [local_embodiments]
        union_embodiments = sorted({e for el in all_embodiments for e in el})

        # Step 2: gather union of sub-loss keys across ranks
        local_sub_keys = sorted({k for d in self._per_embodiment_sub_loss.values() for k in d})
        if dist_available:
            all_sub_keys: list[list[str] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(all_sub_keys, local_sub_keys)
        else:
            all_sub_keys = [local_sub_keys]
        union_sub_keys = sorted({k for kl in all_sub_keys for k in kl})

        # Step 3: insert NaN dummy _LossRecord for missing embodiment/key combos
        for emb in union_embodiments:
            if emb not in self._per_embodiment_loss:
                dummy = _LossRecord()
                dummy.loss += torch.tensor([float("nan")], device="cuda")
                dummy.iter_count += 1
                self._per_embodiment_loss[emb] = dummy
            if emb not in self._per_embodiment_sub_loss:
                self._per_embodiment_sub_loss[emb] = {}
            for key in union_sub_keys:
                if key not in self._per_embodiment_sub_loss[emb]:
                    dummy = _LossRecord()
                    dummy.loss += torch.tensor([float("nan")], device="cuda")
                    dummy.iter_count += 1
                    self._per_embodiment_sub_loss[emb][key] = dummy

        # Step 4: compute distributed averages (all ranks participate in all_reduce)
        log_dict: dict[str, float] = {}
        for emb in union_embodiments:
            avg, valid = self._per_embodiment_loss[emb].get_stat(return_valid_mask_sum=True)
            if valid > 0:
                log_dict[f"{log_prefix}_stats_loss/{emb}"] = avg

        for emb in union_embodiments:
            for key in union_sub_keys:
                avg, valid = self._per_embodiment_sub_loss[emb][key].get_stat(return_valid_mask_sum=True)
                if valid > 0:
                    log_dict[f"{log_prefix}_stats_loss_detail/{emb}_{key}"] = avg

        # Step 5: reset accumulators
        self._per_embodiment_loss = {}
        self._per_embodiment_sub_loss = {}

        return log_dict

    @torch.no_grad()
    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self._accumulate_mode_counts(data_batch)
        self._accumulate_embodiment_counts(data_batch)
        self._accumulate_per_embodiment_loss(data_batch, output_batch, loss)

        if iteration % self.log_freq != 0:
            return

        # All ranks must participate in collective operations below.
        mode_total, mode_counts = self._gather_global_mode_counts()
        embodiment_total, embodiment_counts = self._gather_global_embodiment_counts()
        per_embodiment_loss_dict = self._compute_per_embodiment_loss_stats(log_prefix="train")

        if not distributed.is_rank0():
            return

        if wandb.run is None:
            return

        log_dict: dict[str, float] = {}
        log_dict.update(
            self._build_mode_log_dict(log_prefix="train", global_total=mode_total, global_counts=mode_counts)
        )
        log_dict.update(
            self._build_embodiment_log_dict(
                log_prefix="train", global_total=embodiment_total, global_counts=embodiment_counts
            )
        )
        log_dict.update(per_embodiment_loss_dict)

        wandb.log({k: float(v) for k, v in log_dict.items()}, step=iteration)
