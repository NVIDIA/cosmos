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

"""Per-stream timing callback.

Logs ``forward``, ``backward`` and ``optimizer_step`` wall-clock time broken
down by the data stream that produced each iteration's batch
(``data_batch["dataset_name"]``).

Useful for verifying load-balance hypotheses such as "action_data_slow drives
the long ``optimizer_step`` time observed at large node counts".  Because
:class:`IterativeJointDataLoader` synchronises stream selection across all
ranks via ``seed + global_id``, every rank processes the same stream at the
same iteration, so logging on rank 0 is representative of the global cost.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import wandb

from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.utils import distributed
from cosmos3._src.imaginaire.utils.callback import Callback

_TIMER_KEYS: tuple[str, ...] = ("forward", "backward", "optimizer_step", "dataloader_train")


class PerStreamTiming(Callback):
    """Aggregate ``training_timer`` results by data stream and log to wandb.

    Args:
        log_freq: Number of iterations between wandb logs.  Each log emits
            the per-stream mean time for every key in :data:`_TIMER_KEYS` and
            the per-stream iteration count, then resets the accumulators.
    """

    def __init__(self, log_freq: int = 100) -> None:
        super().__init__()
        self.log_freq = log_freq
        self._sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._counts: dict[str, int] = defaultdict(int)

    @staticmethod
    def _extract_stream_name(data_batch: dict) -> str | None:
        """Return the ``dataset_name`` carried by the packed batch.

        ``IterativeJointDataLoader`` attaches a per-sample ``dataset_name``
        and the collation produces a list of identical names for one stream.
        """
        ds = data_batch.get("dataset_name")
        if ds is None:
            return None
        if isinstance(ds, str):
            return ds
        if isinstance(ds, (list, tuple)) and ds:
            first = ds[0]
            return first if isinstance(first, str) else None
        return None

    @torch.no_grad()
    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, output_batch, loss
        stream = self._extract_stream_name(data_batch)
        if stream is None:
            return

        timer_results = self.trainer.training_timer.results
        for key in _TIMER_KEYS:
            values = timer_results.get(key)
            if not values:
                continue
            self._sums[stream][key] += float(values[-1])
        self._counts[stream] += 1

        if iteration % self.log_freq != 0 or iteration == 0:
            return
        if not distributed.is_rank0() or wandb.run is None:
            self._reset()
            return

        log_dict: dict[str, float] = {}
        for stream_name, key_sums in self._sums.items():
            n = self._counts[stream_name]
            if n == 0:
                continue
            log_dict[f"per_stream_iters/{stream_name}"] = float(n)
            for key, total in key_sums.items():
                log_dict[f"per_stream_timer/{stream_name}/{key}"] = total / n

        wandb.log(log_dict, step=iteration)
        self._reset()

    def _reset(self) -> None:
        self._sums = defaultdict(lambda: defaultdict(float))
        self._counts = defaultdict(int)
