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

"""Callback that saves an emergency checkpoint before a Slurm job is killed.

Slurm signal timeline
---------------------

**Timeout** (job hits ``--time`` limit)::

    T            SIGUSR1 → batch shell  (N is 300 via ``--signal=B:SIGUSR1@300``)
    T+N sec      SIGTERM → all processes
    T+N+KillWait SIGKILL → anything still alive  (KillWait is 30s)

**Preemption** (higher-priority job needs the nodes)::

    T            SIGUSR1 → batch shell
    T+Grace      SIGTERM + SIGKILL  (GraceTime is 300s in GCP-IAD)

**User cancel** (``scancel <jobid>``)::

    T            SIGTERM → all processes  (no SIGUSR1)
    T+KillWait   SIGKILL → anything still alive (KillWait is 30s)

Implementation
--------------

* **How the SIGUSR1 signal is handled:**

  - The Slurm batch script responds to SIGUSR1 by creating a sentinel file
    (``$SLURM_LOG_DIR/SIGUSR1_RECEIVED``) on the shared filesystem.
  - This callback polls for the presence of this sentinel file at the end of
    each training step.
  - When detected, it triggers an emergency checkpoint save.

* **Why the Python process can't receive SIGUSR1 directly:**

  - Pyxis/Enroot containers do not receive SIGUSR1 signals from Slurm (the
    signal is sent to the batch shell, not the container).
  - Attempted to forward SIGUSR1 with both ``srun`` and
    ``scancel --signal`` in batch scripts, proven not working.

* **Why no SIGTERM handler is needed:**

  - The SIGUSR1 signal is the trigger of preemption and timeout, it
    is already able to distinguish them from user cancel.
  - Before SIGTERM arrives there are at least 300 s (``GraceTime=300`` for
    preemption, ``--signal=B:SIGUSR1@N`` for timeout), which is sufficent
    for the poll to detect the sentinel and save a checkpoint.
  - SIGTERM and the subsequent SIGKILL are left to terminate the process
    naturally after the checkpoint has been saved.
  - SIGTERM and SIGUSR1 are logged so we can observe signal delivery into
    Pyxis/Enroot containers for debugging purposes.

To avoid redundant checkpoints, a save is only performed if at least
``save_iter * min_save_fraction`` iterations have elapsed since the last
checkpoint.
"""

from __future__ import annotations

import os
import signal
import sys

import torch

from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.callback import Callback


class TerminationSignalCheckpoint(Callback):
    """Save a checkpoint in response to SIGUSR1 (preemption or timeout).

    Args:
        min_save_fraction: Fraction of the regular checkpoint interval (between 0
            and 1) that must have elapsed since the last checkpoint before an
            emergency save is allowed.  Defaults to 1/3.
    """

    def __init__(self, min_save_fraction: float = 1 / 3):
        super().__init__()
        self._min_save_fraction = min_save_fraction
        self._current_iteration: int = 0
        self._last_checkpoint_iteration: int = 0
        # Captured from on_before_optimizer_step so we can call checkpointer.save().
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._grad_scaler: torch.amp.GradScaler | None = None
        # Sentinel file created by the batch-shell trap when SIGUSR1 arrives.
        # This is the sole detection mechanism because srun/Pyxis does not
        # relay SIGUSR1 into the container.
        slurm_log_dir = os.environ.get("SLURM_LOG_DIR", "")
        self._sigusr1_sentinel = os.path.join(slurm_log_dir, "SIGUSR1_RECEIVED") if slurm_log_dir else ""

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        self._current_iteration = iteration
        self._last_checkpoint_iteration = iteration
        self._install_termination_signal_handlers()

    def on_before_optimizer_step(
        self,
        model_ddp,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._grad_scaler = grad_scaler

    def on_save_checkpoint_success(self, iteration: int = 0, elapsed_time: float = 0) -> None:
        self._last_checkpoint_iteration = iteration

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self._current_iteration = iteration

        if not self._sigusr1_sentinel or not os.path.exists(self._sigusr1_sentinel):
            return

        log.info("[TerminationSignalCheckpoint] Detected SIGUSR1 sentinel file. Will save checkpoint.")

        # Check if the minimum progress has been reached since the last checkpoint.
        min_progress = int(self.config.checkpoint.save_iter * self._min_save_fraction)
        if (iteration - self._last_checkpoint_iteration) < min_progress:
            log.info(
                f"[TerminationSignalCheckpoint] Only {iteration - self._last_checkpoint_iteration} iterations "
                f"since last checkpoint (threshold {min_progress}). Skipping checkpoint save."
            )
            sys.exit(0)

        assert self._optimizer is not None, (
            "[TerminationSignalCheckpoint] Optimizer reference not set — on_before_optimizer_step was never called"
        )

        log.info(f"[TerminationSignalCheckpoint] Saving checkpoint at iteration {iteration}.")
        self.trainer.checkpointer.save(model, self._optimizer, self._scheduler, self._grad_scaler, iteration=iteration)
        # Async DCP checkpointing queues the write to a background process.
        # We must wait for it to finish before exiting.
        self.trainer.checkpointer.finalize()
        log.info(f"[TerminationSignalCheckpoint] Checkpoint saved at iteration {iteration}.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Termination signal handlers
    # ------------------------------------------------------------------

    def _install_termination_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._log_sigterm)
        log.info("[TerminationSignalCheckpoint] Installed SIGTERM handler.")

    def _log_sigterm(self, signum: int, frame: object) -> None:
        log.info(f"[TerminationSignalCheckpoint] Received SIGTERM at iteration {self._current_iteration}.")
