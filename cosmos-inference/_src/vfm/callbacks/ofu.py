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

"""OFU (Operational FLOPs Utilization) callback for OmniMoT training.

Computes and logs OFU metrics by launching ``nvidia-smi dmon`` as a background
subprocess and parsing the Tensor Core activity (mmaact) and processor clock
(pclk) columns.  OFU is defined as::

    OFU = mmaact * (pclk / max_pclk)

where ``max_pclk`` is the max boost clock for the detected hardware (e.g.
1980 MHz for H100, 2062 MHz for GB200).  The result is in the 0-100 range.
"""

from __future__ import annotations

import subprocess
import threading
from collections import defaultdict
from dataclasses import dataclass

import torch
import wandb

from cosmos3._src.imaginaire.attention.utils import is_blackwell_dc
from cosmos3._src.imaginaire.callbacks.every_n import EveryN
from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.trainer import ImaginaireTrainer
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.distributed import is_rank0, rank0_only


@dataclass
class HardwareTarget:
    """Hardware-specific constants for OFU normalisation.

    Attributes:
        name: Human-readable name (used as W&B tag, e.g. "H100").
        max_pclk_mhz: Max boost SM clock in MHz used to normalise OFU.
    """

    name: str
    max_pclk_mhz: float


# Pre-defined hardware targets
H100 = HardwareTarget(name="H100", max_pclk_mhz=1980.0)
GB200 = HardwareTarget(name="GB200", max_pclk_mhz=2062.0)


class OFUCallback(EveryN):
    """Callback that computes and logs Operational FLOPs Utilization (OFU) to W&B.

    OFU = mmaact * (pclk / max_pclk), where mmaact is the MMA activity
    percentage and pclk is the current processor clock from ``nvidia-smi dmon``.
    ``max_pclk`` is determined from the detected hardware (H100 or GB200).
    The result is in the 0-100 range.

    The callback launches ``nvidia-smi dmon`` as a background subprocess on
    ``on_train_start`` and a daemon thread continuously reads its output.
    At every logging interval, accumulated samples are consumed, averaged per GPU
    and overall, and logged to W&B under ``ofu/{hardware_name}``.

    Args:
        hit_thres: Number of warm-up training iterations to skip before logging.
    """

    def __init__(
        self,
        *args,
        hit_thres: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hardware_target = GB200 if is_blackwell_dc() else H100
        self.hit_thres = hit_thres

        # Subprocess state
        self._process: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Buffered samples protected by a lock: list of (gpu_idx, mmaact, pclk)
        self._lock = threading.Lock()
        self._samples: list[tuple[int, float, float]] = []

        # Column indices parsed from the header (set by _reader_loop)
        self._col_gpu: int | None = None
        self._col_mmaact: int | None = None
        self._col_pclk: int | None = None

        # Warm-up counter
        self._hit_counter: int = 0

    # ------------------------------------------------------------------ #
    # Background reader
    # ------------------------------------------------------------------ #

    def _parse_header(self, line: str) -> bool:
        """Parse a dmon header line to locate column indices.

        Called on every ``#`` line because nvidia-smi dmon reprints the header
        every few seconds.  Returns True if ``gpu``, ``mmaact``, and ``pclk``
        columns are all found; warns only when the column-names line (identified
        by the presence of ``gpu``) lacks a required column.  Silently ignores
        the units line (``# Idx  W  C ...``) which does not contain ``gpu``.
        """
        cols = line.lstrip("#").strip().split()
        col_map = {name.lower(): idx for idx, name in enumerate(cols)}
        gpu_idx = col_map.get("gpu")
        mmaact_idx = col_map.get("mmaact")
        pclk_idx = col_map.get("pclk")
        if gpu_idx is not None and mmaact_idx is not None and pclk_idx is not None:
            if self._col_mmaact is None:
                log.info(f"OFUCallback: found mmaact at column {mmaact_idx}, pclk at column {pclk_idx}")
            self._col_gpu = gpu_idx
            self._col_mmaact = mmaact_idx
            self._col_pclk = pclk_idx
            return True
        if gpu_idx is not None:
            missing = [name for name, idx in [("mmaact", mmaact_idx), ("pclk", pclk_idx)] if idx is None]
            log.warning(
                f"OFUCallback: column(s) {missing} not found in nvidia-smi dmon header: {cols}. "
                "OFU metrics will not be available."
            )
        return False

    def _reader_loop(self) -> None:
        """Background thread that reads nvidia-smi dmon output line-by-line."""
        assert self._process is not None and self._process.stdout is not None

        for line in self._process.stdout:
            if self._stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue

            # Header lines repeat every few seconds — always re-parse so that a
            # missed or failed first parse is recovered on the next occurrence.
            if line.startswith("#"):
                self._parse_header(line)
                continue

            # Skip data lines until we have column indices
            if self._col_gpu is None or self._col_mmaact is None or self._col_pclk is None:
                continue

            parts = line.split()
            try:
                gpu_idx = int(parts[self._col_gpu])
                mmaact = float(parts[self._col_mmaact])
                pclk = float(parts[self._col_pclk])
            except (ValueError, IndexError):
                continue

            with self._lock:
                self._samples.append((gpu_idx, mmaact, pclk))

    # ------------------------------------------------------------------ #
    # Lifecycle hooks
    # ------------------------------------------------------------------ #

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if not is_rank0():
            return

        try:
            # --gpm-metrics 5 means that we access Tensor Activity under mmaact column.
            # -d 5 means that we sample the data every 5 seconds.
            cmd = ["nvidia-smi", "dmon", "--gpm-metrics", "5", "-d", "5"]
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
            )
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            log.info(f"OFUCallback: launched nvidia-smi dmon --gpm-metrics 5")
        except FileNotFoundError:
            log.warning("OFUCallback: nvidia-smi not found, OFU metrics will not be available")
        except Exception as e:
            log.warning(f"OFUCallback: failed to launch nvidia-smi dmon: {e}")

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if not is_rank0():
            return
        self._stop_event.set()
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except ProcessLookupError:
                pass  # already exited
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5)
            self._reader_thread = None

    # ------------------------------------------------------------------ #
    # Per-step gating
    # ------------------------------------------------------------------ #

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # All ranks must enter super().on_training_step_end() so they reach the
        # distributed barrier inside EveryN.  Only rank 0 has samples to clear.
        if self._hit_counter < self.hit_thres:
            self._hit_counter += 1
            if self._hit_counter == self.hit_thres:
                # Discard samples collected during warm-up (compilation, allocation, etc.)
                with self._lock:
                    self._samples.clear()
            return
        # Delegate to EveryN for the periodic reporting logic
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    # ------------------------------------------------------------------ #
    # Periodic reporting
    # ------------------------------------------------------------------ #

    @rank0_only
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        if self._process is None:
            return

        # Drain buffered samples
        with self._lock:
            samples = list(self._samples)
            self._samples.clear()

        if not samples:
            log.warning(
                f"OFUCallback: no nvidia-smi samples collected at iteration {iteration}. "
                "Check that the dmon subprocess launched and that the mmaact column is present."
            )
            return

        # Compute per-GPU OFU: mmaact * (pclk / max_pclk)
        max_pclk = self.hardware_target.max_pclk_mhz
        gpu_ofu: dict[int, list[float]] = defaultdict(list)
        gpu_mmaact: dict[int, list[float]] = defaultdict(list)
        gpu_pclk: dict[int, list[float]] = defaultdict(list)
        for gpu_idx, mmaact, pclk in samples:
            gpu_ofu[gpu_idx].append(mmaact * (pclk / max_pclk))
            gpu_mmaact[gpu_idx].append(mmaact)
            gpu_pclk[gpu_idx].append(pclk)

        # Overall averages across all GPUs and samples
        all_ofu = [v for vals in gpu_ofu.values() for v in vals]
        all_mmaact = [v for vals in gpu_mmaact.values() for v in vals]
        all_pclk = [v for vals in gpu_pclk.values() for v in vals]

        log_info: dict[str, float] = {
            f"ofu/{self.hardware_target.name}": sum(all_ofu) / len(all_ofu),
            "ofu/mmaact": sum(all_mmaact) / len(all_mmaact),
            "ofu/avg_pclk_mhz": sum(all_pclk) / len(all_pclk),
            "ofu/num_samples": float(len(samples)),
        }

        if wandb.run is not None:
            wandb.log(log_info, step=iteration)
