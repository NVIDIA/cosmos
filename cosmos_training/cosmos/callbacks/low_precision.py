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

import sys

import torch

from cosmos.trainer import ImaginaireTrainer
from cosmos.utils import log
from cosmos.utils.callback import LowPrecisionCallback as BaseLowPrecisionCallback


class LowPrecisionCallback(BaseLowPrecisionCallback):
    """
    Unified low-precision callback for VFM and VLM.

    Two initialization modes:
    - Config-driven (VLM): pass ``param_torch_dtype`` as a string (e.g. "bfloat16").
      ``precision_type`` is set immediately from the string.
    - Runtime-inspection (VFM): omit ``param_torch_dtype``.
      ``precision_type`` is determined from ``model.precision`` at ``on_train_start``.

    Auto-disabled (``update_iter`` set to maxsize) when fp32 is detected.
    """

    def __init__(
        self,
        config,
        trainer: ImaginaireTrainer,
        update_iter: int,
        param_torch_dtype: str | None = None,
    ):
        self.config = config
        self.trainer = trainer
        self.update_iter = update_iter
        if param_torch_dtype is not None:
            # Config-driven path (VLM)
            self.precision_type = getattr(torch, param_torch_dtype)

    def on_train_start(self, model, iteration: int = 0) -> None:
        if hasattr(self, "precision_type"):
            # Already set from config — nothing to do
            return
        # Runtime-inspection path (VFM): read precision from model
        if not isinstance(model, list):
            model = [model]
        for model_part in model:
            if getattr(model_part, "precision", None) == torch.float32:
                log.critical("Using fp32, should disable master weights.")
                self.update_iter = sys.maxsize
        else:
            precision = getattr(model_part, "precision", None)
            assert precision in [
                torch.bfloat16,
                torch.float16,
                torch.half,
            ], "LowPrecisionCallback must use a low precision dtype."
            self.precision_type = precision
