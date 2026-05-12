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
from typing import Union

import torch

from cosmos3._src.imaginaire.config import Config
from cosmos3._src.imaginaire.trainer import ImaginaireTrainer
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.callback import LowPrecisionCallback as BaseLowPrecisionCallback
from cosmos3._src.vfm.models.omni_mot_model import OmniMoTModel


class LowPrecisionCallback(BaseLowPrecisionCallback):
    """
    Config with non-primitive type makes it difficult to override the option.
    The callback gets precision from model.precision instead.
    It also auto disabled when using fp32.
    """

    def __init__(self, config: Config, trainer: ImaginaireTrainer, update_iter: int):
        self.config = config
        self.trainer = trainer
        self.update_iter = update_iter

    def on_train_start(self, model: Union[OmniMoTModel, list[OmniMoTModel]], iteration: int = 0) -> None:
        if not isinstance(model, list):
            model = [model]
        for model_part in model:
            if model_part.precision == torch.float32:
                log.critical("Using fp32, should disable master weights.")
                self.update_iter = sys.maxsize
        else:
            assert model_part.precision in [
                torch.bfloat16,
                torch.float16,
                torch.half,
            ], "LowPrecisionCallback must use a low precision dtype."
            self.precision_type = model_part.precision
