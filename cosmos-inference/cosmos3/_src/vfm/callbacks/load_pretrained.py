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

from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.utils.callback import Callback


class LoadPretrained(Callback):
    def __init__(self):
        r"""
        This callback enables us to load pretrained model weights if needed.
        Model weights are initialized from safetensors if not loaded already from DCP checkpoint.
        """
        super().__init__()

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        model.load_pretrained_model_if_needed()
