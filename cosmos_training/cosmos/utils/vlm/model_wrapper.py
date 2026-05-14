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

from typing import Any, Union

from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.stateful import Stateful


class ModelWrapper(Stateful):
    """Wrapper for model state dict handling"""

    def __init__(self, model_parts: Union[list[nn.Module], nn.Module]):
        if not isinstance(model_parts, list):
            model_parts = [model_parts]
        self.model_parts = model_parts

    def state_dict(self) -> dict[str, Any]:
        sd = {}
        for model in self.model_parts:
            sd.update(get_model_state_dict(model))
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for model in self.model_parts:
            set_model_state_dict(
                model,
                model_state_dict=state_dict,
                options=StateDictOptions(strict=False),
            )
