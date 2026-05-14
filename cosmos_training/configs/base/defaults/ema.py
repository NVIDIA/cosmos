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

import attrs
from hydra.core.config_store import ConfigStore


@attrs.define(slots=False)
class EMAConfig:
    """
    Config for the EMA.
    """

    enabled: bool = True
    rate: float = 0.1
    iteration_shift: int = 0


PowerEMAConfig: EMAConfig = EMAConfig(
    enabled=True,
    rate=0.10,
    iteration_shift=0,
)


def register_ema():
    cs = ConfigStore.instance()
    cs.store(group="ema", package="model.config.ema", name="power", node=PowerEMAConfig)
