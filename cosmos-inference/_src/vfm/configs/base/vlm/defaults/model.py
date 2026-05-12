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

from hydra.core.config_store import ConfigStore

from cosmos3._src.imaginaire.lazy_config import LazyCall as L
from cosmos3._src.vfm.configs.base.defaults.model_config import VLMModelConfig
from cosmos3._src.vfm.configs.base.vlm.defaults.training import TrainConfig
from cosmos3._src.vfm.models.vlm_model import VLMModel

VLM_FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(VLMModel)(
        config=VLMModelConfig(
            train=TrainConfig(param_dtype="bfloat16"),
        ),
        checkpoint="${checkpoint}",
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="vlm_fsdp", node=VLM_FSDP_CONFIG)
