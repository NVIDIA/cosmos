# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""vLLM plugin: register Cosmos3 models.

See https://docs.vllm.ai/en/latest/design/plugin_system
"""

import logging

logger = logging.getLogger(__name__)


def register():
    from vllm import ModelRegistry

    arch = "Cosmos3ForConditionalGeneration"
    if arch not in ModelRegistry.get_supported_archs():
        logger.info("Registering architecture %s", arch)
        ModelRegistry.register_model(
            arch,
            "vllm_cosmos3.model:Cosmos3ForConditionalGeneration",
        )
