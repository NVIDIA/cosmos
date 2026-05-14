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
"""VLM freeze config (read by ``vlm_model._apply_freeze_config``)."""

from typing import Optional

import attrs

from cosmos.utils.config import make_freezable


@make_freezable
@attrs.define(slots=False)
class VLMFreezeConfig:
    """Selects which parts of a VLM stay trainable.

    Applied at model construction, before the optimizer is built; the optimizer
    only sees the resulting ``requires_grad`` state.
    """

    # Named freeze flags. Supported architectures: Qwen2.5-VL,
    # Qwen3-VL (dense + MoE), InternVL3_5.
    freeze_vision_encoder: bool = False
    freeze_mm_projector: bool = False
    freeze_llm: bool = False

    # Regex-based freeze (mutually exclusive with each other).
    # trainable_params: whitelist — only matching params are trainable.
    # frozen_params:    blacklist — matching params get frozen.
    trainable_params: Optional[list[str]] = None
    frozen_params: Optional[list[str]] = None

    def __attrs_post_init__(self) -> None:
        if self.trainable_params is not None and self.frozen_params is not None:
            raise ValueError("VLMFreezeConfig: set at most one of trainable_params or frozen_params, not both.")
