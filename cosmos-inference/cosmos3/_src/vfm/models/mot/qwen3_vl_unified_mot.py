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

# Backward-compatibility shim: this module was renamed to unified_mot.py.
# Existing serialized configs / checkpoints may reference the old module path,
# so we re-export everything from the new location.
from cosmos3._src.vfm.models.mot.unified_mot import *  # noqa: F401, F403
from cosmos3._src.vfm.models.mot.unified_mot import (  # noqa: F401  # explicit re-exports for type checkers
    LayerTypes,
    MoTDecoderLayer,
    Nemotron3DenseVLTextConfig,
    Nemotron3DenseVLTextForCausalLM,
    Nemotron3DenseVLTextModel,
    PackedAttentionMoT,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeTextForCausalLM,
    Qwen3VLMoeTextModel,
    Qwen3VLTextConfig,
    Qwen3VLTextForCausalLM,
    Qwen3VLTextModel,
    Qwen3VLTextMoTDecoderLayer,
)
