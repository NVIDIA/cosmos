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

"""Load the understanding tower of a Cosmos3 checkpoint."""

import logging

from transformers import Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)

_DROP_PATTERNS: tuple[str, ...] = (
    r"_moe_gen",
    r"^llm2vae\.",
    r"^vae2llm\.",
    r"^time_embedder\.",
)
"""Generation-tower drop patterns (regex, matched via `re.search`)."""

_KEY_MAPPING: dict[str, str] = {
    # Flat Qwen3 -> nested HF Qwen3-VL. Negative lookahead skips already-nested keys.
    r"^model\.(?!language_model\.)(.+)$": r"model.language_model.\1",
}


class Cosmos3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    # Drop-pattern keys don't match any model parameter after rename -- the
    # loader skips them; these patterns silence the resulting warning.
    _keys_to_ignore_on_load_unexpected = list(_DROP_PATTERNS)
    # Drop once a future checkpoint ships `model.visual.*`.
    _keys_to_ignore_on_load_missing = [r"^model\.visual\."]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # `_checkpoint_conversion_mapping` is a global model_type -> mapping
        # registry, so subclassing doesn't register new renames. Inject via
        # the per-call `key_mapping=` kwarg instead, letting callers override.
        merged = {**_KEY_MAPPING, **(kwargs.pop("key_mapping", None) or {})}
        kwargs["key_mapping"] = merged
        logger.info("Cosmos3 transformers shim: applying key_mapping=%s", merged)
        return super().from_pretrained(*args, **kwargs)
