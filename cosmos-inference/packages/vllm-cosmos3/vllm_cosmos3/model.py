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
import re
from collections.abc import Iterable

import torch
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)

_DROP_PATTERNS: tuple[str, ...] = (
    # Generation
    r"_moe_gen",
    r"^llm2vae\.",
    r"^vae2llm\.",
    r"^time_embedder\.",
    # Sound
    r"^llm2sound\.",
    r"^sound2llm\.",
    r"^sound_modality_embed$",
    # Action
    r"^llm2action\.",
    r"^action2llm\.",
    r"^action_modality_embed$",
)
"""Drop patterns (regex, matched via `re.search`)."""
_DROP_RE = re.compile("|".join(_DROP_PATTERNS))

_KEY_MAPPING: dict[str, str] = {
    # Flat Qwen3 -> nested HF Qwen3-VL. Negative lookahead skips already-nested keys.
    r"^model\.(?!language_model\.)(.+)$": r"model.language_model.\1",
    # Flat Qwen3-VL vision component -> nested HF Qwen3-VL.
    r"^(blocks\.|merger\.|patch_embed\.|pos_embed\.|deepstack_merger_list\.)(.*)$": r"model.visual.\1\2",
}
_KEY_MAPPING_RES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(src), tgt) for src, tgt in _KEY_MAPPING.items()
)


def _is_und_tower_weight(name: str) -> bool:
    return _DROP_RE.search(name) is None


def _to_hf_name(name: str) -> str:
    for pat, repl in _KEY_MAPPING_RES:
        name = pat.sub(repl, name)
    return name


class Cosmos3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        overrides = getattr(vllm_config.model_config.hf_config, "allow_patterns_overrides", None)
        if overrides:
            self.allow_patterns_overrides = list(overrides)
            if any(p.endswith(".safetensors") for p in self.allow_patterns_overrides):
                vllm_config.load_config.load_format = "safetensors"

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        # vLLM's per-weight callback: filter generation-tower entries and
        # rename understanding-tower entries before delegating to the base.
        kept = 0
        skipped = 0

        def _iter() -> Iterable[tuple[str, torch.Tensor]]:
            nonlocal kept, skipped
            for name, tensor in weights:
                if _is_und_tower_weight(name):
                    kept += 1
                    yield _to_hf_name(name), tensor
                else:
                    skipped += 1

        loaded = super().load_weights(_iter())
        logger.info(
            "Cosmos3 vllm shim: kept %d understanding weights, skipped %d generation weights",
            kept,
            skipped,
        )
        return loaded
