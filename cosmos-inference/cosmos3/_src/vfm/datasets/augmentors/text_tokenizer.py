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

# Augmentor for tokenizing input text

import json
import random
from typing import Optional

import torch

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos3._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos3._src.vfm.datasets.sequence_packing import add_special_tokens
from cosmos3._src.vfm.models.vlm.qwen3_vl.utils import tokenize_caption

_MAX_NUM_TOKENS = 4096


class TextTokenizerTransform(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

        tokenizer_config = self.args["tokenizer_config"]
        self.cfg_dropout_rate = self.args["cfg_dropout_rate"]
        self.use_system_prompt = self.args.get("use_system_prompt", False)

        # Initialize the text tokenizer
        vlm_tokenizer = lazy_instantiate(tokenizer_config)
        vlm_tokenizer, _ = add_special_tokens(vlm_tokenizer)
        self.vlm_tokenizer = vlm_tokenizer

    def __call__(self, data_dict: dict) -> dict:
        input_caption = data_dict[self.input_keys[0]]

        if isinstance(input_caption, dict):
            # Encode dict into a json string. This json string is then passed to the transformer tokenizer.
            input_caption = json.dumps(input_caption)
            data_dict[self.input_keys[0]] = input_caption

        if self.cfg_dropout_rate > 0:
            # If CFG is used, randomly dropout the input caption
            # We dropout the input caption by replacing it with an empty string
            if random.random() < self.cfg_dropout_rate:
                input_caption = ""
                data_dict[self.input_keys[0]] = input_caption
        text_ids = tokenize_caption(
            input_caption,
            self.vlm_tokenizer,
            is_video=False,
            use_system_prompt=self.use_system_prompt,
        )
        text_ids = text_ids[:_MAX_NUM_TOKENS]  # truncate the text ids to the maximum number of tokens
        # This will take care of wierd edge cases where we generate extremely long captions
        data_dict[self.output_keys[0]] = torch.tensor(text_ids)  # [N_tokens]
        return data_dict


_SYSTEM_PROMPT_IMAGE_EDITING = "You are a helpful assistant who will edit images based on the user's instructions."

_SYSTEM_PROMPT_TRANSFER = "You are a helpful assistant that generates images or videos following the user's instructions and control signals (edge maps, blur, depth, or segmentation)."

_SYSTEM_PROMPTS = {
    "editing": _SYSTEM_PROMPT_IMAGE_EDITING,
    "transfer": _SYSTEM_PROMPT_TRANSFER,
}


def _tokenize_caption_with_system_prompt(caption: str, tokenizer: object, system_prompt: str) -> list[int]:
    """Tokenize a caption with a task-specific system prompt (editing or transfer).

    Args:
        caption: The instruction or caption text.
        tokenizer: The Qwen2 tokenizer (must support ``apply_chat_template``).
        system_prompt: System prompt for the conversation (e.g. editing or transfer).

    Returns:
        List of token IDs.
    """
    conversations = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": caption},
    ]
    tokenizer_output = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        add_vision_id=False,
    )
    return tokenizer_output


class TextTokenizerTransformForEditing(Augmentor):
    """Tokenizer augmentor for interleaved tasks: image editing or transfer (control-conditioned generation).

    Uses a task-specific system prompt. Pass args["task"] = "editing" (default) or "transfer".
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

        tokenizer_config = self.args["tokenizer_config"]
        self.cfg_dropout_rate = self.args.get("cfg_dropout_rate", 0.0)
        task = self.args.get("task", "editing")
        self._system_prompt = _SYSTEM_PROMPTS.get(task, _SYSTEM_PROMPTS["editing"])

        vlm_tokenizer = lazy_instantiate(tokenizer_config)
        vlm_tokenizer, _ = add_special_tokens(vlm_tokenizer)
        self.vlm_tokenizer = vlm_tokenizer

    def __call__(self, data_dict: dict) -> dict | None:
        input_caption = data_dict.get(self.input_keys[0], "")
        if self.cfg_dropout_rate > 0 and random.random() < self.cfg_dropout_rate:
            input_caption = ""
            data_dict[self.input_keys[0]] = input_caption
        text_ids = _tokenize_caption_with_system_prompt(input_caption, self.vlm_tokenizer, self._system_prompt)
        data_dict[self.output_keys[0]] = torch.tensor(text_ids)  # [N_tokens]
        return data_dict


class TextTokenizerTransformForTransfer(TextTokenizerTransformForEditing):
    """Tokenizer augmentor for transfer (control-conditioned) generation. Uses transfer system prompt."""

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        args = dict(args) if args else {}
        args["task"] = "transfer"
        super().__init__(input_keys, output_keys, args)
