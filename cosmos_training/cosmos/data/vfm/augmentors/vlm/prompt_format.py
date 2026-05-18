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

"""Visual-Text Transformations or Augmentations."""

import random
from typing import Dict, Literal

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor

REASONING_SUFFIX = (
    "\nAnswer the question using the following format:\n\n"
    "<think>\nYour reasoning.\n</think>\n\n"
    "Write your final answer immediately after the </think> tag."
)


class PromptFormat(Augmentor):
    def __init__(
        self,
        input_keys: list = ["texts"],
        text_chat_order: Literal["text_end", "text_start", "random"] = "text_end",
    ) -> None:
        """
        Args:
            input_keys (list): List of input keys.
            text_chat_order (Literal["text_end", "text_start", "random"]): Order of text items in user messages.
        """
        self.input_keys = input_keys
        self.text_chat_order = text_chat_order

    def __call__(self, data_dict: Dict) -> Dict:
        conversation_key = self.input_keys[0]

        # retrive conversations from dict
        try:
            list_of_conversation = data_dict[conversation_key]
        except KeyError:
            url = data_dict["__url__"].root + "/" + data_dict["__url__"].path
            print(f"KeyError: {conversation_key} not found in data_dict for url: {url}")
            return None

        # check if this is list of list of dict or list of dict

        if isinstance(list_of_conversation[0], list):
            selected_conversation = random.sample(list_of_conversation, 1)[0]
        elif isinstance(list_of_conversation[0], dict):

            selected_conversation = list_of_conversation
        else:
            raise ValueError(
                f"list_of_conversation is not a list of list of dict or list of dict: {list_of_conversation}"
            )

        # Now it should be list of dict
        assert isinstance(selected_conversation, list) and isinstance(selected_conversation[0], dict), (
            f"selected_conversation is not a list of dict: {selected_conversation}"
        )
        # Normalize all string content to list format
        for message in selected_conversation:
            if "content" in message and isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
            if "reasoning_content" in message and isinstance(message["reasoning_content"], str):
                message["reasoning_content"] = [{"type": "text", "text": message["reasoning_content"]}]

        # Merge reasoning_content into assistant message content
        for i, message in enumerate(selected_conversation):
            if message.get("role") == "assistant" and message.get("reasoning_content"):
                # Append reasoning instruction to the preceding user message
                for j in range(i - 1, -1, -1):
                    if selected_conversation[j].get("role") == "user":
                        selected_conversation[j]["content"].append({"type": "text", "text": REASONING_SUFFIX})
                        break
                # Wrap reasoning items in <think>...</think> tags
                reasoning_items = message["reasoning_content"]
                think_start = [{"type": "text", "text": "<think>\n"}]
                think_end = [{"type": "text", "text": "\n</think>\n\n"}]
                message["content"] = think_start + reasoning_items + think_end + message["content"]
                del message["reasoning_content"]

        data_dict["conversation"] = selected_conversation

        del data_dict[conversation_key]


        # # enforce chat order
        # self._enforce_text_chat_order(selected_conversation)

        return data_dict

    def _enforce_text_chat_order(self, conversation: list) -> None:
        """
        Reorder text content within user messages based on text_chat_order setting.
        NOTE (maxzhaoshuol): this does NOT work for interleaved data!!!!!!

        Args:
            conversation: List of message dictionaries
        """
        for message in conversation:
            if message.get("role") == "user" and "content" in message:
                content = message["content"]
                if isinstance(content, list):
                    # Separate text items from non-text items
                    text_items = [item for item in content if item.get("type") == "text"]
                    non_text_items = [item for item in content if item.get("type") != "text"]

                    if text_items:
                        # Reorder based on text_chat_order
                        if self.text_chat_order == "text_start":
                            # Put text items at the beginning
                            message["content"] = text_items + non_text_items
                        elif self.text_chat_order == "text_end":
                            # Put text items at the end
                            message["content"] = non_text_items + text_items
                        elif self.text_chat_order == "random":
                            print("random")
                            # Randomly put text items at beginning or end
                            if random.random() < 0.5:
                                message["content"] = text_items + non_text_items
                            else:
                                message["content"] = non_text_items + text_items
