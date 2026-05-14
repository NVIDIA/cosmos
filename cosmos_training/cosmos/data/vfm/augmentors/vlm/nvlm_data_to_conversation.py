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

import json
import random
from typing import Dict, Optional

import numpy as np
from PIL import ImageDraw

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.utils import log


def convert_conversation_role(messages):
    """
    The original messages can be in the following format:
    [
        {"from": "human", "value": "Hello, how are you?"},
        {"from": "gpt", "value": "I'm good, thank you!"},
    ]
    or
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "gpt", "content": "I'm good, thank you!"},
    ]

    The target format is:
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm good, thank you!"},
    ]
    """
    role_mapping = {
        "human": "user",
        "gpt": "assistant",
        "user": "user",
        "assistant": "assistant",
        "label": "assistant",
    }
    messages_converted = []
    for message in messages:
        assert "from" in message or "role" in message, f"Invalid message: {message}"
        assert "value" in message or "content" in message, f"Invalid message: {message}"
        role = message["from"] if "from" in message else message["role"]
        content = message["value"] if "value" in message else message["content"]
        role = role_mapping[role]
        messages_converted.append({"role": role, "content": content})
    return messages_converted


class NVLMImageDataConversation(Augmentor):
    """
    This augmentor is used to convert the nvlm data to a conversation format.
    It will take the data_dict with the following keys:
    {
        "data_class": str,
        "images": List[PIL.Image.Image],
        "text": str,
        "words_boxes": Optional[List[List[int]]],
        "words_text": Optional[List[str]],
        "similarity_matrix": Optional[List[List[float]]],
    }
    and convert it to a dictionary with the following keys:
    {
        "conversation": List[Dict],  # Can be taken by TokenizeData augmentors shared with all datasets
        "media": Dict,
    }

    The dataclass includes:
    - SimilarityInterleavedWebdataset
    - CaptioningWebdataset
    - MultiChoiceVQAWebdataset
    - VQAWebdataset
    - OCRWebdataset
    - TextOCRWebdataset

    SimilarityInterleavedWebdataset will come with the conversations
    CaptioningWebdataset will come with the caption
    MultiChoiceVQAWebdataset will come with the question and choices
    VQAWebdataset will come with the question and answer
    OCRWebdataset will come with the text and words_boxes
    TextOCRWebdataset will come with the text and words_boxes
    """

    def __init__(
        self,
        input_keys: list = ["data_class", "images", "text", "words_boxes", "words_text"],
        output_keys: Optional[list] = ["text"],
        media_type: str = "image",
        media_key_in_data_dict: str = "images",
    ) -> None:
        super().__init__(input_keys, output_keys, None)
        self.media_type = media_type
        self.media_key_in_data_dict = media_key_in_data_dict

        self.user_prompt_list = json.load(
            open("projects/cosmos3/vlm/datasets/augmentors/user_prompt_caption_general.json", "r")
        )
        self.user_prompt_ocr_list = json.load(
            open("projects/cosmos3/vlm/datasets/augmentors/user_prompt_ocr.json", "r")
        )

    def __call__(self, data_dict: Dict) -> Dict:
        try:
            return self.try_parse(data_dict)
        except Exception as e:
            log.warning(
                f"Error parsing data_dict: {e} | data_dict: {data_dict.keys()} | __url__: {data_dict['__url__']}"
            )
            return None

    def try_parse(self, data_dict: Dict) -> Dict:
        """
        The output data_dict will has key "conversation" and "media"
        for the "conversation" key, it will be list of dict

        data['conversation'] = [
            {"role": "system", "content": "**"},
            {
                "role": "user",
                "content": [
                    {"type": media_type, media_type: media_dict_key},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {"role": "assistant", "content": caption},
        ]
        """
        data_class = data_dict["data_class"]
        media_dict = {}
        user_content_list = []
        for media_id, media in enumerate(data_dict[self.media_key_in_data_dict]):
            media_dict_key = f"{self.media_type}_{media_id}"
            user_content_list.append({"type": self.media_type, self.media_type: media_dict_key})
            media_dict[media_dict_key] = media

        if data_class == "SimilarityInterleavedWebdataset":
            messages = data_dict["texts"]
            messages = convert_conversation_role(messages)
            # Insert the user_content_list to the user content
            for message_id, message in enumerate(messages):
                if message["role"] == "user":  # Add to the first user message
                    messages[message_id]["content"] = user_content_list + [
                        {"type": "text", "text": messages[message_id]["content"]}
                    ]
                    break

        elif data_class == "CaptioningWebdataset":
            raw_captions = data_dict["caption"]
            user_prompt = random.choice(self.user_prompt_list)
            user_content_list.append({"type": "text", "text": user_prompt})
            messages = [
                {"role": "user", "content": user_content_list},
                {"role": "assistant", "content": f"{raw_captions}"},
            ]

        elif data_class == "MultiChoiceVQAWebdataset":
            if data_dict["correct_choice_idx"] == -1:
                answer = data_dict["choices"]
                user_prompt = "\nAnswer the question using a single word or phrase."
            else:
                answer = data_dict["correct_choice_idx"]
                user_prompt = "\nAnswer with the option's letter from the given choices directly."
            user_content_list.append(
                {"type": "text", "text": f"{data_dict['context']} {data_dict['choices']}. {user_prompt}"}
            )
            messages = [
                {"role": "user", "content": user_content_list},
                {"role": "assistant", "content": f"{answer}"},
            ]
        elif data_class == "VQAWebdataset":
            user_prompt = "\nAnswer the question using a single word or phrase."
            answer = data_dict["answers"]
            if isinstance(answer, list):
                # random sample one answer
                answer = random.choice(answer)
            user_content_list.append({"type": "text", "text": f"{data_dict['context']} {user_prompt}"})
            messages = [
                {"role": "user", "content": user_content_list},
                {"role": "assistant", "content": f"{answer}"},
            ]
        elif data_class == "OCRWebdataset":
            user_prompt = random.choice(self.user_prompt_ocr_list)
            if (
                "words_boxes" in data_dict
                and "words_text" in data_dict
                and isinstance(data_dict["words_boxes"], list)
                and isinstance(data_dict["words_text"], list)
                and len(data_dict["words_boxes"]) == len(data_dict["words_text"])
            ):
                boxes = data_dict["words_boxes"]
                text = data_dict["words_text"]
                # random sample one box and text
                index = random.randint(0, len(boxes) - 1)
                box = boxes[index]
                text = text[index]
                user_prompt = (
                    user_prompt + f"\nbox: {box}; original image size: {np.array(media_dict[media_dict_key]).shape}"
                )
                assert len(media_dict) == 1, (
                    f"media_dict: {media_dict} | user_prompt: {user_prompt} | __url__: {data_dict['__url__']}"
                )
                # Draw the box on the image
                image = media_dict[media_dict_key]

                log.info(
                    f"box: {box} | text: {text} | media_dict_key: {media_dict_key} | __url__: {data_dict['__url__']} | image shape: {np.array(image).shape}"
                )
                if len(box) == 4:
                    draw = ImageDraw.Draw(image)
                    draw.rectangle(box, outline="red", width=2)
                    media_dict[media_dict_key] = image

                reply = text
            elif "words_text" in data_dict:
                reply = data_dict["words_text"]
            else:
                reply = data_dict["text"]
            user_content_list.append({"type": "text", "text": user_prompt})
            messages = [
                {"role": "user", "content": user_content_list},
                {"role": "assistant", "content": f"{reply}"},
            ]
        else:
            log.warning(f"Invalid data class: {data_class}")
            return None

        # Remove image tag in the user content if any
        def remove_image_tag(text):
            text = text.replace("\n<image>", "").replace("</image>", "").replace("<image>", "")
            return text

        for message_id in range(len(messages)):
            if messages[message_id]["role"] == "user" and isinstance(messages[message_id]["content"], list):
                for content_id in range(len(messages[message_id]["content"])):
                    if messages[message_id]["content"][content_id]["type"] == "text":
                        text = messages[message_id]["content"][content_id]["text"]
                        text = remove_image_tag(text)
                        messages[message_id]["content"][content_id]["text"] = text
            elif messages[message_id]["role"] == "user" and isinstance(messages[message_id]["content"], str):
                messages[message_id]["content"] = remove_image_tag(messages[message_id]["content"])

            # Make sure the assistant text content is a string not float or int
            if messages[message_id]["role"] == "assistant" and not isinstance(messages[message_id]["content"], list):
                if isinstance(messages[message_id]["content"], dict):
                    if messages[message_id]["content"]["type"] == "text":
                        messages[message_id]["content"]["text"] = f"{messages[message_id]['content']['text']}"
                else:
                    messages[message_id]["content"] = f"{messages[message_id]['content']}"

        data_dict["conversation"] = messages
        data_dict["media"] = media_dict
        return data_dict
