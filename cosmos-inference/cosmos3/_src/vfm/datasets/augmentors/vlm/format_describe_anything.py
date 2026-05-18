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

"""Augmentors for image captions with SoM prompting.
Copied from projects/cosmos/reason1/datasets/augmentors/format_describe_anything.py
Changes:
    1. Unify system prompt to 'You are a helpful assistant.'
    2. Move task requirements from system prompts to the end of user prompts.
"""

import json
import random
from typing import Dict, List, Literal

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos3._src.vfm.datasets.augmentors.vlm.timestamp import markdown_to_list


# reorder dict entries
def reorder_dict_entries(conversation_data: List[Dict]) -> List[Dict]:
    key_order = ["subject_id", "category", "caption"]
    output_dict = {}
    for key in key_order:
        if key in conversation_data:
            output_dict[key] = conversation_data[key]
    return output_dict


def list_to_markdown(conversation_data: List[Dict]) -> str:
    conversation_data = [reorder_dict_entries(item) for item in conversation_data]
    json_string = json.dumps(conversation_data, indent=2)
    return f"```json\n{json_string}\n```".strip()


def augment_assistant_message(
    assistant_message: List[Dict],
    output_format: Literal[
        "dense_image_caption_json_per_subject",
        "dense_image_caption_plain_per_subject",
        "caption_one_object",
        "location_and_caption_json_one_category",
        "location_and_caption_plain_one_category",
    ],
):
    if output_format == "dense_image_caption_json_per_subject":
        output_message = list_to_markdown(assistant_message)
        return output_message
    elif output_format == "dense_image_caption_plain_per_subject":
        output_message = ""
        for item in assistant_message:
            output_message += f"subject_id = <{item['subject_id']}> category = <{item['category']}> {item['caption']}\n"
        return output_message

    elif output_format == "location_and_caption_json_one_category":
        # remove category
        assistant_message = [
            {"subject_id": item["subject_id"], "caption": item["caption"]} for item in assistant_message
        ]
        output_message = list_to_markdown(assistant_message)
        return output_message
    elif output_format == "location_and_caption_plain_one_category":
        output_message = ""
        for item in assistant_message:
            output_message += f"subject_id = <{item['subject_id']}> {item['caption']}\n"
        return output_message

    elif output_format == "caption_one_object":
        return f"{assistant_message[0]['caption']}"
    else:
        raise ValueError(f"Invalid output format: {output_format}")


def augment_user_prompt(
    assistant_message: List[dict],
    output_format: Literal[
        "dense_image_caption_json_per_subject",
        "dense_image_caption_plain_per_subject",
        "caption_one_object",
        "location_and_caption_json_one_category",
        "location_and_caption_plain_one_category",
    ],
):
    if (
        output_format == "dense_image_caption_json_per_subject"
        or output_format == "dense_image_caption_plain_per_subject"
    ):
        if random.random() < 0.5:
            user_prompt = random.choice(
                [
                    "Caption the notable attributes in the provided image.",
                    "Describe the notable attributes in the provided image.",
                    "Summarize the notable attributes in the provided image.",
                ]
            )
            if random.random() < 0.5:
                user_prompt = "Please " + user_prompt.lower()
        else:
            user_prompt = random.choice(
                [
                    "Can you caption the notable attributes in the provided image?",
                    "Can you describe the notable attributes in the provided image?",
                    "Can you summarize the notable attributes in the provided image?",
                ]
            )
        if output_format == "dense_image_caption_json_per_subject":
            user_prompt += """ List and describe all marked subjects in the image with their categories and detailed captions using the following format:
```json
[
{
"subject_id": <subject id 1>,
"category": <category of subject 1>,
"caption": <detailed caption of subject 1>,
},
{
"subject_id": <subject id 2>,
"category": <category of subject 2>,
"caption": <detailed caption of subject 2>,
},
]
```
"""
        else:
            user_prompt += " Please provide captions of the tracked objects in the images using the following format: \nsubject_id = <subject_id> category = <category> caption of event 1.\nsubject_id = <subject_id> category = <category> caption of event 2.\n"
    elif output_format == "caption_one_object":
        event = assistant_message[0]
        user_prompt = random.choice(
            [
                f"What happen to the object with ID <{event['subject_id']}>?",
                f"Describe the object with ID <{event['subject_id']}>?",
                f"Provide a caption of the object with ID <{event['subject_id']}>?",
            ]
        )
    elif (
        output_format == "location_and_caption_json_one_category"
        or output_format == "location_and_caption_plain_one_category"
    ):
        event = assistant_message[0]

        user_prompt = random.choice(
            [
                f"Caption the attribute of the object with category <{event['category']}>.",
                f"Please describe the attribute of the object with category <{event['category']}>.",
                f"Please caption the attribute of the object with category <{event['category']}>.",
                f"Summarize the attribute of the object with category <{event['category']}>.",
            ]
        )
        if output_format == "location_and_caption_json_one_category":
            user_prompt += """ Find all marked subjects that belong to <category> and describe them in detail using the following format:
```json
[
{
    "subject_id": <subject id 1>,
    "caption": <detailed caption of subject 1>
},
{
    "subject_id": <subject id 2>,
    "caption": <detailed caption of subject 2>
},
]
```"""
        else:
            user_prompt += """ Find all marked subjects that belong to <category> and describe them in detail using the following format:
subject_id = <subject id 1> caption of event 1.
subject_id = <subject id 2> caption of event 2."""
    else:
        raise ValueError(f"Invalid output format: {output_format}")
    return user_prompt


class FormatDescribeAnything(Augmentor):
    def __init__(
        self,
        input_key: list = "media",
        output_format: Literal[
            "dense_image_caption_per_subject",
            "caption_one_object",
            "location_and_caption_one_category",
            "random",
        ] = "random",
        urls_needs_timestamp: list = ["tl_plm_sav_20250714"],
    ) -> None:
        """
        Args:
            input_keys (list): List of input keys.
        """
        self.input_key = input_key
        self.output_format = output_format
        self.urls_needs_timestamp = urls_needs_timestamp

    def __call__(self, data_dict: Dict) -> Dict:
        url = data_dict["__url__"]
        if not any(url_pattern in url.root for url_pattern in self.urls_needs_timestamp):
            return data_dict

        if self.output_format == "random":
            output_format = random.choice(
                [
                    "dense_image_caption_per_subject",
                    "caption_one_object",
                    "location_and_caption_one_category",
                ]
            )
        else:
            output_format = self.output_format

        if output_format == "dense_image_caption_per_subject":
            output_format = random.choice(
                ["dense_image_caption_json_per_subject", "dense_image_caption_plain_per_subject"]
            )
        elif output_format == "location_and_caption_one_category":
            output_format = random.choice(
                ["location_and_caption_json_one_category", "location_and_caption_plain_one_category"]
            )

        # find the assistant message and parse into a list of dictionaries
        for item in data_dict["conversation"]:
            if item["role"] == "assistant":
                """
                content dict:
                ```json
                [
                {
                "subject_id": "4",
                "category": "doughnut",
                "caption": "This doughnut has a golden-brown exterior and a light, airy inside, evenly covered with a shiny, clear sugar glaze."
                },
                {
                "subject_id": "5",
                "category": "tray",
                "caption": "This stainless steel tray is rectangular with rounded edges and includes a sequence of symmetrical cut-outs shaped like simplified flowers or four-leaf clovers."
                },
                ]
                ```
                """
                assistant_message = markdown_to_list(item["content"])
                if assistant_message is None:
                    return None  # skip this sample
                break
        # sort assistant_message by object id
        assistant_message = sorted(assistant_message, key=lambda x: int(x["subject_id"]))

        # if temporal localization or caption, sample one event
        if output_format in ["caption_one_object"]:
            assistant_message = random.sample(assistant_message, 1)
        elif output_format in ["location_and_caption_json_one_category", "location_and_caption_plain_one_category"]:
            available_category = list(
                set([assistant_message_i["category"] for assistant_message_i in assistant_message])
            )
            # sample one subject id
            category = random.choice(available_category)
            assistant_message = [
                assistant_message_i
                for assistant_message_i in assistant_message
                if assistant_message_i["category"] == category
            ]

        # process conversation
        conversation = data_dict["conversation"]
        for item in conversation:
            if item["role"] == "system":
                item["content"] = "You are a helpful assistant."
            elif item["role"] == "user":
                for content in item["content"]:
                    if content["type"] == "text":
                        content["text"] = augment_user_prompt(assistant_message, output_format)
                        if content["text"] is None:  # parse error
                            return None
            elif item["role"] == "assistant":
                assistant_message = augment_assistant_message(assistant_message, output_format)
                item["content"] = assistant_message
        data_dict["conversation"] = conversation

        return data_dict
