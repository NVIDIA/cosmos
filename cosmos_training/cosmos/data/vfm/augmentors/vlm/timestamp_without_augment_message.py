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

"""
Copied from projects/cosmos/reason1/datasets/augmentors/timestamp_without_augment_message.py
Changes:
    - overlay_text is now imported from cosmos3.
"""

import json
import random
import re
from typing import Dict, List, Literal, Tuple

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.data.vfm.augmentors.vlm.timestamp import overlay_text


def list_to_markdown(conversation_data: List[Dict]) -> str:
    json_string = json.dumps(conversation_data, indent=2)
    return f"```json\n{json_string}\n```".strip()


def snap_timestamps_to_existing(assistant_message: List[Dict], existing_timestamps: List[float]) -> List[Dict]:
    """
    Snap conversation start/end timestamps to the nearest existing timestamps.

    Args:
        assistant_message: JSON string containing list of dictionaries with 'start', 'end', and 'caption' fields
        existing_timestamps: List of existing timestamps (floats) to snap to

    Returns:
        List of dictionaries with snapped timestamps
    """
    snapped_message = []

    for item in assistant_message:
        if not isinstance(item, dict) or "start" not in item or "end" not in item:
            raise ValueError("Each item must be a dictionary with 'start' and 'end' fields")

        snapped_item = item.copy()

        # Snap start and end timestamps to existing ones
        snapped_item["start"] = min(existing_timestamps, key=lambda x: abs(x - item["start"]))
        snapped_item["end"] = min(existing_timestamps, key=lambda x: abs(x - item["end"]))

        snapped_message.append(snapped_item)

    # Sort the merged events by start timestamp to ensure chronological order
    # Merge captions that share identical start and end timestamps
    merged_events: Dict[Tuple[float, float], Dict] = {}
    for item in snapped_message:
        item["start"] = round(item["start"], 2)
        item["end"] = round(item["end"], 2)

        key = (item["start"], item["end"])
        if key in merged_events:
            # Concatenate captions for the same time interval.
            merged_events[key]["caption"] = merged_events[key]["caption"].rstrip() + " " + item["caption"].lstrip()
        else:
            merged_events[key] = item

        merged_events[key]["caption"] = merged_events[key]["caption"].strip()

    # Sort the merged events by start timestamp to ensure chronological order
    new_assistant_message = sorted(merged_events.values(), key=lambda x: x["start"])
    if len(new_assistant_message) == 0:
        raise ValueError("No valid assistant message found for data.")

    return new_assistant_message


def augment_assistant_message(
    assistant_message: List[Dict],
    output_format: Literal[
        "dense_video_caption_json", "dense_video_caption_plain", "temporal_localization", "temporal_caption"
    ],
):
    if output_format == "dense_video_caption_json":
        output_message = list_to_markdown(assistant_message)
        return output_message
    elif output_format == "dense_video_caption_plain":
        output_message = ""
        for item in assistant_message:
            output_message += f"<{item['start']}> <{item['end']}> {item['caption']}\n"
        return output_message
    elif output_format == "temporal_localization":
        return f"<{assistant_message[0]['start']}> <{assistant_message[0]['end']}>"
    elif output_format == "temporal_caption":
        return assistant_message[0]["caption"]
    else:
        raise ValueError(f"Invalid output format: {output_format}")


def augment_system_prompt(
    system_prompt: str,
    output_format: Literal[
        "dense_video_caption_json", "dense_video_caption_plain", "temporal_localization", "temporal_caption"
    ],
    need_overlay_text=True,
):
    if output_format == "dense_video_caption_json":
        system_prompt = system_prompt
    elif output_format == "dense_video_caption_plain":
        system_prompt = re.sub(r"Please.*?\]", "", system_prompt, flags=re.DOTALL)  # strip off existing format
        system_prompt += """Please provide captions of all the events in the video with timestamps using the following format:
        <start time> <end time> caption of event 1.\n<start time> <end time> caption of event 2.\n"""
    elif output_format == "temporal_localization":
        system_prompt = re.sub(r"Please.*?\]", "", system_prompt, flags=re.DOTALL)  # strip off existing format
        system_prompt += "Please locate the start and end time of a given event specified by the user using the following format: <start time> <end time>."
    elif output_format == "temporal_caption":
        system_prompt = re.sub(r"Please.*?\]", "", system_prompt, flags=re.DOTALL)  # strip off existing format
        system_prompt += "Please provide a caption of the duration in the video based on the start and end time specified by the user."
    else:
        raise ValueError(f"Invalid output format: {output_format}")

    if need_overlay_text:
        system_prompt = (
            system_prompt
            + "\nAt each frame, the timestamp is embedded at the bottom of the video. You need to extract the timestamp and answer the user question."
        )
    else:
        system_prompt = system_prompt + "\nYou need to extract the timestamp and answer the user question."

    return system_prompt


def augment_user_prompt(
    assistant_message: List[dict],
    output_format: Literal[
        "dense_video_caption_json", "dense_video_caption_plain", "temporal_localization", "temporal_caption"
    ],
):
    if output_format == "dense_video_caption_json" or output_format == "dense_video_caption_plain":
        if random.random() < 0.5:
            user_prompt = random.choice(
                [
                    "Caption the notable events in the provided video.",
                    "Describe the notable events in the provided video.",
                    "Summarize the notable events in the provided video.",
                ]
            )
            if random.random() < 0.5:
                user_prompt = "Please " + user_prompt.lower()
        else:
            user_prompt = random.choice(
                [
                    "Can you caption the notable events in the provided video?",
                    "Can you describe the notable events in the provided video?",
                    "Can you summarize the notable events in the provided video?",
                ]
            )
    elif output_format == "temporal_localization":
        event = assistant_message[0]
        user_prompt = random.choice(
            [
                f"When does the following event happen? {event['caption']}",
                f"When does the event '{event['caption'].lower()[:-1]}' happen?",
                f"Can you find the event '{event['caption'].lower()[:-1]}'?",
            ]
        )
    elif output_format == "temporal_caption":
        event = assistant_message[0]
        if random.random() < 0.5:

            start = round(event["start"])
            end = round(event["end"])
        else:

            start = round(event["start"] * 2) / 2
            end = round(event["end"] * 2) / 2
        if start == end:  # HACK (maxzhaoshuol): remove events with start == end
            raise ValueError("Start and end time are the same for data.")
        user_prompt = random.choice(
            [
                f"Caption the event between {start}s and {end}s.",
                f"Please describe the event between {start} and {end}.",
                f"Please caption the event between the start time {start}s and the end time {end}s.",
                f"Summarize the event between <{start}s, {end}s>.",
            ]
        )
    return user_prompt


class TimeStampWithoutAugmentMessage(Augmentor):
    def __init__(
        self,
        input_key: str = "media",
        output_format: Literal[
            "dense_video_caption", "temporal_localization", "temporal_caption", "random"
        ] = "dense_video_caption",
        urls_needs_timestamp: list = ["av_reasoning_localization_20250627", "tl_activitynet_20250630"],
        processor=None,
    ) -> None:
        """
        Args:
            input_keys (list): List of input keys.
        """
        self.input_key = input_key
        self.output_format = output_format
        self.urls_needs_timestamp = urls_needs_timestamp
        self.processor = processor

    def __call__(self, data_dict: Dict) -> Dict:
        url = data_dict["__url__"]
        if not any(url_pattern in url.root for url_pattern in self.urls_needs_timestamp):
            return data_dict

        media_data = data_dict[self.input_key]
        for k, v in media_data.items():
            if "video" in k:
                video_frames_with_timestamp, timestamps = overlay_text(v["videos"], v["fps"], processor=self.processor)
                media_data[k]["videos"] = video_frames_with_timestamp
        return data_dict
