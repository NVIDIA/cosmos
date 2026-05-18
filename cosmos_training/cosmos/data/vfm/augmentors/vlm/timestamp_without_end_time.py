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

"""Augmentors for video dense caption datasets without end time.
Copied from projects/cosmos/reason1/datasets/augmentors/timestamp_without_end_time.py
Changes:
    1. Unify system prompt to 'You are a helpful assistant.'
    2. Move task requirements from system prompts to user prompts.
    3. Randomly change timestamp formats from ["seconds", "hh:mm:ss", "hh:mm:ss.sss", "mm:ss.sss"]
    4. Add json output format for event temporal localization.
"""

import random
from copy import deepcopy
from typing import Dict, List, Literal, Tuple

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.utils import log
from cosmos.data.vfm.augmentors.vlm.timestamp import (
    json_to_markdown,
    markdown_to_list,
    overlay_text,
    timestamp_convertor,
)


def snap_timestamps_to_existing(assistant_message: List[Dict], existing_timestamps: List[float]) -> List[Dict]:
    """
    Snap conversation start/end timestamps to the nearest existing timestamps.

    Args:
        assistant_message: List of dictionaries with 'start', 'end', and 'caption' fields
        existing_timestamps: List of existing timestamps (floats) to snap to

    Returns:
        List of dictionaries with snapped timestamps
    """
    snapped_message = []
    for item in assistant_message:
        if "caption" not in item and "event" in item:
            # This is the nexar dataset
            item["caption"] = item["event"]
            del item["event"]

        if not isinstance(item, dict) or "start" not in item:
            raise ValueError(f"Each item must be a dictionary with 'start' field. getting {item}")

        snapped_item = item.copy()

        # Snap start and end timestamps to existing ones
        snapped_item["start"] = min(existing_timestamps, key=lambda x: abs(x - item["start"]))

        snapped_message.append(snapped_item)

    # Sort the merged events by start timestamp to ensure chronological order
    # Merge captions that share identical start and end timestamps
    merged_events: Dict[Tuple[float], Dict] = {}
    for item in snapped_message:
        item["start"] = round(item["start"], 2)

        key = (item["start"],)
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
        "dense_video_caption_json",
        "dense_video_caption_plain",
        "single_event_localization",
        "multiple_events_localization_json",
        "temporal_caption",
    ],
    timestamp_format: str = "hh:mm:ss",
):
    # change time stamp format to hh:mm:ss.sss
    assistant_message = deepcopy(assistant_message)
    for item in assistant_message:
        item["start"] = timestamp_convertor[timestamp_format](item["start"])

    if output_format == "dense_video_caption_json":
        output_message = json_to_markdown(assistant_message)
        return output_message
    elif output_format == "dense_video_caption_plain":
        output_message = ""
        for item in assistant_message:
            output_message += f"{item['start']}, {item['caption']}\n"
        return output_message
    elif output_format == "single_event_localization":
        return f"{assistant_message[0]['start']}"
    elif output_format == "multiple_events_localization_json":
        for item in assistant_message:
            if "start" in item:
                item["time"] = item["start"]
                del item["start"]
            if "caption" in item:
                item["event"] = item["caption"]
                del item["caption"]
        output_message = json_to_markdown(assistant_message)
        return output_message
    elif output_format == "temporal_caption":
        return assistant_message[0]["caption"]
    else:
        raise ValueError(f"Invalid output format: {output_format}")


def augment_user_prompt(
    assistant_message: List[dict],
    output_format: Literal[
        "dense_video_caption_json",
        "dense_video_caption_plain",
        "single_event_localization",
        "multiple_events_localization_json",
        "temporal_caption",
    ],
    timestamp_format: str = "hh:mm:ss",
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
        if output_format == "dense_video_caption_json":
            if random.random() < 0.5:
                user_prompt = user_prompt + (
                    "\nPlease identify all the events in the following driving video with timestamps using the following format:\n"
                    "[\n"
                    "  {\n"
                    f'    "start": {timestamp_format},\n'
                    '    "event": <type of event 1>,\n'
                    "  },\n"
                    "  {\n"
                    f'    "start": {timestamp_format},\n'
                    '    "event": <type of event 2>,\n'
                    "  },\n"
                    "]\n"
                    "Each event corresponds to one and only one of the followinig five types: collision, near collision, hard brake, harsh acceleration, sharp cornering.\n"
                )
            else:
                user_prompt = (
                    user_prompt
                    + f"\nPlease provide the result in json format with '{timestamp_format}' for time depiction for each event. Use keywords 'start', 'event' in the json output."
                )
        else:
            user_prompt = (
                user_prompt
                + f"\nPlease provide short descriptions of all the events in the video with timestamps using the following format: \n{timestamp_format}, caption of event 1.\n{timestamp_format}, caption of event 2."
            )
    elif output_format == "single_event_localization":
        event = assistant_message[0]
        event_caption = event["caption"]
        if not event_caption[-1].isalpha():
            event_caption = event_caption[:-1]
        user_prompt = random.choice(
            [
                f"When does the following event happen? {event_caption}.",
                f"When does the event '{event_caption.lower()}' happen?",
                f"Can you find the event '{event_caption.lower()}'?",
            ]
        )
        user_prompt = (
            user_prompt
            + f"\nPlease provide the start timestamp of the event in the following format: {timestamp_format}."
        )

    elif output_format == "multiple_events_localization_json":
        user_prompt = random.choice(
            [
                f"You should find the following {len(assistant_message)} events in the input video:",
                f"Please find the following {len(assistant_message)} events based on descriptions:",
                f"Please identify the following events in the input video:",
            ]
        )
        for i, event in enumerate(assistant_message):
            user_prompt += f"\nEvent {i + 1}: {event['caption']}"
        user_prompt += f"\nPlease provide the result in json format as a list of dictionaries. Use '{timestamp_format}' for time depiction for each event. Use keywords 'time', 'event' in each dictionary."

    elif output_format == "temporal_caption":
        event = assistant_message[0]
        if random.random() < 0.333333:

            start = round(event["start"])
        elif random.random() < 0.666666:

            start = round(event["start"] * 2) / 2
        else:
            start = event["start"]

        if timestamp_format == "seconds":
            if random.random() < 0.5:
                start = f"{start}s"
            else:
                start = f"{start} seconds"
        else:
            start = timestamp_convertor[timestamp_format](start)

        user_prompt = random.choice(
            [
                f"Caption the event starting at {start}.",
                f"Please describe the event starting at {start}.",
                f"Please caption the event starting at {start}.",
                f"Summarize the event starting at {start}.",
            ]
        )
    return user_prompt


class TimeStampWithoutEndTime(Augmentor):
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

        if self.output_format == "random":
            output_format = random.choice(["dense_video_caption", "temporal_localization", "temporal_caption"])
        else:
            output_format = self.output_format

        if output_format == "dense_video_caption":
            output_format = random.choice(["dense_video_caption_json", "dense_video_caption_plain"])

        try:
            # find the assistant message and parse into a list of dictionaries
            for item in data_dict["conversation"]:
                if item["role"] == "assistant":
                    if isinstance(item["content"], list):
                        assert len(item["content"]) == 1
                        assert item["content"][0]["type"] == "text"
                        item["content"] = item["content"][0]["text"]
                    assistant_message = markdown_to_list(item["content"])
                    assistant_message = snap_timestamps_to_existing(assistant_message, timestamps)
                    break

            # remove end time if it exists
            for item in assistant_message:
                if "end" in item:
                    del item["end"]

            # if temporal localization or caption, sample one event
            if output_format == "temporal_localization":
                output_format = random.choice(["single_event_localization", "multiple_events_localization_json"])
            if output_format in ["single_event_localization", "temporal_caption"]:
                assistant_message = random.sample(assistant_message, 1)
            elif output_format == "multiple_events_localization_json":
                assistant_message = random.sample(assistant_message, min(len(assistant_message), random.randint(1, 5)))
                random.shuffle(assistant_message)

            timestamp_format = random.choice(list(timestamp_convertor.keys()))
            # process conversation
            conversation = data_dict["conversation"]
            for item in conversation:
                if item["role"] == "system":
                    item["content"] = "You are a helpful assistant."
                elif item["role"] == "user":
                    for content in item["content"]:
                        if content["type"] == "text":
                            content["text"] = augment_user_prompt(assistant_message, output_format, timestamp_format)
                elif item["role"] == "assistant":
                    assistant_message = augment_assistant_message(assistant_message, output_format, timestamp_format)
                    item["content"] = assistant_message
            data_dict["conversation"] = conversation

            return data_dict

        except Exception as e:
            log.warning(f"Error timestamping: {e}. Skipping this sample {url.root} {data_dict['__key__']}.")
            return None
