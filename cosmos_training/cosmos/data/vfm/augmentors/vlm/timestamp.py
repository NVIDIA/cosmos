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

"""Augmentors for general video dense caption datasets.
Copied from projects/cosmos/reason1/datasets/augmentors/timestamp.py
Changes:
    1. Unify system prompt to 'You are a helpful assistant.'
    2. Move task requirements from system prompts to user prompts.
    3. Randomly change timestamp formats from ["seconds", "hh:mm:ss", "hh:mm:ss.sss", "mm:ss.sss"]
    4. Add json output format for event temporal localization.
"""

import json
import random
from copy import deepcopy
from typing import Dict, List, Literal, Tuple

from PIL import Image, ImageDraw, ImageFont

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.utils import log


def compute_timestamps(frame_index: int, fps: float, processor) -> float:
    if processor is not None and "Qwen3" in processor.name:
        frame_index_start = frame_index // processor.merge_size * processor.merge_size
        frame_index_end = frame_index_start + processor.merge_size - 1
        timestamps_start = frame_index_start / fps
        timestamps_end = frame_index_end / fps
        timestamps = (timestamps_start + timestamps_end) / 2
        timestamps = float(f"{timestamps:.1f}")
        return timestamps
    else:
        return frame_index / fps


def convert_timestamp(seconds: float | str, format: str = "hh:mm:ss") -> str:
    if isinstance(seconds, str):
        seconds = float(seconds)
    # convert seconds to hh:mm:ss.sss format
    minutes = int(seconds // 60)
    hours = int(minutes // 60)
    minutes = minutes % 60
    seconds = seconds % 60
    if format == "hh:mm:ss":
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}"
    elif format == "hh:mm:ss.sss":
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    elif format == "mm:ss.sss":
        return f"{minutes + 60 * hours:02d}:{seconds:06.3f}"
    else:
        raise ValueError(f"Invalid format: {format}")


def check_if_need_overlay_text(processor):
    if processor is not None and ("Qwen3" in processor.name or "Nemotron" in processor.name):
        return False
    return True


timestamp_convertor = {
    "seconds": lambda x: x,
    "hh:mm:ss": lambda x: convert_timestamp(x, format="hh:mm:ss"),
    "hh:mm:ss.sss": lambda x: convert_timestamp(x, format="hh:mm:ss.sss"),
    "mm:ss.sss": lambda x: convert_timestamp(x, format="mm:ss.sss"),
}


def overlay_text(
    images: List[Image.Image],
    fps: float,
    border_height: int = 28,  # this is due to patch size of 28
    temporal_path_size: int = 2,  # Number of positions to cycle through
    font_size: int = 20,
    font_color: str = "white",
    processor=None,
    debug=False,
) -> Tuple[List[Image.Image], List[float]]:
    """
    Overlay text on a list of PIL images with black border.
    The timestamp position cycles through available positions.

    Args:
        images: List of PIL images to process
        fps: Frames per second
        border_height: Height of the black border in pixels (default: 28)
        temporal_path_size: Number of positions to cycle through (default: 2)
        font_size: Font size for the text (default: 20)
        font_color: Color of the text (default: "white")

    Returns:
        List of PIL images with text overlay
        List of timestamps
    """
    if not check_if_need_overlay_text(processor) and not debug:
        # if debug is True, we still need to overlay text for visualization purpose
        return images, [compute_timestamps(i, fps, processor) for i in range(len(images))]

    # Try to use DejaVu Sans Mono font for better readability
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)

    # Process each image
    processed_images = []

    for i, image in enumerate(images):
        # Get original dimensions
        width, height = image.size

        # Create new image with black border at the bottom
        new_height = height + border_height
        if debug:  # add border_height for visualization purpose
            new_height = new_height + border_height
        new_image = Image.new("RGB", (width, new_height), color="black")

        # Paste original image at the top
        new_image.paste(image, (0, 0))

        # Draw text on the black border
        draw = ImageDraw.Draw(new_image)

        # Calculate timestamp for current frame
        total_seconds = compute_timestamps(i, fps, processor)
        text = f"{total_seconds:.2f}s"

        # Get text dimensions
        try:
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(text, font=font)

        # Define available positions (cycling through horizontal positions)
        position_idx = i % temporal_path_size
        section_width = width // temporal_path_size

        # Calculate x position based on cycling position
        section_center_x = position_idx * section_width + section_width // 2
        text_x = section_center_x - text_width // 2

        # Ensure text doesn't go outside bounds
        text_x = max(0, min(text_x, width - text_width))

        # Center vertically in the border
        text_y = height + (border_height - text_height) // 2

        # Draw the single timestamp
        draw.text((text_x, text_y), text, fill=font_color, font=font)

        processed_images.append(new_image)

    return processed_images, [compute_timestamps(i, fps, processor) for i in range(len(images))]


def markdown_to_list(conversation_data: str | List[Dict]) -> List[Dict]:
    if isinstance(conversation_data, list):
        assert (
            isinstance(conversation_data[0], dict)
            and conversation_data[0]["type"] == "text"
            and len(conversation_data) == 1
        )
        conversation_data = conversation_data[0]["text"]
    cleaned_data = conversation_data.strip()
    if cleaned_data.startswith("```json"):
        cleaned_data = cleaned_data[7:]  # Remove '```json'
        if cleaned_data.endswith("```"):
            cleaned_data = cleaned_data[:-3]  # Remove '```'
        cleaned_data = cleaned_data.strip()
    return json.loads(cleaned_data)


def json_to_markdown(conversation_data: List[Dict] | Dict) -> str:
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
        "dense_video_caption_json",
        "dense_video_caption_plain",
        "dense_video_caption_json_with_types",
        "dense_video_caption_plain_with_types",
        "temporal_localization_plain",
        "temporal_localization_json",
        "temporal_caption",
    ],
    timestamp_format: str = "hh:mm:ss",
):
    # change time stamp format to hh:mm:ss.sss
    assistant_message = deepcopy(assistant_message)
    for item in assistant_message:
        item["start"] = timestamp_convertor[timestamp_format](item["start"])
        item["end"] = timestamp_convertor[timestamp_format](item["end"])
    if output_format == "dense_video_caption_json" or output_format == "dense_video_caption_json_with_types":
        output_message = json_to_markdown(assistant_message)
        return output_message
    elif output_format == "dense_video_caption_plain" or output_format == "dense_video_caption_plain_with_types":
        output_message = ""
        for item in assistant_message:
            if output_format == "dense_video_caption_plain":
                output_message += f"{item['start']}, {item['end']}, {item['caption']}\n"
            elif output_format == "dense_video_caption_plain_with_types":
                output_message += f"{item['start']}, {item['end']}, {item['type']}, {item['caption']}\n"
        return output_message
    elif output_format == "temporal_localization_plain":
        return f"{assistant_message[0]['start']}, {assistant_message[0]['end']}"
    elif output_format == "temporal_localization_json":
        output_message = {
            "start": assistant_message[0]["start"],
            "end": assistant_message[0]["end"],
        }
        output_message = json_to_markdown(output_message)
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
        "dense_video_caption_json_with_types",
        "dense_video_caption_plain_with_types",
        "temporal_localization_plain",
        "temporal_localization_json",
        "temporal_caption",
    ],
    timestamp_format: str = "hh:mm:ss",
):
    if (
        output_format == "dense_video_caption_json"
        or output_format == "dense_video_caption_plain"
        or output_format == "dense_video_caption_json_with_types"
        or output_format == "dense_video_caption_plain_with_types"
    ):
        if random.random() < 0.5:
            user_prompt = random.choice(
                [
                    "Caption the notable events in the provided video.",
                    "Describe the notable events in the provided video.",
                    "Summarize the notable events in the provided video.",
                    "Localize a series of activity events in the video, output the start and end timestamp and description for each event.",
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
            # add format requirement.
            if random.random() < 0.5:
                user_prompt = user_prompt + (
                    "\nPlease provide captions of all the events in the video with timestamps using the following format:\n"
                    "[\n"
                    "  {\n"
                    f'    "start": {timestamp_format},\n'
                    f'    "end": {timestamp_format},\n'
                    '    "caption": <caption of event 1>\n'
                    "  },\n"
                    "  {\n"
                    f'    "start": {timestamp_format},\n'
                    f'    "end": {timestamp_format},\n'
                    '    "caption": <caption of event 2>\n'
                    "  }\n"
                    "]"
                )
            else:
                user_prompt = (
                    user_prompt
                    + f"\nProvide the result in json format with '{timestamp_format}' for time depiction for each event. Use keywords 'start', 'end' and 'caption' in the json output."
                )
        elif output_format == "dense_video_caption_json_with_types":
            # add format requirement.
            if random.random() < 0.5:
                user_prompt = user_prompt + (
                    "\nPlease provide captions of all the events in the video with timestamps using the following format:\n"
                    "[\n"
                    "  {\n"
                    f'    "start": {timestamp_format},\n'
                    f'    "end": {timestamp_format},\n'
                    '    "type": <type of event 1>\n'
                    '    "caption": <caption of event 1>\n'
                    "  },\n"
                    "  {\n"
                    f'    "start": {timestamp_format},\n'
                    f'    "end": {timestamp_format},\n'
                    '    "type": <type of event 2>\n'
                    '    "caption": <caption of event 2>\n'
                    "  }\n"
                    "]"
                )
            else:
                user_prompt = (
                    user_prompt
                    + f"\nProvide the result in json format with '{timestamp_format}' for time depiction for each event. Use keywords 'start', 'end', 'type' and 'caption' in the json output."
                )
        elif output_format == "dense_video_caption_plain_with_types":
            user_prompt = (
                user_prompt
                + f"\nPlease provide captions of all the events in the video with start and end timestamps using the following format: \n{timestamp_format}, {timestamp_format}, <type of event 1>, <caption of event 1>.\n{timestamp_format}, {timestamp_format}, <type of event 2>, <caption of event 2>."
            )
        else:  # plain format
            user_prompt = (
                user_prompt
                + f"\nPlease provide captions of all the events in the video with start and end timestamps using the following format: \n{timestamp_format}, {timestamp_format}, caption of event 1.\n{timestamp_format}, {timestamp_format}, caption of event 2."
            )

    elif output_format == "temporal_localization_plain" or output_format == "temporal_localization_json":
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
        if output_format == "temporal_localization_json":
            user_prompt = (
                user_prompt
                + f"\nPlease provide the result in json format with '{timestamp_format}' for time depiction for the event. Use keywords 'start', 'end' in the json output."
            )
        else:
            user_prompt = (
                user_prompt
                + f"\nPlease provide the start and end timestamp of the event in the following format: {timestamp_format}, {timestamp_format}."
            )

    elif output_format == "temporal_caption":
        event = assistant_message[0]
        if random.random() < 0.333333:

            start = round(event["start"])
            end = round(event["end"])
        elif random.random() < 0.666666:

            start = round(event["start"] * 2) / 2
            end = round(event["end"] * 2) / 2
        else:
            start = event["start"]
            end = event["end"]
        if start == end:  # HACK (maxzhaoshuol): remove events with start == end
            raise ValueError("Start and end time are the same for data.")
        if timestamp_format == "seconds":
            if random.random() < 0.5:
                start = f"{start}s"
                end = f"{end}s"
            else:
                start = f"{start} seconds"
                end = f"{end} seconds"
        else:
            start = convert_timestamp(start, format=timestamp_format)
            end = convert_timestamp(end, format=timestamp_format)
        user_prompt = random.choice(
            [
                f"Caption the event between {start} and {end}.",
                f"Please describe the event between {start} and {end}.",
                f"Please caption the event between the start time {start} and the end time {end}.",
                f"Summarize the event between {start} and {end}.",
            ]
        )
    return user_prompt


class TimeStamp(Augmentor):
    def __init__(
        self,
        input_key: list = "media",
        output_format: Literal[
            "dense_video_caption", "temporal_localization", "temporal_caption", "caption", "random"
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
        elif self.output_format == "caption":
            output_format = random.choice(["dense_video_caption", "temporal_caption"])
        else:
            output_format = self.output_format

        if output_format == "dense_video_caption":
            output_format = random.choice(
                [
                    "dense_video_caption_json",
                    "dense_video_caption_plain",
                    "dense_video_caption_json_with_types",
                    "dense_video_caption_plain_with_types",
                ]
            )
        elif output_format == "temporal_localization":
            output_format = random.choice(["temporal_localization_plain", "temporal_localization_json"])

        timestamp_format = random.choice(list[str](timestamp_convertor.keys()))

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

            if "types" in output_format:
                for item in assistant_message:
                    if "type" not in item:  # if type is not provided, use the default format
                        output_format = random.choice(["dense_video_caption_json", "dense_video_caption_plain"])
                        break

            # if temporal localization or caption, sample one event
            if output_format in ["temporal_localization_plain", "temporal_localization_json", "temporal_caption"]:
                assistant_message = random.sample(assistant_message, 1)

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
