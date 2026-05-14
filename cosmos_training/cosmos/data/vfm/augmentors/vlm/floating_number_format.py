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

import re
from typing import Dict

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor


def format_floating_number(text: str, decimal_places: int) -> str:
    """
    Format floating point numbers in text according to the specified format.

    Args:
        text: Input text containing floating point numbers
        floating_number_format: Format string like '.2f', '2.2f', etc.

    Returns:
        Text with floating point numbers formatted according to the format string
    """
    # Pattern to match floating point numbers (including integers that could be floats)
    # Matches: integers, decimals like 123.456, scientific notation, etc.
    pattern = r"-?\d+\.?\d*(?:[eE][+-]?\d+)?"

    def replace_float(match: re.Match) -> str:
        try:
            num = float(match.group())
            # Format the number using the provided format string
            # Handle format strings like '.2f' or '2.2f'
            formatted = f"{num:.{decimal_places}f}".rstrip("0").rstrip(".") if decimal_places > 0 else str(int(num))
            return formatted
        except (ValueError, TypeError):
            # If conversion fails, return the original match
            return match.group()

    # Replace all floating point numbers in the text
    formatted_text = re.sub(pattern, replace_float, text)
    return formatted_text


class FloatingNumberFormat(Augmentor):
    def __init__(
        self,
        input_key: str = "conversation",
        decimal_places: int = 2,
        urls_needs_format: list = [],
        processor=None,
    ) -> None:
        """
        Args:
            input_keys (list): List of input keys.
        """
        self.input_key = input_key
        self.decimal_places = decimal_places
        self.urls_needs_format = urls_needs_format

    def __call__(self, data_dict: Dict) -> Dict:
        url = data_dict["__url__"]
        if not any(url_pattern in url.root for url_pattern in self.urls_needs_format):
            return data_dict

        for item in data_dict[self.input_key]:
            if item["role"] == "user":
                for content in item["content"]:
                    if content["type"] == "text":
                        content["text"] = format_floating_number(content["text"], self.decimal_places)
            elif item["role"] == "assistant":
                if isinstance(item["content"], list):
                    assert len(item["content"]) == 1
                    assert item["content"][0]["type"] == "text"
                    item["content"] = format_floating_number(item["content"][0]["text"], self.decimal_places)
                else:
                    item["content"] = format_floating_number(item["content"], self.decimal_places)
        return data_dict
