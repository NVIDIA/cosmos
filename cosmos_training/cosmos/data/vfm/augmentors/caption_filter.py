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

from typing import Optional

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.utils import log


class CaptionFilter(Augmentor):
    """
    Caption filter augmentor for predict2 training.

    This augmentor filters video samples based on caption content with configurable behavior:
    - contain_keyword=True: Only return videos that contain keywords in captions
    - contain_keyword=False: Only return videos that do NOT contain keywords in captions

    When a sample doesn't match the filter criteria, it returns None, which causes
    the webdataset pipeline to skip that sample and continue to the next one.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        """
        Initialize the caption filter.

        Args:
            input_keys: List containing the caption key (e.g., ["ai_caption"] or text embeddings key)
            output_keys: Not used for filtering, can be None
            args: Dictionary with filtering parameters:
                - "keywords": List of keywords to filter by (e.g., ["camera pan"])
                - "contain_keyword": Boolean flag for filtering behavior:
                    * True: Only return videos that contain keywords
                    * False: Only return videos that do NOT contain keywords
                - "log_filtered": Whether to log filtered samples (default: False)
                - "filter_stats": Whether to track filtering statistics (default: True)
                - "dont_apply_on_webdataset_names": List of webdataset names to not apply the filter on, it will just pass through without checking contain or not contain keywords
        """
        super().__init__(input_keys, output_keys, args)

        # Parse arguments
        if args is None:
            args = {}

        self.keywords = args.get("keywords", [])
        self.contain_keyword = args.get("contain_keyword", False)  # Default to exclude mode
        self.log_filtered = args.get("log_filtered", False)
        self.filter_stats = args.get("filter_stats", True)
        self.dont_apply_on_webdataset_names = args.get("dont_apply_on_webdataset_names", [])

        # Validate input_keys
        if not input_keys or len(input_keys) == 0:
            raise ValueError("CaptionFilter requires at least one input key for the caption field")

        self.caption_key = input_keys[0]  # Use the first input key as the caption key

        # Statistics tracking
        if self.filter_stats:
            self.total_samples = 0
            self.filtered_samples = 0

        # Validate configuration
        if not self.keywords:
            log.warning("CaptionFilter: No keywords provided, filter will not filter any samples")

        mode_str = "contain" if self.contain_keyword else "exclude"
        log.info(
            f"CaptionFilter initialized in '{mode_str}' mode with {len(self.keywords)} keywords using caption key '{self.caption_key}': {self.keywords}"
        )

    def __call__(self, data_dict: dict) -> Optional[dict]:
        """
        Filter data based on caption content.

        This checks the caption field specified by the input_keys parameter.
        Depending on contain_keyword flag:
        - True: Returns data_dict only if caption contains any keyword, None otherwise
        - False: Returns data_dict only if caption contains NO keywords, None otherwise

        Args:
            data_dict: Input data dictionary containing the caption field specified in input_keys

        Returns:
            data_dict: Original data dict if caption passes filter
            None: If caption should be filtered out (causes sample to be skipped)
        """
        data_dict_root = data_dict["__url__"].root
        if any(n in data_dict_root for n in self.dont_apply_on_webdataset_names):
            return data_dict

        if self.filter_stats:
            self.total_samples += 1

        # Check if caption key exists
        if self.caption_key not in data_dict:
            if self.log_filtered:
                log.warning(f"CaptionFilter: No '{self.caption_key}' found in data_dict, passing through")
            return data_dict

        caption = data_dict[self.caption_key]
        if not isinstance(caption, str) or not caption.strip():
            if self.log_filtered:
                log.warning(f"CaptionFilter: '{self.caption_key}' is empty or not a string, got {type(caption)}")
            return data_dict

        # Check if any keywords are found in the caption
        search_caption = caption.lower()
        keyword_found = False
        matched_keyword = None

        for keyword in self.keywords:
            if keyword.lower() in search_caption:
                keyword_found = True
                matched_keyword = keyword
                break

        # Apply filtering logic based on contain_keyword flag
        should_filter = False
        if self.contain_keyword:
            # Include mode: filter out if NO keywords found
            should_filter = not keyword_found
        else:
            # Exclude mode: filter out if ANY keyword found
            should_filter = keyword_found

        if should_filter:
            if self.log_filtered:
                if self.contain_keyword:
                    log.info(f"CaptionFilter: excluded sample (no keywords found) - caption: '{caption[:100]}...'")
                else:
                    log.info(
                        f"CaptionFilter: excluded sample due to keyword '{matched_keyword}' - caption: '{caption[:100]}...'"
                    )

            if self.filter_stats:
                self.filtered_samples += 1
            return None

        # Sample passes filter
        return data_dict

    def get_filter_stats(self) -> dict:
        """
        Get filtering statistics.

        Returns:
            Dictionary with filtering statistics
        """
        if not self.filter_stats:
            return {"stats_disabled": True}

        filter_rate = (self.filtered_samples / self.total_samples * 100) if self.total_samples > 0 else 0
        mode_str = "contain" if self.contain_keyword else "exclude"

        return {
            "total_samples": self.total_samples,
            "filtered_samples": self.filtered_samples,
            "passed_samples": self.total_samples - self.filtered_samples,
            "filter_rate_percent": filter_rate,
            "mode": mode_str,
            "keywords": self.keywords,
        }
