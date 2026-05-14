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

import torch

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.utils import log

# Global template for duration and FPS text timestamps
DEFAULT_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."


class DurationFPSTextTimeStamps(Augmentor):
    """
    Augmentor that appends video duration and FPS as text timestamps to captions.

    This augmentor should run AFTER TextTransformForVideo to append metadata
    to the already-selected caption in data_dict["ai_caption"].

    IMPORTANT: Reads num_frames from the actual video tensor shape to get the
    FINAL frame count after all video processing (subsampling, etc.) is complete.

    Example:
        Original caption: "A cat playing with a ball"
        Augmented caption: "A cat playing with a ball. The video is 1.4 seconds long and is of 24 FPS"

    Args:
        input_keys (list): Input keys (not used, kept for API compatibility)
        output_keys (list): Output keys (not used, kept for API compatibility)
        args (dict): Configuration arguments:
            - caption_key (str): Key for caption in data_dict. Default: "ai_caption"
            - video_key (str): Key for video tensor in data_dict. Default: "video"
            - fps_key (str): Key for FPS value in data_dict. Default: "conditioning_fps"
            - template (str): Format string for metadata text. Default: DEFAULT_TEMPLATE constant
            - separator (str): Separator between caption and metadata. Default: ". "
            - enabled (bool): Whether augmentation is enabled. Default: True
            - skip_on_error (bool): If True, skip on errors and return original data_dict. If False, return None. Default: True
            - num_multiplier_key (str): Key for num_multiplier value in data_dict. Default: "num_multiplier"
    """

    def __init__(
        self, input_keys: Optional[list] = None, output_keys: Optional[list] = None, args: Optional[dict] = None
    ) -> None:
        super().__init__(input_keys, output_keys, args)

        # Configuration with sensible defaults
        self.caption_key = args.get("caption_key", "ai_caption") if args else "ai_caption"
        self.video_key = args.get("video_key", "video") if args else "video"
        self.fps_key = args.get("fps_key", "conditioning_fps") if args else "conditioning_fps"
        self.template = args.get("template", DEFAULT_TEMPLATE) if args else DEFAULT_TEMPLATE
        self.default_separator = args.get("separator", ". ") if args else ". "
        self.enabled = args.get("enabled", True) if args else True
        self.skip_on_error = args.get("skip_on_error", True) if args else True
        self.num_multiplier_key = args.get("num_multiplier_key", "num_multiplier") if args else "num_multiplier"

    def __call__(self, data_dict: dict) -> dict | None:
        """
        Append video duration and FPS as text timestamps to the caption.

        Args:
            data_dict (dict): Input data dict containing caption, fps, and video tensor

        Returns:
            data_dict (dict): Output dict with augmented caption, or None if error and skip_on_error=False
        """
        if not self.enabled:
            return data_dict
        # Get caption - must exist at this point (set by TextTransformForVideo)
        if self.caption_key not in data_dict:
            if self.skip_on_error:
                log.warning(
                    f"DurationFPSTextTimeStamps: '{self.caption_key}' not found in data_dict. Skipping.",
                    rank0_only=False,
                )
                return data_dict
            else:
                return None
        caption = data_dict[self.caption_key]
        if (not isinstance(caption, str) and not isinstance(caption, dict)) or caption == "":
            if self.skip_on_error:
                return data_dict
            else:
                return None

        # Use pre-calculated conditioning_fps from VideoParsing augmentor
        # This already accounts for frame skipping (fps / num_multiplier)
        fps_value = data_dict[self.fps_key]
        if isinstance(fps_value, torch.Tensor):
            fps = fps_value.item() if fps_value.numel() == 1 else fps_value[0].item()
        else:
            fps = float(fps_value)

        # Extract ACTUAL number of frames from the video tensor shape
        # This is critical - we need the final frame count after all processing
        video = data_dict[self.video_key]

        # Video shape is (C, T, H, W)
        num_frames = video.shape[1]

        # Compute duration and append to caption
        if fps > 0:
            duration = int(num_frames / fps)
            if isinstance(caption, str):
                # Case 1: Caption is a string (existing behavior).
                metadata_text = self.template.format(duration=duration, fps=fps)

                # Choose separator based on whether caption ends with a period
                separator = " " if caption.rstrip().endswith(".") else self.default_separator

                # Update caption text
                data_dict[self.caption_key] = caption + separator + metadata_text
            elif isinstance(caption, dict):
                # Case 2: Caption is JSON. Add structured duration/FPS fields.
                data_dict[self.caption_key].update(
                    {
                        "duration": str(duration) + "s",
                        "fps": fps,
                    }
                )

        return data_dict
