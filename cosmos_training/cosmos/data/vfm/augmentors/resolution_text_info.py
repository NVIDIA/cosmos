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

# Default templates for resolution info
DEFAULT_IMAGE_TEMPLATE = "This image is of {height}x{width} resolution."
DEFAULT_VIDEO_TEMPLATE = "This video is of {height}x{width} resolution."


class ResolutionTextInfo(Augmentor):
    """
    Augmentor that appends resolution (height x width) info to captions.

    This augmentor should run AFTER CropToMultiple (which sets final_height/final_width)
    and AFTER text transforms (so ai_caption exists), but BEFORE tokenization.

    Reads resolution from metadata keys (final_height, final_width) set by CropToMultiple.
    Does NOT fall back to tensor shape to avoid incorrect latent dimensions.

    Automatically detects whether the input is an image or video based on which
    key is present in the data_dict, and uses the appropriate template.

    Example:
        Original caption: "A cat playing with a ball"
        Augmented (image): "A cat playing with a ball. This image is 512x512."
        Augmented (video): "A cat playing with a ball. This video is 480x854."

    Args:
        input_keys (list): Input keys (not used, kept for API compatibility)
        output_keys (list): Output keys (not used, kept for API compatibility)
        args (dict): Configuration arguments:
            - caption_key (str): Key for caption in data_dict. Default: "ai_caption"
            - video_key (str): Key for video tensor in data_dict. Default: "video"
            - image_size_key (str): Key for image size tensor in data_dict. Default: "image_size"
            - image_template (str): Format string for image metadata. Default: DEFAULT_IMAGE_TEMPLATE
            - video_template (str): Format string for video metadata. Default: DEFAULT_VIDEO_TEMPLATE
            - separator (str): Separator between caption and metadata. Default: ". "
            - enabled (bool): Whether augmentation is enabled. Default: True
    """

    def __init__(
        self, input_keys: Optional[list] = None, output_keys: Optional[list] = None, args: Optional[dict] = None
    ) -> None:
        super().__init__(input_keys, output_keys, args)

        # Configuration with sensible defaults
        self.caption_key = args.get("caption_key", "ai_caption") if args else "ai_caption"
        self.image_key = args.get("image_key", "images") if args else "images"
        self.video_key = args.get("video_key", "video") if args else "video"
        self.image_size_key = args.get("image_size_key", "image_size") if args else "image_size"
        self.image_template = args.get("image_template", DEFAULT_IMAGE_TEMPLATE) if args else DEFAULT_IMAGE_TEMPLATE
        self.video_template = args.get("video_template", DEFAULT_VIDEO_TEMPLATE) if args else DEFAULT_VIDEO_TEMPLATE
        self.default_separator = args.get("separator", ". ") if args else ". "
        self.enabled = args.get("enabled", True) if args else True

    def __call__(self, data_dict: dict) -> dict | None:
        """
        Append resolution (height x width) as text timestamps to the caption.

        Args:
            data_dict (dict): Input data dict containing caption and image/video tensor

        Returns:
            data_dict (dict): Output dict with augmented caption.
        """
        if not self.enabled:
            return data_dict

        # Get caption - must exist at this point (set by text transforms)
        assert self.caption_key in data_dict, f"caption_key '{self.caption_key}' not found in data_dict."
        caption = data_dict[self.caption_key]

        if (not isinstance(caption, str) and not isinstance(caption, dict)) or caption == "":
            # This is for unconditional case.
            return data_dict

        # Detect image vs video to select template
        is_video = self.video_key in data_dict
        is_image = self.image_key in data_dict

        if isinstance(caption, str):
            # Case 1: Caption is a string. In this case, we create a string template for
            # resolution, aspect ratio info and add it
            if not is_video and not is_image:
                raise ValueError("Neither video_key nor image_key found in data_dict.")

            template = self.video_template if is_video else self.image_template

            # Get dimensions from metadata keys (set by CropToMultiple)
            image_size = data_dict.get(self.image_size_key)
            height = int(image_size[0])
            width = int(image_size[1])

            # Format metadata text
            metadata_text = template.format(height=height, width=width)

            # Choose separator based on whether caption ends with a period
            separator = " " if caption.rstrip().endswith(".") else self.default_separator

            # Update caption
            data_dict[self.caption_key] = caption + separator + metadata_text

        elif isinstance(caption, dict):
            # Case 2: Caption is a dictionary. This is for the json caption case.
            # In this case, we add resolution and aspect ratio in json fields
            aspect_ratio = data_dict["__url__"].meta.opts["aspect_ratio"]
            height = int(data_dict["image_size"][0])
            width = int(data_dict["image_size"][1])
            data_dict[self.caption_key].update(
                {
                    "resolution": {"H": height, "W": width},
                    "aspect_ratio": aspect_ratio,
                }
            )

        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

        return data_dict
