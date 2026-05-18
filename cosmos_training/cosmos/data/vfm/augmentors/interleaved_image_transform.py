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
Visual transformation augmentors for Omni models.
"""

import math
from typing import Dict, List, Optional

import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.data.imaginaire.webdataset.augmentors.image.misc import obtain_image_size


class ResizeToPaddingDivisor(Augmentor):
    """Resize images so that both width and height are multiples of padding_divisor."""

    def __init__(self, input_keys: list, padding_divisor: int = 16) -> None:
        super().__init__(input_keys)
        self.padding_divisor = padding_divisor

    def __call__(self, data_dict: dict) -> dict:
        """Resize images to the nearest multiple of padding_divisor.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with resized images and metadata
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys

        # Get original image size
        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)

        # Calculate new dimensions as multiples of padding_divisor
        new_w = math.ceil(orig_w / self.padding_divisor) * self.padding_divisor
        new_h = math.ceil(orig_h / self.padding_divisor) * self.padding_divisor

        # Resize images
        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=(new_h, new_w),
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )
            if out_key != inp_key:
                del data_dict[inp_key]

        # Store image size information (new_h, new_w, orig_h, orig_w)
        data_dict["image_size"] = torch.tensor([new_h, new_w, orig_h, orig_w], dtype=torch.float)  # [4]

        return data_dict


class InterleavedMediaResize(Augmentor):
    """Resizes interleaved media content (images and videos) for both diffusion and MLLM models.

    This augmentor processes mixed media content containing both images and videos, creating two
    versions of each media item: one optimized for diffusion models and another for Multimodal
    Large Language Models (MLLMs). It preserves aspect ratios while ensuring dimensions meet
    specific constraints for each model type.

    The resizing process follows these steps:
    1. Maintains aspect ratio while ensuring no side exceeds the maximum allowed length
    2. Adjusts dimensions to be divisible by model-specific padding constants
    3. Uses high-quality LANCZOS resampling for optimal visual quality

    Args:
        input_keys (List, optional): List containing the key to access media content in data_dict.
            Must contain exactly one key. Defaults to ['media_list'].
        max_diffusion_image_side_length (int, optional): Maximum side length for diffusion model
            images. Defaults to 1024.
        max_mllm_image_side_length (int, optional): Maximum side length for MLLM images.
            Defaults to 768.
        diffusion_image_padding_constant (int, optional): Divisor for diffusion model image
            dimensions. Both width and height must be divisible by this value. Defaults to 16.
        mllm_image_padding_constant (int, optional): Divisor for MLLM image dimensions.
            Both width and height must be divisible by this value. Defaults to 28.
        use_center_crop (bool, optional): If True, uses center cropping to ensure dimensions
            are divisible by padding constants, avoiding distortion. If False, uses resizing
            which may cause slight distortion. Defaults to False.
        args (Optional[dict], optional): Additional arguments passed to parent class.
            Defaults to None.

    Input Format:
        The data_dict should contain a key (specified in input_keys) with value structured as:
        {
            "image_0": PIL.Image,           # Single image
            "image_1": PIL.Image,           # Another single image
            "video_0": List[PIL.Image],     # Video as list of frames
            "video_1": List[PIL.Image],     # Another video
            ...
        }

    Output Format:
        The method adds two new keys to data_dict:
        - 'diffusion_media_content': Resized media for diffusion models
        - 'mllm_media_content': Resized media for MLLMs

        Both follow the same structure as the input, with resized versions of each media item.

    Example:
        >>> # Using resize (default, may cause slight distortion)
        >>> augmentor = OmniInterleavedMediaResize(
        ...     input_keys=['media_list'],
        ...     max_diffusion_image_side_length=1024,
        ...     max_mllm_image_side_length=768
        ... )
        >>>
        >>> # Using center crop (no distortion)
        >>> augmentor_crop = OmniInterleavedMediaResize(
        ...     input_keys=['media_list'],
        ...     max_diffusion_image_side_length=1024,
        ...     max_mllm_image_side_length=768,
        ...     use_center_crop=True
        ... )
        >>>
        >>> data_dict = {
        ...     'media_list': {
        ...         'image_0': pil_image,
        ...         'video_0': [frame1, frame2, frame3]
        ...     }
        ... }
        >>> result = augmentor(data_dict)
        >>> # result now contains 'diffusion_media_content' and 'mllm_media_content'

    Note:
        - Images are only scaled down, never up, to preserve quality
        - Videos are processed frame by frame, maintaining temporal consistency
        - Unsupported media types will raise a ValueError
        - When use_center_crop=True, images are center-cropped to achieve padding divisibility
          without distortion. When False, images are resized which may cause slight distortion.
    """

    def __init__(
        self,
        input_keys: List = ["media_list"],
        max_diffusion_image_side_length: int = 1024,
        max_mllm_image_side_length: int = 768,
        diffusion_image_padding_constant: int = 16,
        use_center_crop: bool = False,
        args: Optional[dict] = None,
    ) -> None:
        super().__init__(input_keys, None, args)
        self.max_diffusion_image_side_length = max_diffusion_image_side_length
        self.max_mllm_image_side_length = max_mllm_image_side_length
        self.diffusion_image_padding_constant = diffusion_image_padding_constant
        self.use_center_crop = use_center_crop

    def __call__(self, data_dict: Dict) -> Dict:
        assert len(self.input_keys) == 1, (
            "This transform only supports one input key. Try to organize all the media contents under one key."
        )
        if self.input_keys[0] not in data_dict:
            print(f"Input key {self.input_keys[0]} not found in data_dict: {data_dict['__key__']}")
            return None
        original_media_content = data_dict[self.input_keys[0]]

        diffusion_media_content = {}
        mllm_media_content = {}

        for key, media in original_media_content.items():
            # Check if it's an image or video
            if isinstance(media, Image.Image):
                # Process single image
                diffusion_media_content[key] = self._resize_image(
                    media,
                    self.max_diffusion_image_side_length,
                    self.diffusion_image_padding_constant,
                    self.use_center_crop,
                )
                mllm_media_content[key] = self._resize_image(
                    media,
                    self.max_mllm_image_side_length,
                    None,  # we don't need to resize the mllm media content to a specific padding constant since it will be handled by the processor
                    self.use_center_crop,
                )
            elif isinstance(media, list) and all(isinstance(frame, Image.Image) for frame in media):
                # Process video (list of images)
                diffusion_media_content[key] = [
                    self._resize_image(
                        frame,
                        self.max_diffusion_image_side_length,
                        self.diffusion_image_padding_constant,
                        self.use_center_crop,
                    )
                    for frame in media
                ]
                mllm_media_content[key] = [
                    self._resize_image(
                        frame,
                        self.max_mllm_image_side_length,
                        None,  # we don't need to resize the mllm media content to a specific padding constant since it will be handled by the processor
                        self.use_center_crop,
                    )
                    for frame in media
                ]
            else:
                raise ValueError(f"Unsupported media type for key {key}: {type(media)}")

        # Add the resized media content to data_dict
        data_dict["diffusion_media_list"] = diffusion_media_content
        data_dict["mllm_media_list"] = mllm_media_content

        return data_dict

    def _resize_image(
        self, image: Image.Image, max_side_length: int, padding_divisor=None, use_center_crop: bool = False
    ) -> Image.Image:
        """Resize image while preserving aspect ratio and ensuring dimensions are divisible by padding_divisor.

        Args:
            image: Input PIL Image
            max_side_length: Maximum allowed side length
            padding_divisor: Both dimensions must be divisible by this value
            use_center_crop: If True, use center crop to achieve divisibility; if False, use resize

        Returns:
            Resized PIL Image
        """
        # Get original dimensions
        width, height = image.size

        # Calculate scale factor to ensure max side length constraint
        scale_factor = min(max_side_length / width, max_side_length / height)

        # Only scale down, not up
        if scale_factor < 1.0:
            new_width = max(1, int(width * scale_factor))
            new_height = max(1, int(height * scale_factor))
        else:
            new_width = width
            new_height = height

        # Resize image to maintain aspect ratio
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Calculate target dimensions that are divisible by padding_divisor
        if padding_divisor is not None:
            final_width = max(1, (new_width // padding_divisor)) * padding_divisor
            final_height = max(1, (new_height // padding_divisor)) * padding_divisor
        else:
            final_width = new_width
            final_height = new_height

        # If dimensions need adjustment
        if final_width != new_width or final_height != new_height:
            if use_center_crop:
                # Use center crop to achieve target dimensions
                left = (new_width - final_width) // 2
                top = (new_height - final_height) // 2
                right = left + final_width
                bottom = top + final_height
                resized_image = resized_image.crop((left, top, right, bottom))
            else:
                # Use resize (may cause distortion)
                resized_image = resized_image.resize((final_width, final_height), Image.Resampling.LANCZOS)

        return resized_image
