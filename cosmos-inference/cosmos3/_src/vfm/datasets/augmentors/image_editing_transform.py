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
Augmentors for image editing tasks in the cosmos3 VFM pipeline.

These augmentors process conversation-format image editing data and produce
the output format expected by the main training pipeline:
    - images: List[torch.Tensor] (source + target images as a two-frame "video")
    - image_size: List[torch.Tensor]
    - ai_caption: List[str]
    - selected_caption_type: List[str]
    - fps: List[float]
    - num_frames: List[int]
    - dataset_name: str
    - sequence_plan: SequencePlan
"""

from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.vfm.datasets.sequence_packing import SequencePlan


class ExtractImageEditingConversation(Augmentor):
    """Extract and validate image editing conversation from standard annotation format.

    This augmentor processes the cosmos-interleaved conversation format for image editing:
    - Validates that the conversation has exactly one round (user + assistant)
    - User message must contain at least one image and text instruction
    - Assistant message must contain exactly one image (the edited result)
    - If multi-round conversation is found, only the first round is kept

    Input Format (from data_dict):
        - texts: Dict containing "content" with conversation data
        - mllm_media_list: Dict mapping image keys to PIL images (for understanding)
        - diffusion_media_list: Dict mapping image keys to PIL images (for diffusion/VAE)

    Output Format (added to data_dict):
        - source_image: PIL.Image (the input image for editing)
        - target_image: PIL.Image (the edited output image)
        - editing_instruction: str (the user's editing instruction)
    """

    def __init__(
        self,
        input_keys: list | None = None,
        max_round: int = 1,
        args: dict | None = None,
    ) -> None:
        super().__init__(input_keys or [], None, args)
        self.max_round = max_round

    def __call__(self, data_dict: dict) -> dict | None:
        """Extract image editing conversation.

        Args:
            data_dict: Input data dictionary.

        Returns:
            Updated data_dict with source_image, target_image, editing_instruction,
            or None if the data is invalid.
        """
        # Validate required keys
        for required_key in ["mllm_media_list", "diffusion_media_list", "texts"]:
            if required_key not in data_dict:
                log.warning(
                    f"{required_key} not found in data_dict: {data_dict.get('__key__', 'unknown')}",
                    rank0_only=False,
                )
                return None

        mllm_media_list = data_dict["mllm_media_list"]
        diffusion_media_list = data_dict["diffusion_media_list"]

        # Get conversation content
        try:
            texts_content = data_dict["texts"].get("content")
            if texts_content is None:
                log.warning(
                    f"texts.content is None: {data_dict.get('__key__', 'unknown')}",
                    rank0_only=False,
                )
                return None

            # Handle case where content is a list of conversation options
            if isinstance(texts_content, list) and len(texts_content) > 0:
                if isinstance(texts_content[0], list):
                    # Multiple conversation options, randomly select one
                    selected_conversations = random.choice(texts_content)
                else:
                    selected_conversations = texts_content
            else:
                log.warning(
                    f"Unexpected texts.content format: {data_dict.get('__key__', 'unknown')}",
                    rank0_only=False,
                )
                return None
        except Exception as e:
            log.warning(
                f"Error accessing texts.content: {data_dict.get('__key__', 'unknown')}, {str(e)}",
                rank0_only=False,
            )
            return None

        # For image editing, we only keep the first round (user + assistant)
        # Trim to first round if multiple rounds exist
        if len(selected_conversations) > 2:
            log.warning(
                f"Multi-round conversation found ({len(selected_conversations)} messages), "
                f"keeping only first round: {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            selected_conversations = selected_conversations[:2]

        if len(selected_conversations) < 2:
            log.warning(
                f"Expected at least 2 messages (user + assistant), got {len(selected_conversations)}: "
                f"{data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        # Validate roles: first must be user, second must be assistant
        user_msg = selected_conversations[0]
        assistant_msg = selected_conversations[1]

        if user_msg.get("role") != "user":
            log.warning(
                f"First message role is not 'user': {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        if assistant_msg.get("role") != "assistant":
            log.warning(
                f"Second message role is not 'assistant': {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        # Extract user content: must have at least one image and one text
        user_content = user_msg.get("content", [])
        if isinstance(user_content, str):
            user_content = [{"type": "text", "text": user_content}]

        user_text_parts: list[str] = []
        user_image_key: str | None = None

        for item in user_content:
            if not isinstance(item, dict):
                continue
            content_type = item.get("type")
            if content_type == "text":
                user_text_parts.append(item.get("text", ""))
            elif content_type == "image":
                if user_image_key is None:
                    user_image_key = item.get("image")
                # If multiple user images, we only take the first one

        if user_image_key is None:
            log.warning(
                f"No image found in user message: {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        editing_instruction = " ".join(user_text_parts).strip()
        if not editing_instruction:
            log.warning(
                f"No text instruction found in user message: {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        # Extract assistant content: must have exactly one image
        assistant_content = assistant_msg.get("content", [])
        if isinstance(assistant_content, str):
            log.warning(
                f"Assistant content is text-only (no image): {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        assistant_image_key: str | None = None
        for item in assistant_content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                assistant_image_key = item.get("image")
                break

        if assistant_image_key is None:
            log.warning(
                f"No image found in assistant message: {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        # Validate images exist in media lists
        for media_key in [user_image_key, assistant_image_key]:
            if media_key not in diffusion_media_list:
                log.warning(
                    f"Image {media_key} not found in diffusion_media_list: {data_dict.get('__key__', 'unknown')}",
                    rank0_only=False,
                )
                return None

        # Get PIL images
        source_image = diffusion_media_list[user_image_key]
        target_image = diffusion_media_list[assistant_image_key]

        # Handle video (list of frames) - use first frame
        if isinstance(source_image, list):
            source_image = source_image[0] if source_image else None
        if isinstance(target_image, list):
            target_image = target_image[0] if target_image else None

        if source_image is None or target_image is None:
            log.warning(
                f"Source or target image is None: {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        data_dict["source_image"] = source_image
        data_dict["target_image"] = target_image
        data_dict["editing_instruction"] = editing_instruction

        return data_dict


class ImageEditingToTrainingFormat(Augmentor):
    """Convert extracted image editing data to the training-compatible format.

    This augmentor takes the source image, target image, and editing instruction
    and produces the output format expected by the main training pipeline.

    Images are assumed to have been already resized by an upstream augmentor
    (e.g. ``OmniInterleavedMediaResize``).  This augmentor only normalises the
    PIL images to tensors and assembles the remaining metadata fields.

    Input (from data_dict):
        - source_image: PIL.Image (already resized by upstream augmentor)
        - target_image: PIL.Image (already resized by upstream augmentor)
        - editing_instruction: str

    Output (added to data_dict):
        - images: list[torch.Tensor]  — ``[source (C,H_s,W_s), target (C,H_t,W_t)]``
        - ai_caption: str
        - selected_caption_type: str
        - fps: float
        - num_frames: int
        - sequence_plan: SequencePlan
    """

    def __init__(
        self,
        input_keys: list | None = None,
        mean: float = 0.5,
        std: float = 0.5,
        args: dict | None = None,
    ) -> None:
        super().__init__(input_keys or [], None, args)
        self.mean = mean
        self.std = std

    def _normalize_image(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor (C, H, W)."""
        tensor = transforms_F.to_tensor(image)
        tensor = transforms_F.normalize(tensor, mean=[self.mean] * 3, std=[self.std] * 3)
        return tensor

    def __call__(self, data_dict: dict) -> dict | None:
        """Convert image editing data to training format.

        Args:
            data_dict: Input data dictionary with source_image, target_image, editing_instruction.

        Returns:
            Updated data_dict with training-compatible fields, or None on error.
        """
        source_image: Image.Image = data_dict.get("source_image")
        target_image: Image.Image = data_dict.get("target_image")
        editing_instruction: str = data_dict.get("editing_instruction", "")

        if source_image is None or target_image is None:
            return None

        try:
            # Normalize PIL images to tensors (upstream augmentor already handled resizing)
            source_tensor = self._normalize_image(source_image)  # [C,H_s,W_s]
            target_tensor = self._normalize_image(target_image)  # [C,H_t,W_t]

            # Store as list of tensors for the batch collation.
            # Each image keeps its own spatial size; the model encodes them separately.
            data_dict["images"] = [source_tensor, target_tensor]

            # Set text fields
            data_dict["ai_caption"] = editing_instruction
            data_dict["selected_caption_type"] = "editing_instruction"

            # Set metadata
            data_dict["fps"] = 30.0  # Same as standard image training
            data_dict["num_frames"] = 2  # Source + target = 2 frames
            data_dict["image_size"] = [
                torch.tensor(
                    [source_image.height, source_image.width, source_image.height, source_image.width],
                    dtype=torch.float,
                ),  # [4]
                torch.tensor(
                    [target_image.height, target_image.width, target_image.height, target_image.width],
                    dtype=torch.float,
                ),  # [4]
            ]
            # Set the dataset name if not already present
            if "dataset_name" not in data_dict:
                data_dict["dataset_name"] = "image_editing"

            # Build sequence plan for image editing.
            # The number of vision items per sample (e.g. 2 for source + target) is tracked
            # by GenerationDataClean.num_vision_items_per_sample (set in get_data_and_condition).
            # In pack_input_sequence, all items except the last are fully conditioned;
            # the last item uses condition_frame_indexes_vision ([] = fully generated).
            data_dict["sequence_plan"] = SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=[],  # Target (last item) is fully generated
            )

        except Exception as e:
            log.warning(
                f"Error processing image editing data: {data_dict.get('__key__', 'unknown')}, {str(e)}",
                rank0_only=False,
            )
            return None

        return data_dict


class RemoveKeys(Augmentor):
    """Remove specified keys from the data dictionary.

    This is useful for cleaning up intermediate keys that are not needed
    downstream (e.g. raw PIL images, media lists) so that every remaining
    value is a tensor, number, dict, or list — as required by the dataloader
    collation.

    Args:
        input_keys: Keys to remove from ``data_dict``.
    """

    def __init__(
        self,
        input_keys: list | None = None,
        args: dict | None = None,
    ) -> None:
        super().__init__(input_keys or [], None, args)

    def __call__(self, data_dict: dict) -> dict:
        for key in self.input_keys:
            data_dict.pop(key, None)
        return data_dict
