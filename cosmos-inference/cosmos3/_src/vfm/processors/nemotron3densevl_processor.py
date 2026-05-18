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

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize

from cosmos3._src.vfm.processors.base import (
    BaseVLMProcessor,
    convert_string_content_to_list_content,
    maybe_parse_video_content,
)


class Nemotron3DenseVLProcessor(
    BaseVLMProcessor
):
    """Wrapper around the HuggingFace ``AutoProcessor`` for Nemotron3-Dense-VL / Qwen3-2B-ViT."""

    # Nemotron3-Dense / Qwen3-2B-ViT does not expose a single vision-end token;
    # leave ``vision_end_id`` unset (None) rather than the legacy ``</img>``
    # which silently resolved to the UNK token id.
    VISION_END_TOKEN: Optional[str] = None

    def __init__(
        self,
        name: str = "Qwen/Qwen3-VL-2B-Init",
        credentials: str = "./credentials/s3_training.secret",
        bucket: str = "bucket",
        cache_dir: Optional[str] = None,
    ):
        super().__init__(name=name, credentials=credentials, bucket=bucket, cache_dir=cache_dir)
        # Helper attributes consumed by the dataloader video decoding path.
        shortest_edge = self.processor.image_processor.size["shortest_edge"]
        self.min_height_width = int(np.sqrt(shortest_edge))
        self.patch_size = self.processor.video_processor.patch_size
        self.temporal_patch_size = self.processor.video_processor.temporal_patch_size
        self.merge_size = self.processor.video_processor.merge_size
        self.use_smart_resize = True

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=False,
        return_tensors="pt",
        tokenize=True,
        **kwargs,
    ):
        """
        Return:
            inputs: dict
                input_ids: torch.Tensor, shape: (N_token)
                attention_mask: torch.Tensor, shape: (N_token)
                texts: str, the raw text
                image_sizes: torch.Tensor, shape (N_img, 2)
                pixel_values: torch.Tensor, shape (N_img_patch, 3, 224, 224)
        """
        assert tokenize, "tokenize must be True"
        assert return_tensors == "pt", "return_tensors must be pt"
        # Note: this tokenizer does not support "content": str, it always expect "content" entry to be a list of dicts
        messages = convert_string_content_to_list_content(messages)
        kwargs = {}
        # Pre-resize images per-message using smart_resize so the resulting
        # token count matches the configured min/max-pixel budget.
        for message in messages:
            if isinstance(message["content"], list):
                for sub_content in message["content"]:
                    if sub_content.get("type", "") == "image":
                        image = sub_content["image"]
                        max_pixels = sub_content.get("max_pixels", self.processor.image_processor.size["longest_edge"])
                        min_pixels = sub_content.get("min_pixels", self.processor.image_processor.size["shortest_edge"])
                        assert isinstance(image, Image.Image), (
                            "image must be a url string for now, not support list of images for one content"
                        )
                        width, height = image.size
                        resized_height, resized_width = smart_resize(
                            height,
                            width,
                            factor=32,
                            min_pixels=min_pixels,
                            max_pixels=max_pixels,
                        )
                        image = image.resize((resized_width, resized_height))
                        sub_content["image"] = image

        num_video, video_fps, video_total_num_frames, video_frames_indices = maybe_parse_video_content(messages)
        if num_video > 0:
            assert num_video == 1, "only support one video for now"
            fps = video_fps[0]
            total_num_frames = video_total_num_frames[0]
            frames_indices = video_frames_indices[0]
            kwargs.update(
                {
                    "do_sample_frames": False,
                    "video_metadata": dict(fps=fps, total_num_frames=total_num_frames, frames_indices=frames_indices),
                }
            )

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors=return_tensors,
            **kwargs,
        )

        # Convert batch features into single features
        inputs["input_ids"] = inputs["input_ids"][0]  # [N_token]
        inputs["attention_mask"] = inputs["attention_mask"][0]  # [N_token]
        return inputs

    def add_assistant_tokens_mask(self, tokens):
        """
        Add a mask to the assistant tokens.
        This is used to mask out tokens that are not generated by the assistant (e.g.,  system prompts, user prompts, chat templates), such that in the loss computation, only the tokens generated by the assistant are used.
        If there are multiple turns in the conversation, the mask will mask all the assistant tokens in each turn.

        Args:
            tokens (Union[List[int], torch.Tensor]): The tokens to add the mask to.
        Returns:
            Union[List[bool], torch.Tensor]: The mask. True for tokens generated by the assistant (i.e. should apply loss on), False for tokens not generated by the assistant.
        """
        if isinstance(tokens, torch.Tensor) and tokens.ndim == 2:
            mask = torch.stack(
                [self.add_assistant_tokens_mask(tokens[i]) for i in range(tokens.shape[0])]
            )  # [B,N_token]
            assert mask.shape == tokens.shape
            return mask
        np_tokens = tokens.cpu().numpy() if isinstance(tokens, torch.Tensor) else np.array(tokens)
        assert np_tokens.ndim == 1

        # Constants defining bos, eos and fixed offsets.
        BOS_TOKEN = "<|im_start|>"
        EOS_TOKEN = "<|im_end|>"
        ROLE = "assistant"
        # Offsets: skip the bos + "assistant\n" (always 3 tokens) and include the eos (+1) for supervision
        START_OFFSET = 3
        END_OFFSET = 1

        # Retrieve token IDs for the markers and the role.
        bos_token_id = self.processor.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
        eos_token_id = self.processor.tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        role_id = self.processor.tokenizer.convert_tokens_to_ids(ROLE)
        # ``role`` may tokenize into multiple sub-tokens (e.g. Qwen3-2B-ViT
        # splits "assistant"); the multi-token branch below handles that case.
        role_ids = self.processor.tokenizer.encode(ROLE, add_special_tokens=False)
        think_start_id = self.processor.tokenizer.convert_tokens_to_ids("<think>")
        think_end_id = self.processor.tokenizer.convert_tokens_to_ids("</think>")

        # Locate all positions where the start and end markers appear.
        start_indices = np.where(np_tokens == bos_token_id)[0]
        end_indices = np.where(np_tokens == eos_token_id)[0]

        # Initialize the mask with False values.
        masks = np.zeros_like(np_tokens, dtype=bool)
        assert len(start_indices) == len(end_indices)
        # For each pair of bos/eos, check if the role is 'assistant'
        # and apply the mask accordingly.
        for start, end in zip(start_indices, end_indices):
            end_pos = None
            if np_tokens[start + 1] == role_id:
                # Mask tokens from after the assistant header (start+3) to include the end marker (end+1)
                masks[start + START_OFFSET : end + END_OFFSET] = True
                end_pos = start + START_OFFSET
            elif all(np_tokens[start + 1 : start + 1 + len(role_ids)] == role_ids):
                masks[start + START_OFFSET + len(role_ids) - 1 : end + END_OFFSET] = True
                end_pos = start + START_OFFSET + len(role_ids) - 1
            if end_pos is not None and np_tokens[end_pos] == think_start_id:
                masks[end_pos] = False
                if np_tokens[end_pos + 1] == think_end_id:
                    masks[end_pos + 1] = False

        assert masks.shape == np_tokens.shape
        if isinstance(tokens, torch.Tensor):
            return torch.from_numpy(masks)
        else:
            return masks.tolist()
