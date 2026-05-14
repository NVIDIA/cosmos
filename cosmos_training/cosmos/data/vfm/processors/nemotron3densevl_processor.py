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

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize
from transformers.models.auto.processing_auto import AutoProcessor

from cosmos.utils import log
from cosmos.data.vfm.sequence_packing import add_special_tokens
from cosmos.model.vfm.vlm.qwen3_vl.utils import tokenize_caption
from cosmos.utils.vfm.vlm.pretrained_models_downloader import maybe_download_hf_model_from_s3


def convert_string_content_to_list_content(messages: List[Dict]) -> List[Dict]:
    """
    Convert the string content to a list of dicts.
    """
    for message_id, message in enumerate(messages):
        if isinstance(message["content"], str):
            messages[message_id]["content"] = [{"type": "text", "text": message["content"]}]
    return messages


def maybe_parse_video_content(
    messages: List[Dict],
) -> tuple[int, Optional[list[float]], Optional[list[int]], Optional[list[list[int]]]]:
    """
    Convert the string content to a list of dicts.
    """
    num_video = 0
    video_fps = []
    video_total_num_frames = []
    video_frames_indices = []
    for message_id, message in enumerate(messages):
        if isinstance(message["content"], list):
            for sub_content in message["content"]:
                if sub_content.get("type", "") == "video" and isinstance(sub_content["video"], list):
                    num_video += 1
                    fps = sub_content.get("fps", None)
                    if fps is None:
                        log.critical(
                            f"fps is None for video {sub_content}. Better to set the fps explicitly", rank0_only=False
                        )
                    video_fps.append(fps)
                    video_total_num_frames.append(len(sub_content["video"]))
                    video_frames_indices.append(list(range(video_total_num_frames[-1])))
    return num_video, video_fps, video_total_num_frames, video_frames_indices


class Nemotron3DenseVLProcessor:
    # This is a wrapper around the AutoProcessor class to add some helper functions
    is_unified_processor: bool = True  # sentinel for TextTokenizerTransform duck-type check

    def __init__(
        self,
        name="Qwen/Qwen3-VL-2B-Init",
        credentials: str = "./credentials/s3_training.secret",
        bucket: str = "checkpoints-us-east-1",
        cache_dir: str = None,
    ):
        self.name = name
        if os.path.isdir(name):
            model_name_or_path_local = name
        else:
            model_name_or_path_local = maybe_download_hf_model_from_s3(
                name, credentials, bucket, include_model_weights=False
            )

        self.processor = AutoProcessor.from_pretrained(model_name_or_path_local, trust_remote_code=True)
        log.info("Successfully loaded processor from local cache")
        add_special_tokens(self.processor.tokenizer)

        if hasattr(self.processor, "image_token"):
            self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        else:
            self.image_token_id = None
        if hasattr(self.processor, "video_token"):
            self.video_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.video_token)
        else:
            self.video_token_id = None
        self.eos_id = self.processor.tokenizer.eos_token_id
        self.pad_id = self.processor.tokenizer.pad_token_id
        self.vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("</img>")

        # Helper attributes for the dataloader video decoding function
        self.shortest_edge = self.processor.image_processor.size["shortest_edge"]
        self.min_height_width = int(np.sqrt(self.shortest_edge))
        self.patch_size = self.processor.video_processor.patch_size
        self.temporal_patch_size = self.processor.video_processor.temporal_patch_size
        self.merge_size = self.processor.video_processor.merge_size
        self.use_smart_resize = True
        if self.pad_id is None:
            self.pad_id = self.eos_id

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

        # messages = [msg for msg in messages if msg.get("role") != "system"]
        assert tokenize, "tokenize must be True"
        assert return_tensors == "pt", "return_tensors must be pt"
        # Note: this tokenizer does not support "content": str, it always expect "content" entry to be a list of dicts
        messages = convert_string_content_to_list_content(messages)
        kwargs = {}
        for message_id, message in enumerate(messages):
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
            # Here we add the args to avoid the error:
            # File "/usr/local/lib/python3.12/dist-packages/transformers/video_processing_utils.py", line 321, in _decode_and_sample_videos
            #     raise ValueError(
            # ValueError: Sampling frames from a list of images is not supported! Set `do_sample_frames=False`.
            # kwargs["videos_kwargs"] = dict(do_sample_frames=False)
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
            # padding="max_length",
            # max_length=16000,
            # truncation=False,
            **kwargs,
        )

        # Convert batch features into single features
        # By default, the processor returns a batch of features, but we use processor in dataloader, so we need to convert it to single features
        inputs["input_ids"] = inputs["input_ids"][0]  # [N_token]
        inputs["attention_mask"] = inputs["attention_mask"][0]  # [N_token]
        num_image_tokens = inputs["input_ids"] == self.image_token_id  # [N_token]
        num_video_tokens = inputs["input_ids"] == self.video_token_id  # [N_token]
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
        role_ids = self.processor.tokenizer.encode(
            ROLE, add_special_tokens=False
        )  # In case the role_id corresponds to multiple tokens, decode it back to string for accurate comparison
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

    def tokenize_text(
        self,
        caption: str,
        is_video: bool = False,
        use_system_prompt: bool = False,
        system_prompt: Optional[str] = None,
    ) -> list[int]:
        """Tokenize a text caption using the underlying tokenizer.

        Delegates to tokenize_caption so VFM diffusion augmentors and VLM dataloaders
        share the same tokenization logic through a single processor object.
        """
        return tokenize_caption(
            caption,
            self.processor.tokenizer,
            is_video=is_video,
            use_system_prompt=use_system_prompt,
            system_prompt=system_prompt,
        )

    def encode(self, *args, **kwargs):
        return self.processor.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.processor.decode(*args, **kwargs)
