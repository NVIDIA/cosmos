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

"""Visual-Text Transformations or Augmentations."""

import re
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.utils import log
from cosmos.data.vfm.vlm.video_decoder_qwen import token_to_pixels
from cosmos.data.vfm.processors.qwen3vl_processor import Qwen3VLProcessor as Processor
from cosmos.utils.vlm.constant import IGNORE_INDEX, PROCESSOR_KEYS_TO_ADD


def maybe_subsample_frames(model_name_or_path, list_of_pil_image, max_video_token_length, processor):
    """
    Why do we need to subsample frames? For model like eagle_er, it does not support smart downsampling in the processor.
    And all the frames are resized to the same size. There are 2 senerios the context length can easily exceed the limit.
    1: the video has >32 frames, it will create 256*32=8192 tokens which exceeds the limit.
    2: there are multiple images, by default, each image will be tiled into (at most) 13 tiles. Each tile is 256 tokens.
    So if there are multiple images, or many frames in the video, we need to subsample the frames to shorten the context length.
    """
    if "Qwen/Qwen2.5-VL" in model_name_or_path:
        return list_of_pil_image
    elif "eagle_er" in model_name_or_path or "InternVL3_5" in model_name_or_path:
        tokens_per_tile = processor.tokens_per_tile
        # 1 frames map to 256 tokens
        estimated_num_frames = max_video_token_length // tokens_per_tile
        if len(list_of_pil_image) > estimated_num_frames:
            # Evenly sample frames
            sample_idx = np.linspace(0, len(list_of_pil_image) - 1, estimated_num_frames).astype(int)
            return [list_of_pil_image[i] for i in sample_idx]
        else:
            return list_of_pil_image
    else:
        return list_of_pil_image


def convert_all_images_to_rgb(conversation):
    """
    Convert all images to RGB. Otherwise the tokenizer will raise error for image in LA mode.
    """
    new_conversation = []
    for conversation_round in conversation:
        if isinstance(conversation_round["content"], list):
            new_content_list = []
            for content in conversation_round["content"]:
                if "type" not in content:
                    log.critical(
                        f"content: {content} | conversation_round: {conversation_round} | full conversation: {conversation}"
                    )
                    content = {"type": "text", "text": content}
                content_type = content["type"]
                if content_type in ["image", "video"]:
                    if isinstance(content[content_type], Image.Image):
                        content[content_type] = content[content_type].convert("RGB")
                    elif isinstance(content[content_type], list):
                        content_i = content[content_type]
                        new_content_i = []
                        for img in content_i:
                            if isinstance(img, Image.Image):
                                img = img.convert("RGB")
                            new_content_i.append(img)
                        content[content_type] = new_content_i
                new_content_list.append(content)
            conversation_round["content"] = new_content_list
        new_conversation.append(conversation_round)

    return new_conversation


def compress_repeated_tokens(dialog_str):
    pattern = re.compile(r"((<\|[^|]+\|>|<｜[^<>]+｜>|\[[^\]]+\]))\1+")

    def replacer(match):
        token = match.group(1)
        count = len(match.group(0)) // len(token)
        return f"{token}*{count}times"

    # Cap length to avoid regex hang on very long decoded sequences
    max_len = 16 * 1024
    if len(dialog_str) > max_len:
        dialog_str = dialog_str[:max_len] + "...[truncated]"
    return pattern.sub(replacer, dialog_str)


class TokenizeData(Augmentor):
    """
    Image-Text Transform for Supervised Fine-Tuning (SFT) data, for Vision-Language Model training.
    """

    def __init__(
        self,
        processor: Optional[Processor] = None,
        max_video_token_length: int = 8192,
        max_image_token_length: int = 8192,
        add_system_prompt_if_missing: bool = False,
        text_only: bool = False,
    ) -> None:
        """
        Args:
            processor (Processor): Text/Image processor for tokenization.
            max_video_token_length (int): Maximum number of video tokens to use. Defaults to 8192.
        """
        # Create the tokenizer
        self.text_only = text_only
        self.processor = processor  # Expecting a ImageTextTokenizer
        self.max_video_token_length = max_video_token_length
        self.max_image_token_length = max_image_token_length
        self.add_system_prompt_if_missing = add_system_prompt_if_missing

    def __call__(self, data_dict: Dict) -> Dict:
        r"""Tokenize a dialog and pad the sequence.

        "media" is a dict of
        {
            "video_1": {"video": [PIL.Image.Image, ...], "fps": int},
            "image_1": PIL.Image.Image,
        }

        "conversation" is a list of dicts, each dict has the following fields:
        {
            "role": "user" or "assistant",
            "content": [
                {"type": "video", "video": media_key_in_media_dict},
                {"type": "image", "image": media_key_in_media_dict},
                {"type": "text", "text": str},
            ],
        }
        or
        {
            "role": "user" or "assistant",
            "content": str,
        }

        Args:
            data_dict (dict): Input data dict

        Returns:
            data_dict (dict): Output dict
        """
        conversation = data_dict["conversation"]
        processor_kwargs = {}
        total_images = 0
        total_videos = 0
        raw_images = []
        # Pre-compute the total_images and total_videos
        for message in conversation:
            if not isinstance(message, dict):
                raise ValueError(
                    f"message is not a dict: {message} | conversation: {conversation} | data_dict: {data_dict} | __url__: {data_dict['__url__'].root}, {data_dict['__url__'].path}"
                )
            if message["role"] == "user" and isinstance(message["content"], list):
                total_images += len([content for content in message["content"] if content["type"] == "image"])
                total_videos += len([content for content in message["content"] if content["type"] == "video"])

        assert total_videos == 1 or total_videos == 0, "Only one video is supported for now"

        # url
        url = data_dict["__url__"].root + "/" + data_dict["__url__"].path

        # go through each message in the conversation
        for message in conversation:
            # for user message, we insert the media

            if message["role"] == "user" and isinstance(
                message["content"], list
            ):  # Otherwise it's text and content is a string
                images_content_idx_full = [
                    content_idx for content_idx, content in enumerate(message["content"]) if content["type"] == "image"
                ]
                images_content_idx_subsampled = maybe_subsample_frames(
                    self.processor.name, images_content_idx_full, self.max_image_token_length, self.processor
                )
                if (
                    len(images_content_idx_subsampled) > 0
                ):  # for eagle, we need to reduce the max_dynamic_tiles and not use thumbnail. These args only valid for eagle_er processor.
                    processor_kwargs["max_dynamic_tiles"] = 1
                    processor_kwargs["use_thumbnail"] = False

                new_content_list = []
                for content_idx, content in enumerate(message["content"]):
                    if content["type"] == "image":
                        if content_idx not in images_content_idx_subsampled:
                            continue
                        # for image, we do NOT use the temporal patch size, this leads to a smaller max_pixels
                        # Later, each image will be repeated temporal_patch_size times
                        max_total_pixels = token_to_pixels(
                            self.max_image_token_length,
                            patch_size=self.processor.patch_size,
                            temporal_patch_size=1,  # Because this is image, not video
                        )
                        max_pixels_per_image = max_total_pixels // total_images

                        if self.processor.use_smart_resize:
                            min_pixels_per_image = self.processor.processor.image_processor.size["shortest_edge"]
                            if max_pixels_per_image < min_pixels_per_image:
                                log.critical(
                                    f"max_pixels_per_image: {max_pixels_per_image} < min_pixels_per_image: {min_pixels_per_image} | self.max_video_token_length = {self.max_video_token_length} is not enough for total_images: {total_images}, as the default min_pixels is {min_pixels_per_image} | Either increase max_video_token_length or include max_pixels in the content or reduce min_pixels"
                                )
                                return None

                        # Add each image to the content list
                        if "media" not in data_dict:
                            log.critical(
                                f"[TokenizerDataError]media not found in data_dict, available keys: {data_dict.keys()}. url: {url}, content: {message['content']}",
                                rank0_only=False,
                            )
                            return None

                        elif content["image"] not in data_dict["media"]:
                            log.critical(
                                f"[TokenizerDataError]image {content['image']} not found in media, available keys: {data_dict['media'].keys()}. url: {url}",
                                rank0_only=False,
                            )
                            return None
                        image = data_dict["media"][content["image"]]
                        content["image"] = image
                        content["max_pixels"] = max_pixels_per_image
                        raw_images.append(image)

                    elif content["type"] == "video":

                        # as tokenization will NOT upsample the video, we can use a larger value here at the cost of multiple video having 1.5x token length
                        max_total_pixels = token_to_pixels(self.max_video_token_length * 1.5, temporal_patch_size=2)
                        media_key = content["video"]
                        # Add each video to the content list
                        if "media" not in data_dict:
                            log.critical(
                                f"[TokenizerDataError]media not found in data_dict, available keys: {data_dict.keys()}. url: {url}, content: {message['content']}",
                                rank0_only=False,
                            )
                            return None
                        if media_key not in data_dict["media"]:
                            log.info(
                                f"[TokenizerDataError]video {media_key} not found in media, available keys: {data_dict['media'].keys()}. url: {url}"
                            )
                            return None
                        if "videos" not in data_dict["media"][media_key]:
                            log.info(
                                f"[TokenizerDataError]videos not found in media[{media_key}], available keys: {data_dict['media'][media_key].keys()}. url: {url}"
                            )
                            return None
                        videos = data_dict["media"][media_key]["videos"]  # list of PIL images
                        fps = data_dict["media"][media_key]["fps"]

                        # this is because videos are decoded to be around "max_video_token_length" tokens

                        videos = maybe_subsample_frames(
                            self.processor.name, videos, self.max_video_token_length, self.processor
                        )
                        content["video"] = videos

                        max_pixels_per_image = max_total_pixels // total_videos // len(videos)
                        content["fps"] = fps
                        content["max_pixels"] = max_pixels_per_image

                        data_dict["raw_video"] = torch.from_numpy(np.array(videos)).permute(
                            3, 0, 1, 2
                        )  # [3,T,H,W], range [0, 255]
                    new_content_list.append(content)
                message["content"] = new_content_list

        if len(raw_images) > 1:
            # resize the raw_image to the size of the first image
            image_size = raw_images[0].size
            raw_images = [image.resize(image_size) for image in raw_images]

        if len(raw_images) > 0:
            data_dict["raw_image"] = torch.from_numpy(np.array(raw_images)).permute(3, 0, 1, 2)  # [3,num_images,H,W]

        if conversation[0]["role"] != "system" and self.add_system_prompt_if_missing:
            conversation.insert(0, {"role": "system", "content": "You are a helpful assistant."})

        if self.text_only and (total_images > 0 or total_videos > 0):
            log.critical(
                f"Images or videos found in the conversation but expect only text, __url__: {url} | data_dict: {data_dict.keys()} | conversation={conversation}"
            )
            return None

        if total_images > 1 or total_videos > 1:
            add_vision_id = True
        else:
            add_vision_id = False

        try:
            conversation = convert_all_images_to_rgb(conversation)
        except Exception as e:
            log.critical(
                f"Error in convert_all_images_to_rgb: {e} | conversation: {conversation} | __url__: {url} | data_dict: {data_dict.keys()}"
            )
            return None

        try:
            tokenizer_output = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=add_vision_id,
                **processor_kwargs,
            )
        except Exception as e:
            log.critical(
                f"Error in tokenizer_output: {e} | conversation: {conversation} | __url__: {url} | data_dict: {data_dict.keys()}"
            )
            return None
        input_ids = tokenizer_output["input_ids"]
        if "image_grid_thw" in tokenizer_output and "raw_image" in data_dict:
            # image_grid_thw: (1, t, h, w)
            t, h, w = tokenizer_output["image_grid_thw"][0]
            # interpolate raw_image to the size of the image grid * 14
            data_dict["raw_image"] = torch.nn.functional.interpolate(
                data_dict["raw_image"], size=(h * 14, w * 14), mode="bilinear", align_corners=False
            )  # [3,num_images,h*14,w*14]

        try:
            # token_mask: True for tokens to compute loss on; False for tokens to ignore
            token_mask = self.processor.add_assistant_tokens_mask(input_ids)
        except Exception as e:
            log.critical(
                f"Error in add_assistant_tokens_mask: {e} | conversation: {conversation} | __url__: {url} | data_dict: {data_dict.keys()}"
            )
            return None

        input_ids = torch.LongTensor(input_ids)  # [N_token]
        token_mask = torch.BoolTensor(token_mask)  # [N_token]; True = compute loss on this token

        data_dict.update(
            {
                "input_ids": input_ids,
                "token_mask": token_mask,
            }
        )
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in tokenizer_output:
                data_dict[key] = tokenizer_output[key]
        labels = tokenizer_output["input_ids"].clone()  # [N_token]
        labels[~token_mask] = IGNORE_INDEX
        data_dict["labels"] = labels
        data_dict["pad_token_id"] = self.processor.pad_id
        data_dict["ignore_index"] = IGNORE_INDEX

        # keep raw text for debugging/logging purpose. Add \n\n after each <|im_end|>.
        dialog_str = self.processor.decode(input_ids)
        data_dict["dialog_str"] = compress_repeated_tokens(dialog_str.replace("<|im_end|>", "<|im_end|>\n\n"))

        # For debugging purpose
        msg = f"input_ids: {input_ids.shape[-1]} | __url__: {data_dict['__url__'].root}, {data_dict['__url__'].path} | __key__: {data_dict['__key__']}"
        if "raw_video" in data_dict:
            msg += f" | raw_video: {data_dict['raw_video'].shape} "
        if "raw_image" in data_dict:
            msg += f" | raw_image: {data_dict['raw_image'].shape} "
        if "pixel_values" in data_dict:
            msg += f" | pixel_values: {data_dict['pixel_values'].shape} "

        msg += f"original conversation: {data_dict['conversation']}"

        return data_dict
