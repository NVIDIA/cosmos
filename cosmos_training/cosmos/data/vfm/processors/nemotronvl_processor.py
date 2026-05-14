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
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.processing_utils import VideosKwargs
from transformers.video_utils import VideoMetadata

from cosmos.utils import log
from cosmos.data.vfm.sequence_packing import add_special_tokens
from cosmos.model.vfm.vlm.qwen3_vl.utils import tokenize_caption
from cosmos.utils.vfm.vlm.pretrained_models_downloader import maybe_download_hf_model_from_s3

nemotron_chat_template = """
{%- set ns = namespace(enable_thinking=false, has_sys_prompt=false, non_tool_system_content='', has_video=false, explicit_think_requested=false) -%}
{%- set msg = namespace(content='') -%}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {%- set ns.has_sys_prompt = true -%}
        {# Extract system content without tool flags #}
        {%- if message['content'] is string -%}
            {%- set ns.non_tool_system_content = message['content'].replace('</think>', '<_end_think>').replace('/think', '').replace('/no_think', '').replace('<_end_think>', '</think>').strip() -%}
        {%- else -%}
            {%- set ns.non_tool_system_content = '' -%}
            {%- for content in message['content'] -%}
                {%- if content['type'] == 'text' -%}
                    {%- set ns.non_tool_system_content = ns.non_tool_system_content + content['text'].replace('</think>', '<_end_think>').replace('/think', '').replace('/no_think', '').replace('<_end_think>', '</think>') -%}
                {%- endif -%}
            {%- endfor -%}
            {%- set ns.non_tool_system_content = ns.non_tool_system_content.strip() -%}
        {%- endif -%}
    {%- endif -%}
    {# Check for video content in all messages #}
    {%- if message['content'] is not string -%}
        {%- for content in message['content'] -%}
            {%- if content['type'] == 'video' or content['type'] == 'video_url' -%}
                {%- set ns.has_video = true -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {%- if message['content'] is string -%}
        {%- if message['role'] == 'user' or message['role'] == 'system' -%}
            {%- if '/think' in message['content'].replace('</think>', '') -%}
                {%- set ns.enable_thinking = true -%}
                {%- set ns.explicit_think_requested = true -%}
            {%- elif '/no_think' in message['content'] -%}
                {%- set ns.enable_thinking = false -%}
            {%- endif -%}
        {%- endif -%}
    {%- else -%}
        {%- for content in message['content'] -%}
            {%- if content['type'] == 'text' -%}
                {%- if message['role'] == 'user' or message['role'] == 'system' -%}
                    {%- if '/think' in content['text'].replace('</think>', '') -%}
                        {%- set ns.enable_thinking = true -%}
                        {%- set ns.explicit_think_requested = true -%}
                    {%- elif '/no_think' in content['text'] -%}
                        {%- set ns.enable_thinking = false -%}
                    {%- endif -%}
                {%- endif -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endfor -%}

{{- bos_token -}}
{%- if messages[0]['role'] != 'system' -%}
    {{- '<SPECIAL_10>System\n' -}}
{%- else -%}
    {{- '<SPECIAL_10>System\n' + ns.non_tool_system_content }}
{%- endif -%}

{%- if tools -%}
    {%- if ns.non_tool_system_content != '' -%}
        {{- '\n\n' -}}
    {%- endif -%}
    {{- 'You can use the following tools to assist the user if required:\n' -}}
    {{- '<AVAILABLE_TOOLS>[' -}}
    {%- for tool in tools -%}
        {{- (tool.function if tool.function is defined else tool) | tojson -}}
        {{- ', ' if not loop.last else '' -}}
    {%- endfor -%}
    {{- ']</AVAILABLE_TOOLS>\n\n' -}}
    
    {{- 'If you decide to call any tool(s), use the following format:\n' -}}
    {{- '<TOOLCALL>[{"name": "tool_name1", "arguments": "tool_args1"}, ' -}}
    {{- '{"name": "tool_name2", "arguments": "tool_args2"}]</TOOLCALL>\n\n' -}}
    
    {{- 'The user will execute tool-calls and return responses from tool(s) in this format:\n' -}}
    {{- '<TOOL_RESPONSE>[{"response": "tool_response1"}, ' -}}
    {{- '{"response": "tool_response2"}]</TOOL_RESPONSE>\n\n' -}}
    
    {{- 'Based on the tool responses, you can call additional tools if needed, ' -}}
    {{- 'correct tool calls if any errors are found, or just respond to the user.' -}}
{%- endif -%}
{{- '\n' -}}

{%- set messages = messages[1:] if messages[0]['role'] == 'system' else messages -%}

{# Prevent no user or assistant message #}
{%- if messages|length == 0 -%}
    {%- set messages = [{'role': 'user', 'content': ''}] -%}
{%- endif -%}

{%- for message in messages %}
    {%- if message['content'] is string -%}
        {%- set msg.content = message['content'].replace('</think>', '<_end_think>').replace('/think', '').replace('/no_think', '').replace('<_end_think>', '</think>').strip() -%}
    {%- else -%}
        {%- set msg.content = '' -%}
        {%- set mm_content = '' -%}
        {%- set counters = namespace(images=0, videos=0) -%}

        {%- for content in message['content'] -%}
            {%- if content['type'] == 'image' -%}
                {%- set counters.images = counters.images + 1 -%}
            {%- elif content['type'] == 'video' -%}
                {%- set counters.videos = counters.videos + 1 -%}
            {%- elif content['type'] == 'text' -%}
                {%- set msg.content = msg.content + content['text'] -%}
            {%- endif -%}
        {%- endfor -%}
        {%- if '<image>' in msg.content -%}
            {%- set counters.images = 0 -%}
        {%- endif -%}
        {%- if '<video>' in msg.content -%}
            {%- set counters.videos = 0 -%}
        {%- endif -%}
        {%- if counters.images > 1 -%}
            {%- set image_tags = namespace(tags=[]) -%}
            {%- for i in range(counters.images) -%}
                {%- set image_tags.tags = image_tags.tags + ['<image ' + (i + 1)|string + '><image>'] -%}
            {%- endfor -%}
            {%- set mm_content = ' '.join(image_tags.tags) + '\n' -%}
        {%- elif counters.images == 1 -%}
            {%- set mm_content = '<image>\n' -%}
        {%- endif -%}
        {%- set mm_content = mm_content + '<video>\n' * counters.videos -%}
        {%- set msg.content = mm_content + msg.content.lstrip('\n') -%}
    {%- endif -%}

    {%- if message['role'] == 'user' %}
        {{- '<SPECIAL_11>User\n' + msg.content.replace('</think>', '<_end_think>').replace('/think', '').replace('/no_think', '').replace('<_end_think>', '</think>').strip() + '\n' }}
    {%- elif message['role'] == 'tool' %}
        {%- if loop.first or (messages[loop.index0 - 1].role != 'tool') -%}
            {{- '<SPECIAL_11>User\n' + '<TOOL_RESPONSE>[' }}
        {%- endif -%}
        {{- msg.content -}}
        {{- ', ' if not loop.last and (messages[loop.index0 + 1].role == 'tool') else '' -}}
        {%- if loop.last or (messages[loop.index0 + 1].role != 'tool') -%}
            {{- ']</TOOL_RESPONSE>\n' -}}
        {%- endif -%}
    {%- elif message['role'] == 'assistant' %}
        {{- '<SPECIAL_11>Assistant\n' + msg.content.strip() }}
        {%- if message.tool_calls -%}
            {%- if msg.content.strip() != '' -%}
                {{- '\n\n' -}}
            {%- endif -%}
            {{- '<TOOLCALL>[' -}}
            {%- for call in message.tool_calls -%}
                {%- set fn = call.function if call.function is defined else call -%}
                {{- '{"name": "' + fn.name + '", "arguments": ' -}}
                {%- if fn.arguments is string -%}
                    {{- fn.arguments -}}
                {%- else -%}
                    {{- fn.arguments | tojson -}}
                {%- endif -%}
                {{- '}' + (', ' if not loop.last else '') -}}
            {%- endfor -%}
            {{- ']</TOOLCALL>' -}}
        {%- endif -%}
        {{- '\n<SPECIAL_12>\n' -}}
    {%- endif %}
{%- endfor -%}
{%- if add_generation_prompt %}
    {{- '<SPECIAL_11>Assistant\n' }}
    {%- if ns.enable_thinking is defined and ns.enable_thinking is false %}
        {{- '<think></think>' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
"""


def convert_string_content_to_list_content(messages: List[Dict]) -> List[Dict]:
    """
    Convert the string content to a list of dicts.
    """
    for message_id, message in enumerate(messages):
        if isinstance(message["content"], str):
            messages[message_id]["content"] = [{"type": "text", "text": message["content"]}]
    return messages


def maybe_parse_vision_content(
    messages: List[Dict],
) -> tuple[
    int,
    Optional[list[float]],
    Optional[list[int]],
    Optional[list[list[int]]],
    Optional[list[list[np.ndarray]]],
    int,
    Optional[list[Image.Image]],
]:
    """
    Convert the string content to a list of dicts.
    """
    num_video = 0
    video_fps = []
    video_total_num_frames = []
    video_frames_indices = []
    video_frames = []
    images = []
    num_image = 0
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
                    video_frames.append(sub_content["video"])
                elif sub_content.get("type", "") == "image":
                    num_image += 1
                    images.append(sub_content["image"])
    return num_video, video_fps, video_total_num_frames, video_frames_indices, video_frames, num_image, images


def maybe_get_max_pixels_from_images_kwargs(messages: List[Dict]) -> Optional[tuple[int, int]]:
    """
    Get the max pixels from the images_kwargs.
    """
    for message_id, message in enumerate(messages):
        if isinstance(message["content"], list):
            for sub_content in message["content"]:
                if sub_content.get("type", "") == "image" and sub_content.get("max_pixels", None) is not None:
                    return sub_content["max_pixels"], sub_content.get("min_pixels", None)
    return None, None


class NemotronVLProcessor:
    # This is a wrapper around the AutoProcessor class to add some helper functions
    is_unified_processor: bool = True  # sentinel for TextTokenizerTransform duck-type check

    def __init__(
        self,
        name="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        credentials: str = "./credentials/s3_training.secret",
        bucket: str = "checkpoints-us-east-1",
        cache_dir: Optional[str] = None,
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

        self.processor.tokenizer.chat_template = nemotron_chat_template
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
        self.pad_id = self.processor.tokenizer.convert_tokens_to_ids(
            "<SPECIAL_999>"
        )
        self.vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("</img>")

        # Helper attributes for the dataloader video decoding function
        self.shortest_edge = 512  # HACK (xuali): hardcoded based on the model config
        self.min_height_width = 512  # HACK (xuali): hardcoded based on the model config
        self.patch_size = 16  # HACK (xuali): hardcoded based on the model config
        self.temporal_patch_size = 1
        self.merge_size = 1
        self.use_smart_resize = False

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

        has_thinking = False
        for message_id, message in enumerate(messages):
            if message["role"] == "assistant":
                for content in message["content"]:
                    if content.get("type", "") == "text":
                        if "<think>" in content["text"] and "</think>" in content["text"]:
                            has_thinking = True
        for message_id, message in enumerate(messages):
            if message["role"] == "system":
                prefix = "/think " if has_thinking else "/no_think "
                messages[message_id]["content"][0]["text"] = prefix + messages[message_id]["content"][0]["text"]
            if message["role"] == "assistant" and not has_thinking:
                for content in messages[message_id]["content"]:
                    if content.get("type", "text") == "text":
                        content["text"] = "<think></think>" + content["text"]

        num_video, video_fps, video_total_num_frames, video_frames_indices, video_frames, num_image, images = (
            maybe_parse_vision_content(messages)
        )
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        kwargs = {}  # omit kwargs passed in
        if num_video > 0:
            kwargs["videos_kwargs"] = VideosKwargs(do_sample_frames=False)
            assert num_video == 1, "only support one video for now"
            fps = video_fps[0]
            total_num_frames = video_total_num_frames[0]
            frames_indices = video_frames_indices[0]
            inputs = self.processor(
                text=[prompt],
                videos=video_frames,
                videos_kwargs=VideosKwargs(
                    do_sample_frames=False,
                    video_metadata=VideoMetadata(
                        fps=fps, total_num_frames=total_num_frames, duration=total_num_frames / fps, video_backend=None
                    ),
                ),
                return_tensors=return_tensors,
            )
        elif num_image > 0:
            inputs = self.processor(
                text=[prompt],
                images=images,
                return_tensors=return_tensors,
            )

        # Convert batch features into single features
        # By default, the processor returns a batch of features, but we use processor in dataloader, so we need to convert it to single features
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
        BOS_TOKEN = "<SPECIAL_11>"
        EOS_TOKEN = "<SPECIAL_12>"
        ROLE = "Assistant"
        # Offsets: skip the bos + "assistant\n" (always 3 tokens) and include the eos (+1) for supervision
        START_OFFSET = 3
        END_OFFSET = 1

        # Retrieve token IDs for the markers and the role.
        bos_token_id = self.processor.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
        eos_token_id = self.processor.tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        role_id = self.processor.tokenizer.convert_tokens_to_ids(ROLE)

        # Locate all positions where the start and end markers appear.
        start_indices = np.where(np_tokens == bos_token_id)[0].tolist()
        end_indices = np.where(np_tokens == eos_token_id)[0].tolist()[:1]
        for i in range(len(start_indices) - 1, 0, -1):
            end_indices.insert(0, start_indices[i] - 1)
        # Initialize the mask with False values.
        masks = np.zeros_like(np_tokens, dtype=bool)
        assert len(start_indices) == len(end_indices)
        # For each pair of bos/eos, check if the role is 'assistant'
        # and apply the mask accordingly.
        for start, end in zip(start_indices, end_indices):
            if np_tokens[start + 1] == role_id:
                # Mask tokens from after the assistant header (start+3) to include the end marker (end+1)
                masks[start + START_OFFSET : end + END_OFFSET] = True

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
        return self.processor.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.processor.tokenizer.decode(*args, **kwargs)


if __name__ == "__main__":
    """
    PYTHONPATH=. python3 cosmos/data/vlm/processors/nemotronvl_processor.py

    inputs: dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_sizes', 'text'])
        input_ids: type: <class 'torch.Tensor'> shape: torch.Size([6699])
        attention_mask: type: <class 'torch.Tensor'> shape: torch.Size([6699])
        pixel_values: type: <class 'torch.Tensor'> shape: torch.Size([26, 3, 224, 224])
        image_sizes: type: <class 'torch.Tensor'> shape: torch.Size([2, 2])
        text: type: <class 'str'>

    For image, expected output:
        input_ids: type: <class 'torch.Tensor'>
        shape: torch.Size([2772])
        attention_mask: type: <class 'torch.Tensor'>
        shape: torch.Size([2772])
        pixel_values: type: <class 'torch.Tensor'>
        shape: torch.Size([11008, 1536])
        image_grid_thw: type: <class 'torch.Tensor'>
        shape: torch.Size([1, 3])
        image_grid_thw: tensor([[  1,  86, 128]])
        num_image_token_id_tokens: 2752
        num_video_token_id_tokens: 0
        assistant_tokens_mask: 2
        assistant_tokens: tensor([ 59604, 151645])
        decoded_assistant_tokens: Paris<|im_end|>

    For video, expected output:
        input_ids: type: <class 'torch.Tensor'>
        shape: torch.Size([5538])
        attention_mask: type: <class 'torch.Tensor'>
        shape: torch.Size([5538])
        pixel_values_videos: type: <class 'torch.Tensor'>
        shape: torch.Size([22016, 1536])
        video_grid_thw: type: <class 'torch.Tensor'>
        shape: torch.Size([1, 3])
        video_grid_thw: tensor([[  2,  86, 128]])
        num_image_token_id_tokens: 0
        num_video_token_id_tokens: 5504
        assistant_tokens_mask: 2
        assistant_tokens: tensor([ 59604, 151645])
        decoded_assistant_tokens: Paris<|im_end|>
    """
    processor = NemotronVLProcessor("nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16")
    from io import BytesIO

    import requests

    response = requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")
    img = Image.open(BytesIO(response.content))

    # test video
    print("=============== test video ===============")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [img],
                    "fps": 12,
                },
                {"type": "text", "text": "Describe what you see."},
            ],
        },
        {"role": "assistant", "content": "<think> No need to think. </think> A cat is sleeping on a couch."},
    ]
    inputs = processor.apply_chat_template(messages)
    input_ids = inputs["input_ids"]
    decoded_text = processor.decode(input_ids, skip_special_tokens=False)
    print(decoded_text)
    print(list(inputs.keys()))
    for k, v in inputs.items():
        print(f"{k}: type: {type(v)}")
        if isinstance(v, torch.Tensor):
            print(f"shape: {v.shape}")
        if "grid" in k:
            print(f"{k}: {v}")
    num_image_token_id_tokens = inputs["input_ids"] == processor.image_token_id
    print(f"num_image_token_id_tokens: {num_image_token_id_tokens.sum()}")
    num_video_token_id_tokens = inputs["input_ids"] == processor.video_token_id
    print(f"num_video_token_id_tokens: {num_video_token_id_tokens.sum()}")

    assistant_tokens_mask = processor.add_assistant_tokens_mask(inputs["input_ids"])
    print(f"assistant_tokens_mask: {assistant_tokens_mask.sum()}")
    assistant_tokens = inputs["input_ids"][assistant_tokens_mask]
    print(f"assistant_tokens: {assistant_tokens}")
    decoded_assistant_tokens = processor.decode(assistant_tokens, skip_special_tokens=False)
    print(f"decoded_assistant_tokens: {decoded_assistant_tokens}")

    print("\n\n\n\n\n=============== test image ===============")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
            ],
        },
        {"role": "assistant", "content": "<think> No need to think. </think> A cat is sleeping on a couch."},
    ]
    inputs = processor.apply_chat_template(messages)
    input_ids = inputs["input_ids"]
    decoded_text = processor.decode(input_ids, skip_special_tokens=False)
    print(decoded_text)
    print(list(inputs.keys()))
    for k, v in inputs.items():
        print(f"{k}: type: {type(v)}")
        if isinstance(v, torch.Tensor):
            print(f"shape: {v.shape}")
        if "grid" in k:
            print(f"{k}: {v}")
    num_image_token_id_tokens = inputs["input_ids"] == processor.image_token_id
    print(f"num_image_token_id_tokens: {num_image_token_id_tokens.sum()}")
    num_video_token_id_tokens = inputs["input_ids"] == processor.video_token_id
    print(f"num_video_token_id_tokens: {num_video_token_id_tokens.sum()}")

    assistant_tokens_mask = processor.add_assistant_tokens_mask(inputs["input_ids"])
    print(f"assistant_tokens_mask: {assistant_tokens_mask.sum()}")
    assistant_tokens = inputs["input_ids"][assistant_tokens_mask]
    print(f"assistant_tokens: {assistant_tokens}")
    decoded_assistant_tokens = processor.decode(assistant_tokens, skip_special_tokens=False)
    print(f"decoded_assistant_tokens: {decoded_assistant_tokens}")

    print("\n\n\n\n\n=============== done ===============")
