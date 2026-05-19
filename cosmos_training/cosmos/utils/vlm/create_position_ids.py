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

"""Position id generation for Qwen3-VL-MoE multimodal rotary embeddings.

Qwen3-VL-MoE replaces the standard 1D rotary position embedding with a
3-axis Multimodal Rotary Position Embedding (M-RoPE) that encodes the
temporal, height, and width components of every token separately. This
module produces the (T, H, W) position id triple for each token in a
sequence that mixes text, image, and video content.

A Qwen3-VL-MoE-specific quirk shapes the algorithm: video temporal
position is conveyed through interleaved timestamp tokens placed
between frames (the input layout is roughly
    <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>
) rather than via the T axis of the patch grid. Each frame is therefore
treated as t=1 along the M-RoPE T axis, leaving the timestamp tokens to
encode inter-frame timing through ordinary text positions. This differs
from Qwen2-VL, where temporal position rode on the patch-grid T axis.

Currently only model_type == "qwen3_vl_moe" is dispatched. All other
configurations cause get_position_ids to return None, which signals the
caller to fall back on the model's built-in position id handling.
"""

import torch
from transformers import AutoConfig


def get_rope_index_qwen3_vl(
    config,
    input_ids: torch.LongTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the (T, H, W) M-RoPE position id triple for a Qwen3-VL-MoE batch.

    For each input sequence the function alternates between text spans
    and vision spans. Text spans receive standard cumulative position
    ids broadcast across all three axes. Vision spans receive a
    flattened 3D spatial grid: H and W step through the patch grid
    (after spatial_merge_size downsampling), while T steps through the
    temporal grid. For Qwen3-VL-MoE the temporal grid is always 1
    because each video frame is wrapped by an explicit timestamp token,
    so video_grid_thw is expanded to one row per frame with t=1 on
    entry.

    Args:
        config: Qwen3-VL-MoE AutoConfig. Reads vision_config.spatial_merge_size,
            image_token_id, video_token_id, and vision_start_token_id.
        input_ids: LongTensor of shape (B, N_token).
        image_grid_thw: LongTensor of shape (num_images, 3) where each
            row holds the (T, H, W) patch grid of one image. None if no
            images are present in the batch.
        video_grid_thw: LongTensor of shape (num_videos, 3) before
            per-frame expansion. None if no videos are present.
        attention_mask: Tensor of shape (B, N_token), or None. Defaults
            to all ones (i.e. no padding).

    Returns:
        position_ids: LongTensor of shape (3, B, N_token) holding the
            (T, H, W) position id of every token.
        mrope_position_deltas: LongTensor of shape (B, 1) holding
            position_ids.max() + 1 - sequence_length for each batch
            element. The HF generation code uses it to extrapolate
            position ids when extending the sequence past N_token.
    """

    # Expand each video into one row per frame with t=1 so the spatial
    # grid loop below treats every frame as an independent vision span;
    # cross-frame temporal ordering is carried by the timestamp tokens
    # the caller has already interleaved into input_ids.
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)  # [num_frames, 3]
        video_grid_thw[:, 0] = 1

    spatial_merge_size = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    video_token_id = config.video_token_id
    vision_start_token_id = config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        # Mixed text + vision path: walk each sequence span by span.
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)  # [B, N_token]
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )  # [3, B, N_token]
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            # Drop padding tokens before scanning span boundaries.
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            # Each vision span is announced by vision_start_token_id;
            # the token immediately after distinguishes image from video.
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                # Find the next image and video span starting at or after st;
                # whichever comes first is consumed in this iteration.
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                # Convert raw patch grid into the LLM-side grid by
                # collapsing each spatial_merge_size x spatial_merge_size
                # patch block into a single LLM token.
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                # Text span ahead of this vision span: cumulative position
                # ids broadcast across all three axes.
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)  # [3, text_len]

                # Vision span: flatten the (T, H, W) grid in row-major
                # order. t_index is effectively zero here because
                # llm_grid_t == 1 for every Qwen3-VL-MoE vision block;
                # the temporal position rides on the surrounding text
                # tokens (image headers / video timestamps) instead.
                t_index = (
                    torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                )  # [llm_grid_t * llm_grid_h * llm_grid_w]
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )  # [3, llm_grid_t * llm_grid_h * llm_grid_w]
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # Trailing text span after the last vision block.
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)  # [3, text_len]

            # Stitch the per-span position id chunks into one contiguous
            # tensor and scatter it back into the padded slot for this
            # batch element using the attention mask.
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)  # [3, N_token_unpadded]
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)  # [B, 1]
        return position_ids, mrope_position_deltas
    else:
        # Text-only (or no input_ids) path: position ids are identical
        # across the three M-RoPE axes.
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1  # [B, N_token]
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)  # [3, B, N_token]
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]  # [B, 1]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]  # [B, 1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )  # [3, B, N_token]
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )  # [B, 1]

        return position_ids, mrope_position_deltas


def get_position_ids(
    config: AutoConfig | None = None,
    model_name_or_path: str | None = None,
    input_ids: torch.LongTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return M-RoPE position ids for Qwen3-VL-MoE inputs, else None.

    Public dispatcher used by callers that need pre-computed position
    ids for the forward pass (data loaders, benchmark harnesses,
    inference scripts). It loads config from model_name_or_path when
    config is not supplied, dispatches on config.model_type, and
    permutes the M-RoPE output from (3, B, N_token) to the
    (B, 3, N_token) layout the Qwen3-VL-MoE forward pass expects.

    For any other model_type this returns None so the caller can fall
    back on the model's built-in position id handling rather than
    passing custom position ids.

    Args:
        config: AutoConfig of the model, or None. If None,
            model_name_or_path must be provided so the config can be
            loaded from disk or HF hub.
        model_name_or_path: Path or HF id used to load config when
            config is None.
        input_ids: LongTensor of shape (B, N_token).
        image_grid_thw: LongTensor of shape (num_images, 3) or None.
        video_grid_thw: LongTensor of shape (num_videos, 3) or None.
        attention_mask: Tensor of shape (B, N_token), or None.

    Returns:
        Tensor of shape (B, 3, N_token), or None if the model's
        config.model_type is not supported.
    """
    if config is None:
        assert model_name_or_path is not None, "config is None and model_name_or_path is None"
        config = AutoConfig.from_pretrained(model_name_or_path)
    if config.model_type in ["qwen3_vl_moe"]:
        position_ids, _ = get_rope_index_qwen3_vl(config, input_ids, image_grid_thw, video_grid_thw, attention_mask)
        position_ids = position_ids.permute(1, 0, 2).contiguous()  # [B, 3, N_token]
    else:
        position_ids = None
    return position_ids
