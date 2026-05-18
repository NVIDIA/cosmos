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

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from cosmos.data.vfm.sequence_packing import ModalityData, PackedSequence, verify_natten_parameter_list
from cosmos.model.vfm.mot.attention import build_packed_sequence
from cosmos.model.vfm.mot.context_parallel_utils import (
    get_context_parallel_last_hidden_state,
    get_context_parallel_sharded_sequence,
)
from cosmos.model.vfm.mot.domain_aware_linear import DomainAwareLinear
from cosmos.model.vfm.mot.modeling_utils import (
    FlattenedSinCosPositionEmbedding,
    TimestepEmbedder,
    VideoRopePosition3DEmb,
)
from cosmos.model.vfm.utils.memory import MemoryState


class Cosmos3VFMNetworkConfig(PretrainedConfig):
    def __init__(
        self,
        vision_gen=True,
        action_gen=False,
        sound_gen=False,
        vlm_config=None,
        latent_patch_size=2,
        latent_downsample_factor=8,
        latent_channel_size=16,
        position_embedding_type="3d_rope",
        max_latent_h=32,
        max_latent_w=32,
        max_latent_t=32,
        rope_h_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
        rope_t_extrapolation_ratio=1.0,
        enable_fps_modulation=False,
        base_fps=24,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        timestep_scale=0.001,
        predict_text_tokens=False,
        joint_attn_implementation="two_way",
        action_dim=32,
        num_embodiment_domains=32,
        temporal_compression_factor_vision=4,
        temporal_compression_factor_action=1,
        natten_parameter_list=None,
        video_temporal_causal=False,
        # Sound generation parameters
        sound_dim: int | None = None,
        temporal_compression_factor_sound=1,
        sound_latent_fps: int = 25,
        **kwargs,
    ):
        self.vision_gen = vision_gen
        self.sound_gen = sound_gen
        self.vlm_config = vlm_config
        self.latent_patch_size = latent_patch_size
        self.latent_downsample_factor = latent_downsample_factor
        self.latent_channel_size = latent_channel_size
        self.position_embedding_type = position_embedding_type
        self.max_latent_h = max_latent_h
        self.max_latent_w = max_latent_w
        self.max_latent_t = max_latent_t
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.enable_fps_modulation = enable_fps_modulation
        self.base_fps = base_fps
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        self.timestep_scale = timestep_scale
        self.predict_text_tokens = predict_text_tokens
        self.joint_attn_implementation = joint_attn_implementation
        self.temporal_compression_factor_vision = temporal_compression_factor_vision
        self.natten_parameter_list = natten_parameter_list
        self.video_temporal_causal = video_temporal_causal

        # action related parameters
        self.action_gen = action_gen  # whether to generate action tokens
        self.action_dim = action_dim
        self.num_embodiment_domains = num_embodiment_domains
        self.temporal_compression_factor_action = temporal_compression_factor_action
        if self.action_gen:
            assert self.vision_gen, (
                "Action generation requires visual generation! We do NOT support action only training!"
            )

        # sound related parameters
        self.sound_dim = sound_dim
        self.temporal_compression_factor_sound = temporal_compression_factor_sound
        self.sound_latent_fps = sound_latent_fps
        if self.sound_gen:
            assert self.vision_gen, (
                "Sound generation requires visual generation! We do NOT support sound only training!"
            )

        super().__init__(**kwargs)


class Cosmos3VFMNetwork(PreTrainedModel):
    config_class = Cosmos3VFMNetworkConfig
    base_model_prefix = "cosmos3"

    def __init__(self, language_model, config: Cosmos3VFMNetworkConfig):
        super().__init__(config)
        self.language_model = language_model
        self.hidden_size = config.vlm_config.hidden_size
        self.use_moe = "Mo" in config.vlm_config.layer_module
        self.num_heads = config.vlm_config.num_attention_heads
        self.num_kv_heads = config.vlm_config.num_key_value_heads
        self.head_dim = config.vlm_config.head_dim
        self.num_hidden_layers = config.vlm_config.num_hidden_layers
        self.predict_text_tokens = config.predict_text_tokens

        if config.natten_parameter_list is not None and config.joint_attn_implementation != "three_way":
            raise NotImplementedError(
                f"Sparsity is only supported with 'three_way' attention, but got {config.joint_attn_implementation=}, "
                "and 'natten_parameter_list' was not None."
            )
        self.natten_parameter_list = verify_natten_parameter_list(
            config.natten_parameter_list, num_layers=self.num_hidden_layers
        )

        if config.video_temporal_causal and config.joint_attn_implementation != "three_way":
            raise ValueError(
                f"video_temporal_causal=True requires joint_attn_implementation='three_way', "
                f"but got {config.joint_attn_implementation!r}."
            )
        self.video_temporal_causal = config.video_temporal_causal
        self.pad_for_cuda_graphs = False

        if config.vision_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.timestep_scale = config.timestep_scale
            self.latent_downsample = config.latent_downsample_factor * config.latent_patch_size
            self.max_latent_h = config.max_latent_h
            self.max_latent_w = config.max_latent_w
            self.max_latent_t = config.max_latent_t
            self.latent_channel = config.latent_channel_size
            self.patch_latent_dim = self.latent_patch_size**2 * self.latent_channel

            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)

            assert config.position_embedding_type in ["3d_rope", "flattened_sin_cos", "unified_3d_mrope"]
            if config.position_embedding_type == "3d_rope":
                self.latent_pos_embed = VideoRopePosition3DEmb(
                    head_dim=self.hidden_size,
                    len_h=self.max_latent_h,
                    len_w=self.max_latent_w,
                    len_t=self.max_latent_t,
                    h_extrapolation_ratio=config.rope_h_extrapolation_ratio,
                    w_extrapolation_ratio=config.rope_w_extrapolation_ratio,
                    t_extrapolation_ratio=config.rope_t_extrapolation_ratio,
                    enable_fps_modulation=config.enable_fps_modulation,  # fps_modulation scales RoPE by fps. By default, disable FPS RoPE modulation.
                    base_fps=config.base_fps,
                    base_temporal_compression_factor=config.temporal_compression_factor_vision,
                    temporal_compression_factor=config.temporal_compression_factor_vision,
                )
            elif config.position_embedding_type == "flattened_sin_cos":
                self.latent_pos_embed = FlattenedSinCosPositionEmbedding(
                    max_latent_h=self.max_latent_h, max_latent_w=self.max_latent_w, hidden_size=self.hidden_size
                )
            elif config.position_embedding_type == "unified_3d_mrope":
                # No additive position embedding - position info is in 3D position IDs for attention
                self.latent_pos_embed = None
            else:
                raise ValueError(f"Unknown position_embedding_type: {config.position_embedding_type!r}")

        if config.action_gen:
            self.action_dim = config.action_dim
            self.num_embodiment_domains = config.num_embodiment_domains
            self.action2llm = DomainAwareLinear(self.action_dim, self.hidden_size, self.num_embodiment_domains)
            self.llm2action = DomainAwareLinear(self.hidden_size, self.action_dim, self.num_embodiment_domains)

            if config.position_embedding_type == "3d_rope":
                self.action_pos_embed = VideoRopePosition3DEmb(
                    head_dim=self.hidden_size,
                    len_h=1,
                    len_w=1,
                    len_t=self.max_latent_t * config.temporal_compression_factor_vision,
                    h_extrapolation_ratio=config.rope_h_extrapolation_ratio,
                    w_extrapolation_ratio=config.rope_w_extrapolation_ratio,
                    t_extrapolation_ratio=config.rope_t_extrapolation_ratio,
                    enable_fps_modulation=config.enable_fps_modulation,
                    base_fps=config.base_fps,
                    base_temporal_compression_factor=config.temporal_compression_factor_vision,  # vision compression factor is used for base tps
                    temporal_compression_factor=config.temporal_compression_factor_action,  # Action is at frame rate (no temporal compression)
                )
            elif config.position_embedding_type == "unified_3d_mrope":
                # No additive position embedding - position info is in 3D position IDs for attention
                self.action_pos_embed = None
            else:
                raise ValueError(f"Unknown position_embedding_type: {config.position_embedding_type!r}")

            self.action_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

        if config.sound_gen:
            self.sound_dim = config.sound_dim
            self.sound2llm = nn.Linear(config.sound_dim, self.hidden_size)
            self.llm2sound = nn.Linear(self.hidden_size, config.sound_dim)
            self.sound_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

        self.config = config
        self.parallel_dims = None

    def init_weights(self, buffer_device: torch.device | None):
        if self.config.vision_gen or self.config.action_gen or self.config.sound_gen:
            self.time_embedder._init_weights()

        if self.config.vision_gen:
            std = 1.0 / math.sqrt(self.patch_latent_dim)
            torch.nn.init.trunc_normal_(self.vae2llm.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.vae2llm.bias)

            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.llm2vae.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.llm2vae.bias)

            if self.latent_pos_embed is not None:
                self.latent_pos_embed._init_weights()

        if self.config.action_gen:
            # DomainAwareLinear uses embeddings for weights, so we initialize them differently
            # action2llm: input_size=action_dim, output_size=hidden_size
            std = 1.0 / math.sqrt(self.action_dim)
            torch.nn.init.trunc_normal_(self.action2llm.fc.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.action2llm.bias.weight)

            # llm2action: input_size=hidden_size, output_size=action_dim
            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.llm2action.fc.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.llm2action.bias.weight)

            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.action_modality_embed, std=std, a=-3 * std, b=3 * std)

            if self.action_pos_embed is not None:
                self.action_pos_embed._init_weights()

        if self.config.sound_gen:
            # sound2llm: input_size=sound_dim, output_size=hidden_size
            std = 1.0 / math.sqrt(self.sound_dim)
            torch.nn.init.trunc_normal_(self.sound2llm.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.sound2llm.bias)

            # llm2sound: input_size=hidden_size, output_size=sound_dim
            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.llm2sound.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.llm2sound.bias)

            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.sound_modality_embed, std=std, a=-3 * std, b=3 * std)

        self.language_model.init_weights(buffer_device=buffer_device)

    def patchify_and_pack_latents(
        self, tokens_vision: torch.Tensor, token_shapes_vision: List[Tuple[int, int, int]]
    ) -> tuple[torch.Tensor, List[Tuple[int, int, int]]]:
        p = self.latent_patch_size
        # Patchify and pack the latents
        packed_latent = []
        original_latent_shapes = []  # Store original shapes for unpadding later

        # C, T, H, W
        for latent, (t, h, w) in zip(tokens_vision, token_shapes_vision):
            latent = latent.squeeze(0)  # [C,T,H,W]

            # Get original latent dimensions
            _, t_actual, h_actual, w_actual = latent.shape
            original_latent_shapes.append((t_actual, h_actual, w_actual))

            # Compute padded dimensions (must be divisible by p)
            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p

            # Zero-pad if dimensions are not divisible by p
            if h_padded != h_actual or w_padded != w_actual:
                padded = torch.zeros(
                    (self.latent_channel, t_actual, h_padded, w_padded),
                    device=latent.device,
                    dtype=latent.dtype,
                )  # [C,T,H_padded,W_padded]
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded  # [C,T,H_padded,W_padded]

            # Compute number of patches after padding
            h_patches = h_padded // p
            w_patches = w_padded // p

            # Patchify
            latent = latent.reshape(
                self.latent_channel, t_actual, h_patches, p, w_patches, p
            )  # [C,T,h_patches,p,w_patches,p]
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(
                -1, p * p * self.latent_channel
            )  # [T*h_patches*w_patches,patch_latent_dim]
            packed_latent.append(latent)

        # We assumed latents we get to the network is already noised
        packed_latent = torch.cat(packed_latent, dim=0)  # [total_vision_patches,patch_latent_dim]
        return packed_latent, original_latent_shapes

    def unpatchify_and_unpack_latents(
        self,
        packed_mse_preds: torch.Tensor,
        token_shapes_vision: List[Tuple[int, int, int]],
        noisy_frame_indexes_vision: list[torch.Tensor],
        original_latent_shapes: List[Tuple[int, int, int]] | None = None,
    ) -> list[torch.Tensor]:
        p = self.latent_patch_size
        unpatchified_latents = []

        # Split packed_mse_preds back into individual latents based on token_shapes_vision
        start_idx = 0
        for i, (t_c, h_c, w_c) in enumerate(token_shapes_vision):
            # Get original shape for unpadding (if provided)
            if original_latent_shapes is not None:
                t_orig, h_orig, w_orig = original_latent_shapes[i]
                # Compute padded dimensions used during patchify
                h_padded = ((h_orig + p - 1) // p) * p
                w_padded = ((w_orig + p - 1) // p) * p
                h_patches = h_padded // p
                w_patches = w_padded // p
            else:
                # Fallback: use token shapes directly (assumes no padding was needed)
                t_orig, h_orig, w_orig = t_c, h_c * p, w_c * p
                h_patches, w_patches = h_c, w_c

            # noisy_frame_indexes_vision is a list of tensors, each with shape (T,),
            # where the values are the noisy frame indices.
            noisy_frame_indexes = noisy_frame_indexes_vision[i]
            t_n = len(noisy_frame_indexes)

            # Initialize with the original shape (after unpadding), zeros for clean frames
            output_tensor = torch.zeros(
                (self.latent_channel, t_c, h_orig, w_orig),
                device=packed_mse_preds.device,
                dtype=packed_mse_preds.dtype,
            )  # [C,T,H_orig,W_orig]
            num_patches = t_n * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                # Extract patches for this latent
                latent_patches = packed_mse_preds[start_idx:end_idx]  # [num_patches,patch_latent_dim]
                # Reshape back to [t_n, h_patches, w_patches, p, p, channels]
                latent_patches = latent_patches.reshape(
                    t_n, h_patches, w_patches, p, p, self.latent_channel
                )  # [T_n,h_patches,w_patches,p,p,C]
                # Invert the einsum operation: "thwpqc->cthpwq"
                latent = torch.einsum("thwpqc->cthpwq", latent_patches)  # [C,T_n,h_patches,p,w_patches,p]
                # Reshape back to [channels, t_n, h_padded, w_padded]
                latent = latent.reshape(
                    self.latent_channel, t_n, h_patches * p, w_patches * p
                )  # [C,T_n,H_padded,W_padded]

                # Crop to original dimensions (unpad the zeros)
                latent = latent[:, :, :h_orig, :w_orig]  # [C,T_n,H_orig,W_orig]

                # Fill only the noisy frame positions using the actual mask indices
                output_tensor[:, noisy_frame_indexes] = latent

                start_idx = end_idx

            unpatchified_latents.append(output_tensor.unsqueeze(0))  # [1,C,T,H,W]

        # Return list of unpatchified latents (supports variable shapes)
        return unpatchified_latents

    def pack_action(
        self,
        tokens_action: list[torch.Tensor],
        token_shapes_action: list[tuple[int, ...]],
        domain_id_action: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack variable-length action tokens into a 1D sequence for transformer input.

        Args:
            tokens_action: List of action tensors, each [T_i, action_dim] (T_i may vary).
            token_shapes_action: List of (T_i,) tuples per sample.
            domain_id_action: List of domain ID tensors, each of shape [1].

        Returns:
            Tuple of (packed_tokens, per_token_domain_id):
                packed_tokens: [total_action_tokens, action_dim]
                per_token_domain_id: [total_action_tokens]
        """
        packed: list[torch.Tensor] = []
        domain_ids: list[torch.Tensor] = []
        for tokens, shape, d_id in zip(tokens_action, token_shapes_action, domain_id_action):
            T = shape[0]
            packed.append(tokens[:T])
            domain_ids.append(d_id.expand(T))
        return torch.cat(packed, dim=0), torch.cat(domain_ids, dim=0)

    def unpack_action(
        self,
        packed_action_preds: torch.Tensor,
        token_shapes_action: list[tuple[int, ...]],
        noisy_frame_indexes_action: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Unpack action predictions back into per-sample action tensors.

        Args:
            packed_action_preds: Packed action predictions of shape (total_noisy_tokens, action_dim)
            token_shapes_action: Per-sample token shapes, each (T_i,) tuple.
            noisy_frame_indexes_action: List of tensors, each with shape (Tn_i,), where the values
                are the noisy frame indices for sample i.

        Returns:
            List of per-sample tensors, each of shape (T_i, action_dim), with predictions
            placed at noisy positions. Clean positions are left as zeros.
        """
        unpacked: list[torch.Tensor] = []
        start_idx = 0
        for shape, noisy_frame_indexes in zip(token_shapes_action, noisy_frame_indexes_action):
            T = shape[0]
            output = torch.zeros(
                (T, self.action_dim),
                device=packed_action_preds.device,
                dtype=packed_action_preds.dtype,
            )
            t_n = len(noisy_frame_indexes)
            if t_n > 0:
                end_idx = start_idx + t_n
                output[noisy_frame_indexes] = packed_action_preds[start_idx:end_idx]
                start_idx = end_idx
            unpacked.append(output)
        return unpacked

    def pack_sound_latents(
        self,
        tokens_sound: list[torch.Tensor],
        token_shapes_sound: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """Pack sound latents into a 1D sequence for transformer input.

        Args:
            tokens_sound: List of sound latent tensors, each [C, T]
            token_shapes_sound: List of (T, 1, 1) tuples per sample

        Returns:
            Packed tensor of shape [total_sound_tokens, C]
        """
        packed = []
        for sound, shape in zip(tokens_sound, token_shapes_sound):
            T = shape[0]
            # sound: [C, T] → take first T frames → [C, T]
            # Then permute to [T, C] for packing
            sound_tokens = sound[:, :T].permute(1, 0)  # [T,C]
            packed.append(sound_tokens)
        return torch.cat(packed, dim=0)  # [total_sound_tokens,C]

    def unpack_sound_latents(
        self,
        packed_sound_preds: torch.Tensor,
        token_shapes_sound: list[tuple[int, int, int]],
        noisy_frame_indexes_sound: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Unpack sound predictions back into per-sample sound latents.

        Args:
            packed_sound_preds: Packed sound predictions of shape (total_noisy_tokens, sound_dim)
            token_shapes_sound: List of (T, 1, 1) tuples per sample
            noisy_frame_indexes_action: List of tensors, each with shape (T_i,), where the values
                are the noisy frame indices. T_i <= max_T.

        Returns:
            List of per-sample tensors, each [C, T], with predictions placed at noisy positions.
            Clean positions are left as zeros.
        """
        unpacked = []
        start_idx = 0
        for shape, noisy_frame_indexes in zip(token_shapes_sound, noisy_frame_indexes_sound):
            T = shape[0]
            # Initialize output with zeros for clean positions
            output = torch.zeros(
                (self.sound_dim, T),
                device=packed_sound_preds.device,
                dtype=packed_sound_preds.dtype,
            )

            t_n = len(noisy_frame_indexes)

            if t_n > 0:
                end_idx = start_idx + t_n
                # packed_sound_preds: [total_noisy_tokens, C] → transpose and fill at noisy positions
                output[:, noisy_frame_indexes] = packed_sound_preds[
                    start_idx:end_idx
                ].T  # packed_sound_preds[...]: [T_n,C] → .T: [C,T_n]
                start_idx = end_idx

            unpacked.append(output)
        return unpacked

    def _encode_text(
        self,
        packed_seq: PackedSequence,
    ) -> tuple[torch.Tensor, torch.dtype]:
        """Embed text tokens and initialize packed_sequence.

        Args:
            packed_seq: PackedSequence containing text_ids and text_indexes.

        Returns:
            tuple of (packed_sequence, target_dtype) where packed_sequence has text embeddings filled in.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_seq.text_ids)  # [N_text,hidden_size]
        packed_sequence = packed_text_embedding.new_zeros(
            size=(packed_seq.sequence_length, self.hidden_size)
        )  # [N_total,hidden_size]
        packed_sequence[packed_seq.text_indexes] = (
            packed_text_embedding  # [N_text,hidden_size] scattered into [N_total,hidden_size]
        )
        return packed_sequence, packed_text_embedding.dtype

    def _encode_vision(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
        fps: Optional[torch.Tensor] = None,
    ) -> List[Tuple[int, int, int]] | None:
        """Project vision tokens and fill into packed_sequence.

        Args:
            packed_seq: PackedSequence containing vision tokens and metadata.
            packed_sequence: The packed sequence tensor to fill vision embeddings into (modified in-place).
            target_dtype: Target dtype for embeddings (typically from text embedding).
            fps: Optional FPS tensor for RoPE modulation.

        Returns:
            Original latent shapes before padding (for unpadding during decode), or None if no vision tokens.
        """
        if packed_seq.vision is None or packed_seq.vision.tokens is None:
            # No vision tokens in this batch
            return None

        vision = packed_seq.vision
        assert vision.tokens is not None  # Type narrowing (checked above but reassignment loses it)
        assert vision.token_shapes is not None
        assert isinstance(vision.sequence_indexes, torch.Tensor)
        assert isinstance(vision.timesteps, torch.Tensor)
        assert isinstance(vision.mse_loss_indexes, torch.Tensor)

        packed_tokens_vision, original_latent_shapes = self.patchify_and_pack_latents(
            vision.tokens, vision.token_shapes
        )  # packed_tokens_vision: [total_vision_patches,patch_latent_dim]

        packed_tokens_vision = self.vae2llm(packed_tokens_vision)  # [total_vision_patches,hidden_size]

        # Add absolute position embedding only when NOT using unified 3D mRoPE
        # (3D mRoPE provides positional information via rotary embeddings instead)
        if self.latent_pos_embed is not None:
            latent_token_pos_emb = self.latent_pos_embed(vision.token_shapes, fps=fps).to(
                target_dtype
            )  # [total_vision_patches,hidden_size]
            packed_tokens_vision = packed_tokens_vision + latent_token_pos_emb  # [total_vision_patches,hidden_size]

        has_noisy_vision = vision.mse_loss_indexes.numel() > 0

        if has_noisy_vision:
            timesteps_vision = vision.timesteps * self.timestep_scale  # [N_noisy_frames_vision]

            # Timesteps are computed in FP32 for numerical stability.
            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                packed_timestep_embeds_vision = self.time_embedder(
                    timesteps_vision
                )  # [N_noisy_frames_vision,hidden_size]
            packed_timestep_embeds_vision = packed_timestep_embeds_vision.to(
                target_dtype
            )  # [N_noisy_frames_vision,hidden_size]

            packed_tokens_vision = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_vision,
                packed_timestep_embeds=packed_timestep_embeds_vision,
                noisy_frame_indexes=vision.noisy_frame_indexes,
                token_shapes=vision.token_shapes,
            )  # [total_vision_patches,hidden_size]

        packed_sequence[vision.sequence_indexes] = (
            packed_tokens_vision  # [total_vision_patches,hidden_size] scattered into [N_total,hidden_size]
        )
        return original_latent_shapes

    def _decode_vision(
        self,
        packed_seq: PackedSequence,
        last_hidden_state: torch.Tensor,
        output_dict: dict,
        original_latent_shapes: List[Tuple[int, int, int]] | None = None,
    ) -> None:
        """Decode vision tokens from hidden states and update output_dict.

        Args:
            packed_seq: PackedSequence containing mse_loss_indexes_vision and token_shapes_vision.
            last_hidden_state: Hidden states from the transformer.
            output_dict: Output dictionary to update with mse_preds (modified in-place).
            original_latent_shapes: Original latent shapes before padding (for unpadding).
        """
        vision = packed_seq.vision
        # Check if no vision or no noisy vision tokens
        has_noisy_vision = (
            vision is not None
            and vision.tokens is not None
            and isinstance(vision.mse_loss_indexes, torch.Tensor)
            and vision.mse_loss_indexes.numel() > 0
        )
        if not has_noisy_vision:
            # No noisy vision tokens present. The model is predicting actions
            # given clean vision tokens. We need to execute a dummy forward to maintain
            # computation graph consistency across ranks (FSDP should torch all weights).
            preds_vision = torch.zeros(
                [1, self.patch_latent_dim], device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )  # [1,patch_latent_dim]
            preds_vision = self.vae2llm(preds_vision)  # [1,hidden_size]
            preds_vision = self.llm2vae(preds_vision)  # [1,patch_latent_dim]
            # Return a list of per-sample zero tensors with correct shapes (e.g. (C, T, H, W)),
            # so downstream code (_get_velocity, _compute_flow_matching_loss) that iterates over preds_vision
            # gets properly-shaped tensors. Without this, the dummy tensor (1, patch_latent_dim)
            # would cause a size mismatch when concatenating vision+action velocities.
            # When vision is None (no vision in batch), fall back to [preds_vision] purely for
            # gradient graph consistency — it won't be iterated over.
            if vision is not None and vision.tokens is not None:
                preds_vision_list = [torch.zeros_like(tok) for tok in vision.tokens]
                # Inject dummy forward's computation graph so vae2llm/llm2vae params
                # stay in the autograd graph (zeros_like creates detached tensors).
                preds_vision_list[0] = preds_vision_list[0] + 0.0 * preds_vision.sum()
            else:
                preds_vision_list = [preds_vision]
            output_dict.update(preds_vision=preds_vision_list)
        else:
            assert vision is not None  # Type narrowing
            assert isinstance(vision.mse_loss_indexes, torch.Tensor)
            assert vision.noisy_frame_indexes is not None
            preds_vision = self.llm2vae(
                last_hidden_state[vision.mse_loss_indexes]
            )  # [total_noisy_vision_patches,patch_latent_dim]
            preds_vision = self.unpatchify_and_unpack_latents(
                preds_vision,
                token_shapes_vision=vision.token_shapes,
                noisy_frame_indexes_vision=vision.noisy_frame_indexes,
                original_latent_shapes=original_latent_shapes,
            )
            output_dict.update(preds_vision=preds_vision)

    def _encode_action(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
        fps_action: Optional[torch.Tensor] = None,
    ) -> None:
        """Encode action tokens and fill into packed_sequence."""
        if packed_seq.action is None or packed_seq.action.tokens is None:
            # No action tokens in this batch
            return

        action: ModalityData = packed_seq.action
        assert action.token_shapes is not None
        assert isinstance(action.sequence_indexes, torch.Tensor)
        assert isinstance(action.timesteps, torch.Tensor)
        assert isinstance(action.mse_loss_indexes, torch.Tensor)

        # Pack variable-length action tokens into a 1D sequence (same pattern as pack_sound_latents)
        packed_tokens_action, per_token_domain_id = self.pack_action(
            action.tokens, action.token_shapes, action.domain_id
        )
        packed_tokens_action = self.action2llm(packed_tokens_action, per_token_domain_id)

        # Add additive position embedding only if not using unified_3d_mrope
        if self.action_pos_embed is not None:
            # VideoRopePosition3DEmb expects shapes as (t, h, w). For actions we use a 1x1 spatial grid.
            action_shapes_3d = [(ts[0], 1, 1) for ts in action.token_shapes]
            action_token_pos_emb = self.action_pos_embed(
                action_shapes_3d,
                fps=fps_action,
                start_frame_offset=1,
            ).to(target_dtype)  # [B_action*T_action,hidden_size]
            packed_tokens_action = packed_tokens_action + action_token_pos_emb  # [B_action*T_action,hidden_size]

        packed_tokens_action = packed_tokens_action + self.action_modality_embed.view(
            1, -1
        )  # [B_action*T_action,hidden_size]

        has_noisy_actions = action.mse_loss_indexes.numel() > 0
        if has_noisy_actions:
            timesteps_action = action.timesteps * self.timestep_scale  # [N_noisy_frames_action]
            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                packed_timestep_embeds_action = self.time_embedder(
                    timesteps_action
                )  # [N_noisy_frames_action,hidden_size]
            packed_timestep_embeds_action = packed_timestep_embeds_action.to(
                target_dtype
            )  # [N_noisy_frames_action,hidden_size]

            packed_tokens_action = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_action,
                packed_timestep_embeds=packed_timestep_embeds_action,
                noisy_frame_indexes=action.noisy_frame_indexes,
                token_shapes=action.token_shapes,
            )  # [B_action*T_action,hidden_size]

        packed_sequence[action.sequence_indexes] = (
            packed_tokens_action  # [B_action*T_action,hidden_size] scattered into [N_total,hidden_size]
        )

    def _decode_action(
        self,
        packed_seq: PackedSequence,
        last_hidden_state: torch.Tensor,
        output_dict: dict,
    ) -> None:
        """Decode action tokens from hidden states and update output_dict."""
        action = packed_seq.action
        # Check if no action or no noisy action tokens
        has_noisy_action = (
            action is not None
            and action.tokens is not None
            and isinstance(action.mse_loss_indexes, torch.Tensor)
            and action.mse_loss_indexes.numel() > 0
        )
        if not has_noisy_action:
            # dummy forward to maintain computation graph consistency across ranks
            preds_action = torch.zeros(
                [1, self.action_dim], device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )  # [1,action_dim]
            dummy_domain_id = torch.zeros([1], device=last_hidden_state.device, dtype=torch.long)  # [1]
            preds_action = self.action2llm(preds_action, dummy_domain_id) + self.action_modality_embed.view(
                1, -1
            )  # [1,hidden_size]
            preds_action = self.llm2action(preds_action, dummy_domain_id)  # [1,action_dim]
            # Return a list of per-sample zero tensors with correct shapes (e.g. (T, action_dim)),
            # so downstream code (_get_velocity, _compute_flow_matching_loss) that iterates over preds_action
            # gets properly-shaped tensors. Without this, the dummy tensor (1, action_dim)
            # would cause a size mismatch when concatenating vision+action velocities.
            if action is not None and action.tokens is not None:
                preds_action_list = [torch.zeros_like(tok) for tok in action.tokens]
                # Inject dummy forward's computation graph so DomainAwareLinear params
                # stay in the autograd graph (zeros_like creates detached tensors).
                preds_action_list[0] = preds_action_list[0] + 0.0 * preds_action.sum()
            # When action is None (no action in batch), fall back to [preds_action] purely for
            # gradient graph consistency — it won't be iterated over.
            else:
                preds_action_list = [preds_action]
            output_dict.update(preds_action=preds_action_list)
        else:
            assert action is not None  # Type narrowing
            assert isinstance(action.mse_loss_indexes, torch.Tensor)
            assert action.condition_mask is not None
            assert len(action.domain_id) > 0

            action_hidden_states = last_hidden_state[action.mse_loss_indexes]  # [total_noisy_action_tokens,hidden_size]

            # Build per-token domain IDs for the noisy tokens (same expansion logic as pack_action)
            domain_ids: list[torch.Tensor] = []
            for nfi, d_id in zip(action.noisy_frame_indexes, action.domain_id):
                domain_ids.append(d_id.expand(len(nfi)))
            per_token_domain_id = torch.cat(domain_ids, dim=0)

            preds_action = self.llm2action(
                action_hidden_states, per_token_domain_id
            )  # [total_noisy_action_tokens,action_dim]
            preds_action = self.unpack_action(preds_action, action.token_shapes, action.noisy_frame_indexes)
            output_dict.update(preds_action=preds_action)

    def _encode_sound(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
        fps_sound: Optional[torch.Tensor] = None,
    ) -> None:
        """Encode sound tokens and fill into packed_sequence.

        Args:
            packed_seq: PackedSequence containing sound tokens and metadata.
            packed_sequence: The packed sequence tensor to fill sound embeddings into (modified in-place).
            target_dtype: Target dtype for embeddings (typically from text embedding).
            fps_sound: FPS tensor for RoPE modulation. Should be the sound latent rate (e.g., 25 Hz).
        """
        if packed_seq.sound is None or packed_seq.sound.tokens is None:
            # No sound tokens in this batch
            return

        sound = packed_seq.sound
        assert sound.token_shapes is not None
        assert isinstance(sound.sequence_indexes, torch.Tensor)
        assert isinstance(sound.timesteps, torch.Tensor)
        assert isinstance(sound.mse_loss_indexes, torch.Tensor)

        # Pack sound latents: list of [C, T] tensors → [total_tokens, C]
        packed_tokens_sound = self.pack_sound_latents(
            sound.tokens, sound.token_shapes
        )  # [total_sound_tokens,sound_dim]
        packed_tokens_sound = packed_tokens_sound.to(target_dtype)  # [total_sound_tokens,sound_dim]

        # Project sound tokens + modality embedding

        # No additive position embedding is used (unlike legacy video which keeps one for backward compat).
        packed_tokens_sound = (
            self.sound2llm(packed_tokens_sound) + self.sound_modality_embed
        )  # [total_sound_tokens,hidden_size]

        has_noisy_sound = sound.mse_loss_indexes.numel() > 0
        if has_noisy_sound:
            timesteps_sound = sound.timesteps * self.timestep_scale  # [N_noisy_frames_sound]
            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                packed_timestep_embeds_sound = self.time_embedder(timesteps_sound)  # [N_noisy_frames_sound,hidden_size]
            packed_timestep_embeds_sound = packed_timestep_embeds_sound.to(
                target_dtype
            )  # [N_noisy_frames_sound,hidden_size]

            packed_tokens_sound = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_sound,
                packed_timestep_embeds=packed_timestep_embeds_sound,
                noisy_frame_indexes=sound.noisy_frame_indexes,
                token_shapes=sound.token_shapes,
            )  # [total_sound_tokens,hidden_size]

        packed_sequence[sound.sequence_indexes] = (
            packed_tokens_sound  # [total_sound_tokens,hidden_size] scattered into [N_total,hidden_size]
        )

    def _decode_sound(
        self,
        packed_seq: PackedSequence,
        last_hidden_state: torch.Tensor,
        output_dict: dict,
    ) -> None:
        """Decode sound tokens from hidden states and update output_dict.

        Args:
            packed_seq: PackedSequence containing sound modality data.
            last_hidden_state: Hidden states from the transformer.
            output_dict: Output dictionary to update with preds_sound (modified in-place).
        """
        sound = packed_seq.sound
        # Check if no sound or no noisy sound tokens
        has_noisy_sound = (
            sound is not None
            and sound.tokens is not None
            and isinstance(sound.mse_loss_indexes, torch.Tensor)
            and sound.mse_loss_indexes.numel() > 0
        )
        if not has_noisy_sound:
            # dummy forward to maintain computation graph consistency across ranks
            preds_sound = torch.zeros(
                [1, self.sound_dim], device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )  # [1,sound_dim]
            preds_sound = self.sound2llm(preds_sound) + self.sound_modality_embed  # [1,hidden_size]
            preds_sound = self.llm2sound(preds_sound)  # [1,sound_dim]
            if sound is not None and sound.tokens is not None:
                preds_sound_list = [torch.zeros_like(tok) for tok in sound.tokens]
                preds_sound_list[0] = preds_sound_list[0] + 0.0 * preds_sound.sum()
            else:
                preds_sound_list = [preds_sound]
            output_dict.update(preds_sound=preds_sound_list)
        else:
            assert sound is not None  # Type narrowing
            assert isinstance(sound.mse_loss_indexes, torch.Tensor)
            assert sound.condition_mask is not None
            preds_sound = self.llm2sound(
                last_hidden_state[sound.mse_loss_indexes]
            )  # [total_noisy_sound_tokens,sound_dim]
            preds_sound = self.unpack_sound_latents(
                preds_sound, sound.token_shapes, sound.noisy_frame_indexes
            )  # list of [C,T] per sample
            output_dict.update(preds_sound=preds_sound)

    def forward(
        self,
        packed_seq: PackedSequence,
        fps_vision: Optional[torch.Tensor] = None,
        fps_action: Optional[torch.Tensor] = None,
        fps_sound: Optional[torch.Tensor] = None,
        memory: MemoryState | None = None,
    ) -> dict:
        """
        Forward pass for Cosmos3VFMNetwork.

        Args:
            packed_seq: PackedSequence containing all packed tensors and metadata.
                See PackedSequence dataclass for field details.
            fps_vision: Optional FPS tensor for vision RoPE modulation.
            fps_action: Optional FPS tensor for action RoPE modulation.
            fps_sound: Optional FPS tensor for sound RoPE modulation (e.g., sound_latent_fps=25).
            memory: Optional MemoryState for persistent KV-cache memory
                (AR inference or rolling-KV-cache training).  Built by
                ``OmniMoTModel.build_memory_state()``.

        Returns:
            dict with keys:
                - "preds_vision": list[Tensor[C,T,H,W]], one per sample.
                - "preds_action": Velocity predictions for action tokens (if action_gen).
                - "preds_sound": Velocity predictions for sound tokens (if sound_gen).
                - "last_hidden_state": Last hidden state from the transformer.
                - "lbl_metadata_*": Load balancing metadata.
                - "ce_preds": Cross-entropy predictions (if predict_text_tokens is True).
        """
        # Note: During inference with @torch.no_grad(), model may be in training mode
        # This is intentional for proper batch norm / dropout behavior
        # assert self.training, "Cosmos3VFMNetwork only supports training mode"

        packed_sequence, target_dtype = self._encode_text(packed_seq)  # packed_sequence: [N_total,hidden_size]

        # encode vision tokens
        original_latent_shapes: List[Tuple[int, int, int]] | None = None
        if self.config.vision_gen:
            original_latent_shapes = self._encode_vision(packed_seq, packed_sequence, target_dtype, fps_vision)

        # encode action tokens
        if self.config.action_gen:
            self._encode_action(packed_seq, packed_sequence, target_dtype, fps_action)

        # encode sound tokens
        if self.config.sound_gen:
            self._encode_sound(packed_seq, packed_sequence, target_dtype, fps_sound)

        assert self.use_moe
        assert packed_seq.attn_modes is not None
        assert packed_seq.split_lens is not None

        # Get all generation sequence indexes for MoE routing
        # IMPORTANT: Include ALL latent tokens (video + action + sound), not just generation targets.
        # Condition tokens still need to be routed to diffusion experts; they are excluded from
        # LOSS computation, not from routing.
        all_gen_indexes = []
        if packed_seq.vision is not None:
            assert packed_seq.vision.token_shapes is not None
            assert isinstance(packed_seq.vision.sequence_indexes, torch.Tensor)
            all_gen_indexes.append(packed_seq.vision.sequence_indexes)
        if packed_seq.action is not None and isinstance(packed_seq.action.sequence_indexes, torch.Tensor):
            all_gen_indexes.append(packed_seq.action.sequence_indexes)
        if packed_seq.sound is not None and isinstance(packed_seq.sound.sequence_indexes, torch.Tensor):
            all_gen_indexes.append(packed_seq.sound.sequence_indexes)
        vision_sequence_indexes = torch.cat(all_gen_indexes, dim=0) if all_gen_indexes else None  # [N_gen_tokens]

        # When temporal causal is enabled the buffer is [action_t0, vision_t0, action_t1, vision_t1, ...].
        # After torch.cat([vision_indexes, action_indexes]) the interleaved order is lost; sorting restores it.
        if self.video_temporal_causal:
            assert packed_seq.sound is None, "Sound generation is not supported with video_temporal_causal=True."
            if vision_sequence_indexes is not None:
                vision_sequence_indexes = vision_sequence_indexes.sort().values  # [N_gen_tokens]

        vision_token_shapes = packed_seq.vision.token_shapes if packed_seq.vision else None

        # The packer is the single source of truth for the supertoken layout.
        # ``num_action_tokens_per_supertoken`` is stamped onto ``packed_seq`` by
        # ``_pack_supertokens_temporal_causal`` (= tcf when actions are packed
        # inline, 0 otherwise) and read unchanged by the attention builder, the
        # NATTEN metadata generator, and the rolling KV-cache state — keeping
        # all downstream supertoken geometry automatically in sync with the pack.
        num_action_tokens_per_supertoken = packed_seq.num_action_tokens_per_supertoken

        input_pack, attention_meta, natten_metadata_list = build_packed_sequence(
            self.config.joint_attn_implementation,
            packed_sequence=packed_sequence,
            attn_modes=packed_seq.attn_modes,
            split_lens=packed_seq.split_lens,
            sample_lens=packed_seq.sample_lens,
            packed_und_token_indexes=packed_seq.text_indexes,
            packed_gen_token_indexes=vision_sequence_indexes,
            num_heads=self.num_heads,
            is_image_batch=packed_seq.is_image_batch,
            head_dim=self.head_dim,
            num_layers=self.num_hidden_layers,
            token_shapes=packed_seq.vision.token_shapes,
            natten_parameter_list=self.natten_parameter_list,
            cp_world_size=self.parallel_dims.cp_size if self.parallel_dims else 1,
            video_temporal_causal=self.video_temporal_causal,
            use_rolling_kv_cache=memory is not None and memory.uses_rolling_kv_cache,
            vision_token_shapes=vision_token_shapes,
            action_token_shapes=packed_seq.action.token_shapes if packed_seq.action else None,
            num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
            null_action_supertokens=packed_seq.null_action_supertokens,
            pad_for_cuda_graphs=self.pad_for_cuda_graphs,
        )

        input_pack, packed_position_ids = get_context_parallel_sharded_sequence(
            attn_implementation=self.config.joint_attn_implementation,
            input_pack=input_pack,
            position_ids=packed_seq.position_ids,
            parallel_dims=self.parallel_dims,
        )

        packed_outputs, lbl_metadata = self.language_model(
            input_pack,
            attention_mask=attention_meta,
            position_ids=packed_position_ids,
            natten_metadata_list=natten_metadata_list,
            memory=memory,
        )
        last_hidden_state = get_context_parallel_last_hidden_state(
            packed_outputs=packed_outputs,
            parallel_dims=self.parallel_dims,
        )  # [N_total,hidden_size]
        output_dict = dict()

        # decode vision tokens
        if self.config.vision_gen:
            self._decode_vision(packed_seq, last_hidden_state, output_dict, original_latent_shapes)

        # decode action tokens
        if self.config.action_gen:
            self._decode_action(packed_seq, last_hidden_state, output_dict)

        # decode sound tokens
        if self.config.sound_gen:
            self._decode_sound(packed_seq, last_hidden_state, output_dict)

        output_dict.update(last_hidden_state=last_hidden_state)
        for lbl_metadata_key, lbl_metadata_value in lbl_metadata.items():
            output_dict.update({f"lbl_metadata_{lbl_metadata_key}": lbl_metadata_value})
        if self.predict_text_tokens:
            packed_ce_preds = self.language_model.lm_head(
                last_hidden_state[packed_seq.ce_loss_indexes]
            )  # [N_ce_tokens,vocab_size]
            output_dict["ce_preds"] = packed_ce_preds

        return output_dict


def _apply_timestep_embeds_to_noisy_tokens(
    packed_tokens: torch.Tensor,
    packed_timestep_embeds: torch.Tensor,
    noisy_frame_indexes: List[torch.Tensor],
    token_shapes: list[tuple[int, ...]],
) -> torch.Tensor:
    """Apply timestep embeddings to noisy tokens.
    Tn is the number of noisy frames for a given sample.
    Tc is the number of clean frames for a given sample.
    T is the total number of frames for a given sample.
    T = Tn + Tc

    Args:
        packed_tokens: The packed tokens to apply timestep embeddings to.
        packed_timestep_embeds: The packed timestep embeddings to apply.
        noisy_frame_indexes: The frame indices to apply timestep embeddings to
            (list of tensors, each with shape (Tn,)).
        token_shapes: The token shapes for each sample. Each entry is a tuple
            shaped like ``(T, ...)`` where trailing dimensions represent the spatial grid.

    Returns:
        The packed tokens with timestep embeddings applied to the noisy tokens.
    """

    # Handle variable token shapes by processing each sample's noisy_frame_indexes individually.
    # The noisy indices are first expanded to cover the entire spatial grid of each frame.
    #
    # For video frames, the spatial grid is (H, W).
    # For action frames, the spatial grid is ().
    # For sound frames, the spatial grid is (1, 1).
    #
    # The noisy indices are then flattened into a single tensor overall. When flattening,
    # we must ensure that the noisy indices from each sample are unique by adding the
    # cumulative sum of the token shapes of previous samples to the noisy indices for
    # a given sample.
    start_noisy_index = 0
    flattened_noisy_frame_indexes = []

    for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes):
        assert noisy_indexes_i.numel() <= token_shape_i[0]
        spatial_numel_i = math.prod(token_shape_i[1:])
        spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)  # [spatial_numel_i]
        noisy_indexes_i = (
            (noisy_indexes_i * spatial_numel_i).unsqueeze(-1).expand(-1, spatial_numel_i)
        )  # [Tn_i,spatial_numel_i]
        noisy_indexes_i = noisy_indexes_i.clone() + spatial_indexes_i + start_noisy_index  # [Tn_i,spatial_numel_i]
        flattened_noisy_frame_indexes.append(noisy_indexes_i.flatten())  # [Tn_i*spatial_numel_i]
        start_noisy_index += math.prod(token_shape_i)

    flattened_noisy_frame_indexes = torch.cat(flattened_noisy_frame_indexes, dim=0)  # [total_noisy_patches]

    assert packed_tokens.dim() == 2
    assert packed_timestep_embeds.dim() == 2
    assert packed_timestep_embeds.shape[1] == packed_tokens.shape[1]
    assert packed_timestep_embeds.shape[0] <= packed_tokens.shape[0]
    assert flattened_noisy_frame_indexes.dim() == 1
    assert flattened_noisy_frame_indexes.shape[0] == packed_timestep_embeds.shape[0]

    flattened_noisy_frame_indexes = flattened_noisy_frame_indexes.unsqueeze(-1).expand(
        -1,
        packed_tokens.shape[1],
    )  # [total_noisy_patches,hidden_size]

    return packed_tokens.scatter_add(
        dim=0,
        index=flattened_noisy_frame_indexes,
        src=packed_timestep_embeds,
    )  # [total_tokens,hidden_size]
