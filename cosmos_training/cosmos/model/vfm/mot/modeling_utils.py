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
from typing import Optional

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torch.distributed import ProcessGroup
from transformers.activations import ACT2FN

from cosmos.data.vfm.sequence_packing import ModalityData


def has_noisy_tokens(modality_data: ModalityData | None) -> bool:
    """Check if a modality has valid noisy tokens for loss computation."""
    return (
        modality_data is not None
        and modality_data.tokens is not None
        and isinstance(modality_data.mse_loss_indexes, torch.Tensor)
        and modality_data.mse_loss_indexes.numel() > 0
    )


# --------------------------------------------------------
# 2D sine-cosine position embedding (flattened)
# References:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size_h: int, grid_size_w: int, cls_token: bool = False, extra_tokens: int = 0
) -> np.ndarray:
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [H*W,D/2]
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [H*W,D/2]

    emb = np.concatenate([emb_h, emb_w], axis=1)  # [H*W,D]
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size [M]
    out: [M,D]
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # [D/2]

    pos = pos.reshape(-1)  # [M]
    out = np.einsum("m,d->md", pos, omega)  # [M,D/2], outer product

    emb_sin = np.sin(out)  # [M,D/2]
    emb_cos = np.cos(out)  # [M,D/2]

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # [M,D]
    return emb


class FlattenedSinCosPositionEmbedding(nn.Module):
    # This module creates a flattened sin-cos position embedding for a given number of patches per side.
    # Indices are created for 2D array and flattened into 1D array.

    def __init__(self, max_latent_h: int, max_latent_w: int, hidden_size: int, interpolate_pos: bool = False):
        super().__init__()
        self.max_latent_h = max_latent_h
        self.max_latent_w = max_latent_w
        self.hidden_size = hidden_size
        self.interpolate_pos = interpolate_pos
        self.pos_embed = nn.Parameter(torch.zeros(max_latent_h * max_latent_w, hidden_size), requires_grad=False)
        self._init_weights()

    def _get_flattened_position_ids_extrapolate(self, latent_dim_h: int, latent_dim_w: int) -> torch.Tensor:
        coords_h = torch.arange(0, latent_dim_h)  # [H]
        coords_w = torch.arange(0, latent_dim_w)  # [W]
        pos_ids = (coords_h[:, None] * self.max_latent_w + coords_w).flatten()  # [H*W]
        return pos_ids

    def _get_flattened_position_ids_interpolate(self, latent_dim_h: int, latent_dim_w: int) -> torch.Tensor:
        boundaries = torch.arange(1 / self.max_latent_w, 1.0, 1 / self.max_latent_w)  # [max_latent_w-1]
        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / latent_dim_h)  # [H]
        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / latent_dim_w)  # [W]
        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)  # [H]
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)  # [W]
        pos_ids = (bucket_coords_h[:, None] * self.max_latent_w + bucket_coords_w).flatten()  # [H*W]
        return pos_ids

    def _create_flattened_position_ids_packed(self, token_shapes_vision: list[tuple[int, int]]) -> torch.Tensor:
        flattened_position_ids = []
        for t, h, w in token_shapes_vision:
            if self.interpolate_pos:
                flattened_position_ids.append(self._get_flattened_position_ids_interpolate(h, w))  # [H*W]
            else:
                flattened_position_ids.append(self._get_flattened_position_ids_extrapolate(h, w))  # [H*W]
        flattened_position_ids_packed = torch.cat(flattened_position_ids, dim=0)  # [N_vision]
        return flattened_position_ids_packed

    def _init_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.hidden_size, grid_size_h=self.max_latent_h, grid_size_w=self.max_latent_w
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, token_shapes_vision: list[tuple[int, int]], fps: Optional[torch.Tensor] = None) -> torch.Tensor:
        # First create 2D index array
        flattened_position_ids_packed = self._create_flattened_position_ids_packed(token_shapes_vision)  # [N_vision]
        return self.pos_embed[flattened_position_ids_packed]  # [N_vision,hidden_size]


# --------------------------------------------------------
# 2D / 3D RoPE Position Embedding
# --------------------------------------------------------


class VideoRopePosition3DEmb(nn.Module):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        base_temporal_compression_factor: int = 4,
        temporal_compression_factor: int = 4,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = False,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.base_tps = base_fps / base_temporal_compression_factor
        self.temporal_compression_factor = temporal_compression_factor
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"

        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self._init_weights()

    def _init_weights(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.dim_spatial_range.device) / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.dim_spatial_range.device) / dim_t
        )

    def enable_context_parallel(self, process_group: ProcessGroup):
        pass

    def disable_context_parallel(self):
        pass

    def generate_embeddings(
        self,
        latent_shape: torch.Size,
        input_fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
        start_frame_offset: int = 0,
    ):
        """
        Generate embeddings for the given input size.

        Args:
            latent_shape (torch.Size): Input tensor size (Batch, Time, Height, Width).
            input_fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor.
            start_frame_offset (int, optional): Offset for frame indices. Use 1 for action embeddings
                so that action frame indices start at 1 instead of 0. Defaults to 0.

        Returns:
            Not specified in the original code snippet.
        """
        if input_fps is not None:
            tps = input_fps / self.temporal_compression_factor
        else:
            tps = None

        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor
        assert h_ntk_factor is not None and w_ntk_factor is not None and t_ntk_factor is not None

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.float())  # [dim_h/2]
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.float())  # [dim_w/2]
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.float())  # [dim_t/2]

        B, T, H, W = latent_shape
        assert H <= self.max_h and W <= self.max_w, (
            f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        )

        # Re-allocate buffer if current video needs more indices than what we have for self.seq
        # Only rellocate when needed.
        max_needed = max(T, H, W)
        seq = torch.arange(max_needed, device=self.dim_spatial_range.device, dtype=torch.float)

        half_emb_h = torch.outer(seq[:H], h_spatial_freqs)  # [H,dim_h/2]
        half_emb_w = torch.outer(seq[:W], w_spatial_freqs)  # [W,dim_w/2]

        # Frame indices for the embedding (always 0, 1, 2, ...)
        frame_indices = seq[:T]  # [T]

        if self.enable_fps_modulation:
            uniform_tps = tps is None or tps.shape == (1,)
            assert uniform_tps or B == 1 or T == 1, (
                "For video batch, B should be 1 for non-uniform fps. For image batch, T should be 1."
            )

            # apply sequence scaling in temporal dimension
            if tps is None:  # image case
                assert T == 1, "T should be 1 for image batch."
                half_emb_t = torch.outer(frame_indices, temporal_freqs)  # [T,dim_t/2]
            else:
                # Calculate scaled time indices
                # Apply start_frame_offset to the time calculation (not frame indices)
                # This allows one to manipulate the start frame index of embeddings for cross-modality alignment.
                scaled_time = (frame_indices + start_frame_offset) / tps[:1] * self.base_tps  # [T]
                half_emb_t = torch.outer(scaled_time, temporal_freqs)  # [T,dim_t/2]
        else:
            half_emb_t = torch.outer(frame_indices, temporal_freqs)  # [T,dim_t/2]

        rope_embed = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),  # [T,H,W,dim_t/2]
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),  # [T,H,W,dim_h/2]
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),  # [T,H,W,dim_w/2]
            ]
            * 2,
            dim=-1,
        )  # [T,H,W,head_dim]

        return rearrange(rope_embed, "t h w d -> (t h w) d").float()  # [T*H*W,head_dim]

    def forward(
        self,
        token_shapes_vision: list[tuple[int, int, int]],
        fps: Optional[torch.Tensor] = None,
        start_frame_offset: int = 0,
    ) -> torch.Tensor:
        """
        With CP, the function assume that the input tensor is already split.
        It delegates the embedding generation to generate_embeddings function.

        Args:
            token_shapes_vision: List of (t, h, w) tuples for each latent.
            fps: Frames per second tensor.
            start_frame_offset: Offset for frame indices. Use 1 for action embeddings
                so that action frame indices start at 1 instead of 0. Defaults to 0.
        """

        embeddings_packed = []
        for i, latent_shape in enumerate(token_shapes_vision):
            # latent_shape: (t, h, w)
            shape = (1, latent_shape[0], latent_shape[1], latent_shape[2])

            # Extract FPS for this specific video
            video_fps = None
            if fps is not None:
                assert i < fps.shape[0], f"Index {i} out of bounds for fps tensor of shape {fps.shape}"
                video_fps = fps[i : i + 1]

            embeddings = self.generate_embeddings(shape, input_fps=video_fps, start_frame_offset=start_frame_offset)
            embeddings_packed.append(embeddings)

        embeddings_packed = torch.cat(embeddings_packed, dim=0)  # [N_vision,head_dim]
        return embeddings_packed

    @property
    def seq_dim(self):
        return 0


# --------------------------------------------------------
# TimestepEmbedder
# Reference:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.frequency_embedding_size)
        torch.nn.init.trunc_normal_(self.mlp[0].weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.zeros_(self.mlp[0].bias)

        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.mlp[2].weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.zeros_(self.mlp[2].bias)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )  # [D/2]
        args = t[:, None].float() * freqs[None]  # [N,D/2]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [N,D]
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # [N,D+1]
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)  # [N,frequency_embedding_size]
        t_emb = self.mlp(t_freq)  # [N,hidden_size]
        return t_emb


class MLPconnector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_act: str):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)  # [N,out_dim]
        hidden_states = self.activation_fn(hidden_states)  # [N,out_dim]
        hidden_states = self.fc2(hidden_states)  # [N,out_dim]
        return hidden_states
