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

"""Sparse transformer building blocks.

This module provides:
    - AbsolutePositionEmbedder: Sinusoidal position embeddings
    - LearnedPositionEmbedder: 2D learned position embeddings
    - LearnedPositionEmbedder4D: 4D learned position embeddings (t, x, y, z)
    - SparseFeedForwardNet: MLP for sparse tensors
    - SparseTransformerBlock: Standard transformer block (MSA + FFN)
    - SparseMultiheadAttentionPoolingHead: Pooling head using cross-attention
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos.model.tokenizer.models.modules.sparse_ops import (
    LayerNorm32,
    RMSNorm32,
    SparseActivation,
    SparseGELU,
    SparseLinear,
    SparseReLU,
    SparseSiLU,
)
from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

__all__ = [
    "AbsolutePositionEmbedder",
    "LearnedPositionEmbedder",
    "LearnedPositionEmbedder4D",
    "SparseFeedForwardNet",
    "SparseTransformerBlock",
    "SparseMultiheadAttentionPoolingHead",
]


class AbsolutePositionEmbedder(nn.Module):
    """Embeds spatial positions into vector representations using sinusoidal embeddings."""

    def __init__(self, channels: int, in_channels: int = 3):
        """Initialize AbsolutePositionEmbedder.

        Args:
            channels: Output embedding dimension.
            in_channels: Number of spatial dimensions (e.g., 3 for x, y, z).
        """
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        freqs = 1.0 / (10000**freqs)
        # Register as buffer so it stays on the correct device
        self.register_buffer("freqs", freqs, persistent=False)

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal position embeddings.

        Args:
            x: 1-D Tensor of N indices.

        Returns:
            (N, D) Tensor of positional embeddings.
        """
        # freqs is now a registered buffer, already on correct device
        out = torch.outer(x.to(self.freqs.dtype), self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute position embeddings.

        Args:
            x: (N, D) tensor of spatial positions.

        Returns:
            (N, channels) tensor of position embeddings.
        """
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat(
                [
                    embed,
                    torch.zeros(N, self.channels - embed.shape[1], device=embed.device),
                ],
                dim=-1,
            )
        return embed


class SparseFeedForwardNet(nn.Module):
    """Feed-forward network for sparse tensors (MLP with GELU activation)."""

    def __init__(self, channels: int, mlp_channels: int = 2048, use_bias: bool = False):
        """Initialize SparseFeedForwardNet.

        Args:
            channels: Number of input/output channels.
            mlp_channels: Hidden layer size.
            use_bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, mlp_channels, bias=use_bias),
            SparseGELU(approximate="tanh"),
            SparseLinear(mlp_channels, channels, bias=use_bias),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        """Apply MLP to sparse tensor.

        Args:
            x: Input SparseTensor.

        Returns:
            Output SparseTensor.
        """
        return self.mlp(x)

    def forward_tensor(self, feats: torch.Tensor) -> torch.Tensor:
        """Apply the same MLP directly to flat token features."""
        fc1 = self.mlp[0]
        activation = self.mlp[1]
        fc2 = self.mlp[2]
        hidden = F.linear(feats, fc1.weight, fc1.bias)
        if isinstance(activation, SparseGELU):
            hidden = F.gelu(hidden, approximate=activation.approximate)
        elif isinstance(activation, SparseSiLU):
            hidden = F.silu(hidden, inplace=activation.inplace)
        elif isinstance(activation, SparseReLU):
            hidden = F.relu(hidden, inplace=activation.inplace)
        elif isinstance(activation, SparseActivation):
            hidden = activation.activation(hidden)
        else:
            hidden = activation(hidden)
        return F.linear(hidden, fc2.weight, fc2.bias)


class SparseTransformerBlock(nn.Module):
    """Sparse Transformer block (MSA + FFN) with optional gradient checkpointing."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_channels: int = 2048,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        use_bias: bool = False,
        use_rms_norm: bool = True,
        ln_affine: bool = True,
        multiscale: Any | None = None,
        layer_idx: int | None = None,
    ):
        """Initialize SparseTransformerBlock.

        Args:
            channels: Number of input/output channels.
            num_heads: Number of attention heads.
            mlp_channels: Hidden layer size for MLP.
            use_checkpoint: Whether to use gradient checkpointing.
            use_rope: Whether to use rotary position embeddings.
            qk_rms_norm: Whether to apply RMS normalization to Q and K.
            use_bias: Whether to use bias in linear layers.
            use_rms_norm: Whether to use RMSNorm (vs LayerNorm).
            ln_affine: Whether to use affine parameters in LayerNorm.
            multiscale: Configuration for multiscale expansion (optional).
            layer_idx: Optional layer index used for debug artifact stamping.
        """
        super().__init__()
        # Import here to avoid circular imports
        from cosmos.model.tokenizer.models.modules.attention.modules import SparseMultiHeadAttention

        self.use_checkpoint = use_checkpoint
        self.multiscale = multiscale
        self.layer_idx = layer_idx

        if use_rms_norm:
            self.norm1 = RMSNorm32(channels, eps=1e-6)
            self.norm2 = RMSNorm32(channels, eps=1e-6)
        else:
            self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
            self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)

        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            use_bias=use_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.attn.layer_idx = layer_idx
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_channels=mlp_channels,
            use_bias=use_bias,
        )

    def _forward(
        self,
        x: SparseTensor,
        kv_cache: dict[str, SparseTensor] | None = None,
        kv_cache_size: int | None = None,
        kv_cache_detach: bool = True,
        temporal_causal_mask: bool = False,
    ) -> tuple[SparseTensor, dict[str, SparseTensor]]:
        """Internal forward pass.

        Args:
            x: Input SparseTensor.
            kv_cache: Optional KV cache for attention.
            kv_cache_size: Size limit for KV cache.
            kv_cache_detach: Whether tensors stored in the KV cache should be detached.
            temporal_causal_mask: Whether to apply temporal-causal, same-timestep
                bidirectional self-attention.

        Returns:
            Tuple of (output SparseTensor, updated kv_cache).
        """
        if kv_cache is None:
            kv_cache = {}

        if self.multiscale is not None and "factor" in self.multiscale:
            x = x.expand_by_factors(
                self.multiscale["factor"],
                channel_duplicate=self.multiscale["channel_duplication"],
            )

        h = x.replace(self.norm1(x.feats))
        h, kv_cache = self.attn(
            h,
            kv_cache=kv_cache,
            kv_cache_size=kv_cache_size,
            kv_cache_detach=kv_cache_detach,
            temporal_causal_mask=temporal_causal_mask,
        )
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.mlp(h)
        x = x + h

        return x, kv_cache

    def _forward_no_cache(self, x: SparseTensor, temporal_causal_mask: bool = False) -> SparseTensor:
        """Internal forward pass for the common no-cache path."""
        if self.multiscale is not None and "factor" in self.multiscale:
            x = x.expand_by_factors(
                self.multiscale["factor"],
                channel_duplicate=self.multiscale["channel_duplication"],
            )

        h = x.replace(self.norm1(x.feats))
        h = self.attn.forward_no_cache(h, temporal_causal_mask=temporal_causal_mask)
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.mlp(h)
        x = x + h
        return x

    def forward(
        self,
        x: SparseTensor,
        kv_cache: dict[str, SparseTensor] | None = None,
        kv_cache_size: int | None = None,
        kv_cache_detach: bool = True,
        temporal_causal_mask: bool = False,
    ) -> tuple[SparseTensor, dict[str, SparseTensor]]:
        """Forward pass with optional gradient checkpointing.

        Args:
            x: Input SparseTensor.
            kv_cache: Optional KV cache for attention.
            kv_cache_size: Size limit for KV cache.
            kv_cache_detach: Whether tensors stored in the KV cache should be detached.
            temporal_causal_mask: Whether to apply temporal-causal, same-timestep
                bidirectional self-attention.

        Returns:
            Tuple of (output SparseTensor, updated kv_cache).
        """
        if self.training and self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                x,
                kv_cache,
                kv_cache_size,
                kv_cache_detach,
                temporal_causal_mask,
                use_reentrant=False,
            )
        else:
            return self._forward(x, kv_cache, kv_cache_size, kv_cache_detach, temporal_causal_mask)

    def forward_no_cache(self, x: SparseTensor, temporal_causal_mask: bool = False) -> SparseTensor:
        """Forward pass without KV-cache plumbing."""
        if self.training and self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward_no_cache,
                x,
                temporal_causal_mask,
                use_reentrant=False,
            )
        return self._forward_no_cache(x, temporal_causal_mask)

    def forward_tensor_no_cache(
        self,
        feats: torch.Tensor,
        q_seqlen: list[int],
        cu_seqlens_q: torch.Tensor,
        max_q_seqlen: int,
        q_freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass directly on flat token features for the eval fast path."""
        if self.training and self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                lambda input_feats: self._forward_tensor_no_cache(
                    input_feats,
                    q_seqlen=q_seqlen,
                    cu_seqlens_q=cu_seqlens_q,
                    max_q_seqlen=max_q_seqlen,
                    q_freqs_cis=q_freqs_cis,
                ),
                feats,
                use_reentrant=False,
            )
        return self._forward_tensor_no_cache(
            feats,
            q_seqlen=q_seqlen,
            cu_seqlens_q=cu_seqlens_q,
            max_q_seqlen=max_q_seqlen,
            q_freqs_cis=q_freqs_cis,
        )

    def _forward_tensor_no_cache(
        self,
        feats: torch.Tensor,
        q_seqlen: list[int],
        cu_seqlens_q: torch.Tensor,
        max_q_seqlen: int,
        q_freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass directly on flat token features for tensor-varlen execution."""
        if self.multiscale is not None and "factor" in self.multiscale:
            raise NotImplementedError("Tensor no-cache block path does not support multiscale expansion.")

        h = self.norm1(feats)
        h = self.attn.forward_tensor_no_cache(
            h,
            q_seqlen=q_seqlen,
            cu_seqlens_q=cu_seqlens_q,
            max_q_seqlen=max_q_seqlen,
            q_freqs_cis=q_freqs_cis,
        )
        feats = feats + h
        h = self.norm2(feats)
        h = self.mlp.forward_tensor(h)
        return feats + h

    def forward_tensor_flat_kv(
        self,
        feats: torch.Tensor,
        current_times: torch.Tensor,
        q_seqlen: list[int],
        cu_seqlens_q: torch.Tensor,
        max_q_seqlen: int,
        kv_cache: dict[str, object] | None = None,
        kv_cache_size: int | None = None,
        kv_cache_detach: bool = True,
        q_freqs_cis: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Forward pass on flat query tensors with the flat KV-cache attention fast path."""
        if self.multiscale is not None and "factor" in self.multiscale:
            raise NotImplementedError("Tensor flat-KV block path does not support multiscale expansion.")

        h = self.norm1(feats)
        h, kv_cache = self.attn.forward_tensor_flat_kv(
            h,
            current_times=current_times,
            q_seqlen=q_seqlen,
            cu_seqlens_q=cu_seqlens_q,
            max_q_seqlen=max_q_seqlen,
            kv_cache=kv_cache,
            kv_cache_size=kv_cache_size,
            kv_cache_detach=kv_cache_detach,
            q_freqs_cis=q_freqs_cis,
        )
        feats = feats + h
        h = self.norm2(feats)
        h = self.mlp.forward_tensor(h)
        return feats + h, {} if kv_cache is None else kv_cache


class LearnedPositionEmbedder(nn.Module):
    """Learned 2D position embeddings for sparse tensors."""

    def __init__(self, hidden_size: int, num_patches: int = 256, max_t: int = 32, max_z: int = 16):
        """Initialize LearnedPositionEmbedder.

        Args:
            hidden_size: Embedding dimension.
            num_patches: Number of 2D patches (assumes square grid).
            max_t: Maximum temporal dimension (unused, for API compatibility).
            max_z: Maximum depth dimension (unused, for API compatibility).
        """
        super().__init__()
        self.embed_dim = hidden_size
        self.num_patches = num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self._resized_position_embedding_cache: dict[tuple[int, int, str, torch.dtype, int, int], torch.Tensor] = {}

    def train(self, mode: bool = True) -> "LearnedPositionEmbedder":
        """Switch training mode and clear eval-only interpolation cache."""
        self._resized_position_embedding_cache.clear()
        return super().train(mode)

    def _get_interpolated_position_embedding(
        self,
        positional_embeddings: torch.Tensor,
        target_height: int,
        target_width: int,
        target_device: torch.device,
    ) -> torch.Tensor:
        """Return a resized positional embedding grid for one spatial shape."""
        compute_dtype = positional_embeddings.dtype
        if positional_embeddings.device.type == "cpu":
            compute_dtype = torch.float32

        cache_key = (
            target_height,
            target_width,
            str(target_device),
            compute_dtype,
            self.position_embedding.weight.data_ptr(),
            self.position_embedding.weight._version,
        )
        if not self.training:
            cached = self._resized_position_embedding_cache.get(cache_key)
            if cached is not None:
                return cached

        if target_height == self.position_embedding_size and target_width == self.position_embedding_size:
            interpolated = positional_embeddings
            if interpolated.device != target_device or interpolated.dtype != compute_dtype:
                interpolated = interpolated.to(device=target_device, dtype=compute_dtype)
        else:
            pos_emb = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
            if pos_emb.device != target_device or pos_emb.dtype != compute_dtype:
                pos_emb = pos_emb.to(device=target_device, dtype=compute_dtype)
            interpolated = (
                F.interpolate(
                    pos_emb,
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                .squeeze(0)
                .permute(1, 2, 0)
            )

        if not self.training:
            self._resized_position_embedding_cache[cache_key] = interpolated
        return interpolated

    def infer_spatial_shapes(self, sparse_patches: SparseTensor) -> torch.LongTensor:
        """Infer spatial shapes from sparse tensor (vectorized implementation).

        Args:
            sparse_patches: Sparse tensor with coordinates.

        Returns:
            Spatial shapes of shape (batch_size, 2) with [height, width].
        """
        batch_size = sparse_patches.shape[0]
        device = sparse_patches.device

        if sparse_patches.coords.shape[0] == 0:
            return torch.ones((batch_size, 2), dtype=torch.long, device=device)

        batch_indices = sparse_patches.coords[:, 0].long()
        x_coords = sparse_patches.coords[:, 2].long()
        y_coords = sparse_patches.coords[:, 3].long()

        max_x = torch.zeros(batch_size, dtype=torch.long, device=device)
        max_y = torch.zeros(batch_size, dtype=torch.long, device=device)

        max_x = max_x.scatter_reduce(0, batch_indices, x_coords, reduce="amax", include_self=True)
        max_y = max_y.scatter_reduce(0, batch_indices, y_coords, reduce="amax", include_self=True)

        spatial_shapes = torch.stack([max_x + 1, max_y + 1], dim=1)

        return spatial_shapes

    def resize_positional_embeddings_sparse(
        self,
        positional_embeddings: torch.Tensor,
        sparse_patches: SparseTensor,
    ) -> torch.Tensor:
        """Resize positional embeddings for sparse tensor format.

        Optimized implementation that groups patches by unique target shapes,
        calling F.interpolate once per unique shape instead of once per batch.

        Args:
            positional_embeddings: Base positional embeddings (height, width, embed_dim).
            sparse_patches: Sparse tensor with coordinates.

        Returns:
            Resized positional embeddings of shape (num_patches, embed_dim).
        """
        spatial_shapes = self.infer_spatial_shapes(sparse_patches)

        num_patches = sparse_patches.coords.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype
        device = sparse_patches.device

        if num_patches == 0:
            return torch.empty((0, embed_dim), device=device, dtype=source_dtype)

        resized_embeddings = torch.empty(
            (num_patches, embed_dim),
            device=device,
            dtype=source_dtype,
        )

        batch_indices = sparse_patches.coords[:, 0].long()
        sort_order = torch.argsort(batch_indices, stable=True)
        sorted_batch_indices = batch_indices[sort_order]
        unique_batch_indices, batch_counts = torch.unique_consecutive(sorted_batch_indices, return_counts=True)

        # Materialize the small batch-level shape metadata on CPU once so the
        # per-batch loop does not repeatedly scalarize GPU tensors into Python.
        spatial_shape_values = spatial_shapes.to(device="cpu").tolist()
        unique_batch_values = unique_batch_indices.to(device="cpu").tolist()
        batch_count_values = batch_counts.to(device="cpu").tolist()
        interpolated_cache: dict[tuple[int, int], torch.Tensor] = {}

        start = 0
        for batch_index, patch_count in zip(unique_batch_values, batch_count_values):
            end = start + patch_count
            patch_indices = sort_order[start:end]
            start = end

            height_val, width_val = spatial_shape_values[batch_index]
            shape_key = (height_val, width_val)
            interpolated = interpolated_cache.get(shape_key)
            if interpolated is None:
                interpolated = self._get_interpolated_position_embedding(
                    positional_embeddings,
                    height_val,
                    width_val,
                    device,
                )
                interpolated_cache[shape_key] = interpolated

            selected_coords = sparse_patches.coords[patch_indices]
            patch_x = torch.clamp(selected_coords[:, 2].long(), 0, height_val - 1)
            patch_y = torch.clamp(selected_coords[:, 3].long(), 0, width_val - 1)

            selected_embeddings = interpolated[patch_x, patch_y].to(source_dtype)
            resized_embeddings[patch_indices] = selected_embeddings

        return resized_embeddings

    def forward(self, sparse_patches: SparseTensor) -> SparseTensor:
        """Forward pass for sparse vision embeddings.

        Args:
            sparse_patches: Sparse tensor with patch features.

        Returns:
            Patches with positional embeddings added.
        """
        if sparse_patches.coords.shape[0] == 0:
            return sparse_patches

        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )

        resized_positional_embeddings = self.resize_positional_embeddings_sparse(positional_embeddings, sparse_patches)

        return sparse_patches.replace(resized_positional_embeddings)


class LearnedPositionEmbedder4D(nn.Module):
    """Learned 4D position embeddings (2D spatial + temporal + depth)."""

    def __init__(self, hidden_size: int, num_patches_2d: int = 256, max_t: int = 32, max_z: int = 16):
        """Initialize LearnedPositionEmbedder4D.

        Args:
            hidden_size: Embedding dimension.
            num_patches_2d: Number of 2D patches (assumes square grid).
            max_t: Maximum temporal dimension.
            max_z: Maximum depth dimension.
        """
        super().__init__()
        self.embed_dim = hidden_size

        self.num_patches_2d = num_patches_2d
        self.spatial_embedding_size_2d = int(round(self.num_patches_2d**0.5))

        expected_patches = self.spatial_embedding_size_2d**2
        if abs(expected_patches - num_patches_2d) > 1:
            import warnings

            warnings.warn(
                f"num_patches_2d={num_patches_2d} is not a perfect square. "
                f"Using spatial_embedding_size_2d={self.spatial_embedding_size_2d} "
                f"(square={expected_patches})"
            )

        self.position_embedding = nn.Embedding(self.num_patches_2d, self.embed_dim)
        self.position_embedding_t = nn.Embedding(max_t, self.embed_dim)
        self.position_embedding_z = nn.Embedding(max_z, self.embed_dim)

        self.position_embedding_t.weight.data.zero_()
        self.position_embedding_z.weight.data.zero_()

    def infer_shapes(self, sparse_patches: SparseTensor) -> torch.LongTensor:
        """Infer 4D spatial shapes from sparse tensor (vectorized implementation).

        Args:
            sparse_patches: Sparse tensor with 4D coordinates [batch_idx, t, x, y, z].

        Returns:
            Spatial shapes of shape (batch_size, 4) with [max_t, max_x, max_y, max_z].
        """
        batch_size = sparse_patches.shape[0]
        device = sparse_patches.device
        coords = sparse_patches.coords

        if coords.shape[0] == 0:
            return torch.ones((batch_size, 4), dtype=torch.long, device=device)

        batch_indices = coords[:, 0].long()

        max_t = torch.zeros(batch_size, dtype=torch.long, device=device)
        max_x = torch.zeros(batch_size, dtype=torch.long, device=device)
        max_y = torch.zeros(batch_size, dtype=torch.long, device=device)
        max_z = torch.zeros(batch_size, dtype=torch.long, device=device)

        if coords.shape[1] > 1:
            t_coords = coords[:, 1].long()
            max_t = max_t.scatter_reduce(0, batch_indices, t_coords, reduce="amax", include_self=True)

        x_coords = coords[:, 2].long()
        y_coords = coords[:, 3].long()
        max_x = max_x.scatter_reduce(0, batch_indices, x_coords, reduce="amax", include_self=True)
        max_y = max_y.scatter_reduce(0, batch_indices, y_coords, reduce="amax", include_self=True)

        if coords.shape[1] > 4:
            z_coords = coords[:, 4].long()
            max_z = max_z.scatter_reduce(0, batch_indices, z_coords, reduce="amax", include_self=True)

        spatial_shapes = torch.stack([max_t + 1, max_x + 1, max_y + 1, max_z + 1], dim=1)

        return spatial_shapes

    def resize_spatial_embeddings_2d(
        self,
        spatial_embeddings: torch.Tensor,
        sparse_patches: SparseTensor,
    ) -> torch.Tensor:
        """Resize 2D spatial (x,y) positional embeddings for sparse tensor format.

        Optimized implementation that groups patches by unique target shapes,
        calling F.interpolate once per unique shape instead of once per batch.

        Args:
            spatial_embeddings: Base 2D spatial embeddings (height, width, embed_dim).
            sparse_patches: Sparse tensor with coordinates.

        Returns:
            Resized spatial embeddings of shape (num_patches, embed_dim).
        """
        spatial_shapes = self.infer_shapes(sparse_patches)

        num_patches = sparse_patches.coords.shape[0]
        embed_dim = spatial_embeddings.shape[-1]
        source_dtype = spatial_embeddings.dtype
        device = sparse_patches.device

        if num_patches == 0:
            return torch.zeros((0, embed_dim), device=device, dtype=source_dtype)

        resized_embeddings = torch.zeros(
            (num_patches, embed_dim),
            device=device,
            dtype=source_dtype,
        )

        spatial_emb = spatial_embeddings.permute(2, 0, 1).unsqueeze(0)

        if spatial_emb.device.type == "cpu":
            spatial_emb = spatial_emb.to(device=device, dtype=torch.float32)

        batch_indices = sparse_patches.coords[:, 0].long()
        patch_target_shapes = spatial_shapes[batch_indices][:, 1:3]

        unique_shapes, inverse_indices = torch.unique(patch_target_shapes, dim=0, return_inverse=True)

        for shape_idx in range(unique_shapes.shape[0]):
            height, width = unique_shapes[shape_idx]
            height_val, width_val = height.item(), width.item()

            patch_mask = inverse_indices == shape_idx
            patch_indices = patch_mask.nonzero(as_tuple=True)[0]

            if patch_indices.shape[0] == 0:
                continue

            interpolated = (
                F.interpolate(
                    spatial_emb,
                    size=(height_val, width_val),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                .squeeze(0)
                .permute(1, 2, 0)
            )

            selected_coords = sparse_patches.coords[patch_indices]
            patch_x = selected_coords[:, 2].long()
            patch_y = selected_coords[:, 3].long()

            selected_embeddings = interpolated[patch_x, patch_y].to(source_dtype)
            resized_embeddings[patch_indices] = selected_embeddings

        return resized_embeddings

    def get_1d_embeddings(
        self,
        sparse_patches: SparseTensor,
        coord_index: int,
        shape_index: int,
        embedding_layer: nn.Embedding,
        dim_name: str = "1D",
    ) -> torch.Tensor:
        """Get 1D positional embeddings (temporal or depth) with interpolation support.

        Optimized implementation that groups patches by unique max dimensions,
        calling F.interpolate once per unique dimension instead of once per batch.

        Args:
            sparse_patches: Sparse tensor with coordinates.
            coord_index: Index in coordinates for this dimension (1 for t, 4 for z).
            shape_index: Index in spatial_shapes for this dimension (0 for t, 3 for z).
            embedding_layer: The embedding layer to use.
            dim_name: Name for debugging/logging.

        Returns:
            1D embeddings of shape (num_patches, embed_dim).
        """
        coords = sparse_patches.coords
        device = sparse_patches.device
        num_patches = coords.shape[0]

        embeddings = torch.zeros(
            (num_patches, self.embed_dim),
            device=device,
            dtype=embedding_layer.weight.dtype,
        )

        if num_patches == 0 or coords.shape[1] <= coord_index:
            return embeddings

        shapes = self.infer_shapes(sparse_patches)
        batch_indices = coords[:, 0].long()
        dim_coords = coords[:, coord_index].long()
        patch_max_dims = shapes[batch_indices, shape_index]

        unique_max_dims = torch.unique(patch_max_dims)

        for max_dim in unique_max_dims:
            max_dim_val = max_dim.item()

            patch_mask = patch_max_dims == max_dim
            patch_indices = patch_mask.nonzero(as_tuple=True)[0]

            if patch_indices.shape[0] == 0:
                continue

            selected_dim_coords = dim_coords[patch_indices]

            if max_dim_val <= embedding_layer.num_embeddings:
                embeddings[patch_indices] = embedding_layer(selected_dim_coords)
            else:
                base_emb = embedding_layer.weight.transpose(0, 1).unsqueeze(0)

                if base_emb.device.type == "cpu":
                    base_emb = base_emb.to(device=device, dtype=torch.float32)

                resized_emb = (
                    F.interpolate(
                        base_emb,
                        size=max_dim_val,
                        mode="linear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                    .to(embedding_layer.weight.dtype)
                )

                embeddings[patch_indices] = resized_emb[selected_dim_coords]

        return embeddings

    def get_temporal_embeddings(self, sparse_patches: SparseTensor) -> torch.Tensor:
        """Get temporal positional embeddings with interpolation support.

        Args:
            sparse_patches: Sparse tensor with coordinates.

        Returns:
            Temporal embeddings of shape (num_patches, embed_dim).
        """
        return self.get_1d_embeddings(
            sparse_patches,
            coord_index=1,
            shape_index=0,
            embedding_layer=self.position_embedding_t,
            dim_name="temporal",
        )

    def get_depth_embeddings(self, sparse_patches: SparseTensor) -> torch.Tensor:
        """Get depth (z) positional embeddings with interpolation support.

        Args:
            sparse_patches: Sparse tensor with coordinates.

        Returns:
            Depth embeddings of shape (num_patches, embed_dim).
        """
        return self.get_1d_embeddings(
            sparse_patches,
            coord_index=4,
            shape_index=3,
            embedding_layer=self.position_embedding_z,
            dim_name="depth",
        )

    def forward(self, sparse_patches: SparseTensor) -> SparseTensor:
        """Forward pass for 4D sparse vision embeddings.

        Args:
            sparse_patches: Sparse tensor with patch features.
                - coords: (num_patches, 5) with [batch_idx, t, x, y, z]
                - feats: (num_patches, embed_dim)

        Returns:
            Patches with all positional embeddings added.
        """
        spatial_embeddings_2d = self.position_embedding.weight.reshape(
            self.spatial_embedding_size_2d, self.spatial_embedding_size_2d, -1
        )

        resized_spatial_embeddings = self.resize_spatial_embeddings_2d(spatial_embeddings_2d, sparse_patches)

        temporal_embeddings = self.get_temporal_embeddings(sparse_patches)
        depth_embeddings = self.get_depth_embeddings(sparse_patches)

        total_positional_embeddings = resized_spatial_embeddings + temporal_embeddings + depth_embeddings

        return sparse_patches.replace(total_positional_embeddings)


class SparseMultiheadAttentionPoolingHead(nn.Module):
    """Sparse Multihead Attention Pooling Head using cross-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int | None = None,
        layer_norm_eps: float = 1e-6,
        use_checkpoint: bool = False,
        use_bias: bool = True,
        use_rms_norm: bool = False,
        qk_rms_norm: bool = False,
    ):
        """Initialize SparseMultiheadAttentionPoolingHead.

        Args:
            hidden_size: Hidden dimension size.
            num_attention_heads: Number of attention heads.
            intermediate_size: MLP intermediate size (defaults to hidden_size * 4).
            layer_norm_eps: Layer norm epsilon.
            use_checkpoint: Whether to use gradient checkpointing.
            use_bias: Whether to use bias in linear layers.
            use_rms_norm: Whether to use RMSNorm (vs LayerNorm).
            qk_rms_norm: Whether to apply RMS norm to Q/K.
        """
        super().__init__()
        from cosmos.model.tokenizer.models.modules.attention.modules import SparseMultiHeadAttention

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.use_checkpoint = use_checkpoint

        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        self.probe = nn.Parameter(torch.randn(hidden_size))

        self.attention = SparseMultiHeadAttention(
            hidden_size,
            num_heads=num_attention_heads,
            ctx_channels=hidden_size,
            type="cross",
            use_bias=use_bias,
            qk_rms_norm=qk_rms_norm,
        )
        self.attention.layer_idx = -1

        if use_rms_norm:
            self.layernorm = RMSNorm32(hidden_size, eps=layer_norm_eps)
        else:
            self.layernorm = LayerNorm32(hidden_size, eps=layer_norm_eps)

        self.mlp = SparseFeedForwardNet(
            hidden_size,
            mlp_channels=intermediate_size,
            use_bias=use_bias,
        )

    def _forward(self, hidden_state: SparseTensor) -> SparseTensor:
        """Forward pass using cross-attention between probe tokens and input features.

        Args:
            hidden_state: SparseTensor with input features.

        Returns:
            Pooled features as SparseTensor.
        """
        batch_size = hidden_state.shape[0]
        device = hidden_state.device

        probe_coords = torch.full((batch_size, 4), -1, dtype=torch.int32, device=device)
        probe_coords[:, 0] = torch.arange(batch_size, device=device)

        probe_feats = self.probe.unsqueeze(0).repeat(batch_size, 1)
        probe_tokens = SparseTensor(feats=probe_feats, coords=probe_coords)

        hidden_state, _ = self.attention(probe_tokens, context=hidden_state)

        residual = hidden_state

        hidden_state = hidden_state.replace(self.layernorm(hidden_state.feats))
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state

    def forward(self, hidden_state: SparseTensor) -> SparseTensor:
        """Forward pass with optional checkpointing.

        Args:
            hidden_state: SparseTensor with input features.

        Returns:
            Pooled features as SparseTensor.
        """
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, hidden_state, use_reentrant=False)
        else:
            return self._forward(hidden_state)
