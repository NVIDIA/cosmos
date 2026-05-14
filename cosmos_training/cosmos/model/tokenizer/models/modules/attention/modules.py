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

"""Multi-head attention modules for sparse tensors.

This module provides high-level attention modules including:
    - RotaryPositionEmbedder: Rotary position embeddings (RoPE)
    - SparseMultiHeadRMSNorm: RMS normalization for attention
    - SparseMultiHeadAttention: Full multi-head attention
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos.model.tokenizer.models.modules import DEBUG as SPARSE_DEBUG
from cosmos.model.tokenizer.models.modules.attention.full_attn import (
    sparse_scaled_dot_product_attention,
    tensor_varlen_scaled_dot_product_attention,
)

if TYPE_CHECKING:
    from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor


__all__ = [
    "RotaryPositionEmbedder",
    "SparseMultiHeadRMSNorm",
    "SparseMultiHeadAttention",
]

AttentionDebugPayload = dict[str, object]


def _assert_temporal_coords_sorted(times: torch.Tensor, context: str) -> None:
    """Validate the searchsorted temporal-order invariant in debug mode."""
    if not SPARSE_DEBUG or times.numel() < 2:
        return
    if not bool(torch.all(times[1:] >= times[:-1]).item()):
        raise AssertionError(f"temporal coords must be sorted for {context}")


def _manual_segmented_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_seqlen: list[int],
    kv_seqlen: list[int],
    *,
    store_attn: bool,
) -> tuple[
    torch.Tensor, list[torch.Tensor] | None
]:  # q: [Tq,H,D], k: [Tk,H,D], v: [Tk,H,D], returns ([Tq,H,D], optional [[Q,H,K]])
    """Compute segmented attention eagerly in fp32 for debug capture."""
    if len(q_seqlen) != len(kv_seqlen):
        raise ValueError(
            f"Manual debug attention requires matching segment counts, got {len(q_seqlen)} and {len(kv_seqlen)}."
        )

    out_chunks: list[torch.Tensor] = []
    attn_chunks: list[torch.Tensor] = []
    q_offset = 0
    kv_offset = 0
    scale = 1.0 / math.sqrt(float(q.shape[-1]))

    for q_len, kv_len in zip(q_seqlen, kv_seqlen, strict=True):
        q_chunk = q[q_offset : q_offset + q_len].to(dtype=torch.float32)  # [Q,H,D]
        k_chunk = k[kv_offset : kv_offset + kv_len].to(dtype=torch.float32)  # [K,H,D]
        v_chunk = v[kv_offset : kv_offset + kv_len].to(dtype=torch.float32)  # [K,H,D]
        scores = torch.einsum("qhd,khd->qhk", q_chunk, k_chunk) * scale  # [Q,H,K]
        weights = torch.softmax(scores, dim=-1)  # [Q,H,K]
        out_chunk = torch.einsum("qhk,khd->qhd", weights, v_chunk)  # [Q,H,D]
        out_chunks.append(out_chunk.to(dtype=v.dtype))  # [Q,H,D]
        if store_attn:
            attn_chunks.append(weights.detach().clone())  # [Q,H,K]
        q_offset += q_len
        kv_offset += kv_len

    if q_offset != q.shape[0] or kv_offset != k.shape[0]:
        raise AssertionError(
            "Manual debug attention consumed inconsistent token counts: "
            f"q={q.shape[0]} vs {q_offset}, k={k.shape[0]} vs {kv_offset}."
        )

    if out_chunks:
        output = torch.cat(out_chunks, dim=0)  # [Tq,H,D]
    else:
        output = q.new_empty((0, q.shape[1], v.shape[-1]))  # [0,H,D]
    captured_attn = attn_chunks if store_attn else None
    return output, captured_attn


class RotaryPositionEmbedder(nn.Module):
    """Rotary Position Embedding (RoPE) for sparse tensors.

    Computes position-dependent rotation matrices for query and key vectors.
    Uses all 4 position dimensions (t, h, w, z) for proper 4D position encoding.
    """

    def __init__(self, head_dim: int, pos_cls_token: int = 0):
        """Initialize RotaryPositionEmbedder.

        Args:
            head_dim: Dimension of each attention head.
            pos_cls_token: Position to use for special/CLS sparse tokens.
        """
        super().__init__()
        self.head_dim = head_dim
        self.pos_cls_token = pos_cls_token

        self.dim_rope = head_dim // 8

        # Register freqs as a buffer so it stays on the correct device
        freqs = 1.0 / (10000.0 ** (torch.arange(0, self.dim_rope, dtype=torch.float32) / head_dim))
        self.register_buffer("freqs", freqs, persistent=False)

    def _normalize_positions(
        self,
        positions: torch.Tensor,
        has_special_tokens: bool | None = None,
    ) -> torch.Tensor:
        """Normalize sparse coordinates for RoPE.

        Sparse tokenizer tensors may carry either 3 position columns
        ``[t, h, w]`` or 4 columns ``[t, h, w, z]``. RoPE always operates on
        4 dimensions, so missing depth is padded with zeros.

        Special sparse tokens use negative coordinates. Those positions are
        remapped to ``pos_cls_token`` so CLS-style tokens get a deterministic
        phase instead of being rotated by negative indices.
        """
        if positions.shape[-1] == 3:
            positions = torch.cat(
                [
                    positions,
                    torch.zeros(
                        positions.shape[:-1] + (1,),
                        dtype=positions.dtype,
                        device=positions.device,
                    ),
                ],
                dim=-1,
            )
        elif positions.shape[-1] != 4:
            raise ValueError(f"RoPE expects 3D or 4D positions, got shape {positions.shape}")

        if has_special_tokens is None:
            special_mask = (positions < 0).all(dim=-1, keepdim=True)
            has_special_tokens = bool(special_mask.any().item())
        elif has_special_tokens:
            special_mask = (positions < 0).all(dim=-1, keepdim=True)
        else:
            special_mask = None

        if has_special_tokens and special_mask is not None:
            cls_positions = torch.full_like(positions, self.pos_cls_token)
            positions = torch.where(special_mask, cls_positions, positions)

        return positions

    def compute_freqs_cis(
        self,
        positions: torch.Tensor,
        has_special_tokens: bool | None = None,
    ) -> torch.Tensor:
        """Compute RoPE frequencies for given positions on-the-fly.

        Args:
            positions: [..., 3 or 4] tensor containing t, h, w[, z] positions.

        Returns:
            Complex frequency tensor for rotary embeddings.
        """
        positions = self._normalize_positions(positions, has_special_tokens=has_special_tokens)
        positions_fp = positions.to(dtype=torch.float32)
        freqs = self.freqs if self.freqs.dtype == torch.float32 else self.freqs.float()

        # Calculate frequencies for all 4 dimensions
        t_freq = torch.outer(positions_fp[..., 0], freqs)
        h_freq = torch.outer(positions_fp[..., 1], freqs)
        w_freq = torch.outer(positions_fp[..., 2], freqs)
        z_freq = torch.outer(positions_fp[..., 3], freqs)

        # Convert to complex numbers
        magnitudes = torch.ones_like(t_freq)
        freqs_cis = torch.empty(
            (t_freq.shape[0], self.dim_rope * 4),
            dtype=torch.complex64,
            device=t_freq.device,
        )
        freqs_cis[:, 0 : self.dim_rope] = torch.polar(magnitudes, t_freq)
        freqs_cis[:, self.dim_rope : self.dim_rope * 2] = torch.polar(magnitudes, h_freq)
        freqs_cis[:, self.dim_rope * 2 : self.dim_rope * 3] = torch.polar(magnitudes, w_freq)
        freqs_cis[:, self.dim_rope * 3 :] = torch.polar(magnitudes, z_freq)

        return freqs_cis

    @staticmethod
    def _get_cacheable_tensor_version(tensor: torch.Tensor) -> int:
        """Return a stable tensor version when available.

        Inference-mode tensors do not track version counters and raise on
        access. Those tensors are treated as immutable for our cache purposes,
        so a sentinel version is sufficient. In-place mutation under
        inference_mode would bypass cache invalidation.
        """
        try:
            return tensor._version
        except RuntimeError:
            return -1

    def get_cached_freqs_cis(self, sparse_tensor: "SparseTensor", positions: torch.Tensor) -> torch.Tensor:
        """Return cached RoPE frequencies for a sparse tensor's current coordinates."""
        has_special_tokens = sparse_tensor.has_special_tokens()
        cache_key = (
            "rope_freqs_cis",
            self.head_dim,
            self.pos_cls_token,
            has_special_tokens,
            positions.data_ptr(),
            self._get_cacheable_tensor_version(positions),
            tuple(positions.shape),
            str(positions.device),
        )
        freqs_cis = sparse_tensor.get_spatial_cache(cache_key)
        if freqs_cis is None:
            freqs_cis = self.compute_freqs_cis(positions, has_special_tokens=has_special_tokens)
            sparse_tensor.register_spatial_cache(cache_key, freqs_cis)
        return freqs_cis

    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
        xk_freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            xq: Query tensor.
            xk: Key tensor.
            freqs_cis: Frequency tensor for queries.
            xk_freqs_cis: Frequency tensor for keys.

        Returns:
            Tuple of rotated (query, key) tensors.
        """
        reshape_xq = xq.to(torch.float32).reshape(*xq.shape[:-1], -1, 2)
        reshape_xk = xk.to(torch.float32).reshape(*xk.shape[:-1], -1, 2)
        xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
        xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])

        # add head dim
        freqs_cis = freqs_cis.unsqueeze(-2)
        xk_freqs_cis = xk_freqs_cis.unsqueeze(-2)

        xq_out = xq_ * freqs_cis
        xq_out = torch.view_as_real(xq_out).reshape(*xq_out.shape[:-1], -1)

        xk_out = xk_ * xk_freqs_cis
        xk_out = torch.view_as_real(xk_out).reshape(*xk_out.shape[:-1], -1)

        return xq_out.to(xq.dtype), xk_out.to(xk.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        indices: torch.Tensor,
        k_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to queries and keys.

        Args:
            q: Query tensor.
            k: Key tensor.
            indices: Position indices for queries.
            k_indices: Position indices for keys (defaults to indices).

        Returns:
            Tuple of position-embedded (query, key) tensors.
        """
        # freqs is now a registered buffer, already on correct device
        q_freqs = self.compute_freqs_cis(indices)

        if k_indices is None:
            k_freqs = q_freqs
        else:
            k_freqs = self.compute_freqs_cis(k_indices)

        q_embed, k_embed = self.apply_rotary_emb(q, k, freqs_cis=q_freqs, xk_freqs_cis=k_freqs)

        return q_embed, k_embed


class SparseMultiHeadRMSNorm(nn.Module):
    """Multi-head RMS normalization for sparse tensors.

    Applies RMS normalization with per-head learnable scale parameters.
    """

    def __init__(self, dim: int, heads: int):
        """Initialize SparseMultiHeadRMSNorm.

        Args:
            dim: Dimension per head.
            heads: Number of attention heads.
        """
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: "SparseTensor" | torch.Tensor) -> "SparseTensor" | torch.Tensor:
        """Apply multi-head RMS normalization.

        Args:
            x: Input tensor or SparseTensor.

        Returns:
            Normalized tensor with same type as input.
        """
        from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

        if isinstance(x, SparseTensor):
            output_dtype = x.dtype
            if not self.training and output_dtype != torch.float32 and self.gamma.dtype == output_dtype:
                normalized_feats = F.normalize(x.feats.float(), dim=-1).to(output_dtype)
                return x.replace(normalized_feats * self.gamma * self.scale)
            normalized_feats = F.normalize(x.feats.float(), dim=-1)
            scaled_feats = (normalized_feats * self.gamma * self.scale).to(output_dtype)
            return x.replace(scaled_feats)
        else:
            output_dtype = x.dtype
            if not self.training and output_dtype != torch.float32 and self.gamma.dtype == output_dtype:
                normalized = F.normalize(x.float(), dim=-1).to(output_dtype)
                return normalized * self.gamma * self.scale
            normalized = F.normalize(x.float(), dim=-1)
            return (normalized * self.gamma * self.scale).to(output_dtype)


class SparseMultiHeadAttention(nn.Module):
    """Multi-head attention over SparseTensor token layouts.

    The tokenizer is sparse in its token representation: activations are stored
    as ``SparseTensor`` objects and packed into contiguous per-sequence token
    lists before attention. The attention kernel itself is standard 1D varlen
    attention, not a block-sparse or multi-dimensional sparse attention kernel.

    Supports self-attention and cross-attention with tokenizer full attention.

    Also supports:
        - Rotary position embeddings (RoPE)
        - QK RMS normalization
        - KV caching for inference
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: int | None = None,
        type: Literal["self", "cross"] = "self",
        use_bias: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        out_channels: int | None = None,
        pos_cls_token: int = 0,
    ):
        """Initialize SparseMultiHeadAttention.

        Args:
            channels: Number of input channels.
            num_heads: Number of attention heads.
            ctx_channels: Number of context channels for cross-attention.
            type: Attention type ("self" or "cross").
            use_bias: Whether to use bias in linear projections.
            use_rope: Whether to use rotary position embeddings.
            qk_rms_norm: Whether to apply RMS normalization to Q and K.
            out_channels: Number of output channels (defaults to channels).
            pos_cls_token: Position used for special/CLS sparse tokens in RoPE.
        """
        super().__init__()

        if out_channels is None:
            out_channels = channels
        assert channels % num_heads == 0
        assert out_channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm
        self.layer_idx: int | None = None
        self._debug_capture_enabled = False
        self._debug_capture_store_qkv = True
        self._debug_capture_store_attn = False
        self._debug_capture_assert_single_segment = False
        self._last_attention_debug: AttentionDebugPayload | None = None

        head_dim = channels // num_heads
        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=use_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=use_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=use_bias)

        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(head_dim, num_heads)

        self.to_out = nn.Linear(channels, out_channels, bias=use_bias)

        if use_rope:
            self.rope = RotaryPositionEmbedder(head_dim, pos_cls_token=pos_cls_token)

    def set_debug_capture(
        self,
        enabled: bool,
        *,
        store_qkv: bool = True,
        store_attn: bool = False,
        assert_single_segment: bool = False,
    ) -> None:
        """Enable or disable eager debug capture for attention visualization."""
        self._debug_capture_enabled = enabled
        self._debug_capture_store_qkv = store_qkv
        self._debug_capture_store_attn = store_attn
        self._debug_capture_assert_single_segment = assert_single_segment
        if not enabled:
            self._last_attention_debug = None

    def clear_last_attention_debug(self) -> None:
        """Clear the last captured debug-attention payload."""
        self._last_attention_debug = None

    def get_last_attention_debug(self) -> AttentionDebugPayload | None:
        """Return the last captured debug-attention payload."""
        return self._last_attention_debug

    def _maybe_assert_single_debug_segment(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_seqlen: list[int],
        kv_seqlen: list[int],
    ) -> None:  # q: [Tq,H,D], k: [Tk,H,D]
        """Validate the v1 single-segment assumption when requested."""
        if not self._debug_capture_assert_single_segment:
            return
        if len(q_seqlen) != 1 or len(kv_seqlen) != 1:
            raise AssertionError(
                "Debug attention capture requires one query segment and one KV segment in v1, "
                f"got q={q_seqlen}, kv={kv_seqlen}."
            )
        if q_seqlen[0] != q.shape[0] or kv_seqlen[0] != k.shape[0]:
            raise AssertionError(
                "Debug attention capture requires flat token counts to match the single segment lengths, "
                f"got q={q.shape[0]} vs {q_seqlen[0]} and k={k.shape[0]} vs {kv_seqlen[0]}."
            )

    def _record_debug_capture(
        self,
        *,
        mode: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_seqlen: list[int],
        kv_seqlen: list[int],
        attn_chunks: list[torch.Tensor] | None,
    ) -> None:  # q: [Tq,H,D], k: [Tk,H,D], v: [Tk,H,D]
        """Persist one eager debug-attention payload on the module."""
        payload: AttentionDebugPayload = {
            "mode": mode,
            "layer_idx": self.layer_idx,
            "q_seqlen": list(q_seqlen),
            "kv_seqlen": list(kv_seqlen),
            "i4_attn_backends": os.environ.get("I4_ATTN_BACKENDS"),
            "sparse_backend": os.environ.get("SPARSE_BACKEND"),
        }
        if self._debug_capture_store_qkv:
            payload["q"] = q.detach().clone()  # [Tq,H,D]
            payload["k"] = k.detach().clone()  # [Tk,H,D]
            payload["v"] = v.detach().clone()  # [Tk,H,D]
        if attn_chunks is not None:
            payload["attn"] = attn_chunks
        self._last_attention_debug = payload

    @staticmethod
    def _linear(module: nn.Linear, x: "SparseTensor" | torch.Tensor) -> "SparseTensor" | torch.Tensor:
        """Apply linear layer to tensor or SparseTensor."""
        from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

        if isinstance(x, SparseTensor):
            return x.replace(module(x.feats))
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(x: "SparseTensor" | torch.Tensor, shape: tuple[int, ...]) -> "SparseTensor" | torch.Tensor:
        """Reshape channel dimensions of tensor or SparseTensor."""
        from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

        if isinstance(x, SparseTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x: "SparseTensor" | torch.Tensor, num_fused: int) -> "SparseTensor" | torch.Tensor:
        """Reshape for fused QKV or KV projections."""
        from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

        if isinstance(x, SparseTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, SparseTensor) else x_feats

    def _qkv_rope(self, qkv: "SparseTensor") -> "SparseTensor":
        """Apply RoPE to packed QKV tensor."""
        q, k, v = qkv.feats.unbind(dim=1)
        freqs_cis = self.rope.get_cached_freqs_cis(qkv, qkv.coords[:, 1:])
        q, k = self.rope.apply_rotary_emb(q, k, freqs_cis=freqs_cis, xk_freqs_cis=freqs_cis)
        qkv = qkv.replace(torch.stack([q, k, v], dim=1))
        return qkv

    def _q_kv_rope(
        self,
        q: "SparseTensor",
        k: "SparseTensor",
        q_freqs_cis: torch.Tensor | None = None,
        k_freqs_cis: torch.Tensor | None = None,
    ) -> tuple["SparseTensor", "SparseTensor"]:
        """Apply RoPE to separate Q and K tensors."""
        if q_freqs_cis is None:
            q_freqs_cis = self.rope.get_cached_freqs_cis(q, q.coords[:, 1:])
        if k_freqs_cis is None:
            k_freqs_cis = self.rope.get_cached_freqs_cis(k, k.coords[:, 1:])
        q_feats, k_feats = self.rope.apply_rotary_emb(
            q.feats,
            k.feats,
            freqs_cis=q_freqs_cis,
            xk_freqs_cis=k_freqs_cis,
        )
        return q.replace(q_feats), k.replace(k_feats)

    def _q_flat_k_rope(
        self,
        q: "SparseTensor",
        k_feats: torch.Tensor,
        q_freqs_cis: torch.Tensor | None = None,
        k_freqs_cis: torch.Tensor | None = None,
    ) -> tuple["SparseTensor", torch.Tensor]:
        """Apply RoPE when K is already packed as a flat tensor."""
        if q_freqs_cis is None:
            q_freqs_cis = self.rope.get_cached_freqs_cis(q, q.coords[:, 1:])
        if k_freqs_cis is None:
            raise ValueError("Packed KV RoPE path requires precomputed key frequencies.")
        q_feats, k_feats = self.rope.apply_rotary_emb(
            q.feats,
            k_feats,
            freqs_cis=q_freqs_cis,
            xk_freqs_cis=k_freqs_cis,
        )
        return q.replace(q_feats), k_feats

    def forward_no_cache(
        self,
        x: "SparseTensor" | torch.Tensor,
        context: "SparseTensor" | torch.Tensor | None = None,
        temporal_causal_mask: bool = False,
    ) -> "SparseTensor" | torch.Tensor:
        """Apply attention without constructing or updating KV-cache state."""
        from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)

            q, k, v = qkv.unbind(dim=1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)

            if self.use_rope and isinstance(q, SparseTensor):
                q, k = self._q_kv_rope(q, k)

            h = sparse_scaled_dot_product_attention(q, k, v, temporal_causal_mask=temporal_causal_mask)
        else:
            if temporal_causal_mask:
                raise NotImplementedError("temporal_causal_mask is only implemented for self-attention.")
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)

            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=1)
                k = self.k_rms_norm(k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))

            if self.use_rope:
                k, v = kv.unbind(dim=1)
                q, k = self._q_kv_rope(q, k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))

            h = sparse_scaled_dot_product_attention(q, kv)

        h = self._reshape_chs(h, (-1,))
        return self._linear(self.to_out, h)

    def forward_tensor_no_cache(
        self,
        feats: torch.Tensor,
        q_seqlen: list[int],
        cu_seqlens_q: torch.Tensor,
        max_q_seqlen: int,
        q_freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply self-attention directly on flat token features for the eval fast path."""
        if self._type != "self":
            raise NotImplementedError("Tensor no-cache attention is only implemented for self-attention.")
        self._last_attention_debug = None

        qkv = self.to_qkv(feats).reshape(feats.shape[0], 3, self.num_heads, -1)  # [T,3,H,D]
        q, k, v = qkv.unbind(dim=1)  # q: [T,H,D], k: [T,H,D], v: [T,H,D]
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)  # [T,H,D]
            k = self.k_rms_norm(k)  # [T,H,D]

        if self.use_rope:
            if q_freqs_cis is None:
                raise ValueError("Tensor no-cache RoPE path requires precomputed q_freqs_cis.")
            q, k = self.rope.apply_rotary_emb(
                q, k, freqs_cis=q_freqs_cis, xk_freqs_cis=q_freqs_cis
            )  # q_: [T,H,D], k_: [T,H,D]

        if self._debug_capture_enabled:
            self._maybe_assert_single_debug_segment(q, k, q_seqlen, q_seqlen)
            h, attn_chunks = _manual_segmented_scaled_dot_product_attention(
                q,
                k,
                v,
                q_seqlen=q_seqlen,
                kv_seqlen=q_seqlen,
                store_attn=self._debug_capture_store_attn,
            )
            self._record_debug_capture(
                mode="self_tensor_no_cache",
                q=q,
                k=k,
                v=v,
                q_seqlen=q_seqlen,
                kv_seqlen=q_seqlen,
                attn_chunks=attn_chunks,
            )
        else:
            h = tensor_varlen_scaled_dot_product_attention(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_q,
                max_q_seqlen=max_q_seqlen,
                max_kv_seqlen=max_q_seqlen,
            )  # [T,H,D]
        h = h.reshape(h.shape[0], -1)  # [T,HD]
        return self.to_out(h)  # [T,C]

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
        """Apply self-attention on flat query tensors with the flat KV-cache fast path.

        This path is only valid for batch size 1 packed sequences with full
        self-attention and no special tokens. Callers are expected to enforce
        those constraints before entering the method.
        """
        if self._type != "self":
            raise NotImplementedError("Tensor flat-KV attention is only implemented for self-attention.")
        if kv_cache is None:
            kv_cache = {}

        qkv = self.to_qkv(feats).reshape(feats.shape[0], 3, self.num_heads, -1)
        q, k, v = qkv.unbind(dim=1)
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        current_k_for_cache = k
        if self.use_rope:
            if q_freqs_cis is None:
                raise ValueError("Tensor flat-KV RoPE path requires precomputed q_freqs_cis.")
            q, current_k_for_cache = self.rope.apply_rotary_emb(
                q,
                current_k_for_cache,
                freqs_cis=q_freqs_cis,
                xk_freqs_cis=q_freqs_cis,
            )

        (
            flat_k_with_cache,
            flat_v_with_cache,
            combined_flat_times,
            flat_kv_seqlen,
            flat_cu_seqlens_kv,
        ) = self._prepare_flat_kv_cache(
            current_k_feats=current_k_for_cache,
            current_v_feats=v,
            current_times=current_times.contiguous(),
            kv_cache=kv_cache,
        )
        updated_kv_cache = self._update_flat_kv_cache(
            flat_k_with_cache,
            flat_v_with_cache,
            combined_flat_times,
            kv_cache,
            kv_cache_size,
            kv_cache_detach=kv_cache_detach,
        )

        h = tensor_varlen_scaled_dot_product_attention(
            q=q,
            k=flat_k_with_cache,
            v=flat_v_with_cache,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=flat_cu_seqlens_kv,
            max_q_seqlen=max_q_seqlen,
            max_kv_seqlen=flat_kv_seqlen[0],
        )
        h = h.reshape(h.shape[0], -1)
        return self.to_out(h), updated_kv_cache

    @staticmethod
    def _has_compatible_flat_kv_cache_state(kv_cache: dict[str, object]) -> bool:
        """Return whether an existing KV cache dict can resume the flat tensor fast path."""
        if kv_cache.get("cached_k") is not None or kv_cache.get("cached_v") is not None:
            return False

        flat_keys_present = any(key in kv_cache for key in ("cached_k_feats", "cached_v_feats", "cached_times"))
        if not flat_keys_present:
            return True

        cached_k_feats = kv_cache.get("cached_k_feats")
        cached_v_feats = kv_cache.get("cached_v_feats")
        cached_times = kv_cache.get("cached_times")
        return (
            isinstance(cached_k_feats, torch.Tensor)
            and isinstance(cached_v_feats, torch.Tensor)
            and isinstance(cached_times, torch.Tensor)
        )

    def _can_use_flat_kv_cache_fast_path(
        self,
        k: "SparseTensor",
        v: "SparseTensor",
        kv_cache: dict[str, object],
    ) -> bool:
        """Return whether the tensor-only KV fast path can be used safely."""
        if k.shape[0] != 1 or v.shape[0] != 1 or k.has_special_tokens():
            return False
        return self._has_compatible_flat_kv_cache_state(kv_cache)

    @staticmethod
    def _prepare_flat_kv_cache(
        current_k_feats: torch.Tensor,
        current_v_feats: torch.Tensor,
        current_times: torch.Tensor,
        kv_cache: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
        """Prepare packed flat KV tensors for the single-batch cache fast path.

        The flat cache stores keys in the exact representation consumed by
        attention: post-RoPE when rotary embeddings are enabled, otherwise the
        raw RMS-normalized keys.
        """
        cached_k_feats = kv_cache.get("cached_k_feats")
        cached_v_feats = kv_cache.get("cached_v_feats")
        cached_times = kv_cache.get("cached_times")

        if (
            isinstance(cached_k_feats, torch.Tensor)
            and isinstance(cached_v_feats, torch.Tensor)
            and isinstance(cached_times, torch.Tensor)
        ):
            combined_k_feats = torch.cat([cached_k_feats, current_k_feats], dim=0)
            combined_v_feats = torch.cat([cached_v_feats, current_v_feats], dim=0)
            combined_times = torch.cat([cached_times, current_times], dim=0)
        else:
            combined_k_feats = current_k_feats
            combined_v_feats = current_v_feats
            combined_times = current_times

        total_tokens = int(combined_k_feats.shape[0])
        kv_seqlen = [total_tokens]
        cu_seqlens_kv = torch.tensor([0, total_tokens], dtype=torch.int32, device=current_k_feats.device)
        return combined_k_feats, combined_v_feats, combined_times, kv_seqlen, cu_seqlens_kv

    @staticmethod
    def _update_flat_kv_cache(
        combined_k_feats: torch.Tensor,
        combined_v_feats: torch.Tensor,
        combined_times: torch.Tensor,
        kv_cache: dict[str, object] | None = None,
        kv_cache_size: int | None = None,
        kv_cache_detach: bool = True,
    ) -> dict[str, object]:
        """Update the tensor-only KV cache for the single-batch fast path."""
        if kv_cache is None:
            kv_cache = {}

        if kv_cache_size is None or kv_cache_size <= 0:
            kv_cache.pop("cached_k", None)
            kv_cache.pop("cached_v", None)
            kv_cache.pop("cached_k_freqs_cis", None)
            kv_cache.pop("cached_k_feats", None)
            kv_cache.pop("cached_v_feats", None)
            kv_cache.pop("cached_times", None)
            return kv_cache

        if combined_times.numel() == 0:
            start_idx = 0
            cached_times = combined_times.new_empty((0,))
        else:
            _assert_temporal_coords_sorted(combined_times, "flat KV cache updates")
            max_time = combined_times[-1]
            time_threshold = max_time - kv_cache_size + 1
            start_idx = int(torch.searchsorted(combined_times, time_threshold, right=False).item())
            cached_times = combined_times[start_idx:].clone()
            cached_times -= time_threshold.to(dtype=cached_times.dtype)

        cached_k_feats = combined_k_feats[start_idx:]
        cached_v_feats = combined_v_feats[start_idx:]
        if kv_cache_detach:
            cached_k_feats = cached_k_feats.detach()
            cached_v_feats = cached_v_feats.detach()
            cached_times = cached_times.detach()

        kv_cache["cached_k_feats"] = cached_k_feats
        kv_cache["cached_v_feats"] = cached_v_feats
        kv_cache["cached_times"] = cached_times
        kv_cache.pop("cached_k_freqs_cis", None)
        kv_cache.pop("cached_k", None)
        kv_cache.pop("cached_v", None)
        return kv_cache

    def _prepare_kv_cache(
        self,
        k: "SparseTensor",
        v: "SparseTensor",
        cached_k: "SparseTensor" | None = None,
        cached_v: "SparseTensor" | None = None,
        current_k_freqs_cis: torch.Tensor | None = None,
        cached_k_freqs_cis: torch.Tensor | None = None,
    ) -> tuple["SparseTensor", "SparseTensor", torch.Tensor | None]:
        """Prepare K and V tensors by concatenating with cached values.

        Args:
            k: Current key tensor.
            v: Current value tensor.
            cached_k: Cached key tensor from previous timesteps.
            cached_v: Cached value tensor from previous timesteps.

        Returns:
            Tuple of (combined_k, combined_v, combined_k_freqs_cis).
        """
        if cached_k is not None and cached_v is not None:
            combined_k = self._concat_temporal_sparse(cached_k, k)
            combined_v = self._concat_temporal_sparse(cached_v, v)
            if current_k_freqs_cis is not None and cached_k_freqs_cis is not None:
                combined_k_freqs_cis = self._concat_temporal_freqs(
                    cached_k,
                    k,
                    cached_k_freqs_cis,
                    current_k_freqs_cis,
                )
            else:
                combined_k_freqs_cis = None
        else:
            combined_k = k
            combined_v = v
            combined_k_freqs_cis = current_k_freqs_cis

        return combined_k, combined_v, combined_k_freqs_cis

    @staticmethod
    def _concat_temporal_freqs(
        cached: "SparseTensor",
        current: "SparseTensor",
        cached_freqs_cis: torch.Tensor,
        current_freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate cached/current RoPE frequencies in batch-local temporal order."""
        if not cached.has_special_tokens() and not current.has_special_tokens():
            combined_parts = []
            current_batch_size = current.shape[0]

            for batch_idx in range(current_batch_size):
                if batch_idx < cached.shape[0]:
                    cached_slice = cached.layout[batch_idx]
                    if cached_slice.start < cached_slice.stop:
                        combined_parts.append(cached_freqs_cis[cached_slice])

                current_slice = current.layout[batch_idx]
                if current_slice.start < current_slice.stop:
                    combined_parts.append(current_freqs_cis[current_slice])

            if not combined_parts:
                return current_freqs_cis.new_empty((0,) + current_freqs_cis.shape[1:])
            return torch.cat(combined_parts, dim=0)

        combined_parts = []
        current_batch_size = current.shape[0]

        for batch_idx in range(current_batch_size):
            cached_special_parts = []
            cached_normal_parts = []
            current_special_parts = []
            current_normal_parts = []

            if batch_idx < cached.shape[0]:
                cached_slice = cached.layout[batch_idx]
                if cached_slice.start < cached_slice.stop:
                    cached_batch_coords = cached.coords[cached_slice]
                    cached_batch_freqs = cached_freqs_cis[cached_slice]
                    cached_special_mask = (cached_batch_coords[:, 1:] == -1).all(dim=1)
                    if cached_special_mask.any():
                        cached_special_parts.append(cached_batch_freqs[cached_special_mask])
                    if (~cached_special_mask).any():
                        cached_normal_parts.append(cached_batch_freqs[~cached_special_mask])

            current_slice = current.layout[batch_idx]
            if current_slice.start < current_slice.stop:
                current_batch_coords = current.coords[current_slice]
                current_batch_freqs = current_freqs_cis[current_slice]
                current_special_mask = (current_batch_coords[:, 1:] == -1).all(dim=1)
                if current_special_mask.any():
                    current_special_parts.append(current_batch_freqs[current_special_mask])
                if (~current_special_mask).any():
                    current_normal_parts.append(current_batch_freqs[~current_special_mask])

            combined_parts.extend(cached_special_parts)
            combined_parts.extend(current_special_parts)
            combined_parts.extend(cached_normal_parts)
            combined_parts.extend(current_normal_parts)

        if not combined_parts:
            return current_freqs_cis.new_empty((0,) + current_freqs_cis.shape[1:])
        return torch.cat(combined_parts, dim=0)

    def _concat_temporal_sparse(self, cached: "SparseTensor", current: "SparseTensor") -> "SparseTensor":
        """Concatenate cached and current sparse tensors along temporal dimension.

        Injects cached data at the beginning of each batch to maintain proper
        temporal ordering.
        """
        return current.concat_temporal_at_batch_start(cached)

    def _update_kv_cache(
        self,
        k: "SparseTensor",
        v: "SparseTensor",
        kv_cache: dict[str, object] | None = None,
        kv_cache_size: int | None = None,
        kv_cache_detach: bool = True,
        k_freqs_cis: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Update the KV cache with the last k timesteps from current K and V.

        Args:
            k: Current key tensor.
            v: Current value tensor.
            kv_cache: Existing cache dictionary.
            kv_cache_size: Number of timesteps to cache.
            kv_cache_detach: Whether to detach tensors stored in the cache.

        Returns:
            Updated cache dictionary.
        """
        if kv_cache is None:
            kv_cache = {}

        if kv_cache_size is not None and kv_cache_size > 0:
            sparse_tensor_cls = type(k)
            if k.coords.shape[0] == 0:
                selected_coords = torch.empty(
                    (0, k.coords.shape[1]),
                    dtype=k.coords.dtype,
                    device=k.coords.device,
                )
                valid_mask = torch.zeros(0, dtype=torch.bool, device=k.coords.device)
            elif not k.has_special_tokens():
                max_time = k.coords[:, 1].max()
                time_threshold = max_time - kv_cache_size + 1
                cached_coords_parts = []
                cached_k_parts = []
                cached_v_parts = []
                cached_freq_parts = []
                cached_layout = []
                offset = 0

                for batch_slice in k.layout:
                    if batch_slice.start >= batch_slice.stop:
                        cached_layout.append(slice(offset, offset))
                        continue

                    batch_times = k.coords[batch_slice, 1].contiguous()
                    _assert_temporal_coords_sorted(batch_times, "sparse KV cache updates")
                    relative_start = int(torch.searchsorted(batch_times, time_threshold, right=False).item())
                    absolute_start = batch_slice.start + relative_start
                    absolute_end = batch_slice.stop
                    batch_len = max(0, absolute_end - absolute_start)

                    if batch_len > 0:
                        batch_coords = k.coords[absolute_start:absolute_end].clone()
                        batch_coords[:, 1] -= time_threshold.to(dtype=batch_coords.dtype)
                        cached_coords_parts.append(batch_coords)
                        cached_k_parts.append(k.feats[absolute_start:absolute_end])
                        cached_v_parts.append(v.feats[absolute_start:absolute_end])
                        if k_freqs_cis is not None:
                            cached_freq_parts.append(k_freqs_cis[absolute_start:absolute_end])

                    cached_layout.append(slice(offset, offset + batch_len))
                    offset += batch_len

                if cached_coords_parts:
                    selected_coords = torch.cat(cached_coords_parts, dim=0)
                    cached_k_feats = torch.cat(cached_k_parts, dim=0)
                    cached_v_feats = torch.cat(cached_v_parts, dim=0)
                else:
                    selected_coords = torch.empty(
                        (0, k.coords.shape[1]),
                        dtype=k.coords.dtype,
                        device=k.coords.device,
                    )
                    cached_k_feats = torch.empty(
                        (0,) + k.feats.shape[1:],
                        dtype=k.feats.dtype,
                        device=k.feats.device,
                    )
                    cached_v_feats = torch.empty(
                        (0,) + v.feats.shape[1:],
                        dtype=v.feats.dtype,
                        device=v.feats.device,
                    )

                cached_k = sparse_tensor_cls(
                    feats=cached_k_feats,
                    coords=selected_coords,
                    shape=torch.Size([k.shape[0]] + list(k.feats.shape[1:])),
                    layout=cached_layout,
                    has_special_tokens=False,
                )
                cached_v = sparse_tensor_cls(
                    feats=cached_v_feats,
                    coords=selected_coords,
                    shape=torch.Size([v.shape[0]] + list(v.feats.shape[1:])),
                    layout=list(cached_layout),
                    has_special_tokens=False,
                )

                if kv_cache_detach:
                    cached_k = cached_k.detach()
                    cached_v = cached_v.detach()

                kv_cache["cached_k"] = cached_k
                kv_cache["cached_v"] = cached_v
                if k_freqs_cis is not None:
                    if cached_freq_parts:
                        cached_k_freqs_cis = torch.cat(cached_freq_parts, dim=0)
                    else:
                        cached_k_freqs_cis = k_freqs_cis.new_empty((0,) + k_freqs_cis.shape[1:])
                    if kv_cache_detach:
                        cached_k_freqs_cis = cached_k_freqs_cis.detach()
                    kv_cache["cached_k_freqs_cis"] = cached_k_freqs_cis
                else:
                    kv_cache.pop("cached_k_freqs_cis", None)
                kv_cache.pop("cached_k_feats", None)
                kv_cache.pop("cached_v_feats", None)
                kv_cache.pop("cached_times", None)
                return kv_cache
            else:
                max_time = k.coords[:, 1].max()
                time_threshold = max_time - kv_cache_size + 1
                valid_mask = k.coords[:, 1] >= time_threshold
                selected_coords = k.coords[valid_mask].clone()
                selected_coords[:, 1] -= time_threshold.to(dtype=selected_coords.dtype)

            cached_layout = sparse_tensor_cls._build_layout_from_batch_indices(selected_coords[:, 0], k.shape[0])
            cached_k = sparse_tensor_cls(
                feats=k.feats[valid_mask],
                coords=selected_coords,
                shape=torch.Size([k.shape[0]] + list(k.feats.shape[1:])),
                layout=cached_layout,
            )
            cached_v = sparse_tensor_cls(
                feats=v.feats[valid_mask],
                coords=selected_coords,
                shape=torch.Size([v.shape[0]] + list(v.feats.shape[1:])),
                layout=list(cached_layout),
            )
            if kv_cache_detach:
                cached_k = cached_k.detach()
                cached_v = cached_v.detach()

            # Store in cache
            kv_cache["cached_k"] = cached_k
            kv_cache["cached_v"] = cached_v
            if k_freqs_cis is not None:
                cached_k_freqs_cis = k_freqs_cis[valid_mask]
                if kv_cache_detach:
                    cached_k_freqs_cis = cached_k_freqs_cis.detach()
                kv_cache["cached_k_freqs_cis"] = cached_k_freqs_cis
            else:
                kv_cache.pop("cached_k_freqs_cis", None)
            kv_cache.pop("cached_k_feats", None)
            kv_cache.pop("cached_v_feats", None)
            kv_cache.pop("cached_times", None)
        else:
            # Clear cache if temporal size is 0 or None
            kv_cache.pop("cached_k", None)
            kv_cache.pop("cached_v", None)
            kv_cache.pop("cached_k_freqs_cis", None)
            kv_cache.pop("cached_k_feats", None)
            kv_cache.pop("cached_v_feats", None)
            kv_cache.pop("cached_times", None)

        return kv_cache

    def forward(
        self,
        x: "SparseTensor" | torch.Tensor,
        context: "SparseTensor" | torch.Tensor | None = None,
        kv_cache: dict[str, object] | None = None,
        kv_cache_size: int | None = None,
        kv_cache_detach: bool = True,
        temporal_causal_mask: bool = False,
    ) -> tuple["SparseTensor" | torch.Tensor, dict[str, object]]:
        """Apply multi-head attention.

        Args:
            x: Input tensor or SparseTensor.
            context: Context for cross-attention (required if type="cross").
            kv_cache: Optional KV cache for inference.
            kv_cache_size: Number of timesteps to cache.
            kv_cache_detach: Whether tensors stored in the KV cache should be detached.
            temporal_causal_mask: Whether to apply temporal-causal, same-timestep
                bidirectional attention. Only supported for self-attention in full
                attention mode without KV cache.

        Returns:
            Tuple of (attention_output, updated_kv_cache).
        """
        from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

        self._last_attention_debug = None

        if kv_cache is None:
            kv_cache = {}

        if self._type == "self":
            if temporal_causal_mask and kv_cache_size not in (None, 0):
                raise ValueError("temporal_causal_mask cannot be combined with KV cache.")

            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)

            q, k, v = qkv.unbind(dim=1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)

            q_freqs_cis = None
            current_k_freqs_cis = None
            k_with_cache_freqs_cis = None
            if self.use_rope and isinstance(q, SparseTensor):
                q_freqs_cis = self.rope.get_cached_freqs_cis(q, q.coords[:, 1:])
                current_k_freqs_cis = self.rope.get_cached_freqs_cis(k, k.coords[:, 1:])

            flat_k_with_cache = None
            flat_v_with_cache = None
            flat_kv_seqlen = None
            flat_cu_seqlens_kv = None
            combined_flat_times = None

            # Handle KV cache AFTER RMS norm, BEFORE RoPE
            if (
                isinstance(k, SparseTensor)
                and kv_cache_size is not None
                and kv_cache_size > 0
                and self._can_use_flat_kv_cache_fast_path(k, v, kv_cache)
            ):
                current_k_for_cache = k.feats
                if self.use_rope:
                    q, current_k_for_cache = self._q_flat_k_rope(
                        q,
                        current_k_for_cache,
                        q_freqs_cis=q_freqs_cis,
                        k_freqs_cis=current_k_freqs_cis,
                    )
                (
                    flat_k_with_cache,
                    flat_v_with_cache,
                    combined_flat_times,
                    flat_kv_seqlen,
                    flat_cu_seqlens_kv,
                ) = self._prepare_flat_kv_cache(
                    current_k_feats=current_k_for_cache,
                    current_v_feats=v.feats,
                    current_times=k.coords[:, 1].contiguous(),
                    kv_cache=kv_cache,
                )
            elif isinstance(k, SparseTensor) and kv_cache_size is not None:
                cached_k = kv_cache.get("cached_k", None)
                cached_v = kv_cache.get("cached_v", None)
                cached_k_freqs_cis = kv_cache.get("cached_k_freqs_cis", None)

                # Prepare KV cache (concatenate cached with current)
                k_with_cache, v_with_cache, k_with_cache_freqs_cis = self._prepare_kv_cache(
                    k,
                    v,
                    cached_k,
                    cached_v,
                    current_k_freqs_cis=current_k_freqs_cis,
                    cached_k_freqs_cis=cached_k_freqs_cis,
                )
            else:
                k_with_cache, v_with_cache = k, v
                k_with_cache_freqs_cis = current_k_freqs_cis

            # Update cache after RMS norm. The flat tensor fast path stores
            # attention-ready post-RoPE keys, while the sparse fallback stores
            # pre-RoPE keys plus cached frequencies.
            if flat_k_with_cache is not None and flat_v_with_cache is not None and combined_flat_times is not None:
                kv_cache = self._update_flat_kv_cache(
                    flat_k_with_cache,
                    flat_v_with_cache,
                    combined_flat_times,
                    kv_cache,
                    kv_cache_size,
                    kv_cache_detach=kv_cache_detach,
                )
            else:
                kv_cache = self._update_kv_cache(
                    k_with_cache,
                    v_with_cache,
                    kv_cache,
                    kv_cache_size,
                    kv_cache_detach=kv_cache_detach,
                    k_freqs_cis=k_with_cache_freqs_cis,
                )

            if self.use_rope:
                if flat_k_with_cache is None:
                    q, k_with_cache = self._q_kv_rope(
                        q,
                        k_with_cache,
                        q_freqs_cis=q_freqs_cis,
                        k_freqs_cis=k_with_cache_freqs_cis,
                    )

            # Perform attention
            if (
                flat_k_with_cache is not None
                and flat_v_with_cache is not None
                and flat_kv_seqlen is not None
                and flat_cu_seqlens_kv is not None
            ):
                h = sparse_scaled_dot_product_attention(
                    q,
                    flat_k_with_cache,
                    flat_v_with_cache,
                    temporal_causal_mask=temporal_causal_mask,
                    kv_seqlen=flat_kv_seqlen,
                    cu_seqlens_kv=flat_cu_seqlens_kv,
                    max_kv_seqlen=flat_kv_seqlen[0],
                )
            else:
                h = sparse_scaled_dot_product_attention(
                    q,
                    k_with_cache,
                    v_with_cache,
                    temporal_causal_mask=temporal_causal_mask,
                )
        else:
            if temporal_causal_mask:
                raise NotImplementedError("temporal_causal_mask is only implemented for self-attention.")
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)

            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=1)
                k = self.k_rms_norm(k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))

            if self.use_rope:
                k, v = kv.unbind(dim=1)
                q, k = self._q_kv_rope(q, k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))

            if self._debug_capture_enabled:
                if not isinstance(q, SparseTensor) or not isinstance(kv, SparseTensor):
                    raise NotImplementedError("Debug capture for dense cross-attention inputs is not implemented.")
                k, v = kv.unbind(dim=1)
                q_seqlen = q.get_batch_seq_lens()
                kv_seqlen = k.get_batch_seq_lens()
                self._maybe_assert_single_debug_segment(q.feats, k.feats, q_seqlen, kv_seqlen)
                h_feats, attn_chunks = _manual_segmented_scaled_dot_product_attention(
                    q.feats,
                    k.feats,
                    v.feats,
                    q_seqlen=q_seqlen,
                    kv_seqlen=kv_seqlen,
                    store_attn=self._debug_capture_store_attn,
                )
                self._record_debug_capture(
                    mode="cross_sparse_forward",
                    q=q.feats,
                    k=k.feats,
                    v=v.feats,
                    q_seqlen=q_seqlen,
                    kv_seqlen=kv_seqlen,
                    attn_chunks=attn_chunks,
                )
                h = q.replace(h_feats)
            else:
                h = sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h, kv_cache
