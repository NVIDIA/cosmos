# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Backend selection helpers for the dense tokenizer runtime."""

from __future__ import annotations

from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos.model.tokenizer.models.modules.attention.full_attn import tensor_dense_scaled_dot_product_attention

DenseRuntimeBackend = Literal["varlen", "batched", "auto"]
DenseResolvedBackend = Literal["varlen", "batched"]


def resolve_dense_backend(backend: DenseRuntimeBackend, use_compile: bool) -> DenseResolvedBackend:
    """Resolve the dense-runtime backend for the current execution mode.

    Args:
        backend: Requested backend mode.
        use_compile: Whether the caller intends to run under ``torch.compile``.

    Returns:
        Concrete backend name.

    Raises:
        ValueError: If ``backend`` is not one of the supported values.
    """
    if backend == "auto":
        return "batched" if use_compile else "varlen"
    if backend in ("varlen", "batched"):
        return backend
    raise ValueError(f"Unsupported dense runtime backend: {backend}")


def run_varlen_block_stack(
    blocks: nn.ModuleList,
    feats: torch.Tensor,
    q_seqlen: list[int],
    cu_seqlens_q: torch.Tensor,
    max_q_seqlen: int,
    q_freqs_cis: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the existing tensor no-cache block path over dense `[B, S, D]` chunks."""
    if feats.ndim != 3:
        raise ValueError(f"Varlen dense backend expects [B, S, D] features, got shape {tuple(feats.shape)}.")

    if len(blocks) == 0:
        return feats

    batch_size, seq_len, hidden_size = feats.shape
    flat_feats = feats.reshape(batch_size * seq_len, hidden_size)
    for block in blocks:
        flat_feats = block.forward_tensor_no_cache(
            flat_feats,
            q_seqlen=q_seqlen,
            cu_seqlens_q=cu_seqlens_q,
            max_q_seqlen=max_q_seqlen,
            q_freqs_cis=q_freqs_cis,
        )
    return flat_feats.reshape(batch_size, seq_len, hidden_size)


def run_batched_block_stack(
    blocks: nn.ModuleList,
    feats: torch.Tensor,
    q_freqs_cis: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the dense batched block path over uniform `[B, S, D]` chunks."""
    if feats.ndim != 3:
        raise ValueError(f"Batched dense backend expects [B, S, D] features, got shape {tuple(feats.shape)}.")

    output = feats
    for block in blocks:
        if block.training and getattr(block, "use_checkpoint", False):
            output = torch.utils.checkpoint.checkpoint(
                partial(run_batched_block, block, q_freqs_cis=q_freqs_cis),
                output,
                use_reentrant=False,
            )
        else:
            output = run_batched_block(block, output, q_freqs_cis=q_freqs_cis)
    return output


def run_batched_block(
    block: nn.Module,
    feats: torch.Tensor,
    q_freqs_cis: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one transformer block with the dense batched attention path."""
    if getattr(block, "multiscale", None) is not None:
        raise NotImplementedError("Dense runtime batched backend does not support multiscale blocks.")
    if getattr(block.attn, "_type", None) != "self":
        raise NotImplementedError("Dense runtime batched backend only supports self-attention blocks.")

    residual = feats
    h = block.norm1(feats)
    h = run_batched_attention(block.attn, h, q_freqs_cis=q_freqs_cis)
    feats = residual + h
    residual = feats
    h = block.norm2(feats)
    h = block.mlp.forward_tensor(h)
    return residual + h


def run_batched_attention(
    attention: nn.Module,
    feats: torch.Tensor,
    q_freqs_cis: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one dense self-attention layer via the imaginaire attention frontend."""
    if not hasattr(attention, "to_qkv"):
        raise ValueError("Dense runtime batched backend requires fused to_qkv linear projections.")
    if not hasattr(attention, "to_out"):
        raise ValueError("Dense runtime batched backend requires an output projection linear layer.")

    batch_size, seq_len, hidden_size = feats.shape
    qkv = F.linear(feats, attention.to_qkv.weight, attention.to_qkv.bias).reshape(
        batch_size,
        seq_len,
        3,
        attention.num_heads,
        -1,
    )
    q, k, v = qkv.unbind(dim=2)

    if getattr(attention, "qk_rms_norm", False):
        flat_q = q.reshape(batch_size * seq_len, attention.num_heads, -1)
        flat_k = k.reshape(batch_size * seq_len, attention.num_heads, -1)
        q = attention.q_rms_norm(flat_q).reshape(batch_size, seq_len, attention.num_heads, -1)
        k = attention.k_rms_norm(flat_k).reshape(batch_size, seq_len, attention.num_heads, -1)

    if getattr(attention, "use_rope", False):
        if q_freqs_cis is None:
            raise ValueError("Dense runtime batched backend requires precomputed q_freqs_cis when RoPE is enabled.")
        flat_q = q.reshape(batch_size * seq_len, attention.num_heads, -1)
        flat_k = k.reshape(batch_size * seq_len, attention.num_heads, -1)
        flat_q, flat_k = attention.rope.apply_rotary_emb(
            flat_q,
            flat_k,
            freqs_cis=q_freqs_cis,
            xk_freqs_cis=q_freqs_cis,
        )
        q = flat_q.reshape(batch_size, seq_len, attention.num_heads, -1)
        k = flat_k.reshape(batch_size, seq_len, attention.num_heads, -1)

    h = tensor_dense_scaled_dot_product_attention(q=q, k=k, v=v)
    h = h.reshape(batch_size, seq_len, hidden_size)
    return F.linear(h, attention.to_out.weight, attention.to_out.bias)
