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

"""Full scaled dot-product attention for sparse tensors.

This module provides full attention implementations supporting:
    - Self-attention with packed QKV
    - Cross-attention with separate Q and KV
    - Separate Q, K, V tensors
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from cosmos.model.attention.frontend import attention as i4_attention
from cosmos.model.attention.varlen import generate_varlen_parameters

if TYPE_CHECKING:
    from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor


__all__ = [
    "sparse_scaled_dot_product_attention",
    "tensor_dense_scaled_dot_product_attention",
    "tensor_varlen_scaled_dot_product_attention",
]


def _generate_varlen_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_seqlen: list[int],
    kv_seqlen: list[int],
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build varlen metadata once via cosmos.model.attention utilities.

    This path is intended for the generic sparse-attention codepaths that do
    not already receive precomputed varlen metadata from upstream. Tensor fast
    paths should continue to pass cumulative seqlens and max lengths directly.
    """
    q_seqlens_tensor = torch.tensor(q_seqlen, dtype=torch.int32, device=q.device)
    kv_seqlens_tensor = torch.tensor(kv_seqlen, dtype=torch.int32, device=q.device)
    cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen = generate_varlen_parameters(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        q_seqlens_tensor,
        kv_seqlens_tensor,
    )
    assert cu_seqlens_q is not None
    assert cu_seqlens_kv is not None
    return cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen


def tensor_varlen_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_q_seqlen: int,
    max_kv_seqlen: int,
) -> torch.Tensor:
    """Apply tokenizer packed varlen attention through cosmos.model.attention."""
    if q.shape[0] == 0:
        return q.new_empty((0, q.shape[1], v.shape[-1]))

    out = i4_attention(
        query=q.unsqueeze(0).contiguous(),
        key=k.unsqueeze(0).contiguous(),
        value=v.unsqueeze(0).contiguous(),
        cumulative_seqlen_Q=cu_seqlens_q,
        cumulative_seqlen_KV=cu_seqlens_kv,
        max_seqlen_Q=max_q_seqlen,
        max_seqlen_KV=max_kv_seqlen,
    )
    return out.squeeze(0)


def tensor_dense_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Apply dense batched attention via the imaginaire attention frontend."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "Dense tensor attention expects [B, S, H, D]-style tensors, "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
        )
    if q.shape[0] == 0:
        return q.new_empty((0, q.shape[1], q.shape[2], v.shape[-1]))

    return i4_attention(
        query=q.contiguous(),
        key=k.contiguous(),
        value=v.contiguous(),
    )


def _pack_sparse_temporal_causal_qkv(
    q: "SparseTensor",
    k: "SparseTensor",
    v: "SparseTensor",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int], torch.Tensor]:
    """Pack sparse Q/K/V into timestep-prefix attention segments.

    Each timestep becomes one query segment. Its matching KV segment contains all
    tokens from the same batch with temporal index less than or equal to that
    timestep, plus any special tokens with negative temporal indices. This is
    equivalent to a temporal causal mask with full visibility inside the current
    timestep.
    """
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(
            f"Batch size mismatch for temporal causal attention: q={q.shape[0]}, k={k.shape[0]}, v={v.shape[0]}"
        )

    q_index_chunks: list[torch.Tensor] = []
    q_feat_chunks: list[torch.Tensor] = []
    k_feat_chunks: list[torch.Tensor] = []
    v_feat_chunks: list[torch.Tensor] = []
    q_seqlen: list[int] = []
    kv_seqlen: list[int] = []

    for batch_idx in range(q.shape[0]):
        q_slice = q.layout[batch_idx]
        k_slice = k.layout[batch_idx]

        if q_slice.start == q_slice.stop:
            continue

        q_batch_coords = q.coords[q_slice]
        k_batch_coords = k.coords[k_slice]

        if k_batch_coords.shape[0] == 0:
            raise ValueError(f"Temporal causal attention requires non-empty KV for batch {batch_idx}.")

        q_batch_times = q_batch_coords[:, 1]
        k_batch_times = k_batch_coords[:, 1]

        q_special_mask = q_batch_times < 0
        if q_special_mask.any():
            q_idx = q_special_mask.nonzero(as_tuple=True)[0] + q_slice.start
            kv_idx = (k_batch_times < 0).nonzero(as_tuple=True)[0] + k_slice.start
            if kv_idx.numel() == 0:
                kv_idx = q_idx

            q_index_chunks.append(q_idx)
            q_feat_chunks.append(q.feats.index_select(0, q_idx))
            k_feat_chunks.append(k.feats.index_select(0, kv_idx))
            v_feat_chunks.append(v.feats.index_select(0, kv_idx))
            q_seqlen.append(int(q_idx.numel()))
            kv_seqlen.append(int(kv_idx.numel()))

        valid_q_times = q_batch_times[q_batch_times >= 0]
        if valid_q_times.numel() == 0:
            continue

        for timestep in torch.unique(valid_q_times, sorted=True).tolist():
            q_local_idx = (q_batch_times == timestep).nonzero(as_tuple=True)[0]
            kv_local_idx = ((k_batch_times < 0) | (k_batch_times <= timestep)).nonzero(as_tuple=True)[0]

            q_idx = q_local_idx + q_slice.start
            kv_idx = kv_local_idx + k_slice.start

            q_index_chunks.append(q_idx)
            q_feat_chunks.append(q.feats.index_select(0, q_idx))
            k_feat_chunks.append(k.feats.index_select(0, kv_idx))
            v_feat_chunks.append(v.feats.index_select(0, kv_idx))
            q_seqlen.append(int(q_idx.numel()))
            kv_seqlen.append(int(kv_idx.numel()))

    if not q_index_chunks:
        empty_q = q.feats.new_empty((0,) + q.feats.shape[1:])
        empty_k = k.feats.new_empty((0,) + k.feats.shape[1:])
        empty_v = v.feats.new_empty((0,) + v.feats.shape[1:])
        empty_idx = torch.empty(0, dtype=torch.long, device=q.device)
        return empty_q, empty_k, empty_v, [], [], empty_idx

    return (
        torch.cat(q_feat_chunks, dim=0),
        torch.cat(k_feat_chunks, dim=0),
        torch.cat(v_feat_chunks, dim=0),
        q_seqlen,
        kv_seqlen,
        torch.cat(q_index_chunks, dim=0),
    )


def _sparse_temporal_causal_scaled_dot_product_attention(
    q: "SparseTensor",
    k: "SparseTensor",
    v: "SparseTensor",
) -> "SparseTensor":
    """Apply temporal-causal attention with full visibility inside each timestep."""
    q_pack, k_pack, v_pack, q_seqlen, kv_seqlen, q_indices = _pack_sparse_temporal_causal_qkv(q, k, v)

    if q_indices.numel() == 0:
        return q.replace(q.feats.clone())

    cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen = _generate_varlen_metadata(
        q=q_pack,
        k=k_pack,
        v=v_pack,
        q_seqlen=q_seqlen,
        kv_seqlen=kv_seqlen,
    )
    out = tensor_varlen_scaled_dot_product_attention(
        q=q_pack,
        k=k_pack,
        v=v_pack,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_q_seqlen=max_q_seqlen,
        max_kv_seqlen=max_kv_seqlen,
    )

    out_feats = torch.empty_like(q.feats)
    out_feats[q_indices] = out
    return q.replace(out_feats)


@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, k: "SparseTensor", v: "SparseTensor") -> torch.Tensor:
    """Apply scaled dot product attention to a sparse tensor.

    Args:
        q: A [N, L, H, Ci] dense tensor containing Qs.
        k: A [N, *, H, Ci] sparse tensor containing Ks.
        v: A [N, *, H, Co] sparse tensor containing Vs.
    """
    ...


def sparse_scaled_dot_product_attention(*args, **kwargs):
    """Flexible scaled dot-product attention for sparse tensors.

    Supports three calling conventions:
        1. Single packed QKV tensor: qkv of shape [N, *, 3, H, C]
        2. Separate Q and packed KV: q of shape [N, *, H, C], kv of shape [N, *, 2, H, C]
        3. Separate Q, K, V tensors: q, k, v each of shape [N, *, H, C]

    Args:
        *args: Positional arguments (qkv, or q+kv, or q+k+v).
        **kwargs: Keyword arguments for the above.

    Returns:
        Attention output with same structure as query input.
    """
    from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

    temporal_causal_mask = kwargs.pop("temporal_causal_mask", False)
    kv_seqlen_override = kwargs.pop("kv_seqlen", None)
    cu_seqlens_kv_override = kwargs.pop("cu_seqlens_kv", None)
    max_kv_seqlen_override = kwargs.pop("max_kv_seqlen", None)
    arg_names_dict = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args) :]:
        assert key in kwargs, f"Missing argument {key}"

    if temporal_causal_mask:
        q_arg = args[0] if len(args) > 0 else kwargs["q"]
        k_arg = args[1] if len(args) > 1 else kwargs["k"]
        v_arg = args[2] if len(args) > 2 else kwargs["v"]
        if num_all_args != 3:
            raise ValueError("temporal_causal_mask only supports separate q, k, v inputs.")
        if not (
            isinstance(q_arg, SparseTensor) and isinstance(k_arg, SparseTensor) and isinstance(v_arg, SparseTensor)
        ):
            raise ValueError("temporal_causal_mask requires sparse q, k, v inputs.")
        return _sparse_temporal_causal_scaled_dot_product_attention(q_arg, k_arg, v_arg)

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs["qkv"]
        assert isinstance(qkv, SparseTensor), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, (
            f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        )
        device = qkv.device

        s = qkv
        q_seqlen = qkv.get_batch_seq_lens()
        kv_seqlen = q_seqlen
        qkv = qkv.feats  # [T, 3, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        assert (
            isinstance(q, SparseTensor)
            and isinstance(kv, (SparseTensor, torch.Tensor))
            or isinstance(q, torch.Tensor)
            and isinstance(kv, SparseTensor)
        ), f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = q.get_batch_seq_lens()
            q = q.feats  # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)  # [T_Q, H, C]

        if isinstance(kv, SparseTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, (
                f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            )
            kv_seqlen = kv.get_batch_seq_lens()
            kv = kv.feats  # [T_KV, 2, H, C]
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)  # [T_KV, 2, H, C]

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
        assert (
            isinstance(q, SparseTensor)
            and isinstance(k, (SparseTensor, torch.Tensor))
            and type(k) == type(v)
            or isinstance(q, torch.Tensor)
            and isinstance(k, SparseTensor)
            and isinstance(v, SparseTensor)
        ), f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        packed_flat_kv = (
            isinstance(q, SparseTensor)
            and isinstance(k, torch.Tensor)
            and isinstance(v, torch.Tensor)
            and len(k.shape) == 3
            and len(v.shape) == 3
        )
        if packed_flat_kv:
            assert kv_seqlen_override is not None or cu_seqlens_kv_override is not None, (
                "Packed flat KV tensors require kv_seqlen or cu_seqlens_kv overrides."
            )
            if kv_seqlen_override is not None:
                assert q.shape[0] == len(kv_seqlen_override), (
                    f"Batch size mismatch, got {q.shape[0]} query batches and {len(kv_seqlen_override)} KV segments."
                )
        else:
            assert q.shape[0] == k.shape[0] == v.shape[0], (
                f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
            )
        device = q.device

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = q.get_batch_seq_lens()
            q = q.feats  # [T_Q, H, Ci]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if isinstance(k, SparseTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = k.get_batch_seq_lens()
            k = k.feats  # [T_KV, H, Ci]
            v = v.feats  # [T_KV, H, Co]
        else:
            if len(k.shape) == 3 and len(v.shape) == 3:
                if kv_seqlen_override is None:
                    assert cu_seqlens_kv_override is not None
                    kv_seqlen_override = (
                        (cu_seqlens_kv_override[1:] - cu_seqlens_kv_override[:-1]).to(dtype=torch.int64).tolist()
                    )
                kv_seqlen = kv_seqlen_override
                if max_kv_seqlen_override is None:
                    max_kv_seqlen_override = max(kv_seqlen) if kv_seqlen else 0
            else:
                assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
                assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
                N, L, H, CI, CO = *k.shape, v.shape[-1]
                kv_seqlen = [L] * N
                k = k.reshape(N * L, H, CI)  # [T_KV, H, Ci]
                v = v.reshape(N * L, H, CO)  # [T_KV, H, Co]

    if num_all_args == 1:
        q, k, v = qkv.unbind(dim=1)
    elif num_all_args == 2:
        k, v = kv.unbind(dim=1)

    if num_all_args in [1, 2, 3]:
        cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen = _generate_varlen_metadata(
            q=q,
            k=k,
            v=v,
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
        )
        if cu_seqlens_kv_override is not None:
            cu_seqlens_kv = cu_seqlens_kv_override.to(device=device, dtype=torch.int32)
            max_kv_seqlen = max_kv_seqlen_override if max_kv_seqlen_override is not None else max_kv_seqlen

    out = tensor_varlen_scaled_dot_product_attention(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_q_seqlen=max_q_seqlen,
        max_kv_seqlen=max_kv_seqlen,
    )

    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)
