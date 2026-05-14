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

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Flash Attention v3 (flash3) Backend: intermediate APIs
Only safe to import when FLASH3_SUPPORTED is True.
"""

import inspect

# pyrefly: ignore  # missing-import
from flash_attn_3_nv.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from torch import Tensor


# reflection of the commit hash in the version, so we have to manually inspect the signatures
HAS_RETURN_ATTN_PROBS = "return_attn_probs" in inspect.signature(flash_attn_func).parameters

from cosmos.model.attention.checks import assert_universal_tensor_checks
from cosmos.model.attention.flash3.checks import flash3_attention_check
from cosmos.model.attention.masks import CausalType
from cosmos.model.attention.utils.environment import is_torch_compiling


def flash3_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    causal_type: CausalType | None = None,
    scale: float | None = None,
    cumulative_seqlen_Q: Tensor | None = None,
    cumulative_seqlen_KV: Tensor | None = None,
    max_seqlen_Q: int | None = None,
    max_seqlen_KV: int | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Runs Flash Attention v3 on given operands (Q, K, V) with the heads-last contiguous layout
        (`[batch, seqlen, heads, head_dim]`).

    Parameters:
        query (Tensor): 4-D query tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim]`)

        key (Tensor): 4-D key tensor, with the heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim]`)

        value (Tensor): 4-D value tensor, with heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`)

        is_causal (bool): whether or not causal masking is enabled. Default is False.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`. Required when `is_causal = True`.

        scale (float | None): Dot product scale (attention scale). Defaults to head_dim ** -0.5.

        cumulative_seqlen_Q (Tensor | None): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of query tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_KV` and `max_seqlen_{Q,KV}`.

        cumulative_seqlen_KV (Tensor | None): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of key/value tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_Q` and `max_seqlen_{Q,KV}`.

        max_seqlen_Q (int | None): (varlen) Optional integer indicating the maximum query
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_KV`.

        max_seqlen_KV (int | None): (varlen) Optional integer indicating the maximum key/value
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_Q`.

    Other Parameters:
        return_lse (bool): Whether to return the logsumexp values. Default is False.

        backend_kwargs (dict | None): Key-value pair for passing arguments specific to Flash's
            attention operator, if any.

        deterministic (bool): Deterministic backward pass required.

    Returns:
        output (Tensor): 4-D output tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, 1]`). Only returned when return_lse is True.
            NOTE: this tensor is not contiguous in this backend (Flash3) and it should not be made
            contiguous unless we can guarantee its results aren't merged via `merge_attentions`.
    """

    is_varlen = cumulative_seqlen_Q is not None
    assert_universal_tensor_checks(query, key, value)

    backend_kwargs = backend_kwargs.copy() if backend_kwargs is not None else {}
    # Determinism in backend_kwargs supersedes primary flag, if set to True
    if "deterministic" in backend_kwargs:
        deterministic = deterministic or backend_kwargs["deterministic"]
        del backend_kwargs["deterministic"]

    assert flash3_attention_check(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        device=query.device,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        is_causal=is_causal,
        causal_type=causal_type,
        is_varlen=is_varlen,
        deterministic=deterministic,
        raise_error=True,
    )

    # This check introduces recompiles
    if not is_torch_compiling():
        if is_varlen and max_seqlen_Q == max_seqlen_KV == 0:
            raise NotImplementedError(
                "You're trying to use varlen attention with the flash3 backend and "
                "an empty batch, which is not yet supported by flash3."
            )

    scale = scale if scale is not None else query.shape[-1] ** -0.5

    if HAS_RETURN_ATTN_PROBS:
        backend_kwargs["return_attn_probs"] = True

    if is_varlen:
        assert query.shape[0] == key.shape[0] == value.shape[0] == 1
        q = query.squeeze(0)  # [total_tokens,H,D]
        k = key.squeeze(0)  # [total_tokens,Hkv,D]
        v = value.squeeze(0)  # [total_tokens,Hkv,Dv]
        assert q.dim() == k.dim() == v.dim() == 3
        out, lse_ = flash_attn_varlen_func(
            q=query.squeeze(0),
            k=key.squeeze(0),
            v=value.squeeze(0),
            cu_seqlens_q=cumulative_seqlen_Q,
            cu_seqlens_k=cumulative_seqlen_KV,
            max_seqlen_q=max_seqlen_Q,
            max_seqlen_k=max_seqlen_KV,
            softmax_scale=scale,
            causal=is_causal,
            deterministic=deterministic,
            **backend_kwargs,
            # qv=None,
            # q_descale=None, k_descale=None, v_descale=None,
            # attention_chunk=0,
            # num_splits=1,
            # pack_gqa=None,
            # sm_margin=0,
            # window_size=(-1, -1),
            # softcap=0.0, # 0.0 means deactivated
        )
        assert out.dim() == 3  # [total_tokens,H,Dv]
        assert lse_.dim() == 2  # [H,total_tokens]

        output = out.unsqueeze(0)  # [1,total_tokens,H,Dv]
        lse = lse_.unsqueeze(0)  # [1,H,total_tokens]

    else:
        output, lse = flash_attn_func(  # output: [B,N,H,Dv], lse: [B,H,N]
            q=query,
            k=key,
            v=value,
            softmax_scale=scale,
            causal=is_causal,
            deterministic=deterministic,
            **backend_kwargs,
            # qv=None,
            # q_descale=None, k_descale=None, v_descale=None,
            # attention_chunk=0,
            # num_splits=1,
            # pack_gqa=None,
            # sm_margin=0,
            # window_size=(-1, -1),
            # softcap=0.0, # 0.0 means deactivated
        )

    assert isinstance(output, Tensor)
    assert isinstance(lse, Tensor)
    assert output.dim() == 4  # [B,N,H,Dv] or [1,total_tokens,H,Dv]
    assert lse.dim() == 3  # [B,H,N] or [1,H,total_tokens]


    # incorrect. All output and lse tensors passed into `merge_attentions` must have the same data
    # pointer as their corresponding attention autograd ops!
    lse = lse.permute(0, 2, 1)  # [B,N,H] or [1,total_tokens,H]

    if return_lse:
        return output, lse

    return output
