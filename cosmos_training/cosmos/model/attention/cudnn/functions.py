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

cuDNN Backend: intermediate APIs
Only safe to import when CUDNN_SUPPORTED is True.
"""

import time
from functools import partial

import torch
from torch import Tensor
from torch.amp import custom_bwd, custom_fwd
from torch.autograd import Function

from cosmos.model.attention.checks import assert_universal_tensor_checks
from cosmos.model.attention.cudnn.checks import cudnn_attention_check
from cosmos.model.attention.cudnn.cudnn_forward import (
    cudnn_sdpa_fwd_generate_op,
    cudnn_sdpa_fwd_generate_operands,
    cudnn_sdpa_fwd_post_process,
)
from cosmos.model.attention.masks import CausalType
from cosmos.model.attention.utils.safe_ops import log

amp_fwd = partial(custom_fwd, device_type="cuda")
amp_bwd = partial(custom_bwd, device_type="cuda")


CUDNN_PADDING_REQUIRED = False


class CudnnAttentionAutogradFn(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        num_heads: int,
        is_causal: bool,
        scale: float,
    ) -> tuple[Tensor, Tensor]:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        seqlen_Q = None
        seqlen_KV = None
        padding_Q = 0
        padding_KV = 0

        if CUDNN_PADDING_REQUIRED:
            Q_multiplier = 256
            KV_multiplier = 256

            if query.shape[1] % Q_multiplier != 0:
                seqlen_Q = query.shape[1]
                padding_Q = Q_multiplier - (seqlen_Q % Q_multiplier)

                old_shape = query.shape
                query = torch.nn.functional.pad(query, (0, 0, 0, 0, 0, padding_Q), "constant", 0)
                log.debug(f"cuDNN Attention: padded query from {old_shape} to {query.shape}.")

            if key.shape[1] % KV_multiplier != 0:
                seqlen_KV = key.shape[1]
                padding_KV = KV_multiplier - (seqlen_KV % KV_multiplier)

                old_shape = key.shape
                key = torch.nn.functional.pad(key, (0, 0, 0, 0, 0, padding_KV), "constant", 0)
                value = torch.nn.functional.pad(value, (0, 0, 0, 0, 0, padding_KV), "constant", 0)
                log.debug(f"cuDNN Attention: padded KV from {old_shape} to {key.shape}.")

        # Transform operands to cuDNN-compatible layouts, make output tensors
        (q_cudnn_layout, k_cudnn_layout, v_cudnn_layout, output_cudnn_layout, lse_cudnn_layout) = (
            cudnn_sdpa_fwd_generate_operands(q=query, k=key, v=value, num_heads=num_heads, return_lse=True)
        )

        # Construct graph
        assert q_cudnn_layout.device == k_cudnn_layout.device == v_cudnn_layout.device == output_cudnn_layout.device
        assert q_cudnn_layout.dtype == k_cudnn_layout.dtype == v_cudnn_layout.dtype == output_cudnn_layout.dtype
        cudnn_graph_gen_start = time.time() * 1e3
        cudnn_sdpa = cudnn_sdpa_fwd_generate_op(
            dtype=q_cudnn_layout.dtype,
            device=q_cudnn_layout.device,
            q_shape=q_cudnn_layout.shape,
            q_stride=q_cudnn_layout.stride(),
            k_shape=k_cudnn_layout.shape,
            k_stride=k_cudnn_layout.stride(),
            v_shape=v_cudnn_layout.shape,
            v_stride=v_cudnn_layout.stride(),
            output_shape=output_cudnn_layout.shape,
            output_stride=output_cudnn_layout.stride(),
            lse_shape=None if lse_cudnn_layout is None else lse_cudnn_layout.shape,
            lse_stride=None if lse_cudnn_layout is None else lse_cudnn_layout.stride(),
            is_causal=is_causal,
            attn_scale=scale,
            seqlen_Q=seqlen_Q,
            seqlen_KV=seqlen_KV,
        )
        cudnn_graph_gen_time = time.time() * 1e3 - cudnn_graph_gen_start
        log.debug(f"cuDNN Attention forward graph generation took {cudnn_graph_gen_time:.1f} ms.")

        # Execute graph
        cudnn_sdpa(
            q=q_cudnn_layout,
            k=k_cudnn_layout,
            v=v_cudnn_layout,
            output=output_cudnn_layout,
            lse=lse_cudnn_layout,
        )

        # Transform outputs back to torch contiguous layouts
        output, logsumexp = cudnn_sdpa_fwd_post_process(
            output_cudnn_layout=output_cudnn_layout,
            lse_cudnn_layout=lse_cudnn_layout,
        )

        ctx.save_for_backward(q_cudnn_layout, k_cudnn_layout, v_cudnn_layout, lse_cudnn_layout, output_cudnn_layout)
        ctx.num_heads = num_heads
        ctx.scale = scale

        if padding_Q > 0:
            old_shape = output.shape
            output = output[:, :seqlen_Q, :, :]
            logsumexp = logsumexp[:, :seqlen_Q, :, :]
            assert output.shape[1] == seqlen_Q
            assert logsumexp.shape[1] == seqlen_Q
            log.debug(f"cuDNN Attention: unpadded output from {old_shape} to {output.shape}.")

        return output, logsumexp

    @staticmethod
    @amp_bwd
    def backward(
        ctx, grad_out: Tensor, grad_lse: Tensor
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        None,
        None,
        None,
    ]:
        raise NotImplementedError()


def cudnn_attention(
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
    Runs cuDNN Attention on given operands (Q, K, V) with the heads-last contiguous layout
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

        backend_kwargs (dict | None): Key-value pair for passing arguments specific to cuDNN's
            attention operator, if any.

        deterministic (bool): Deterministic backward pass required.

    Returns:
        output (Tensor): 4-D output tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, 1]`). Only returned when return_lse is True.
    """

    is_varlen = cumulative_seqlen_Q is not None
    assert_universal_tensor_checks(query, key, value)
    assert cudnn_attention_check(
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


    assert not is_varlen  # cudnn_attention_check should prevent this assertion failing

    num_heads = query.shape[-2]
    scale = scale if scale is not None else query.shape[-1] ** -0.5

    output, lse = CudnnAttentionAutogradFn.apply(
        query,
        key,
        value,
        num_heads,
        is_causal,
        scale,
    )

    if return_lse:
        return output, lse

    return output
