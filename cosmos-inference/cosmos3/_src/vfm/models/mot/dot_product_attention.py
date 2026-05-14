# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Simplified wrapper around TransformerEngine's C++ pytorch backend.
This supports torch.compile(fullgraph=True).
Lowers to cudnn ultimately.
Only bf16 / fp16 is supported.
Only THD layout is supported.
Currently, tensors are made contiguous -- packed th2d, th3d not supported yet.
"""

import math
from typing import List, Optional, Tuple

import torch
import transformer_engine

_TE_VER = tuple(int(x) for x in transformer_engine.__version__.split(".")[:2])


try:
    # transformer_engine 2.8.0
    import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
except ImportError:
    # older transformer_engine
    import transformer_engine.pytorch.dot_product_attention.utils as dpa_utils  # type: ignore

import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import (
    TE_DType,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    AttnBiasType,
    AttnMaskType,
    QKVLayout,
)

if _TE_VER >= (2, 8):
    from transformer_engine.pytorch.cpp_extensions.fused_attn import SoftmaxType


__all__ = ["cudnn_fused_attention"]


def get_window_size(attn_mask_type: str) -> Tuple[int, int]:
    return dpa_utils.check_set_window_size(attn_mask_type)


def cudnn_fused_attention(
    query_layer: torch.Tensor,  # [total_tokens_q,num_heads,head_dim]
    key_layer: torch.Tensor,  # [total_tokens_kv,num_heads,head_dim]
    value_layer: torch.Tensor,  # [total_tokens_kv,num_heads,head_dim]
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    attn_mask_type: str = "causal",
    attention_dropout: float = 0.0,
    training: bool = True,
) -> torch.Tensor:  # [total_tokens_q,num_heads*head_dim]
    """fused attention fprop"""

    deterministic = torch.are_deterministic_algorithms_enabled()
    window_size = get_window_size(attn_mask_type)
    softmax_scale = 1.0 / math.sqrt(key_layer.shape[-1])

    output_tensors = cudnn_fused_attn(
        training,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        query_layer,
        key_layer,
        value_layer,
        window_size,
        softmax_scale,
        attention_dropout if training else 0.0,
        attn_mask_type,
        deterministic,
    )
    output = output_tensors[0]  # [total_tokens_q,num_heads,head_dim]

    # ...hd -> ...(hd)
    return output.view(*output.shape[:-2], -1)  # [total_tokens_q,num_heads*head_dim]


BACKEND_F16arb_ELTS_PER_THREADS = 16


@torch.library.custom_op("cosmos3::cudnn_fused_attn", mutates_args=())
def cudnn_fused_attn(
    is_training: bool,
    max_seqlen_q: torch.Tensor,
    max_seqlen_kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,  # [total_tokens_q,num_heads,head_dim]
    k: torch.Tensor,  # [total_tokens_kv,num_heads,head_dim]
    v: torch.Tensor,  # [total_tokens_kv,num_heads,head_dim]
    window_size: List[int],
    attn_scale: float,
    dropout: float,
    attn_mask_type: str,
    deterministic: bool,
) -> List[torch.Tensor]:
    attn_bias = None
    attn_bias_type = "no_bias"
    fast_zero_fill = True
    softmax_offset = None
    softmax_type = "vanilla"
    fake_dtype = q.dtype

    rng_elts_per_thread = BACKEND_F16arb_ELTS_PER_THREADS
    s_quantizer = None
    o_quantizer = None
    rng_gen = None


    # "thd_thd_thd" format requires contiguous tensors.
    # We should benchmark thd_th2d / th3d formats as well.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    qkv_layout = "thd_thd_thd"

    cu_seqlens_q_padded = cu_seqlens_q
    cu_seqlens_kv_padded = cu_seqlens_kv

    args = (
        max_seqlen_q.item(),
        max_seqlen_kv.item(),
        is_training,
        attn_scale,
        dropout,
        fast_zero_fill,
        QKVLayout[qkv_layout],
        AttnBiasType[attn_bias_type],
        AttnMaskType[attn_mask_type],
    )

    if _TE_VER >= (2, 8):
        args += (SoftmaxType[softmax_type],)

    args += (
        tuple(window_size),
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        fake_dtype,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        None,  # page_table_k,
        None,  # page_table_v,
        s_quantizer,
        o_quantizer,
        attn_bias,
    )

    if _TE_VER >= (2, 8):
        args += (softmax_offset,)

    args += (
        rng_gen,
        rng_elts_per_thread,
    )

    if _TE_VER >= (2, 9):
        # return_max_logit
        args += (False,)

    if _TE_VER >= (2, 10):
        # is_cuda_graph
        args += (False,)


    # I'd have to create DotProductAttention class and somehow pass it in here, but argument types for these torch.ops are very strict.
    # Moreover, back-propagation would still need additional tweaks to work properly.
    output_tensors = tex.fused_attn_fwd(*args)
    return output_tensors


import math


def _get_max_tokens(num_tokens: int) -> int:
    """
    Quantize token count:
    - t = 0, ..., 1024   -> max_t = 1024
    - t = 1025, ..., 32k -> max_t = next power of 2
    - t = 32k+1, ...     -> max_t = increment by 32k steps

    Note: translated from transformer_engine/common/fused_attn/utils.cu::get_max_tokens
    """
    if num_tokens <= 0:
        return 1024
    log2_t = math.ceil(math.log2(num_tokens))
    if log2_t <= 10:
        max_t = 1024
    elif log2_t <= 15:
        max_t = 2**log2_t
    else:
        max_t = ((num_tokens + 32767) // 32768) * 32768
    return max_t



# The goal for this function is to return fake tensors of the correct shape and dtype
# without having to run the actual operator.


@cudnn_fused_attn.register_fake
def _(
    is_training: bool,
    max_seqlen_q: torch.Tensor,
    max_seqlen_kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,  # [total_tokens_q,num_heads,head_dim]
    k: torch.Tensor,  # [total_tokens_kv,num_heads,head_dim]
    v: torch.Tensor,  # [total_tokens_kv,num_heads,head_dim]
    window_size: List[int],
    attn_scale: float,
    dropout: float,
    attn_mask_type: str,
    deterministic: bool,
) -> List[torch.Tensor]:
    max_tokens = _get_max_tokens(q.shape[0])
    return [
        q.new_empty(tuple(q.shape[:-1]) + (v.shape[-1],)),  # [total_tokens_q,num_heads,head_dim]
        q.new_empty(
            max_tokens, q.shape[1], 1, dtype=torch.float32
        ),  # these are the softmax outputs from cudnn; will always be float32
        q.new_empty((2,)),
    ]


def cudnn_fused_attn_bwd_setup_context(ctx, inputs, output) -> None:
    (
        _,  # is_training
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        window_size,
        attn_scale,
        dropout,
        attn_mask_type,
        deterministic,
    ) = inputs

    out = output[0]
    aux_ctx_tensors = output[1:]
    qkvo_tensors = (q, k, v, out)

    # assume fwd and bwd always use the same high precision, i.e. torch.float16 or torch.bfloat16
    # used when some tensors are base tensors and loose the "dtype" attribute
    ctx.nominal_dtype = q.dtype

    ctx.save_for_backward(
        *qkvo_tensors,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        *aux_ctx_tensors,
    )

    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_kv = max_seqlen_kv
    ctx.attn_scale = attn_scale
    ctx.dropout_p = dropout
    ctx.fast_zero_fill = True
    ctx.attn_bias_type = "no_bias"
    ctx.attn_mask_type = attn_mask_type
    ctx.softmax_type = "vanilla"
    ctx.window_size = window_size
    ctx.deterministic = deterministic


@torch.library.custom_op("cosmos3::cudnn_fused_attn_bwd_op", mutates_args=())
def cudnn_fused_attn_bwd_op(
    max_seqlen_q: torch.Tensor,
    max_seqlen_kv: torch.Tensor,
    attn_scale: float,
    dropout: float,
    fast_zero_fill: bool,
    attn_bias_type: str,
    attn_mask_type: str,
    softmax_type: str,
    window_size: List[int],
    deterministic: bool,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,  # [total_tokens_q,num_heads,head_dim]
    k: torch.Tensor,  # [total_tokens_kv,num_heads,head_dim]
    v: torch.Tensor,  # [total_tokens_kv,num_heads,head_dim]
    out: torch.Tensor,  # [total_tokens_q,num_heads,head_dim]
    d_out: torch.Tensor,  # [total_tokens_q,num_heads,head_dim]
    dqkv_nominal_dtype: torch.dtype,
    dqkv_te_dtype: torch.dtype,
    aux_ctx_tensors: List[torch.Tensor],
    cu_seqlens_q_padded: torch.Tensor,
    cu_seqlens_kv_padded: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # dq,dk,dv each [total_tokens,num_heads,head_dim]
    qkv_layout = "thd_thd_thd"
    args = (
        max_seqlen_q.item(),
        max_seqlen_kv.item(),
        attn_scale,
        dropout,
        fast_zero_fill,
        QKVLayout[qkv_layout],
        AttnBiasType[attn_bias_type],
        AttnMaskType[attn_mask_type],
    )

    if _TE_VER >= (2, 8):
        args += (SoftmaxType[softmax_type],)

    args += (
        window_size,
        deterministic,
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        out,
        d_out,
        dqkv_nominal_dtype,
        TE_DType[dqkv_te_dtype],
        aux_ctx_tensors,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        None,  # s_quantizer,
        None,  # dp_quantizer,
        None,  # dqkv_quantizer,
    )

    if _TE_VER >= (2, 10):
        # is_cuda_graph
        args += (False,)

    dq, dk, dv, *rest = tex.fused_attn_bwd(*args)
    return dq, dk, dv


@cudnn_fused_attn_bwd_op.register_fake
def _(
    max_seqlen_q: torch.Tensor,
    max_seqlen_kv: torch.Tensor,
    attn_scale: float,
    dropout: float,
    fast_zero_fill: bool,
    attn_bias_type: str,
    attn_mask_type: str,
    softmax_type: str,
    window_size: List[int],
    deterministic: bool,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,  # [total_tokens_q,num_heads,head_dim]
    k: torch.Tensor,  # [total_tokens_kv,num_heads,head_dim]
    v: torch.Tensor,  # [total_tokens_kv,num_heads,head_dim]
    out: torch.Tensor,  # [total_tokens_q,num_heads,head_dim]
    d_out: torch.Tensor,  # [total_tokens_q,num_heads,head_dim]
    dqkv_nominal_dtype: torch.dtype,
    dqkv_te_dtype: torch.dtype,
    aux_ctx_tensors: List[torch.Tensor],
    cu_seqlens_q_padded: torch.Tensor,
    cu_seqlens_kv_padded: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # dq,dk,dv each [total_tokens,num_heads,head_dim]
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def cudnn_fused_attn_bwd_impl(ctx, grad):
    d_out, _, _ = grad
    d_out = d_out.contiguous()

    (
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        *aux_ctx_tensors,
    ) = ctx.saved_tensors

    if not aux_ctx_tensors[0].is_contiguous():
        aux_ctx_tensors[0] = aux_ctx_tensors[0].contiguous()

    with torch.cuda.nvtx.range("FusedAttnFunc.backward"):
        # get nominal data type of dq, dk, dv
        # FP16/BF16 attention: torch.float16 or torch.bfloat16
        dqkv_nominal_dtype = ctx.nominal_dtype

        # q, k, v, out, d_out, dq, dk, dv: torch.Tensor; torch.float16 or torch.bfloat16
        dq, dk, dv = cudnn_fused_attn_bwd_op(
            ctx.max_seqlen_q,
            ctx.max_seqlen_kv,
            ctx.attn_scale,
            ctx.dropout_p,
            ctx.fast_zero_fill,
            ctx.attn_bias_type,
            ctx.attn_mask_type,
            ctx.softmax_type,
            ctx.window_size,
            ctx.deterministic,
            cu_seqlens_q,
            cu_seqlens_kv,
            q,
            k,
            v,
            out,
            d_out,
            dqkv_nominal_dtype,
            d_out.dtype,
            aux_ctx_tensors,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
        )

        output = (
            None,  # is_training
            None,  # max_seqlen_q
            None,  # max_seqlen_kv
            None,  # cu_seqlens_q
            None,  # cu_seqlens_kv
            dq,
            dk,
            dv,
            None,  # window_size
            None,  # attn_scale
            None,  # dropout
            None,  # attn_mask_type
            None,  # deterministic
        )
        return output


cudnn_fused_attn.register_autograd(cudnn_fused_attn_bwd_impl, setup_context=cudnn_fused_attn_bwd_setup_context)
