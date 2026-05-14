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

Frontend APIs
"""

import torch
from torch import Tensor

from cosmos.model.attention.backends import choose_backend, choose_multi_dim_backend
from cosmos.model.attention.checks import (
    attention_param_checks,
    attention_tensor_checks,
    multi_dim_attention_param_checks,
    multi_dim_attention_param_filter,
    multi_dim_attention_tensor_checks,
    universal_tensor_checks,
    varlen_tensor_checks,
)
from cosmos.model.attention.flash2 import flash2_attention
from cosmos.model.attention.flash3 import flash3_attention
from cosmos.model.attention.masks import CausalType
from cosmos.model.attention.natten import natten_attention, natten_multi_dim_attention
from cosmos.model.attention.utils.environment import filter_attention_merge_backends
from cosmos.model.attention.utils.safe_ops import log


# Map backend names to their frontend attention API
BACKEND_MAP = {
    "natten": natten_attention,
    "flash2": flash2_attention,
    "flash3": flash3_attention,
}

MULTI_DIM_BACKEND_MAP = {
    "natten": natten_multi_dim_attention,
}


def attention(
    query: Tensor,  # [B,S_Q,H,D]
    key: Tensor,  # [B,S_KV,H_KV,D]
    value: Tensor,  # [B,S_KV,H_KV,D_V]
    is_causal: bool = False,
    causal_type: CausalType | None = None,
    scale: float | None = None,
    # varlen parameters
    seqlens_Q: Tensor | None = None,  # [B]
    seqlens_KV: Tensor | None = None,  # [B]
    cumulative_seqlen_Q: Tensor | None = None,  # [B+1]
    cumulative_seqlen_KV: Tensor | None = None,  # [B+1]
    max_seqlen_Q: int | None = None,
    max_seqlen_KV: int | None = None,
    # backend & misc parameters
    backend: str | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:  # [B,S_Q,H,D_V] or ([B,S_Q,H,D_V], [B,S_Q,H,1])
    """
    Runs Attention on given operands (Q, K, V) with the heads-last contiguous layout
        (`[batch, seqlen, heads, head_dim]`).

    Varlen Attention is only supported for the sequence-packed layout: QKV tensors have batch size
    1, and tokens from different batches are concatenated without any padding along the sequence
    dimension. Sequence lengths for different batches can be provided in two ways:
        1. `seqlens_Q` and `seqlens_KV` (less efficient): only provide the sequence lengths as
            integer tensors (must be on the same device as QKV), and cumulative and maximum sequence
            lengths are recomputed on each call.
        2. `cumulative_seqlen_{Q,KV}` and `max_seqlen_{Q,KV}` (more efficient):
            compute cumulative and maximum sequence lengths. `cumulative_seqlen_{Q,KV}` are integer
            tensors on the same device as QKV containing the cumulative sum of `seqlens_{Q,KV}`,
            with an additional `0` element in the beginning, therefore sized `batch+1`.
            `max_seqlen_{Q,KV}` are integers (not Tensors) that represent the maximum sequence
            lengths for Q and KV among all sequence batches.
            You can use `generate_varlen_parameters` to generate these
            parameters:
                ```python3
                from cosmos.model.attention.varlen import generate_varlen_parameters
                (
                    cumulative_seqlen_Q,
                    cumulative_seqlen_KV,
                    max_seqlen_Q,
                    max_seqlen_KV,
                ) = generate_varlen_parameters(q, k, v, seqlens_Q, seqlens_KV)
                ```

    Parameters:
        query (Tensor): 4-D query tensor, with the heads-last contiguous layout
            (`[batch, seqlen_q, heads, head_dim]`)

        key (Tensor): 4-D key tensor, with the heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim]`)

        value (Tensor): 4-D value tensor, with heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`)

        is_causal (bool): whether or not causal masking is enabled. Default is False.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`, `CausalType.DontCare` (only valid when seqlen_q == seqlen_kv).
            Required when `is_causal = True`.

        scale (float | None): Dot product scale (attention scale). Defaults to head_dim ** -0.5.

        seqlens_Q (Tensor | None): (varlen) Optional 1-D tensor with size `batch`
            indicating the number of query tokens in each batch. Must be passed together with
            `seqlens_KV`.

        seqlens_KV (Tensor | None): (varlen) Optional 1-D tensor with size `batch`
            indicating the number of key/value tokens in each batch. Must be passed together with
            `seqlens_Q`.

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
        backend (str | None): Backend to run with. If unspecified (default), it will try to
            select the best available.

        return_lse (bool): Whether to return the logsumexp values. Default is False.

        backend_kwargs (dict | None): Key-value pair for passing arguments specific to the backend's
            attention operator, if any. Only valid when a specific backend is selected (backend is
            not None).

        deterministic (bool): Whether to enforce deterministic backward pass. Default is False.
            When True, backends are selected based on deterministic support.

    Returns:
        output (Tensor): 4-D output tensor, with the heads-last contiguous layout
            (`[batch, seqlen_q, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor, with the heads-last contiguous layout
            (`[batch, seqlen_q, heads, 1]`). Only returned when return_lse is True.
            NOTE: this tensor is not guaranteed to be contiguous with some backends and it should
            not be made contiguous unless we can guarantee its results aren't merged via
            `merge_attentions`.
    """

    assert universal_tensor_checks(query=query, key=key, value=value, raise_error=True)

    assert attention_tensor_checks(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        raise_error=True,
    )

    attention_param_checks(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        is_causal=is_causal,
        causal_type=causal_type,
    )

    (
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        max_seqlen_Q,
        max_seqlen_KV,
    ) = varlen_tensor_checks(
        query=query,
        key=key,
        value=value,
        seqlens_Q=seqlens_Q,
        seqlens_KV=seqlens_KV,
        cumulative_seqlen_Q=cumulative_seqlen_Q,
        cumulative_seqlen_KV=cumulative_seqlen_KV,
        max_seqlen_Q=max_seqlen_Q,
        max_seqlen_KV=max_seqlen_KV,
    )
    is_varlen = cumulative_seqlen_Q is not None

    scale = scale if scale is not None else query.shape[-1] ** -0.5

    if backend is None and backend_kwargs is not None:
        backend_kwargs = None
        log.debug("A backend was not specified, but got backend_kwargs. Ignoring... ")

    if backend is not None and backend not in BACKEND_MAP:
        raise ValueError(f"Selected {backend=}, but available choices are {BACKEND_MAP.keys()}. ")

    compatible_backend = choose_backend(
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
        backend=backend,
        raise_error=False,
    )

    # Either incompatible backend specified by user, or no compatible backends found
    if compatible_backend is None and backend is None:
        raise ValueError(
            "Could not find a compatible Attention backend for this use case / device. "
            "Try running with debug logs to find out why."
        )
    elif compatible_backend is None:
        raise ValueError(
            f"Selected Attention backend {backend} is incompatible with this use case / device. "
            "Try running with debug logs to find out why."
        )

    assert compatible_backend in BACKEND_MAP
    return BACKEND_MAP[compatible_backend](
        query=query,
        key=key,
        value=value,
        is_causal=is_causal,
        causal_type=causal_type,
        scale=scale,
        cumulative_seqlen_Q=cumulative_seqlen_Q,
        cumulative_seqlen_KV=cumulative_seqlen_KV,
        max_seqlen_Q=max_seqlen_Q,
        max_seqlen_KV=max_seqlen_KV,
        return_lse=return_lse,
        backend_kwargs=backend_kwargs,
        deterministic=deterministic,
    )


def multi_dimensional_attention(
    query: Tensor,  # [B,*token_layout_shape,H,D]
    key: Tensor,  # [B,*token_layout_shape,H_KV,D]
    value: Tensor,  # [B,*token_layout_shape,H_KV,D_V]
    window_size: tuple | int = -1,
    stride: tuple | int = 1,
    dilation: tuple | int = 1,
    is_causal: tuple | bool = False,
    scale: float | None = None,
    # backend & misc parameters
    backend: str | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> (
    Tensor | tuple[Tensor, Tensor]
):  # [B,*token_layout_shape,H,D_V] or ([B,*token_layout_shape,H,D_V], [B,*token_layout_shape,H])
    """
    Runs Multi-Dimensional Attention on given operands (Q, K, V) with the heads-last contiguous
    layout (`[batch, *, heads, head_dim]`). Supports up to and including 3 dimensions:
        * 1-D: `[batch, X, heads, head_dim]`, with masking arguments expecting tuples of size 1.
        * 2-D: `[batch, X, Y, heads, head_dim]`, with masking arguments expecting tuples of size 2.
        * 3-D: `[batch, X, Y, Z, heads, head_dim]`, with masking arguments expecting tuples of size 3.

    The dimensions here refer to the layout of tokens; that is the arrangement of tokens for each
    batch/head, or the `[X]`, `[X, Y]`, `[X, Y, Z]` part of the input shape.
    We refer to these as the "token layout shape".

    For now, it is always expected that Q, K, and V match in the sizes of those dimensions.

    Masking arguments, all of which can be set uniformly across all dimensions or per dimension, are:
        * `window_size`: determines the sliding window size. -1 is interpreted as the maximum window
            size. Must be either -1 or at least 2 and at most the token layout shape.
            For example, if inputs are `[batch, X, Y, Z, heads_{q,kv}, head_dim_{qk,v}]`,
            `window_size` must be either an integer == -1 or an integer <= `min(X, Y, Z)`,
            or a tuple of size 3 corresponding to the three dimensions / axes, where:
                * `window_size[0] == -1 or 2 <= window_size[0] <= X`
                * `window_size[1] == -1 or 2 <= window_size[1] <= Y`
                * `window_size[2] == -1 or 2 <= window_size[2] <= Z`
            When `window_size` is set to the maximum for any dimension, we're effectively performing
            self attention (no sparsity) along that dimension.
            Default is -1 (self attention).

        * `stride`: determines the step size of the sliding window. Only matters when the
            corresponding `window_size` is not -1 / maximum (self attention).
            Default is 1, indicating the smallest sliding window delay.
            Larger values trade off translational equivariance for potentially improved efficiency.
            Maximum value for `stride` along each dimension is the corresponding `window_size`.
            If `stride == window_size` along any dimension, it is equivalent to blocked / windowed
            attention (from works such as Swin Transformer, SAM, ViTDet, etc) along that dimension,
            meaning no overlap between windows.
            For more details, please refer to the GNA paper:
            https://arxiv.org/abs/2504.16922

        * `dilation`: introduces gaps between tokens in a sliding window, similarly to dilated
            convolution.
            Default is 1, indicating no gaps.
            Maximum value is the largest positive integer that satisfies
            `window_size * dilation <= token_layout_shape` along that dimension.
            Higher dilation means more sparse and global context. Lower dilation means more
            locality.
            For more details, please refer to the DiNAT paper:
            https://arxiv.org/abs/2209.15001

        * `is_causal`: per-dimension causal mask.

    Parameters:
        query (Tensor): 4-D, 5-D, or 6-D query tensor, with the heads-last contiguous layout
            (`[batch, *token_layout_shape, heads, head_dim]`)

        key (Tensor): 4-D, 5-D, or 6-D key tensor, with the heads-last contiguous layout
            (`[batch, *token_layout_shape, heads_kv, head_dim]`)

        value (Tensor): 4-D, 5-D, or 6-D value tensor, with heads-last contiguous layout
            (`[batch, *token_layout_shape, heads_kv, head_dim_v]`)

        window_size (tuple | int): Attention window (kernel) size / shape. If an
            integer, it will be repeated for all dimensions. For example `window_size=3`, when
            `len(token_layout_shape) == 3`, is interpreted as `window_size=(3, 3, 3)`.
            `-1`s are replaced with the corresponding `token_layout_shape`.
            Final window size must satisfy `2 <= window_size <= token_layout_shape`.
            Default is -1 (no sparsity).

        stride (tuple | int): Sliding window step size/shape. If an integer, it will be repeated
            for all dimensions. For example `stride=2`, when `len(token_layout_shape) == 3`, is
            interpreted as `stride=(2, 2, 2)`.
            Final stride must satisfy `1 <= stride <= window_size`.
            Default is 1.

        dilation (tuple | int): Dilation step size/shape. If an integer, it will be repeated for
            all dimensions. For example `dilation=4`, when `len(token_layout_shape) == 3`, is
            interpreted as `dilation=(4, 4, 4)`.
            Final dilation must satisfy `2 <= dilation * window_size <= token_layout_shape`.
            Default is 1.

        is_causal (tuple | bool): Toggle causal masking. If a boolean, it will be repeated for all
            dimensions. For example `is_causal=True`, when `len(token_layout_shape) == 3`, is
            interpreted as `is_causal=(True, True, True)`.
            Default is False.

        scale (float | None): Dot product scale (attention scale). Defaults to head_dim ** -0.5.

    Other Parameters:
        backend (str | None): Backend to run with. If unspecified (default), it will try to
            select the best available.

        return_lse (bool): Whether to return the logsumexp values. Default is False.

        backend_kwargs (dict | None): Key-value pair for passing arguments specific to the backend's
            multi-dim / sparse attention operator, if any. Only valid when a specific backend is
            selected (backend is not None).

        deterministic (bool): Whether to enforce deterministic backward pass. Default is False.
            When True, backends are selected based on deterministic support.

    Returns:
        output (Tensor): 4-D, 5-D, or 6-D output tensor, with the heads-last contiguous layout
            (`[batch, *token_layout_shape, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor, with the heads-last contiguous layout
            (`[batch, *token_layout_shape, heads]`). Only returned when return_lse is True.
    """

    assert universal_tensor_checks(query=query, key=key, value=value, raise_error=True)

    assert multi_dim_attention_tensor_checks(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        raise_error=True,
    )

    token_layout_shape, window_size, stride, dilation, is_causal = multi_dim_attention_param_filter(
        query,
        window_size=window_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
    )
    num_dims = len(token_layout_shape)

    # Automatic transformation for 1s in token layout
    # I.e. Attention over a (1, 16, 32) token layout is identical to over a (16, 32)

    token_layout_ones = [i for i in range(num_dims) if token_layout_shape[i] == 1]
    if len(token_layout_ones) > 0:
        token_layout_t = tuple(s for i, s in enumerate(token_layout_shape) if i not in token_layout_ones)
        window_size_t = tuple(w for i, w in enumerate(window_size) if i not in token_layout_ones)
        stride_t = tuple(s for i, s in enumerate(stride) if i not in token_layout_ones)
        dilation_t = tuple(d for i, d in enumerate(dilation) if i not in token_layout_ones)
        is_causal_t = tuple(c for i, c in enumerate(is_causal) if i not in token_layout_ones)

        assert all(x >= 2 for x in token_layout_t)
        assert all(w >= 2 for w in window_size_t)

        query_t = query.reshape(
            query.shape[0], *token_layout_t, query.shape[-2], query.shape[-1]
        )  # [B,*token_layout_t,H,D]
        key_t = key.reshape(key.shape[0], *token_layout_t, key.shape[-2], key.shape[-1])  # [B,*token_layout_t,H_KV,D]
        value_t = value.reshape(
            value.shape[0], *token_layout_t, value.shape[-2], value.shape[-1]
        )  # [B,*token_layout_t,H_KV,D_V]
        output_shape = [x for x in query.shape[:-1]] + [value.shape[-1]]

        if not torch.compiler.is_compiling():
            log.debug(
                "This Multi-Dimensional Attention problem has 1s in the token layout, which can be simplified from "
                f"<{token_layout_shape=}, {window_size=}, {stride=}, {dilation=}, {is_causal=}> into "
                f"<{token_layout_t=}, {window_size_t=}, {stride_t=}, {dilation_t=}, {is_causal_t=}>."
            )

        output_t, lse_t = multi_dimensional_attention(
            query=query_t,
            key=key_t,
            value=value_t,
            window_size=window_size_t,
            stride=stride_t,
            dilation=dilation_t,
            is_causal=is_causal_t,
            scale=scale,
            backend=backend,
            return_lse=True,
            backend_kwargs=backend_kwargs,
            deterministic=deterministic,
        )
        output = output_t.reshape(*output_shape)  # [B,*token_layout_shape,H,D_V]
        lse = lse_t.reshape(*output_shape[:-1])  # [B,*token_layout_shape,H]
        if return_lse:
            return output, lse
        return output

    multi_dim_attention_param_checks(
        query,
        window_size=window_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
    )

    # Fast path for self attention problems
    if all(x == w for x, w in zip(token_layout_shape, window_size)) and (
        not any(c for c in is_causal) or num_dims == 1
    ):
        if not torch.compiler.is_compiling():
            log.debug(
                "This Multi-Dimensional Attention problem is implementable with standard Attention: "
                f"{token_layout_shape=}, {window_size=}, {is_causal=}."
            )
            if backend is not None:
                log.debug(f"Ignoring {backend=} and backend args...")

        query_1d = query.flatten(1, num_dims)  # [B,S,H,D]
        key_1d = key.flatten(1, num_dims)  # [B,S,H_KV,D]
        value_1d = value.flatten(1, num_dims)  # [B,S,H_KV,D_V]
        is_causal_1d = is_causal[0]
        output_shape = [x for x in query.shape[:-1]] + [value.shape[-1]]

        output_1d, lse_1d = attention(
            query_1d,
            key_1d,
            value_1d,
            scale=scale,
            is_causal=is_causal_1d,
            causal_type=CausalType.DontCare,
            return_lse=True,
            deterministic=deterministic,
        )
        output = output_1d.reshape(*output_shape)  # [B,*token_layout_shape,H,D_V]
        lse = lse_1d.reshape(*output_shape[:-1])  # [B,*token_layout_shape,H]
        if return_lse:
            return output, lse
        return output

    scale = scale if scale is not None else query.shape[-1] ** -0.5

    if backend is None and backend_kwargs is not None:
        backend_kwargs = None
        log.debug("A backend was not specified, but got backend_kwargs. Ignoring... ")

    backend = choose_multi_dim_backend(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        device=query.device,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        deterministic=deterministic,
        backend=backend,
    )

    if backend not in MULTI_DIM_BACKEND_MAP:
        raise ValueError(f"Selected {backend=}, but available choices are {MULTI_DIM_BACKEND_MAP.keys()}. ")

    return MULTI_DIM_BACKEND_MAP[backend](
        query=query,
        key=key,
        value=value,
        window_size=window_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        return_lse=return_lse,
        backend_kwargs=backend_kwargs,
        deterministic=deterministic,
    )


def multi_dimensional_attention_varlen(
    query: Tensor,  # [1,S_total,H,D]
    key: Tensor,  # [1,S_total,H_KV,D]
    value: Tensor,  # [1,S_total,H_KV,D_V]
    metadata: dict,
    scale: float | None = None,
    backend: str | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:  # [1,S_total,H,D_V] or ([1,S_total,H,D_V], [1,S_total,H])
    """
    Runs Variable-Length Multi-Dimensional Attention on sequence-packed QKV tensors.

    This operation performs sparse/multi-dimensional attention on variable-length sequences
    where tokens from different samples with different spatial layouts are concatenated
    along the sequence dimension. Each sample can have its own spatial dimensions
    (e.g., different height/width for 2D layouts).

    The metadata should be pre-computed using `configure_varlen_metadata` and reused
    across forward/backward passes for efficiency.

    **Requires NATTEN >= 0.21.9.dev0 and either Hopper or Blackwell DC-class architecture**

    Parameters:
        query (Tensor): 4-D query tensor with sequence-packed layout
            (`[1, seqlen_total, heads, head_dim]`)

        key (Tensor): 4-D key tensor with sequence-packed layout
            (`[1, seqlen_total, heads_kv, head_dim]`)

        value (Tensor): 4-D value tensor with sequence-packed layout
            (`[1, seqlen_total, heads_kv, head_dim_v]`)

        metadata (dict): Pre-computed varlen metadata from `imaginaire.varlen.generate_multi_dim_varlen_parameters`.

        scale (float | None): Attention scale. Defaults to head_dim ** -0.5.

    Other Parameters:
        backend (str | None): Backend to run with. If unspecified (default), it will try to
            select the best available.

        return_lse (bool): Whether to return logsumexp values. Default is False.

        backend_kwargs (dict | None): Backend-specific arguments.

        deterministic (bool): Whether to enforce deterministic backward pass. Default is False.
            Not supported for this operation (Hopper and Blackwell FNA backends do not support determinism).

    Returns:
        output (Tensor): 4-D output tensor with sequence-packed layout
            (`[1, seqlen_total, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor (`[1, seqlen_total, heads]`).
            Only returned when return_lse is True.
    """
    # For now, NATTEN is the only backend that supports varlen multi-dimensional attention
    from cosmos.model.attention.natten import natten_supported

    if not natten_supported():
        raise RuntimeError("merge_attentions requires NATTEN. Please upgrade NATTEN to use attention merging.")

    if backend is not None and backend != "natten":
        raise ValueError(
            f"multi_dimensional_attention_varlen currently only supports 'natten' backend, got {backend=}."
        )

    # Import NATTEN's varlen function
    from cosmos.model.attention.natten.functions import natten_multi_dim_attention_varlen

    return natten_multi_dim_attention_varlen(
        query=query,
        key=key,
        value=value,
        metadata=metadata,
        scale=scale,
        return_lse=return_lse,
        backend_kwargs=backend_kwargs,
        deterministic=deterministic,
    )


def spatio_temporal_attention(
    query: Tensor,  # [B,T,H_spatial,W_spatial,H,D]
    key: Tensor,  # [B,T,H_spatial,W_spatial,H_KV,D]
    value: Tensor,  # [B,T,H_spatial,W_spatial,H_KV,D_V]
    window_size: tuple | int = -1,
    stride: tuple | int = 1,
    dilation: tuple | int = 1,
    scale: float | None = None,
    # backend & misc parameters
    backend: str | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> (
    Tensor | tuple[Tensor, Tensor]
):  # [B,T,H_spatial,W_spatial,H,D_V] or ([B,T,H_spatial,W_spatial,H,D_V], [B,T,H_spatial,W_spatial,H])
    """
    Runs Spatio-Temporal Attention on unflattened QKV with the heads-last contiguous layout
    (`[batch, T, H, W, heads, head_dim]`).
    For now, it is always expected that Q, K, and V match in their shapes.

    Parameters:
        query (Tensor): 6-D query tensor, with the heads-last contiguous layout
            (`[batch, T, H, W, heads, head_dim]`)

        key (Tensor): 6-D key tensor, with the heads-last contiguous layout
            (`[batch, T, H, W, heads_kv, head_dim]`)

        value (Tensor): 6-D value tensor, with heads-last contiguous layout
            (`[batch, T, H, W, heads_kv, head_dim_v]`)

        window_size (tuple | int): Attention window (kernel) size / shape. If an
            integer, it will be repeated for all dimensions. For example `window_size=3` is
            interpreted as `window_size=(3, 3, 3)`.
            `-1`s are replaced with the corresponding value in `(T, H, W)`.
            Default is -1 (no sparsity).

        stride (tuple | int): Sliding window step size/shape. If an integer, it will be repeated
            for all dimensions. For example `stride=2` is interpreted as `stride=(2, 2, 2)`.
            Final stride must satisfy `1 <= stride <= window_size`.
            Default is 1.

        dilation (tuple | int): Dilation step size/shape. If an integer, it will be repeated for
            all dimensions. For example `dilation=4` is interpreted as `dilation=(4, 4, 4)`.
            Final dilation must satisfy `2 <= dilation * window_size <= (T, H, W)`.
            Default is 1.

        scale (float | None): Dot product scale (attention scale). Defaults to head_dim ** -0.5.

    Other Parameters:
        backend (str | None): Backend to run with. If unspecified (default), it will try to
            select the best available.

        return_lse (bool): Whether to return the logsumexp values. Default is False.

        backend_kwargs (dict | None): Key-value pair for passing arguments specific to the backend's
            multi-dim / sparse attention operator, if any. Only valid when a specific backend is
            selected (backend is not None).

        deterministic (bool): Whether to enforce deterministic backward pass. Default is False.
            When True, backends are selected based on deterministic support.

    Returns:
        output (Tensor): 6-D output tensor, with the heads-last contiguous layout
            (`[batch, T, H, W, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor, with the heads-last contiguous layout
            (`[batch, T, H, W, heads]`). Only returned when return_lse is True.
    """
    if query.dim() != 6:
        raise ValueError(
            "Spatio-Temporal Attention requires 6-D input tensors ([batch, T, H, W, heads, head_dim]), "
            f"got {query.shape=})."
        )

    return multi_dimensional_attention(
        query=query,
        key=key,
        value=value,
        window_size=window_size,
        stride=stride,
        dilation=dilation,
        is_causal=(True, False, False),
        scale=scale,
        backend=backend,
        return_lse=return_lse,
        backend_kwargs=backend_kwargs,
        deterministic=deterministic,
    )


def merge_attentions(
    outputs: list[Tensor],  # list of [B,S,H,D_V]
    lse_tensors: list[Tensor],  # list of [B,S,H]
    torch_compile: bool = False,
) -> tuple[Tensor, Tensor]:  # ([B,S,H,D_V], [B,S,H])
    """
    Merges multiple attention outputs that share the same query.

    **NOTE: the user is responsible for ensuring ALL output and LSE tensors have the same data
    pointer as the outputs from the corresponding Attention operations for correct backpropagation!**

    **NOTE: requires NATTEN**

    **NOTE: for backpropagation, only two outputs can be merged for now.**

    Takes multiple attention outputs computed from the same set of query but w.r.t. different
    key/value pairs, and merges them as if all key/value pairs had been concatenated.
    This enables patterns like:
    - Combining local and global attention (e.g., sparse + dense context)
    - Pipelined context parallelism

    The NATTEN backend can be controlled via environment variable filtering.
    See `filter_attention_merge_backends` for details.

    Parameters:
        outputs (list[Tensor]): List of 4-D attention output tensors, with the heads-last layout
            (`[batch, seqlen, heads, head_dim]`). Must contain at least 2 tensors.

        lse_tensors (list[Tensor]): List of 3-D logsumexp tensors, with the heads-last layout
            (`[batch, seqlen, heads]`). Must match length of `outputs`.

        torch_compile (bool): Attempt to use `torch.compile` to fuse the underlying elementwise
            operations. Default is False.

    Returns:
        output (Tensor): Merged attention output tensor (`[batch, seqlen, heads, head_dim]`).

        logsumexp (Tensor): Updated logsumexp tensor (`[batch, seqlen, heads]`).
    """
    # For now, NATTEN is the only backend that provides merge_attentions

    # Check if NATTEN is allowed by environment variable
    allowed_backends = filter_attention_merge_backends(["natten"])

    if "natten" not in allowed_backends:
        raise RuntimeError(
            "merge_attentions requires NATTEN backend, but it has been disabled via environment variable. "
            "NATTEN is currently the only supported backend for merge_attentions."
        )

    from cosmos.model.attention.natten import natten_supported

    if not natten_supported():
        raise RuntimeError("merge_attentions requires NATTEN. Please upgrade NATTEN to use attention merging.")

    # Import and use NATTEN's merge_attentions
    from natten.functional import merge_attentions as natten_merge_attentions

    return natten_merge_attentions(
        outputs=outputs,
        lse_tensors=lse_tensors,
        torch_compile=torch_compile,
        use_autograd_fix=True,  # Always use autograd fix for correct backprop
    )
