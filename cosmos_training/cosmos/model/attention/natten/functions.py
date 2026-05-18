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

NATTEN Backend: intermediate APIs
Only safe to import when NATTEN_SUPPORTED is True.
"""

from contextlib import nullcontext

from natten.context import set_memory_usage_preference, use_kv_parallelism_in_fused_na
from natten.functional import attention as _natten_attention
from natten.functional import neighborhood_attention_generic as _natten_multi_dim_attention
from torch import Tensor

from cosmos.model.attention.checks import (
    assert_universal_tensor_checks,
    multi_dim_attention_param_checks,
    multi_dim_attention_param_filter,
)
from cosmos.model.attention.masks import CausalType
from cosmos.model.attention.natten import NATTEN_VARLEN_MULTI_DIM_VERSION, natten_version_satisfies
from cosmos.model.attention.natten.checks import (
    choose_natten_backend,
    choose_natten_multi_dim_backend,
    natten_attention_check,
    natten_multi_dim_attention_check,
)
from cosmos.model.attention.utils import torch_deterministic_mode
from cosmos.model.attention.utils.environment import is_torch_compiling

set_memory_usage_preference("unrestricted")
use_kv_parallelism_in_fused_na(True)


def natten_attention(
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
    Runs NATTEN Attention on given operands (Q, K, V) with the heads-last contiguous layout
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

        backend_kwargs (dict | None): Key-value pair for passing arguments specific to NATTEN's
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
    assert natten_attention_check(
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
        if is_varlen and max_seqlen_Q == max_seqlen_KV == 0 and not natten_version_satisfies("0.21.6.dev7"):
            raise NotImplementedError(
                "You're trying to use varlen attention with the NATTEN backend and "
                "an empty batch, which is only supported since version "
                "0.21.6.dev7. Please upgrade NATTEN."
            )

    scale = scale if scale is not None else query.shape[-1] ** -0.5

    backend_kwargs = backend_kwargs.copy() if backend_kwargs is not None else {}

    natten_backend = None
    if "backend" in backend_kwargs:
        natten_backend = backend_kwargs["backend"]
        del backend_kwargs["backend"]
    else:
        natten_backend = choose_natten_backend(
            query_shape=query.shape,
            key_shape=key.shape,
            value_shape=value.shape,
            dtype=query.dtype,
            device=query.device,
            requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
            is_causal=is_causal,
            is_varlen=is_varlen,
            deterministic=deterministic,
            raise_error=True,
        )

    assert natten_backend is not None

    # Override NATTEN's default delta reduction method: using PyTorch
    # is more accurate, but slightly slower.
    # Only affects NATTEN's "cutlass-fmha" backend (Ampere kernels)
    backward_use_pt_reduction = True
    if "backward_use_pt_reduction" in backend_kwargs:
        backward_use_pt_reduction = backend_kwargs["backward_use_pt_reduction"]
        del backend_kwargs["backward_use_pt_reduction"]

    with torch_deterministic_mode() if deterministic else nullcontext():
        return _natten_attention(
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            scale=scale,
            cumulative_seqlen_Q=cumulative_seqlen_Q,
            cumulative_seqlen_KV=cumulative_seqlen_KV,
            max_seqlen_Q=max_seqlen_Q,
            max_seqlen_KV=max_seqlen_KV,
            return_lse=return_lse,
            backend=natten_backend,
            backward_use_pt_reduction=backward_use_pt_reduction,
            **backend_kwargs,
        )


def natten_multi_dim_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    window_size: tuple | int = -1,
    stride: tuple | int = 1,
    dilation: tuple | int = 1,
    is_causal: tuple | bool = False,
    scale: float | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Runs NATTEN's Multi-Dimensional Attention on given operands (Q, K, V) with the heads-last
    contiguous layout (`[batch, *, heads, head_dim]`). Supports up to and including 3 dimensions:
        * 1-D: `[batch, X, heads, head_dim]`, with masking arguments expecting tuples of size 1.
        * 2-D: `[batch, X, Y, heads, head_dim]`, with masking arguments expecting tuples of size 2.
        * 3-D: `[batch, X, Y, Z, heads, head_dim]`, with masking arguments expecting tuples of size 3.

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
            for all dimensions.  For example `stride=2`, when `len(token_layout_shape) == 3`, is
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
        return_lse (bool): Whether to return the logsumexp values. Default is False.

        backend_kwargs (dict | None): Key-value pair for passing arguments specific to NATTEN's
            multi-dim / sparse attention operator, if any.

        deterministic (bool): Deterministic backward pass required.

    Returns:
        output (Tensor): 4-D, 5-D, or 6-D output tensor, with the heads-last contiguous layout
            (`[batch, *token_layout_shape, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor, with the heads-last contiguous layout
            (`[batch, *token_layout_shape, heads, 1]`). Only returned when return_lse is True.
    """

    assert_universal_tensor_checks(query, key, value)
    assert natten_multi_dim_attention_check(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        device=query.device,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        deterministic=deterministic,
        raise_error=True,
    )

    token_layout, window_size, stride, dilation, is_causal = multi_dim_attention_param_filter(
        query,
        window_size=window_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
    )

    multi_dim_attention_param_checks(
        query,
        window_size=window_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
    )

    scale = scale if scale is not None else query.shape[-1] ** -0.5

    backend_kwargs = backend_kwargs.copy() if backend_kwargs is not None else {}

    natten_backend = None
    if "backend" in backend_kwargs:
        natten_backend = backend_kwargs["backend"]
        del backend_kwargs["backend"]
    else:
        natten_backend = choose_natten_multi_dim_backend(
            query_shape=query.shape,
            key_shape=key.shape,
            value_shape=value.shape,
            dtype=query.dtype,
            device=query.device,
            requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
            deterministic=deterministic,
            raise_error=True,
        )

    assert natten_backend is not None

    # Override NATTEN's default delta reduction method: using PyTorch
    # is more accurate, but slightly slower.
    # Only affects NATTEN's "cutlass-fmha" backend (Ampere kernels)
    backward_use_pt_reduction = True
    if "backward_use_pt_reduction" in backend_kwargs:
        backward_use_pt_reduction = backend_kwargs["backward_use_pt_reduction"]
        del backend_kwargs["backward_use_pt_reduction"]

    with torch_deterministic_mode() if deterministic else nullcontext():
        return _natten_multi_dim_attention(
            query=query,
            key=key,
            value=value,
            kernel_size=window_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            scale=scale,
            backend=natten_backend,
            backward_use_pt_reduction=backward_use_pt_reduction,
            return_lse=return_lse,
            **backend_kwargs,
        )


def natten_multi_dim_attention_varlen(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    metadata: dict,
    scale: float | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Runs NATTEN's Variable-Length Multi-Dimensional Attention on given operands (Q, K, V) with
    sequence-packed layout (`[batch=1, seqlen, heads, head_dim]`).

    This operation is used for sparse/multi-dimensional attention on variable-length sequences,
    where tokens from different batches with different spatial layouts are concatenated along
    the sequence dimension.

    **Requires NATTEN >= 0.21.9.dev0**

    Parameters:
        query (Tensor): 4-D query tensor, with the sequence-packed layout
            (`[1, seqlen_total, heads, head_dim]`)

        key (Tensor): 4-D key tensor, with the sequence-packed layout
            (`[1, seqlen_total, heads_kv, head_dim]`)

        value (Tensor): 4-D value tensor, with sequence-packed layout
            (`[1, seqlen_total, heads_kv, head_dim_v]`)

        metadata (dict): Pre-computed varlen metadata from `generate_multi_dim_varlen_parameters`.

        scale (float | None): Dot product scale (attention scale). Defaults to head_dim ** -0.5.

    Other Parameters:
        return_lse (bool): Whether to return the logsumexp values. Default is False.

        backend_kwargs (dict | None): Additional backend-specific arguments.

        deterministic (bool): Deterministic backward pass required.

    Returns:
        output (Tensor): 4-D output tensor, with the sequence-packed layout
            (`[1, seqlen_total, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor, with the sequence-packed layout
            (`[1, seqlen_total, heads]`). Only returned when return_lse is True.
    """
    # Check if NATTEN version supports varlen features
    if not natten_version_satisfies(NATTEN_VARLEN_MULTI_DIM_VERSION):
        raise RuntimeError(
            f"NATTEN's varlen/varsized attention requires NATTEN >= {NATTEN_VARLEN_MULTI_DIM_VERSION}. "
            "Please upgrade NATTEN to use this feature."
        )

    # Import NATTEN's varlen function (only available in NATTEN_VARLEN_MULTI_DIM_VERSION+)
    from natten.varlen import neighborhood_attention_varlen

    backend_kwargs = backend_kwargs.copy() if backend_kwargs is not None else {}

    if deterministic:
        raise ValueError(
            "Deterministic mode is not supported for varlen multi-dimensional attention. "
            "This operation requires Hopper / Blackwell FNA backends which do not support deterministic mode."
        )

    # Parameter mapping: NATTEN uses kernel_size instead of window_size
    outputs = neighborhood_attention_varlen(
        query=query,
        key=key,
        value=value,
        metadata=metadata,
        scale=scale,
        return_lse=return_lse,
        **backend_kwargs,
    )

    return outputs
