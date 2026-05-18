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

cudNN backend checks
"""

from functools import partial

import torch

from cosmos.model.attention.checks import attention_param_checks, attention_tensor_checks
from cosmos.model.attention.cudnn import CUDNN_DISALLOWED, CUDNN_SUPPORTED
from cosmos.model.attention.cudnn.meta import get_bwd_dtypes, get_fwd_dtypes
from cosmos.model.attention.masks import CausalType
from cosmos.model.attention.utils import get_arch_tag, is_torch_compiling, log_or_raise_error


def cudnn_attention_check(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    is_causal: bool,
    causal_type: CausalType,
    is_varlen: bool,
    deterministic: bool = False,
    raise_error: bool = False,
) -> bool:
    """
    Input validation function for the cuDNN backend.
    Runs the common and cuDNN-specific checks. Returns False if any checks fail, otherwise True.

    Parameters:
        query_shape (torch.Size): Shape of 4-D query tensor (`[batch, seqlen, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D key tensor (`[batch, seqlen_kv, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D value tensor (`[batch, seqlen_kv, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        is_causal (bool): whether or not causal masking is enabled.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`. Required when `is_causal = True`.

        is_varlen (bool): whether or not a variable length (varlen) use case. Must be inferred
            beforehand based on arguments such as seqlens_{Q,KV} or cumulative_seqlen_{Q,KV} being
            passed.

        deterministic (bool): Deterministic backward pass required.

        raise_error (bool): whether to raise an error if any checks fail or no backend is selected,
            instead of just returning False. Default is False.

    Returns:
        success (bool): whether use case is compatible with cuDNN backend.

    """
    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    if not CUDNN_SUPPORTED:
        target_fn(
            "cuDNN is not supported in this environment. Run with debug logs to find out why, or choose another backend.",
            exception=RuntimeError,
        )
        return False

    if CUDNN_DISALLOWED:
        target_fn("cuDNN backend is disabled. Please choose another backend.", exception=RuntimeError)
        return False

    if deterministic:
        target_fn("cuDNN Attention does not support deterministic mode.", exception=RuntimeError)
        return False

    if is_torch_compiling():
        target_fn(
            "cuDNN backend does not support torch.compile yet.",
            exception=RuntimeError,
        )
        return False

    arch_tag = get_arch_tag(device)
    fwd_dtypes = get_fwd_dtypes(arch_tag)
    bwd_dtypes = get_bwd_dtypes(arch_tag)
    if not attention_tensor_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        requires_grad=requires_grad,
        supported_dtypes_forward=fwd_dtypes,
        supported_dtypes_backward=bwd_dtypes,
        supports_mla=False,
        supports_gqa_mqa=False,
        raise_error=raise_error,
        backend_name="cuDNN Attention",
    ):
        target_fn("cuDNN does not support the given inputs.", exception=RuntimeError)
        return False

    if is_varlen:
        target_fn("Varlen for cuDNN Attention is not integrated yet.", exception=RuntimeError)
        return False

    # Verifies causal_type is a CausalType instance when is_causal
    # Verifies DontCare is not used unless seqlen_q == seqlen_kv
    attention_param_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        is_causal=is_causal,
        causal_type=causal_type,
    )


    if is_causal and causal_type not in [CausalType.TopLeft, CausalType.DontCare]:
        target_fn("cuDNN Attention only supports top-left causal masking for now.", exception=RuntimeError)
        return False

    return True
