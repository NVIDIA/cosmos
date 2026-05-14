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

Flash Attention v3 (flash3) backend checks
"""

from functools import partial

import torch

from cosmos3._src.imaginaire.attention.checks import attention_param_checks, attention_tensor_checks
from cosmos3._src.imaginaire.attention.flash3 import FLASH3_SUPPORTED
from cosmos3._src.imaginaire.attention.flash3.meta import get_bwd_dtypes, get_fwd_dtypes
from cosmos3._src.imaginaire.attention.masks import CausalType
from cosmos3._src.imaginaire.attention.utils import get_arch_tag, log_or_raise_error


def flash3_attention_check(
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
    Input validation function for the flash3 backend.

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
        success (bool): whether use case is compatible with flash3 backend.

    """
    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    if not FLASH3_SUPPORTED:
        target_fn(
            "Flash Attention v3 (flash3) is not supported in this environment. Run with debug logs to find out why, or choose another backend.",
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
        # flash3 supports MLA, unlike flash2, but with some constraints
        # disabled for now due to API bug
        supports_mla=False,
        supports_gqa_mqa=True,
        raise_error=raise_error,
        backend_name="Flash Attention v3 (flash3)",
    ):
        target_fn("Flash Attention v3 (flash3) does not support the given inputs.", exception=RuntimeError)
        return False

    # MLA constraints
    if query_shape[-1] != value_shape[-1]:
        head_dim_q = query_shape[-1]
        head_dim_v = value_shape[-1]
        if not ((head_dim_q <= 64 and head_dim_v <= 512) or (128 <= head_dim_q <= 192 and 96 <= head_dim_v <= 128)):
            target_fn(
                "Flash Attention v3 (flash3) does not support this head dim combination. "
                "Expected either head_dim_qk <= 64 and head_dim_v <= 512, or 128 <= head_dim_qk <= 192 "
                f"and 96 <= head_dim_v <= 128, got {head_dim_q=}, {head_dim_v=}.",
                exception=ValueError,
            )
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

    if is_causal and causal_type not in [CausalType.BottomRight, CausalType.DontCare]:
        target_fn("Flash Attention v3 only supports bottom-right causal masking.", exception=ValueError)
        return False

    return True
