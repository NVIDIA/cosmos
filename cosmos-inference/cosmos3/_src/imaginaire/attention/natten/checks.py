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

NATTEN backend checks
"""

from functools import partial

import torch

from cosmos3._src.imaginaire.attention.checks import (
    attention_param_checks,
    attention_tensor_checks,
    multi_dim_attention_tensor_checks,
)
from cosmos3._src.imaginaire.attention.masks import CausalType
from cosmos3._src.imaginaire.attention.natten import (
    NATTEN_BLACKWELL_DETERMINISTIC_VERSION,
    NATTEN_BLACKWELL_PARTIAL_HEAD_DIM_VERSION,
    NATTEN_HOPPER_CAUSAL_VARLEN_VERSION,
    NATTEN_SUPPORTED,
    get_natten_version,
    natten_version_satisfies,
)
from cosmos3._src.imaginaire.attention.natten.meta import get_bwd_dtypes, get_fwd_dtypes
from cosmos3._src.imaginaire.attention.utils import get_arch_tag, is_fp8, log_or_raise_error
from cosmos3._src.imaginaire.attention.utils.safe_ops import log
from cosmos3._src.imaginaire.attention.utils.safe_ops.functools import lru_cache


def dtype_supported(
    dtype: torch.dtype, requires_grad: bool, dtypes_fwd: list[torch.dtype], dtypes_bwd: list[torch.dtype] | None = None
) -> bool:
    """
    Helper determining whether dtype is supported with different sets of supported dtypes for
    training and inference (forward+backward and forward).

    Parameters:
        dtype (torch.dtype): tensor element type.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        dtypes_fwd (list[torch.dtype]): list of dtypes allowed for inference only (when not
            tensor.requires_grad).

        dtypes_bwd (list[torch.dtype] | None): Optional list of dtypes allowed for training only
            (when tensor.requires_grad), if different from dtypes_fwd.

    """
    if requires_grad and dtypes_bwd is not None:
        return dtype in dtypes_bwd
    return dtype in dtypes_fwd


@lru_cache
def choose_natten_backend(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    is_causal: bool,
    is_varlen: bool,
    deterministic: bool = False,
    requires_fna: bool = False,
    raise_error: bool = False,
) -> str | None:
    """
    Chooses an FMHA backend in NATTEN (cutlass-fmha, hopper-fmha, blackwell-fmha) for the current
    use case based on features needed and current GPU architecture.

    Using tensor shapes, it infers whether MLA (head_dim_value != head_dim_qk) or
    GQA/MQA (heads_kv != heads_q) are required.
    Using device, it infers GPU architecture and compatible backends.
    Using arguments is_causal and is_varlen, and other inferred features, it picks the best
    available backend.

    It is possible for no backend to be selected, if the combination of features is not available in
    any one of the NATTEN backends, in which case it will return None.

    Parameters:
        query_shape (torch.Size): Shape of 4-D query tensor (`[batch, seqlen, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D key tensor (`[batch, seqlen_kv, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D value tensor (`[batch, seqlen_kv, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        is_causal (bool): whether or not causal masking is enabled.

        is_varlen (bool): whether or not a variable length (varlen) use case. Must be inferred
            beforehand based on arguments such as seqlens_{Q,KV} or cumulative_seqlen_{Q,KV} being
            passed.

        deterministic (bool): Deterministic backward pass required.

        requires_fna (bool): Whether the selection is for FNA kernels (sometimes they have different
            feature coverage compared to their FMHA counterparts.)

        raise_error (bool): whether to raise an error if no backend is selected, instead of just
            returning None. Default is False.

    Returns:
        backend (str | None): selected NATTEN backend, if any compatible.

    """
    natten_version = get_natten_version()

    target_fn = partial(log_or_raise_error, raise_error=raise_error)


    arch_tag = get_arch_tag(device)

    is_mla = query_shape[-1] != value_shape[-1]
    head_dim = max(query_shape[-1], value_shape[-1])

    # banning devices not supported since CUDA 13.0 for simplicity
    if arch_tag < 75:
        log.debug("NATTEN is not supported because compute capability is below the minimum (7.5).")
        return None

    # blackwell-fmha: sm100 and sm103 only.
    # limitations: no mla (TBD).
    blackwell_fmha_fwd_dtypes = [torch.float16, torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
    blackwell_fmha_bwd_dtypes = [torch.float16, torch.bfloat16]
    blackwell_dtype_supported = dtype_supported(
        dtype=dtype,
        requires_grad=requires_grad,
        dtypes_fwd=blackwell_fmha_fwd_dtypes,
        dtypes_bwd=blackwell_fmha_bwd_dtypes,
    )

    blackwell_deterministic_supported = natten_version_satisfies(NATTEN_BLACKWELL_DETERMINISTIC_VERSION)
    blackwell_deterministic_blocked = deterministic and (requires_fna or not blackwell_deterministic_supported)

    blackwell_partial_head_dim_support = natten_version_satisfies(NATTEN_BLACKWELL_PARTIAL_HEAD_DIM_VERSION)
    blackwell_head_dim_alignment_constraint = 16 if is_fp8(dtype) else 8
    blackwell_head_dim_in_range = 0 < head_dim and head_dim <= 128
    blackwell_head_dim_alignment_met = head_dim % blackwell_head_dim_alignment_constraint == 0
    blackwell_head_dim_supported = head_dim in [32, 64, 128] or (
        blackwell_partial_head_dim_support and blackwell_head_dim_in_range and blackwell_head_dim_alignment_met
    )

    if (
        arch_tag in [100, 103]
        and not is_mla
        and blackwell_dtype_supported
        and blackwell_head_dim_supported
        and not blackwell_deterministic_blocked
    ):
        return "blackwell-fmha"
    else:
        reason = ""
        if blackwell_deterministic_blocked:
            reason += "Deterministic mode requested but not supported. "
        if arch_tag not in [100, 103]:
            reason += f"Incompatible architecture ({arch_tag}, expected 100 or 103). "
        if is_mla:
            reason += "Use case is MLA (head_dim_qk != head_dim_value). "
        if not blackwell_dtype_supported:
            if requires_grad:
                reason += (
                    f"Data type {dtype} is not in list of supported dtypes for training: {blackwell_fmha_bwd_dtypes}. "
                )
            else:
                reason += (
                    f"Data type {dtype} is not in list of supported dtypes for inference: {blackwell_fmha_fwd_dtypes}. "
                )
        if not blackwell_head_dim_supported:
            reason += f"{head_dim=} is not supported with {dtype=} (natten {natten_version})"
        log.debug(f"NATTEN backend blackwell-fmha is not compatible. Reason: {reason}")

    # hopper-fmha: sm90 only.
    # limitations: no mla.
    # varlen and causal masking support was added in NATTEN_HOPPER_CAUSAL_VARLEN_VERSION
    hopper_fmha_dtypes = [torch.float16, torch.bfloat16]
    dtype_supported_hopper = dtype_supported(dtype=dtype, requires_grad=requires_grad, dtypes_fwd=hopper_fmha_dtypes)
    head_dim_supported_hopper = (head_dim in [32, 64, 128, 256] and not requires_grad) or head_dim in [32, 64, 128]
    hopper_varlen_causal_supported = natten_version_satisfies(NATTEN_HOPPER_CAUSAL_VARLEN_VERSION)
    hopper_varlen_causal_check = hopper_varlen_causal_supported or (not is_varlen and not is_causal)
    if (
        arch_tag == 90
        and hopper_varlen_causal_check
        and not is_mla
        and dtype_supported_hopper
        and head_dim_supported_hopper
        and not deterministic
    ):
        return "hopper-fmha"
    else:
        reason = ""
        if deterministic:
            reason += "Deterministic mode requested but hopper-fmha does not support it. "
        if arch_tag != 90:
            reason += f"Incompatible architecture ({arch_tag}, expected 90). "
        if is_causal and not hopper_varlen_causal_supported:
            reason += (
                "Use case is causal, which is only supported since natten "
                + f"{NATTEN_HOPPER_CAUSAL_VARLEN_VERSION}, detected version: {natten_version}. "
            )
        if is_varlen and not hopper_varlen_causal_supported:
            reason += (
                "Use case is varlen, which is only supported since natten "
                + f"{NATTEN_HOPPER_CAUSAL_VARLEN_VERSION}, detected version: {natten_version}. "
            )
        if is_mla:
            reason += "Use case is MLA (head_dim_qk != head_dim_value). "
        if not dtype_supported_hopper:
            reason += f"Data type {dtype} is not in list of supported dtypes: {hopper_fmha_dtypes}. "
        if not head_dim_supported_hopper:
            reason += f"{head_dim=} with {requires_grad=} is not supported. "
        log.debug(f"NATTEN backend hopper-fmha is not compatible. Reason: {reason}")

    # cutlass-fmha: targets sm50, sm70, sm75, sm80 (supports sm80+)
    # limitations: none.
    cutlass_fmha_dtypes = [torch.float32, torch.float16, torch.bfloat16]
    dtype_supported_cutlass = dtype_supported(dtype=dtype, requires_grad=requires_grad, dtypes_fwd=cutlass_fmha_dtypes)
    head_dim_supported_cutlass = head_dim % 8 == 0
    if dtype_supported_cutlass and head_dim_supported_cutlass:
        return "cutlass-fmha"
    else:
        reason = ""
        if not dtype_supported_cutlass:
            reason += f"Data type {dtype} is not in list of supported dtypes: {cutlass_fmha_dtypes}. "
        if not head_dim_supported_cutlass:
            reason += f"{head_dim=} is not supported. "
        log.debug(f"NATTEN backend cutlass-fmha is not compatible. Reason: {reason}")

    target_fn(
        f"Could not find a compatible NATTEN FMHA backend for {arch_tag=}, {is_causal=}, {is_varlen=}, {is_mla=}.",
        exception=RuntimeError,
    )
    return None


def natten_attention_check(
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
    Input validation function for the NATTEN backend.
    Runs the common checks in addition to trying to find a compatible NATTEN backend. If any checks
    fail, or no compatible backend is found in NATTEN, returns False.

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
        success (bool): whether use case is compatible with NATTEN backend.

    """
    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    if not NATTEN_SUPPORTED:
        target_fn(
            "NATTEN is not supported in this environment. Run with debug logs to find out why, or choose another backend.",
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
        supports_mla=True,
        supports_gqa_mqa=True,
        raise_error=raise_error,
        backend_name="NATTEN Attention",
    ):
        target_fn("NATTEN does not support the given inputs.", exception=RuntimeError)
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
        target_fn("NATTEN Attention only supports top-left causal masking for now.", exception=RuntimeError)
        return False

    natten_backend = choose_natten_backend(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        is_causal=is_causal,
        is_varlen=is_varlen,
        deterministic=deterministic,
        raise_error=raise_error,
    )

    if natten_backend is None:
        return False

    return True


@lru_cache
def choose_natten_multi_dim_backend(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    deterministic: bool = False,
    raise_error: bool = False,
) -> str | None:
    """
    Chooses an FNA backend in NATTEN (cutlass-fna, hopper-fna, blackwell-fna) for the current
    use case based on features needed and current GPU architecture.

    Using tensor shapes, it infers whether MLA (head_dim_value != head_dim_qk) or
    GQA/MQA (heads_kv != heads_q) are required.
    Using device, it infers GPU architecture and compatible backends.
    Using arguments is_causal and is_varlen, and other inferred features, it picks the best
    available backend.

    It is possible for no backend to be selected, if the combination of features is not available in
    any one of the NATTEN backends, in which case it will return None.

    Parameters:
        query_shape (torch.Size): Shape of 4-D, 5-D, or 6-D query tensor (`[batch, *token_layout_shape, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D, 5-D, or 6-D key tensor (`[batch, *token_layout_shape, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D, 5-D, or 6-D value tensor (`[batch, *token_layout_shape, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        deterministic (bool): Deterministic backward pass required.

        raise_error (bool): whether to raise an error if no backend is selected, instead of just
            returning None. Default is False.

    Returns:
        backend (str | None): selected NATTEN backend, if any compatible.

    """

    # Reuse choose_natten_backend instead of duplicating code
    # NATTEN specifically makes sure the FNA counterparts cover all the features the FMHA kernels
    # do.
    fmha_backend = choose_natten_backend(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        is_causal=False,  # causal masking in supported across all multi-dim (FNA) backends
        is_varlen=False,  # varlen is undefined (so far) for multi-dim
        deterministic=deterministic,
        requires_fna=True,
        raise_error=raise_error,
    )

    natten_fmha_backend_to_fna_backend = {
        "cutlass-fmha": "cutlass-fna",
        "hopper-fmha": "hopper-fna",
        "blackwell-fmha": "blackwell-fna",
    }

    assert fmha_backend in natten_fmha_backend_to_fna_backend
    return natten_fmha_backend_to_fna_backend[fmha_backend]


def natten_multi_dim_attention_check(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    deterministic: bool = False,
    raise_error: bool = False,
) -> bool:
    """
    Input validation function for the NATTEN multi-dimensional backend.
    Runs the common checks in addition to trying to find a compatible NATTEN backend. If any checks
    fail, or no compatible backend is found in NATTEN, returns False.

    Parameters:
        query_shape (torch.Size): Shape of 4-D, 5-D, or 6-D query tensor (`[batch, *token_layout_shape, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D, 5-D, or 6-D key tensor (`[batch, *token_layout_shape, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D, 5-D, or 6-D value tensor (`[batch, *token_layout_shape, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        deterministic (bool): Deterministic backward pass required.

        raise_error (bool): whether to raise an error if any checks fail or no backend is selected,
            instead of just returning False. Default is False.

    Returns:
        success (bool): whether use case is compatible with NATTEN backend.

    """
    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    if not NATTEN_SUPPORTED:
        target_fn(
            "NATTEN is not supported in this environment. Run with debug logs to find out why, or choose another backend.",
            exception=RuntimeError,
        )
        return False

    arch_tag = get_arch_tag(device)
    fwd_dtypes = get_fwd_dtypes(arch_tag)
    bwd_dtypes = get_bwd_dtypes(arch_tag)
    if not multi_dim_attention_tensor_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        requires_grad=requires_grad,
        supported_dtypes_forward=fwd_dtypes,
        supported_dtypes_backward=bwd_dtypes,
        supports_mla=True,
        supports_gqa_mqa=True,
        raise_error=raise_error,
        backend_name="NATTEN Multi-Dimensional Attention",
    ):
        target_fn("NATTEN does not support the given inputs.", exception=RuntimeError)
        return False

    natten_backend = choose_natten_multi_dim_backend(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        deterministic=deterministic,
        raise_error=raise_error,
    )

    if natten_backend is None:
        return False

    return True
