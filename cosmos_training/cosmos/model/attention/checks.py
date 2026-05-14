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

Common, op-specific, and backend-specific checks
"""

from collections.abc import Sequence
from functools import partial
from typing import Any

import torch
from torch import Tensor

from cosmos.model.attention.masks import CausalType
from cosmos.model.attention.utils import log_or_raise_error
from cosmos.model.attention.utils.environment import is_torch_compiling
from cosmos.model.attention.varlen import generate_varlen_parameters


def universal_tensor_checks(
    query: Tensor, key: Tensor, value: Tensor, raise_error: bool = True
) -> bool:  # query/key/value: [B,*,H,D]
    """
    Universal tensor validation: checks sparse/nested tensors and ensures device/dtype consistency.
    This should be called by users before extracting tensor properties for tensorless APIs.

    Parameters:
        query (Tensor): Query tensor.
        key (Tensor): Key tensor.
        value (Tensor): Value tensor.
        raise_error (bool): Whether to raise an error if checks fail. Default is True.

    Returns:
        success (bool): Whether all checks pass.
    """
    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    if query.is_sparse or key.is_sparse or value.is_sparse:
        target_fn("This operation does not support sparse tensors.", exception=NotImplementedError)
        return False

    if query.is_nested or key.is_nested or value.is_nested:
        target_fn("This operation does not support nested tensors.", exception=NotImplementedError)
        return False

    if query.device != key.device or query.device != value.device:
        target_fn(
            f"Query, key, and value must be on the same device, got {query.device=}, {key.device=}, {value.device=}.",
            exception=ValueError,
        )
        return False

    if query.dtype != key.dtype or query.dtype != value.dtype:
        target_fn(
            f"Query, key, and value must assume the same data type, got {query.dtype=}, {key.dtype=}, {value.dtype=}.",
            exception=ValueError,
        )
        return False

    return True


def assert_universal_tensor_checks(query: Tensor, key: Tensor, value: Tensor) -> None:  # query/key/value: [B,*,H,D]
    """
    Universal tensor validation using assertions for backend functions.
    Checks sparse/nested tensors and ensures device/dtype/requires_grad consistency.

    This is intended for internal backend use only. Users should not call backend functions directly.
    Assertions are disabled in production (-O flag), so this is appropriate for post-frontend checks.

    Parameters:
        query (Tensor): Query tensor.
        key (Tensor): Key tensor.
        value (Tensor): Value tensor.
    """
    assert not query.is_sparse and not key.is_sparse and not value.is_sparse, "Sparse tensors not supported"
    assert not query.is_nested and not key.is_nested and not value.is_nested, "Nested tensors not supported"
    assert query.device == key.device == value.device, (
        f"Device mismatch: {query.device=}, {key.device=}, {value.device=}"
    )
    assert query.dtype == key.dtype == value.dtype, f"Dtype mismatch: {query.dtype=}, {key.dtype=}, {value.dtype=}"
    # Disabled: requires_grad may differ if differentiable queries attend to non-differentiable
    # keys, e.g. when attending to a KV-cache during training.
    # assert query.requires_grad == key.requires_grad == value.requires_grad, (
    #     f"requires_grad mismatch: {query.requires_grad=}, {key.requires_grad=}, {value.requires_grad=}"
    # )


def _universal_attention_checks(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    requires_grad: bool,
    supported_dtypes_forward: list[torch.dtype] | None = None,
    supported_dtypes_backward: list[torch.dtype] | None = None,
    supports_mla: bool = True,
    supports_gqa_mqa: bool = True,
    raise_error: bool = True,
    backend_name: str | None = None,
) -> bool:
    backend_name = backend_name or "Attention"

    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    query_dim = len(query_shape)
    key_dim = len(key_shape)
    value_dim = len(value_shape)

    if query_dim != key_dim or query_dim != value_dim:
        target_fn(
            f"Q, K, and V must have the same rank, got {query_dim=}, {key_dim=}, {value_dim=}.",
            exception=ValueError,
        )
        return False

    if query_shape[0] != key_shape[0] or query_shape[0] != value_shape[0]:
        target_fn(
            f"Q, K, and V must match in batch size, got {query_shape[0]=}, {key_shape[0]=}, {value_shape[0]=}.",
            exception=ValueError,
        )
        return False

    if query_shape[-1] != key_shape[-1]:
        target_fn(
            f"Q and K head dims must match, got {query_shape[-1]=}, {key_shape[-1]=}.",
            exception=ValueError,
        )
        return False

    if key_shape[-2] != value_shape[-2]:
        target_fn(
            f"K and V must always have the same number of heads, got {key_shape[-2]=}, {value_shape[-2]=}.",
            exception=ValueError,
        )
        return False

    if not supports_mla and query_shape[-1] != value_shape[-1]:
        target_fn(
            f"{backend_name} does not support different head dims for QK and V, got "
            f"{query_shape[-1]=}, {value_shape[-1]=}.",
            exception=ValueError,
        )
        return False

    if not supports_gqa_mqa and (query_shape[-2] != key_shape[-2] or query_shape[-2] != value_shape[-2]):
        target_fn(
            f"{backend_name} does not support GQA/MQA, therefore number of heads in Q, K, and V "
            f"must match, got {query_shape[-2]=}, {key_shape[-2]=}, {value_shape[-2]=}.",
            exception=ValueError,
        )
        return False

    if supports_gqa_mqa:
        heads_q = query_shape[-2]
        heads_kv = key_shape[-2]

        if heads_q < heads_kv or heads_q % heads_kv != 0:
            target_fn(
                f"KV heads must evenly divide Q heads, got {heads_q=}, {heads_kv=}.",
                exception=ValueError,
            )
            return False

    # Caller must ensure dtype consistency via universal_tensor_checks
    if supported_dtypes_forward is not None and dtype not in supported_dtypes_forward:
        target_fn(
            f"{backend_name} does not support forward pass (inference) with data type {dtype}; "
            f"supported dtypes: {supported_dtypes_forward}.",
            exception=ValueError,
        )
        return False

    if supported_dtypes_backward is not None and requires_grad and dtype not in supported_dtypes_backward:
        target_fn(
            f"{backend_name} does not support backward pass (training) with data type {dtype}; "
            f"supported dtypes: {supported_dtypes_backward}.",
            exception=ValueError,
        )
        return False

    return True


def attention_tensor_checks(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    requires_grad: bool,
    supported_dtypes_forward: list[torch.dtype] | None = None,
    supported_dtypes_backward: list[torch.dtype] | None = None,
    supports_mla: bool = True,
    supports_gqa_mqa: bool = True,
    raise_error: bool = True,
    backend_name: str | None = None,
) -> bool:
    backend_name = backend_name or "Attention"

    if not _universal_attention_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        requires_grad=requires_grad,
        supported_dtypes_forward=supported_dtypes_forward,
        supported_dtypes_backward=supported_dtypes_backward,
        supports_mla=supports_mla,
        supports_gqa_mqa=supports_gqa_mqa,
        raise_error=raise_error,
        backend_name=backend_name,
    ):
        return False

    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    query_dim = len(query_shape)
    if query_dim != 4:
        target_fn(
            f"Attention expects 4-D tensors as inputs, got {query_dim=}.",
            exception=ValueError,
        )
        return False

    if key_shape[1] != value_shape[1]:
        target_fn(
            f"K and V must match in sequence length, got {key_shape[1]=}, {value_shape[1]=}.",
            exception=ValueError,
        )
        return False

    return True


def varlen_tensor_checks(
    query: Tensor,  # [1,S_total_Q,H,D]
    key: Tensor,  # [1,S_total_KV,H_KV,D]
    value: Tensor,  # [1,S_total_KV,H_KV,D_V]
    seqlens_Q: Tensor | None = None,  # [B]
    seqlens_KV: Tensor | None = None,  # [B]
    cumulative_seqlen_Q: Tensor | None = None,  # [B+1]
    cumulative_seqlen_KV: Tensor | None = None,  # [B+1]
    max_seqlen_Q: int | None = None,
    max_seqlen_KV: int | None = None,
) -> (
    tuple[None, None, int, int] | tuple[Tensor, Tensor, int, int]
):  # (cumseqlen_Q[B+1], cumseqlen_KV[B+1], max_seqlen_Q, max_seqlen_KV)
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError(
            f"Q, K, and V must match in batch size, got {query.shape[0]=}, {key.shape[0]=}, {value.shape[0]=}."
        )


    if not is_torch_compiling():
        # Validate max_seqlen values: neither can be negative, and they must be
        # both zero/None (not varlen) or both positive (varlen).
        if (max_seqlen_Q is not None and max_seqlen_Q < 0) or (max_seqlen_KV is not None and max_seqlen_KV < 0):
            raise ValueError(
                f"max_seqlen_Q and max_seqlen_KV cannot be negative, got {max_seqlen_Q=}, {max_seqlen_KV=}."
            )

        if (max_seqlen_Q == 0) != (max_seqlen_KV == 0):
            raise ValueError(
                "max_seqlen_Q and max_seqlen_KV must either both be 0/None (not varlen) or both be positive "
                f"(varlen), got {max_seqlen_Q=}, {max_seqlen_KV=}."
            )

    if all(
        x is None
        for x in [
            seqlens_Q,
            seqlens_KV,
            cumulative_seqlen_Q,
            cumulative_seqlen_KV,
        ]
    ) and all(
        x is None or x == 0
        for x in [
            max_seqlen_Q,
            max_seqlen_KV,
        ]
    ):
        # Not varlen
        return None, None, 0, 0

    if seqlens_Q is not None or seqlens_KV is not None:
        # Generate cumulative_seqlen_{Q,KV}, max_seqlen_{Q,KV}, total_seqlen_{Q,KV}
        # based on user input
        return generate_varlen_parameters(
            query=query,
            key=key,
            value=value,
            seqlens_Q=seqlens_Q,
            seqlens_KV=seqlens_KV,
        )

    # Validate user-input cumulative_seqlen_{Q,KV}, max_seqlen_{Q,KV}, total_seqlen_{Q,KV}

    # Mismatch (one 0, the other positive) is already caught by the early check above.
    # This feature may require support in the backends themselves; see NATTEN PR:
    # https://github.com/SHI-Labs/NATTEN/pull/327
    if any(
        x is None
        for x in [
            cumulative_seqlen_Q,
            cumulative_seqlen_KV,
            max_seqlen_Q,
            max_seqlen_KV,
        ]
    ):
        raise ValueError(
            "Variable length Attention requires all of cumulative_seqlen_{Q,KV} and max_seqlen_{Q,KV} to be set."
        )

    if query.shape[0] != 1:
        raise ValueError(
            f"Variable length Attention only supports sequence-packed memory layout (batch = 1), got {query.shape[0]=}."
        )

    assert cumulative_seqlen_Q is not None
    assert cumulative_seqlen_KV is not None
    assert max_seqlen_Q is not None
    assert max_seqlen_KV is not None

    if not isinstance(max_seqlen_Q, int) or not isinstance(max_seqlen_KV, int):
        raise ValueError(
            f"max_seqlen_Q and max_seqlen_KV must be ints, got {type(max_seqlen_Q)=}, {type(max_seqlen_KV)=}."
        )

    total_seqlen_Q = query.shape[1]
    total_seqlen_KV = key.shape[1]


    if not is_torch_compiling():
        # When both max_seqlens are 0, skip bounds checks (skip kernel / empty-batch case).
        # Mismatch is already caught by the early check, so at this point either both are 0 or both are positive.
        if max_seqlen_Q > 0 or max_seqlen_KV > 0:
            if max_seqlen_Q > total_seqlen_Q:
                raise ValueError(
                    f"Maximum sequence length cannot exceed total, got {max_seqlen_Q=}, {total_seqlen_Q=}."
                )

            if max_seqlen_KV > total_seqlen_KV:
                raise ValueError(
                    f"Maximum sequence length cannot exceed total, got {max_seqlen_KV=}, {total_seqlen_KV=}."
                )

            if max_seqlen_Q < 1 or max_seqlen_KV < 1:
                raise ValueError(
                    f"Maximum sequence length cannot be less than 1, got {max_seqlen_Q=}, {max_seqlen_KV=}."
                )

    if not isinstance(cumulative_seqlen_Q, Tensor) or not isinstance(cumulative_seqlen_KV, Tensor):
        raise ValueError("cumulative_seqlen_Q and cumulative_seqlen_KV must both be tensors.")

    if cumulative_seqlen_Q.device != query.device or cumulative_seqlen_KV.device != query.device:
        raise ValueError(
            "cumulative_seqlen_Q and cumulative_seqlen_KV must be on the same device as QKV, but "
            f"{cumulative_seqlen_Q.device=}, {cumulative_seqlen_KV.device=}, {query.device=}."
        )

    if cumulative_seqlen_Q.dtype != torch.int32 or cumulative_seqlen_KV.dtype != torch.int32:
        raise ValueError(
            "cumulative_seqlen_Q and cumulative_seqlen_KV must both be torch.int32 tensors, got "
            f"{cumulative_seqlen_Q.dtype=}, {cumulative_seqlen_KV.dtype=}."
        )

    if cumulative_seqlen_Q.dim() != 1 or cumulative_seqlen_KV.dim() != 1:
        raise ValueError(
            "cumulative_seqlen_Q and cumulative_seqlen_KV must both be 1-D tensors, got "
            f"{cumulative_seqlen_Q.dim()=}, {cumulative_seqlen_KV.dim()=}."
        )

    if cumulative_seqlen_Q.shape[0] != cumulative_seqlen_KV.shape[0]:
        raise ValueError(
            "cumulative_seqlen_Q and cumulative_seqlen_KV must match in size, got "
            f"{cumulative_seqlen_Q.shape=}, {cumulative_seqlen_KV.shape=}."
        )

    if cumulative_seqlen_Q.shape[0] < 2:
        raise ValueError(
            "cumulative_seqlen_Q and cumulative_seqlen_KV must contain at least 2 elements, got "
            f"{cumulative_seqlen_Q.shape=}, {cumulative_seqlen_KV.shape=}."
        )

    return (
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        max_seqlen_Q,
        max_seqlen_KV,
    )


def attention_param_checks(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    is_causal: bool,
    causal_type: CausalType,
):
    if is_causal and (causal_type is None or not isinstance(causal_type, CausalType)):
        raise ValueError(
            f"Argument causal_type must be specified as an enum instance of CausalType when is_causal=True, got {causal_type=}."
        )

    assert len(query_shape) == len(key_shape) == len(value_shape) == 4
    assert key_shape[1] == value_shape[1]
    if is_causal and causal_type == CausalType.DontCare and query_shape[1] != key_shape[1]:
        raise ValueError(
            "Causal mask type DontCare is only valid when seqlen_q == seqlen_kv, got "
            f"{query_shape[1]=}, {key_shape[1]=}."
        )


def multi_dim_attention_tensor_checks(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    requires_grad: bool,
    supported_dtypes_forward: list[torch.dtype] | None = None,
    supported_dtypes_backward: list[torch.dtype] | None = None,
    supports_mla: bool = True,
    supports_gqa_mqa: bool = True,
    raise_error: bool = True,
    backend_name: str | None = None,
) -> bool:
    backend_name = backend_name or "Multi-Dimensional Attention"

    if not _universal_attention_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        requires_grad=requires_grad,
        supported_dtypes_forward=supported_dtypes_forward,
        supported_dtypes_backward=supported_dtypes_backward,
        supports_mla=supports_mla,
        supports_gqa_mqa=supports_gqa_mqa,
        raise_error=raise_error,
        backend_name=backend_name,
    ):
        return False

    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    query_dim = len(query_shape)
    if query_dim not in [4, 5, 6]:
        target_fn(
            f"Multi-Dimensional Attention supports 4-D, 5-D, or 6-D tensors as inputs, got {query_dim=}.",
            exception=ValueError,
        )
        return False

    num_dims = query_dim - 3  # minus batch, heads, head_dim

    q_token_layout_shape = query_shape[1 : 1 + num_dims]
    k_token_layout_shape = key_shape[1 : 1 + num_dims]
    v_token_layout_shape = value_shape[1 : 1 + num_dims]

    if q_token_layout_shape != k_token_layout_shape or q_token_layout_shape != v_token_layout_shape:
        target_fn(
            "Q, K and V must match in their token layout shapes in multi-dimensional attention, "
            f"got {q_token_layout_shape=}, {k_token_layout_shape=}, {v_token_layout_shape=}.",
            exception=ValueError,
        )
        return False

    return True


def check_valid_tuple_or_element(
    param: Any, num_dims: int, typename: type, raise_error: bool = False, param_name: str = "unknown"
) -> tuple | None:
    if isinstance(param, typename):
        return tuple(param for _ in range(num_dims))

    if isinstance(param, Sequence) and len(param) == num_dims and all(isinstance(x, typename) for x in param):
        return tuple(x for x in param)

    if raise_error:
        raise ValueError(f"Invalid value for parameter {param_name}: {param}.")
    return None


def multi_dim_attention_param_filter_tensorless(
    token_layout_shape: tuple,
    window_size: tuple | int = -1,
    stride: tuple | int = 1,
    dilation: tuple | int = 1,
    is_causal: tuple | bool = False,
) -> tuple[tuple, tuple, tuple, tuple]:
    """
    Converts all multi-dimensional parameters to standard types.
    """

    if not isinstance(token_layout_shape, tuple) or any(not isinstance(x, int) for x in token_layout_shape):
        raise ValueError(f"token_layout_shape must be an integer tuple, got {token_layout_shape=}.")

    num_dims = len(token_layout_shape)
    assert num_dims in [1, 2, 3]

    window_size_ = check_valid_tuple_or_element(window_size, num_dims, int)
    if window_size_ is None:
        raise ValueError(
            f"Parameter 'window_size' must be either an int or tuple of {num_dims} ints, got {window_size=}."
        )

    stride_ = check_valid_tuple_or_element(stride, num_dims, int)
    if stride_ is None:
        raise ValueError(f"Parameter 'stride' must be either an int or tuple of {num_dims} ints, got {stride=}.")

    dilation_ = check_valid_tuple_or_element(dilation, num_dims, int)
    if dilation_ is None:
        raise ValueError(f"Parameter 'dilation' must be either an int or tuple of {num_dims} ints, got {dilation=}.")

    is_causal_ = check_valid_tuple_or_element(is_causal, num_dims, bool)
    if is_causal_ is None:
        raise ValueError(
            f"Parameter 'is_causal' must be either a boolean or tuple of {num_dims} booleans, got {is_causal=}."
        )

    # Map -1 windows to corresponding size in token layout
    window_size_ = tuple(w if w != -1 else x for x, w in zip(token_layout_shape, window_size_))

    return window_size_, stride_, dilation_, is_causal_


def multi_dim_attention_param_checks_tensorless(
    token_layout_shape: tuple,
    window_size: tuple,
    stride: tuple,
    dilation: tuple,
    is_causal: tuple,
):
    """
    Validates multi-dimensional parameters.
    """

    if not isinstance(token_layout_shape, tuple) or any(not isinstance(x, int) for x in token_layout_shape):
        raise ValueError(f"token_layout_shape must be an integer tuple, got {token_layout_shape=}.")

    num_dims = len(token_layout_shape)
    assert num_dims in [1, 2, 3]

    if any(x <= 1 for x in token_layout_shape):
        raise ValueError(f"Token layout dimensions must all be >= 2, got {token_layout_shape=}.")

    if any(w <= 1 for w in window_size):
        raise ValueError(
            "Parameter 'window_size' must be either -1 (no sparsity) or >= 2 along every dimension, "
            f"got {window_size=}."
        )

    if any(w * d > x for x, w, d in zip(token_layout_shape, window_size, dilation)):
        raise ValueError(
            "The product of 'window_size' and 'dilation' cannot be greater than the input "
            f"(token layout shape), got {window_size=}, {dilation=}, {token_layout_shape=}."
        )

    if any(s < 1 for s in stride):
        raise ValueError(f"Parameter 'stride' allows positive integers only, got {stride=}.")

    if any(s > w for w, s in zip(window_size, stride)):
        raise ValueError(
            f"Parameter 'stride' cannot be greater than window size along any dimension, got {window_size=}, {stride=}."
        )

    if any(d < 1 for d in dilation):
        raise ValueError(f"Parameter 'dilation' allows positive integers only, got {dilation=}.")


def multi_dim_attention_param_filter(
    query: Tensor,  # [B,*token_layout_shape,H,D]
    window_size: tuple | int = -1,
    stride: tuple | int = 1,
    dilation: tuple | int = 1,
    is_causal: tuple | bool = False,
) -> tuple[tuple, tuple, tuple, tuple, tuple]:
    """
    Converts all multi-dimensional parameters to standard types.
    """
    assert query.dim() in [4, 5, 6]
    num_dims = query.dim() - 3
    token_layout_shape = tuple(s for s in query.shape[1 : 1 + num_dims])

    window_size_, stride_, dilation_, is_causal_ = multi_dim_attention_param_filter_tensorless(
        token_layout_shape=token_layout_shape,
        window_size=window_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
    )

    return token_layout_shape, window_size_, stride_, dilation_, is_causal_


def multi_dim_attention_param_checks(
    query: Tensor,  # [B,*token_layout_shape,H,D]
    window_size: tuple,
    stride: tuple,
    dilation: tuple,
    is_causal: tuple,
):
    """
    Validates multi-dimensional parameters.
    """
    assert query.dim() in [4, 5, 6]
    num_dims = query.dim() - 3
    token_layout_shape = tuple(s for s in query.shape[1 : 1 + num_dims])

    multi_dim_attention_param_checks_tensorless(
        token_layout_shape=token_layout_shape,
        window_size=window_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
    )
