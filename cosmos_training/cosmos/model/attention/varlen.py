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

Varlen utilities
"""

import torch
from torch import Tensor

from cosmos.model.attention.utils import is_torch_compiling


def generate_varlen_parameters(
    query: Tensor,  # [1,S_total_Q,H,D]
    key: Tensor,  # [1,S_total_KV,H_KV,D]
    value: Tensor,  # [1,S_total_KV,H_KV,D_V]
    seqlens_Q: Tensor | None = None,  # [B]
    seqlens_KV: Tensor | None = None,  # [B]
) -> (
    tuple[None, None, int, int] | tuple[Tensor, Tensor, int, int]
):  # (cumseqlen_Q[B+1], cumseqlen_KV[B+1], max_seqlen_Q, max_seqlen_KV)

    # which we launch the varlen kernel) and not device tensors.
    # .item() introduces control flow and breaks the graph.
    # It is also inefficient to repeat this per-op, and mostly there for convenience.
    # generate_varlen_parameters should ideally always be called by the user ahead of model
    # forward / backward.
    if is_torch_compiling():
        raise RuntimeError(
            "Running 'generate_varlen_parameters' in a torch-compiled region is disallowed as it "
            "results in graph breaks. Please consider calling ahead of time and pass "
            "'cumulative_seqlen_{Q,KV}' and 'max_seqlen_{Q,KV}' instead of 'seqlens_{Q,KV}' to "
            "'attention'. "
        )

    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError(
            f"Q, K, and V must match in batch size, got {query.shape[0]=}, {key.shape[0]=}, {value.shape[0]=}."
        )

    if (seqlens_Q is None) ^ (seqlens_KV is None):
        raise ValueError(
            "Variable length Attention requires both of seqlens_Q and seqlens_KV to be set, got "
            f"{seqlens_Q=}, {seqlens_KV=}."
        )

    if seqlens_Q is None and seqlens_KV is None:
        # Not varlen
        return None, None, 0, 0

    assert seqlens_Q is not None
    assert seqlens_KV is not None

    if not isinstance(seqlens_Q, Tensor) or not isinstance(seqlens_KV, Tensor):
        raise ValueError("seqlens_Q and seqlens_KV must both be tensors.")

    if seqlens_Q.device != query.device or seqlens_KV.device != query.device:
        raise ValueError(
            "seqlens_Q and seqlens_KV must be on the same device as QKV, but "
            f"{seqlens_Q.device=}, {seqlens_KV.device=}, {query.device=}."
        )

    if seqlens_Q.dtype != torch.int32 or seqlens_KV.dtype != torch.int32:
        raise ValueError(
            f"seqlens_Q and seqlens_KV must both be torch.int32 tensors, got {seqlens_Q.dtype=}, {seqlens_KV.dtype=}."
        )

    if seqlens_Q.dim() != 1 or seqlens_KV.dim() != 1:
        raise ValueError(
            f"seqlens_Q and seqlens_KV must both be 1-D tensors, got {seqlens_Q.dim()=}, {seqlens_KV.dim()=}."
        )

    if seqlens_Q.shape[0] != seqlens_KV.shape[0]:
        raise ValueError(f"seqlens_Q and seqlens_KV must match in size, got {seqlens_Q.shape=}, {seqlens_KV.shape=}.")

    if seqlens_Q.shape[0] < 1:
        raise ValueError(
            f"seqlens_Q and seqlens_KV must contain at least one element, got {seqlens_Q.shape=}, {seqlens_KV.shape=}."
        )

    if query.shape[0] != 1:
        raise ValueError(
            f"Variable length attention only supports sequence-packed memory layout (batch = 1), got {query.shape[0]=}."
        )

    assert seqlens_Q.dim() == seqlens_KV.dim() == 1
    assert seqlens_Q.shape[0] == seqlens_KV.shape[0] >= 1
    assert seqlens_Q.dtype == seqlens_KV.dtype == torch.int32

    max_seqlen_Q = seqlens_Q.max().item()  # type: ignore
    max_seqlen_KV = seqlens_KV.max().item()  # type: ignore

    if max_seqlen_Q < 0 or max_seqlen_KV < 0:
        raise ValueError(f"max_seqlen_Q and max_seqlen_KV cannot be negative, got {max_seqlen_Q=}, {max_seqlen_KV=}.")


    # This feature may require support in the backends themselves; see NATTEN PR:
    # https://github.com/SHI-Labs/NATTEN/pull/327
    if (max_seqlen_Q == 0) != (max_seqlen_KV == 0):
        raise ValueError(
            "max_seqlen_Q and max_seqlen_KV must either both be 0 or both be positive, "
            f"but computed {max_seqlen_Q=}, {max_seqlen_KV=} from provided seqlens."
        )


    z = torch.tensor([0], dtype=torch.int32, device=seqlens_Q.device)  # [1]
    cumulative_seqlen_Q = torch.cat([z, seqlens_Q.cumsum(0).to(torch.int32)], dim=0)  # [B+1]
    cumulative_seqlen_KV = torch.cat([z, seqlens_KV.cumsum(0).to(torch.int32)], dim=0)  # [B+1]

    assert isinstance(max_seqlen_Q, int)
    assert isinstance(max_seqlen_KV, int)

    return (
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        max_seqlen_Q,
        max_seqlen_KV,
    )


def generate_multi_dim_varlen_parameters(
    token_layout_list: list,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    window_size_list: list | None = None,
    stride_list: list | None = None,
    dilation_list: list | None = None,
    is_causal: tuple | bool = False,
    *args,
    **kwargs,
) -> dict:
    """
    Configures metadata for variable-length multi-dimensional attention operations.

    This function prepares the metadata needed for varlen/varsized sparse attention,
    including backend selection and tile configurations. The metadata should be generated
    ahead of time (outside of torch.compile regions) and reused across forward/backward passes.

    **Requires NATTEN >= 0.21.9.dev0**

    Parameters:
        token_layout_list (list): List of token layout tuples describing the spatial arrangement
            of tokens for each sequence. For example, for 2D attention with two sequences of
            sizes (H1, W1) and (H2, W2), pass [(H1, W1), (H2, W2)].

        head_dim (int): Attention head dimension.

        device (torch.device): Target device for runtime.

        dtype (torch.dtype): Tensor element type.

        requires_grad (bool): Whether tensors will require backward pass.

        window_size_list (list | None): Per-sequence window sizes for variable kernel sizes.

        stride_list (list | None): Per-sequence stride values for variable strides.

        dilation_list (list | None): Per-sequence dilation values for variable dilations.

        is_causal (tuple | bool): Toggle causal masking. Default is False.

    Returns:
        dict: Runtime metadata for varlen operations. This dict should be passed to
            `natten_multi_dimensional_attention_varlen` as the `metadata` parameter.
    """
    # For now, NATTEN is the only backend that supports varlen multi-dimensional attention

    from cosmos.model.attention.natten import NATTEN_VARLEN_MULTI_DIM_VERSION, natten_supported, natten_version_satisfies

    if not natten_supported():
        raise RuntimeError("generate_multi_dim_varlen_parameters requires NATTEN.")

    if not natten_version_satisfies(NATTEN_VARLEN_MULTI_DIM_VERSION):
        raise RuntimeError(
            f"generate_multi_dim_varlen_parameters requires NATTEN >= {NATTEN_VARLEN_MULTI_DIM_VERSION}. "
            "Please upgrade NATTEN to use varlen/varsized attention features."
        )

    from natten.varlen import configure_varlen

    # Map -1s in window size list to full attention
    if window_size_list is None:
        window_size_list_filtered = [token_layout for token_layout in token_layout_list]
    else:
        window_size_list_filtered = []
        for window_size, token_layout in zip(window_size_list, token_layout_list):
            window_size_filtered = tuple(k if k > 0 else x for k, x in zip(window_size, token_layout))
            window_size_list_filtered.append(window_size_filtered)

    metadata = configure_varlen(
        token_layout_list=token_layout_list,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        is_causal=is_causal,
        kernel_size=None,
        stride=None,
        dilation=None,
        kernel_size_list=window_size_list_filtered,
        stride_list=stride_list,
        dilation_list=dilation_list,
        *args,
        **kwargs,
    )

    return metadata
