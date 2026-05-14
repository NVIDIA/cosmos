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

from typing import Callable

import cudnn
import torch
from torch import Size, Tensor

from cosmos.model.attention.utils import get_arch_tag
from cosmos.model.attention.utils.safe_ops import log
from cosmos.model.attention.utils.safe_ops.functools import lru_cache


# Force using padded mask as a potential workaround for failing use cases
FORCE_PADDED_MASK = False

CUDNN_GRAPH_CACHE_SIZE = 64

log.debug(f"cuDNN Attention graphs are cached using an LRU cache with capacity {CUDNN_GRAPH_CACHE_SIZE}.")
log.debug(f"cuDNN Attention {FORCE_PADDED_MASK=}.")


def get_dtype_choices(arch_tag: int) -> dict:
    """
    Returns data type choices according to arch tag (attention.utils.get_arch_tag).

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        data_type_choices (dict): a map from PyTorch data types to cuDNN data types. Empty if device
        is not supported.

    """

    if arch_tag < 80:
        log.debug("cuDNN Attention is not supported because compute capability is below the minimum (8.0).")
        return {}


    log.debug(f"cuDNN Attention only supports FP16 and BF16 for {arch_tag=}.")
    return {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
    }


def cudnn_sdpa_fwd_generate_operands(
    q: Tensor, k: Tensor, v: Tensor, num_heads: int, return_lse: bool = False
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """
    Takes torch input operands (Q, K, V), validates them, and returns views compatible with cuDNN
    APIs ("strided" view with heads-first logical layout but heads-last physical layout.

    NOTE: this operation tries to specifically avoid memory copies and express everything as tensor
    views, therefore it is crucial to not manipulate the outputs in __any way__ after this point,
    and directly call cuDNN SDPA operations on it.
    This is also what makes this operation very efficient and low in overhead, as there are no
    device/CUDA operations or barriers with host/CPU.

    Parameters:
        query (Tensor): 4-D query tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim]`)

        key (Tensor): 4-D key tensor, with the heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim]`)

        value (Tensor): 4-D value tensor, with heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`)

        num_heads (int): Number of attention heads. Used for layout validation.

    Other Parameters:
        return_lse (bool): Whether to store and return the logsumexp values. Default is False.

    Returns:
        query_cudnn_layout (Tensor): 4-D query tensor, with the cuDNN strided layout
            (`[batch, heads, seqlen, head_dim]`).

        key_cudnn_layout (Tensor): 4-D key tensor, with the cuDNN strided layout
            (`[batch, heads_kv, seqlen_kv, head_dim]`).

        value_cudnn_layout (Tensor): 4-D output tensor, with the cuDNN strided layout
            (`[batch, heads_kv, seqlen_kv, head_dim_v]`).

        output_cudnn_layout (Tensor): 4-D output tensor, with the cuDNN strided layout
            (`[batch, heads, seqlen, head_dim_v]`).

        logsumexp_cudnn_layout (Tensor | None): only returned when return_lse is True. logsumexp
            tensor, with the cuDNN strided layout (`[batch, heads, seqlen, 1]`).
    """

    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(
            f"All attention operands must match in batch size, got {q.shape[0]=}, {k.shape[0]=}, {v.shape[0]=}."
        )

    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"Query and key must match in head dim, got {q.shape[-1]=}, {k.shape[-1]=}.")

    if q.shape[-2] != num_heads:
        raise ValueError(
            f"The heads-last layout considers q.shape[-2] as number of heads, got {q.shape[-2]=} but {num_heads=}."
        )

    if k.shape[-2] != num_heads:
        raise ValueError(
            f"The heads-last layout considers k.shape[-2] as number of heads, got {k.shape[-2]=} but {num_heads=}."
        )

    if v.shape[-2] != num_heads:
        raise ValueError(
            f"The heads-last layout considers v.shape[-2] as number of heads, got {v.shape[-2]=} but {num_heads=}."
        )

    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError(
            "All attention operands must be contiguous, got "
            f"{q.is_contiguous()=}, {k.is_contiguous()=}, {v.is_contiguous()=}."
        )

    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"All attention operands must match in dtype, got {q.dtype=}, {k.dtype=}, {v.dtype=}.")

    if q.device != k.device or q.device != v.device:
        raise ValueError(
            f"All attention operands must be on the same device, got {q.device=}, {k.device=}, {v.device=}."
        )

    dtype = q.dtype
    device = q.device
    arch_tag = get_arch_tag(device)
    dtype_choices = get_dtype_choices(arch_tag)

    if dtype not in dtype_choices:
        raise ValueError(f"Data type {dtype} is not supported; choices are: {dtype_choices.keys()}.")

    if arch_tag < 80:
        raise NotImplementedError(f"cuDNN Attention only supports SM80 and later, but {device=} is SM{arch_tag}.")

    batch, seqlen_q, _, head_dim_qk = q.shape
    _, _, _, head_dim_v = v.shape

    output = torch.empty([batch, seqlen_q, num_heads, head_dim_v], dtype=dtype, device=device)  # [B,N,H,Dv]
    lse = None
    if return_lse:
        lse = torch.empty([batch, seqlen_q, num_heads, 1], dtype=dtype, device=device)  # [B,N,H,1]

    q_cudnn_layout = q.permute(0, 2, 1, 3)  # [B,H,N,D]
    k_cudnn_layout = k.permute(0, 2, 1, 3)  # [B,Hkv,Nkv,D]
    v_cudnn_layout = v.permute(0, 2, 1, 3)  # [B,Hkv,Nkv,Dv]
    output_cudnn_layout = output.permute(0, 2, 1, 3)  # [B,H,N,Dv]
    lse_cudnn_layout = None
    assert q_cudnn_layout.data_ptr() == q.data_ptr()
    assert k_cudnn_layout.data_ptr() == k.data_ptr()
    assert v_cudnn_layout.data_ptr() == v.data_ptr()
    assert output_cudnn_layout.data_ptr() == output.data_ptr()

    if return_lse:
        lse_cudnn_layout = lse.permute(0, 2, 1, 3)  # [B,H,N,1]
        assert lse_cudnn_layout.data_ptr() == lse.data_ptr()

    return q_cudnn_layout, k_cudnn_layout, v_cudnn_layout, output_cudnn_layout, lse_cudnn_layout


@lru_cache(maxsize=CUDNN_GRAPH_CACHE_SIZE)
def cudnn_sdpa_fwd_generate_op(
    dtype: torch.dtype,
    device: torch.device,
    q_shape: Size,
    q_stride: Size,
    k_shape: Size,
    k_stride: Size,
    v_shape: Size,
    v_stride: Size,
    output_shape: Size,
    output_stride: Size,
    lse_shape: Size | None = None,
    lse_stride: Size | None = None,
    is_causal: bool = False,
    attn_scale: float | None = None,
    seqlen_Q: int | None = None,
    seqlen_KV: int | None = None,
) -> Callable:
    """
    Takes use case metadata that has been validated and generated by cudnn_sdpa_fwd_generate_operands
    and returns a callable cuDNN SDPA forward operation.
    This function does NOT perform attention and rather prepares and builds the cuDNN graph
    responsible for doing so.

    The final callable that it returns will take any q, k, v, output and (optionally) lse tensors
    matching the same attributes (shape, device, dtype, etc) and call cuDNN SDPA forward on them.

    Parameters:
        dtype (torch.dtype): Tensor data type for Q, K, V, and output.

        device (torch.device): Torch (CUDA) device where tensors are and where Attention will run.

        q_shape (Size): The shape of the 4-D query tensor with the cuDNN strided layout
            (`[batch, heads, seqlen, head_dim]`).

        q_stride (Size): The stride of the 4-D query tensor with the cuDNN strided layout.

        k_shape (Size): The shape of the 4-D key tensor with the cuDNN strided layout
            (`[batch, heads_kv, seqlen_kv, head_dim]`).

        k_stride (Size): The stride of the 4-D key tensor with the cuDNN strided layout.

        v_shape (Size): The shape of the 4-D value tensor with the cuDNN strided layout
            (`[batch, heads_kv, seqlen_kv, head_dim_v]`).

        v_stride (Size): The stride of the 4-D value tensor with the cuDNN strided layout.

        output_shape (Size): The shape of the 4-D output tensor with the cuDNN strided layout
            (`[batch, heads, seqlen, head_dim_v]`).

        output_stride (Size): The stride of the 4-D output tensor with the cuDNN strided layout.

        lse_shape (Size | None): The shape of the 4-D logsumexp tensor with the cuDNN strided
            layout (`[batch, heads, seqlen, 1]`).

        lse_stride (Size | None): The stride of the 4-D logsumexp tensor with the cuDNN strided
            layout.

    Other Parameters:
        is_causal (bool): whether or not causal masking is enabled. Default is False.

        attn_scale (float | None): Dot product scale (attention scale). Defaults to
            head_dim ** -0.5.

    Returns:
        cudnn_sdpa_forward_exec (Callable): Function executing the cuDNN graph with the SDPA
            forward operation. Function signature:

                query_cudnn_layout (Tensor): 4-D query tensor, with the cuDNN strided layout
                    (`[batch, heads, seqlen, head_dim]`).

                key_cudnn_layout (Tensor): 4-D key tensor, with the cuDNN strided layout
                    (`[batch, heads_kv, seqlen_kv, head_dim]`).

                value_cudnn_layout (Tensor): 4-D output tensor, with the cuDNN strided layout
                    (`[batch, heads_kv, seqlen_kv, head_dim_v]`).

                output_cudnn_layout (Tensor): 4-D output tensor, with the cuDNN strided layout
                    (`[batch, heads, seqlen, head_dim_v]`).

                logsumexp_cudnn_layout (Tensor | None): Optional logsumexp tensor, with the
                    cuDNN strided layout (`[batch, heads, seqlen, 1]`).
    """

    attn_scale = attn_scale if attn_scale is not None else q_shape[-1] ** -0.5

    arch_tag = get_arch_tag(device)
    dtype_choices = get_dtype_choices(arch_tag)

    assert dtype in dtype_choices
    cudnn_dtype = dtype_choices[dtype]

    graph = cudnn.pygraph(
        io_data_type=cudnn_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q_cudnn = graph.tensor(dim=q_shape, stride=q_stride, data_type=cudnn_dtype)
    k_cudnn = graph.tensor(dim=k_shape, stride=k_stride, data_type=cudnn_dtype)
    v_cudnn = graph.tensor(dim=v_shape, stride=v_stride, data_type=cudnn_dtype)

    assert (lse_shape is None and lse_stride is None) or (lse_shape is not None and lse_stride is not None)
    generate_stats = lse_shape is not None

    seqlen_q_cudnn = None
    seqlen_kv_cudnn = None
    use_padding_mask = FORCE_PADDED_MASK or seqlen_Q is not None or seqlen_KV is not None
    if use_padding_mask:
        seqlen_Q = seqlen_Q if seqlen_Q is not None else q_shape[2]
        seqlen_KV = seqlen_KV if seqlen_KV is not None else k_shape[2]

        seqlen_q_cudnn = graph.tensor(dim=[q_shape[0], 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        seqlen_kv_cudnn = graph.tensor(dim=[k_shape[0], 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)

    o_cudnn, lse_cudnn = graph.sdpa(
        q=q_cudnn,
        k=k_cudnn,
        v=v_cudnn,
        generate_stats=generate_stats,
        attn_scale=attn_scale,
        use_causal_mask=is_causal,
        use_padding_mask=use_padding_mask,
        seq_len_q=seqlen_q_cudnn,
        seq_len_kv=seqlen_kv_cudnn,
    )

    o_cudnn.set_output(True).set_data_type(cudnn_dtype).set_dim(output_shape).set_stride(output_stride)
    if generate_stats:
        lse_cudnn.set_output(True).set_dim(lse_shape).set_stride(lse_stride)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    workspace_size_bytes = graph.get_workspace_size()
    log.debug(f"Generated cuDNN Attention graph. Scratch space required: {workspace_size_bytes} bytes.")

    handle = cudnn.create_handle()

    def cudnn_operation(q: Tensor, k: Tensor, v: Tensor, output: Tensor, lse: Tensor | None = None):

        # with random NaNs and illegal memory accesses and things of that nature.
        stream = torch.cuda.current_stream(q.device)
        cudnn.set_stream(handle=handle, stream=stream.cuda_stream)

        workspace = torch.zeros(workspace_size_bytes, device=device, dtype=torch.uint8)  # [workspace_size_bytes]

        variant_pack = {
            q_cudnn: q,
            k_cudnn: k,
            v_cudnn: v,
            o_cudnn: output,
        }

        if use_padding_mask:
            batch = k.shape[0]
            seqlen_q_cu = torch.tensor([seqlen_Q for _ in range(batch)]).to(device).reshape(batch, 1, 1, 1)  # [B,1,1,1]
            seqlen_kv_cu = (
                torch.tensor([seqlen_KV for _ in range(batch)]).to(device).reshape(batch, 1, 1, 1)
            )  # [B,1,1,1]
            log.debug(f"{q.shape=}, {k.shape=}, {seqlen_q_cu=}, {seqlen_kv_cu=}")
            variant_pack[seqlen_q_cudnn] = seqlen_q_cu
            variant_pack[seqlen_kv_cudnn] = seqlen_kv_cu

        if generate_stats:
            assert lse is not None
            assert lse_cudnn is not None
            variant_pack[lse_cudnn] = lse
        else:
            assert lse is None and lse_cudnn is None

        log.debug(f"Generated cuDNN Attention graph executed")
        return graph.execute(variant_pack, workspace, handle=handle)

    return cudnn_operation


def cudnn_sdpa_fwd_post_process(
    output_cudnn_layout: Tensor,
    lse_cudnn_layout: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """
    Takes torch tensor views validated and generated by cudnn_sdpa_fwd_generate_operands and
    maps back to torch contiguous layout (heads-last both logical and physical).
    It should be called after the cuDNN operation.

    Like cudnn_sdpa_fwd_generate_operands, this function is expected to be minimal overhead.

    Parameters:
        output_cudnn_layout (Tensor): 4-D output tensor, with the cuDNN strided layout
            (`[batch, heads, seqlen, head_dim_v]`).

        logsumexp_cudnn_layout (Tensor | None): Optional logsumexp tensor, with the cuDNN
            strided layout (`[batch, heads, seqlen, 1]`).

    Returns:
        output (Tensor): 4-D output tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim_v]`).

        logsumexp (Tensor | None): Optional logsumexp tensor, with the heads-last contiguous
            layout (`[batch, seqlen, heads, 1]`).
    """

    output = output_cudnn_layout.permute(0, 2, 1, 3)  # [B,N,H,Dv]
    lse = None
    assert output.data_ptr() == output_cudnn_layout.data_ptr()

    if lse_cudnn_layout is not None:
        lse = lse_cudnn_layout.permute(0, 2, 1, 3)  # [B,N,H,1]
        assert lse.data_ptr() == lse_cudnn_layout.data_ptr()

    return output, lse
