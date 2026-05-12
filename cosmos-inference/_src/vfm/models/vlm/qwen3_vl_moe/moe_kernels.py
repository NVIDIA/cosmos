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

from typing import Callable, Literal

import torch
import triton
import triton.language as tl

# Set the token group alignment size for experts in MoE. This is implemented by
# padding each expert size to the next multiple of TOKEN_GROUP_ALIGN_SIZE_M.

# Valid values are: 8, 16, or 32.
# Different values are needed for different cases:

# * For bf16, 8 is enough (16 byte alignment / 2 bytes per elem = 8 elements).
# * For fp8, 16 byte alignment / 1 byte per elem = 16 elements.
# * For mxfp8, we need 32 (or block_size) because scaling block size is (1 x 32),
#   so when doing per-token-group quantization on each logically distinct subtensor,
#   we need to ensure the contracting dim is divisible by block_size.
#   In the backward pass, grad_weight = (grad_output_t @ input).t() has gemm dims
#   of (N, M) @ (M, K) so M is the contracting dim, and group offsets are along M,
#   so we need 32 element alignment.
TOKEN_GROUP_ALIGN_SIZE_M = 16
ValidTokenGroupAlignmentSize = Literal[8, 16, 32]


def _permute(
    x: torch.Tensor,
    num_tokens_per_expert: int,
    num_experts: int,
    alignment: int = TOKEN_GROUP_ALIGN_SIZE_M,
):
    x_padded_size = x.shape[0] + num_experts * alignment
    padded_max_len = ((x_padded_size + alignment - 1) // alignment) * alignment

    with torch.no_grad():
        (
            permuted_indices,
            padded_num_tokens_per_expert,
        ) = _generate_permute_indices(
            num_tokens_per_expert=num_tokens_per_expert,
            num_experts=num_experts,
            max_len=padded_max_len,
            alignment=alignment,
        )

    x = torch.vstack((x, x.new_zeros(x.shape[-1])))
    input_shape = x.shape
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, padded_num_tokens_per_expert


def _unpermute(out, input_shape, permuted_indices):
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    return out_unpermuted[:-1]


def indices_padding_wrapper(func: Callable) -> Callable:
    """
    In order to use torch._grouped_mm, we need to make sure the number of
    tokens each expert gets is a multiple of TOKEN_GROUP_ALIGN_SIZE_M. The
    generate_permute_indices kernel also helps achieve this via padding,
    without incurring synchronization between device and host.
    """

    def wrapper(
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        num_experts = num_tokens_per_expert.shape[0]

        input_shape, x, permuted_indices, padded_num_tokens_per_expert = _permute(x, num_tokens_per_expert, num_experts)

        out = func(gate_up_proj, down_proj, act_fn, x, padded_num_tokens_per_expert)

        out = _unpermute(out, input_shape, permuted_indices)
        return out

    return wrapper


@triton.jit
def _fill_indices_kernel(
    num_tokens_per_expert_ptr,
    start_index_values_ptr,
    write_offsets_ptr,
    output_ptr,
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Number of threads per block
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # map programs (blocks) to the experts and loop (grid stride) if needed
    for expert_id in range(pid, num_experts, num_programs):
        # read this experts write offset
        write_offset = tl.load(write_offsets_ptr + expert_id)

        # load number of tokens for this expert
        start_index = tl.load(start_index_values_ptr + expert_id)
        length = tl.load(num_tokens_per_expert_ptr + expert_id)

        # each thread in block processes tokens in parallel
        offsets = tl.arange(0, BLOCK_SIZE)

        # tokens are processed in chunks of BLOCK_SIZE
        for chunk_start in range(0, length, BLOCK_SIZE):
            chunk_offsets = chunk_start + offsets

            # mask valid indices
            mask = chunk_offsets < length

            values = start_index + chunk_offsets

            # destination
            dest_indices = write_offset + chunk_offsets

            # store
            tl.store(output_ptr + dest_indices, values, mask=mask)


def _fill_indices_wrapper(
    num_tokens_per_expert: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    num_experts: int,
    max_len: int,
    block_size: int = 128,
    max_blocks: int = 1024,  # cap on total number of blocks to launch
):
    # preallocate output
    permuted_indices = torch.full((max_len,), -1, dtype=torch.int32, device=num_tokens_per_expert.device)

    # write offsets is per local expert...
    num_blocks = min(num_experts, max_blocks)
    # grid = one block per expert unless capped and then we loop...
    grid = (num_blocks,)

    # launch kernel
    _fill_indices_kernel[grid](
        num_tokens_per_expert,
        start_index_values,
        write_offsets,
        permuted_indices,
        num_experts,
        BLOCK_SIZE=block_size,
    )
    return permuted_indices


def _generate_permute_indices(
    num_tokens_per_expert: torch.Tensor,
    num_experts: int,
    max_len: int,
    alignment: int,
):
    """
    Prepare permutation indices and the number of tokens for each expert.

    Args:
        num_tokens_per_expert: number of tokens for each expert.
        num_experts: number of experts.
        max_len: maximum length of the output index vector.
        alignment: alignment for each returned element in `m_sizes` and padding min for zero token experts.

    Returns:
        permuted_indices: Tensor of indices that map original token order to the expert-grouped order.
        m_sizes: aligned number of tokens for each expert (padded to alignment boundary).
        m_offsets: Cumulative sum of m_sizes. The exclusive ending position for each expert's tokens.

    Explanatory details:
        `tokens_per_expert_group` is of shape (num_ranks * experts_per_rank,), for example:
        From: |       rank 0      |       rank 1      |
        To:   | E0 | E1 | E2 | E3 | E0 | E1 | E2 | E3 |
              |  4 |  2 |  1 |  3 |  1 |  2 |  3 |  4 |
    """
    start_index_values = torch.cumsum(num_tokens_per_expert, dim=0) - num_tokens_per_expert

    # pad out empty experts to alignment requirement
    m_sizes = torch.clamp_min(num_tokens_per_expert, alignment)

    # align the chunk sizes (cdiv)
    m_sizes = (m_sizes.to(torch.int32) + alignment - 1) // alignment * alignment

    # additional prefix sum to get write offset of each expert in permuted_indices
    # write offsets is per local expert, not global
    write_offsets = torch.cumsum(m_sizes, dim=0) - m_sizes

    # Select the implementation to use
    permuted_indices = _fill_indices_wrapper(
        num_tokens_per_expert=num_tokens_per_expert,
        start_index_values=start_index_values,
        write_offsets=write_offsets,
        num_experts=num_experts,
        max_len=max_len,
    )

    return permuted_indices, m_sizes
