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

import torch
from torch.distributed.tensor import DTensor, Partial
from torch.distributed.tensor.device_mesh import DeviceMesh

from cosmos3._src.vfm.models.vlm.qwen3_vl_moe.qwen3_vl_moe import LBLMetadata


def compute_load_balancing_loss(
    lbl_metadata: LBLMetadata | None,
    coeff: float | None,
    method: str,
    device_mesh: DeviceMesh | None,
) -> torch.Tensor | None:
    """
    Compute the load balancing loss. We compute the load balancing loss
    for each layer, and then average the loss across all layers.

    For computing the load balancing loss for each layer, we can either
    use the fraction of tokens routed to each expert for this rank ("local" method), or
    use the fraction of tokens routed to each expert across all ranks ("global" method).

    Args:
        lbl_metadata: The load balancing metadata. Contains the following tensors
            - num_tokens_per_expert: [num_layers, num_experts] - The number of
              tokens routed to each expert for this rank for each layer.
            - num_tokens: [num_layers, 1] - The total number of tokens in the
              batch for each layer.
            - mean_router_prob_per_expert: [num_layers, num_experts] - The average
              probability of routing to each expert for this rank for each layer.
        coeff: The coefficient for the load balancing loss.
        method: The method for the load balancing loss. Can be "local" or "global".
        device_mesh: The device mesh. Only needed if method is "global".

    Returns:
        The load balancing loss. None if lbl_metadata is None or coeff is None.
    """
    if lbl_metadata is None or coeff is None:
        return None
    assert method in ["local", "global"], "Invalid method"

    num_tokens_per_expert = lbl_metadata.num_tokens_per_expert
    num_experts = num_tokens_per_expert.shape[-1]
    num_tokens = lbl_metadata.num_tokens
    mean_router_prob_per_expert = lbl_metadata.mean_router_prob_per_expert

    if method == "global":
        # Note that these collectives must be executed outside a torch compiled region
        # since torch compile could reorder the collectives and cause deadlocks.
        assert device_mesh is not None, "MoE models require multiple GPUs."

        num_tokens_per_expert = DTensor.from_local(
            num_tokens_per_expert,
            device_mesh=device_mesh,
            placements=[Partial()] * device_mesh.ndim,
        ).full_tensor()
        num_tokens = DTensor.from_local(
            num_tokens,
            device_mesh=device_mesh,
            placements=[Partial()] * device_mesh.ndim,
        ).full_tensor()

    # Compute the fraction of tokens routed to each experts.
    # Summing over all experts should be equal to self.top_k.
    mean_tokens_per_expert = num_tokens_per_expert.float() / num_tokens.float()

    lbl = torch.mean(torch.sum(mean_tokens_per_expert * mean_router_prob_per_expert, dim=-1) * num_experts)
    return lbl * coeff
