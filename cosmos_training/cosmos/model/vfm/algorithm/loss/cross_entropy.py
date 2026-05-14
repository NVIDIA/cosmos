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
"""CE loss for VLM training.

Ported from cosmos_rl.policy.trainer.llm_trainer.sft_trainer.async_safe_ce
(packages/cosmos-rl/cosmos_rl/policy/trainer/llm_trainer/sft_trainer.py).

The reduction formula must match async_safe_ce exactly to preserve loss parity
between Phase 0 (cosmos-rl path) and Phase 2 (this module).

Two reduction paths — determined by cp_group presence:

  CP enabled (cp_group.size() > 1):
    Per-rank mean CE loss × loss_scaling_factor.
    Rationale: each CP rank sees a different segment of the sequence; computing
    a weighted-mean here would require knowing each rank's valid-token count,
    which is expensive. The simpler per-rank mean × scaling is consistent with
    cosmos-rl's implementation.

  CP disabled (cp_group is None or cp_group.size() == 1):
    Sum CE loss / (global_n_valid_tokens + 1e-8) × (num_dp_workers × scaling).
    The ×num_dp_workers compensates for FSDP's gradient averaging across DP
    ranks, ensuring the effective gradient equals the gradient of the global
    mean loss even with unbalanced per-rank token counts.
    Reference: async_safe_ce:97-109 in the source file above.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from cosmos.utils.vfm.vlm.constant import IGNORE_INDEX


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_scaling_factor: float = 1.0,
    dp_group: Optional[dist.ProcessGroup] = None,
    cp_group: Optional[dist.ProcessGroup] = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Next-token-prediction CE loss with DP/CP group reduction.

    Matches the behavior of cosmos_rl.policy.trainer.llm_trainer.sft_trainer.async_safe_ce
    with the TORCH_CROSS_ENTROPY backend (F.cross_entropy with float32 cast).

    Args:
        logits: (B, T, V) float tensor — raw model output before softmax.
        labels: (B, T) long tensor — ground-truth token ids.
                Positions equal to ignore_index are excluded from the loss.
        loss_scaling_factor: scalar multiplied into the returned loss.
        dp_group: FSDP data-parallel shard group for loss normalization.
                  None = no DP reduction (single-GPU or replicate-only).
        cp_group: Context-parallel group. If size > 1, use per-rank mean.
                  None = no CP reduction.
        ignore_index: label value to exclude (defaults to ``IGNORE_INDEX``, -100).

    Returns:
        Scalar loss tensor.
    """
    # Shift for next-token prediction: predict token[t+1] using hidden state[t].
    # logits[:, :-1] aligns with labels[:, 1:].
    # Reference: async_safe_ce:63-73 (output[:, :-1], target[:, 1:])
    shifted_logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
    shifted_labels = labels[:, 1:].contiguous().view(-1)

    if cp_group is not None and cp_group.size() > 1:
        # CP path: each rank sees a different sequence segment.
        # Use simple mean reduction; nan_to_num handles fully-ignored batches.
        # Reference: async_safe_ce:74-88
        loss = F.cross_entropy(
            shifted_logits.float(),
            shifted_labels,
            ignore_index=ignore_index,
            reduction="mean",
        )
        loss = torch.nan_to_num(loss, nan=0.0)
        return loss * loss_scaling_factor

    # No-CP path: per-token loss, then normalize over the global valid-token count.
    # Reference: async_safe_ce:89-109
    per_token_loss = F.cross_entropy(
        shifted_logits.float(),
        shifted_labels,
        ignore_index=ignore_index,
        reduction="none",
    )
    n_valid_tokens = (shifted_labels != ignore_index).sum()
    num_dp_workers = 1
    if dp_group is not None:
        dist.all_reduce(n_valid_tokens, op=dist.ReduceOp.SUM, group=dp_group)
        num_dp_workers = dist.get_world_size(group=dp_group)

    loss = per_token_loss.sum() / (n_valid_tokens + 1e-8) * (num_dp_workers * loss_scaling_factor)
    return loss
