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
Unified data and condition interface where we save the tokenized states and/or
noised latent states for diffusion/flow-matching training.
Used for the VFM generation model.
"""

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class GenerationDataClean:
    """
    Container for tokenized states and conditioning info (clean states)
    for the multi-modal (vision, sound, action) MoT training.
    Used for the VFM generation model.
    """

    batch_size: int
    # Vision (list of per-sample tensors)
    is_image_batch: bool
    raw_state_vision: list[torch.Tensor] | None = None  # raw state in pixel space
    x0_tokens_vision: list[torch.Tensor] | None = None  # tokenized latent state
    fps_vision: torch.Tensor | None = None

    # Image editing: number of vision items per sample.
    # When set, x0_tokens_vision is a flat list of individually-encoded image latents
    # (e.g. [src1, tgt1, src2, tgt2, ...]) and this field records how many items belong
    # to each sample (e.g. [2, 2, ...]).  None for standard T2I/T2V (one item per sample).
    num_vision_items_per_sample: list[int] | None = None

    # Audio (Sound)
    raw_state_sound: torch.Tensor | None = None
    x0_tokens_sound: torch.Tensor | None = None
    fps_sound: torch.Tensor | None = None

    # Action (dense list of per-sample tensors, only action-having samples)
    raw_state_action: list[torch.Tensor] | None = None
    x0_tokens_action: list[torch.Tensor] | None = None
    fps_action: torch.Tensor | None = None
    action_domain_id: list[torch.Tensor] | None = None  # per-sample domain IDs, None when no action samples
    raw_action_dim: list[torch.Tensor] | None = None  # raw action dimension, used adding masks to loss calculation


@dataclass(slots=True)
class GenerationDataNoised:
    """Container for states after noise addition, along with other
    helper attributes for the flow-matching (gt velocity and noise)
    for the multi-modal (vision, sound, action) MoT training.
    Used for the VFM generation model.
    """

    batch_size: int
    # Vision
    epsilon_vision: torch.Tensor  # unit gaussian noise tensor
    xt_tokens_vision: torch.Tensor  # tokens added with noise level t per flow-matching formulation
    vt_target_vision: torch.Tensor  # gt rectified flow field
    sigmas_vision: torch.Tensor | None = None  # SNR to add to the vision tokens

    # Audio (Sound)
    epsilon_sound: torch.Tensor | None = None
    xt_tokens_sound: torch.Tensor | None = None
    vt_target_sound: torch.Tensor | None = None
    sigmas_sound: torch.Tensor | None = None

    # Action
    epsilon_action: torch.Tensor | None = None
    xt_tokens_action: torch.Tensor | None = None
    vt_target_action: torch.Tensor | None = None
    sigmas_action: torch.Tensor | None = None
    raw_action_dim: list[torch.Tensor] | None = None  # raw action dimension, used adding masks to loss calculation


def unwrap_and_densify(raw: list | torch.Tensor | None, to_kwargs: dict) -> list[torch.Tensor] | None:
    """Unwrap nested single-element lists and filter ``None`` entries.

    The joint dataloader can produce data as nested single-element lists
    (e.g. ``[[t1], [None], [t2]]``).  This helper flattens the nesting,
    drops ``None`` entries, and moves the remaining tensors to the target
    device/dtype.

    Args:
        raw: The raw value from ``data_batch``.  May be ``None``, a bare
            tensor, or a (possibly nested) list of tensors / ``None`` s.
            Each tensor in the list has shape ``(...)``.
        to_kwargs: Keyword arguments forwarded to ``torch.Tensor.to``
            (e.g. ``{"device": "cuda"}`` or ``{"device": "cuda", "dtype": torch.bfloat16}``).

    Returns:
        A dense list of device tensors each with shape ``(...)``, or ``None``
        if the input is ``None`` or every entry is ``None``.

    Examples:
        >>> unwrap_and_densify([[t1], [None], [t2]], {"device": "cuda"})
        [t1.cuda(), t2.cuda()]
        >>> unwrap_and_densify(None, {"device": "cuda"})
        None
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        return [raw.to(**to_kwargs)]  # list of 1 tensor: [(...)]
    # Unwrap single-element inner lists: [[t], [None]] -> [t, None]
    if len(raw) > 0 and isinstance(raw[0], list):
        raw = [item[0] if isinstance(item, list) else item for item in raw]
    # Filter None entries and move to device
    dense = [x.to(**to_kwargs) for x in raw if x is not None]  # list of B tensors: [(...), ...]
    return dense if dense else None


def _expand_per_sample_to_per_vision_item(
    tensor: torch.Tensor,  # [B,...]
    num_vision_items_per_sample: list[int] | None,
) -> torch.Tensor:  # [N_vision_items,...]
    """Expand a per-sample tensor to a per-vision-item tensor.

    For image editing, each sample may contribute multiple vision items
    (e.g. source + target).  This helper repeats each sample's value for
    all of its vision items so that downstream indexing by vision-item
    position works correctly.

    Args:
        tensor: Per-sample tensor of shape ``(N, ...)``.
        num_vision_items_per_sample: Number of vision items per sample,
            e.g. ``[2, 2, ...]``.  If ``None``, the tensor is returned as-is
            (standard single-item-per-sample case).

    Returns:
        Tensor of shape ``(sum(num_vision_items_per_sample), ...)``, or the
        original tensor when ``num_vision_items_per_sample`` is ``None``.
    """
    if num_vision_items_per_sample is None:
        return tensor  # [B,...]
    expanded = []
    for sample_idx, num_items in enumerate(num_vision_items_per_sample):
        for _ in range(
            num_items
        ):  # torch.stack(tensor[idx].repeat(num_vision_items_per_sample[idx]) for idx in range(len(num_vision_items_per_sample)))
            expanded.append(tensor[sample_idx])  # [...]
    return torch.stack(expanded)  # [N_vision_items,...]
