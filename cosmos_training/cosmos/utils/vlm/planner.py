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

from typing import Optional

import torch
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata

from cosmos.utils import log


class RenameLoadPlanner(DefaultLoadPlanner):
    """
    RenameLoadPlanner that renames variables during checkpoint load.
    """

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Optional[Metadata] = None,
        is_coordinator: bool = False,
    ) -> None:
        super().set_up_planner(
            state_dict=state_dict,
            metadata=metadata,
            is_coordinator=is_coordinator,
        )

        self.state_dict = remove_extra_state_in_transformer_engine(self.state_dict)

        if needs_remapping_for_routed_experts(metadata):
            log.critical("Old checkpoint, requires remapping of tensors")
            self.state_dict = remap_state_dict_for_routed_experts(self.state_dict)

        self.state_dict = remap_model_state_for_deepep(self.state_dict, metadata)

        # Do an early check to see if the checkpoint is valid and print the missing
        # keys in the state dict if not. The reason is the original default planner's
        # error message is not helpful enough when the keys are mismatched.
        missing_keys = get_missing_keys(self.state_dict, metadata)

        if missing_keys:
            log.critical(f"Missing keys in checkpoint: {missing_keys}...")
            log.critical(f"Checkpoint keys: {list(metadata.state_dict_metadata)}...")


def get_missing_keys(
    state_dict: dict[str, torch.Tensor],
    metadata: Metadata,
) -> list[str]:
    missing_keys = []
    for fqn, obj in state_dict.items():
        # ignore state_dict keys which do not exist in `state_dict` if strict=False
        if fqn not in metadata.state_dict_metadata:
            missing_keys.append(fqn)
    return missing_keys


def remove_extra_state_in_transformer_engine(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    modified_state_dict: dict[str, torch.Tensor] = {}
    for fqn, obj in state_dict.items():
        # "_extra_state" is an nn.Parameter within transformer_engine.DotProductAttention
        # which is needed for supporting FP8. Since we don't need FP8 support here, we remove
        # keys containing "_extra_state" from the state_dict.
        if "_extra_state" in fqn:
            continue
        modified_state_dict[fqn] = obj
    return modified_state_dict


def needs_remapping_for_routed_experts(metadata: Metadata) -> bool:
    # Check if there is substring "mlp.down_projs" in any key of metadata.state_dict_metadata
    # If yes, do a remapping of state_dict keys
    for key in metadata.state_dict_metadata.keys():
        if "mlp.down_projs" in key:
            # Means this is old checkpoint
            return True
    return False


def remap_state_dict_for_routed_experts(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Remap the state dict by separating the weights for grouped experts as a
    separate module under MoE.

    Args:
        state_dict (Dict[str, torch.Tensor]): The state dict to remap.

    Returns:
        Dict[str, torch.Tensor]: The remapped state dict.
    """

    def unmapped(key):
        """Map new model keys back to old checkpoint keys."""
        import re

        # New checkpoint format.
        moe_pattern = r"^([\w.]*)model\.layers\.(\d+)\.mlp\.experts\.(gate|up|down)_projs$"
        match = re.match(moe_pattern, key)
        if match:
            prefix = match.group(1)
            layer_num = match.group(2)
            proj_type = match.group(3)
            log.info(f"Remapping {key} to {prefix}model.layers.{layer_num}.mlp.{proj_type}_projs")
            # Old checkpoint format.
            return f"{prefix}model.layers.{layer_num}.mlp.{proj_type}_projs"
        return key

    return {unmapped(k): v for k, v in state_dict.items()}


def remap_model_state_for_deepep(
    state_dict: dict[str, torch.Tensor],
    metadata: Metadata,
) -> dict[str, torch.Tensor]:
    """
    Remap the state dict by removing the "gate_and_up_projs" key.
    And add the "gate_projs" and "up_projs" keys to the state dict.
    """
    import re

    missing_keys = get_missing_keys(state_dict, metadata)

    # Check if there is substring "gate_and_up_projs" in any key of missing_keys
    # If yes, do a remapping of state_dict keys
    needs_remapping = any(["gate_and_up_projs" in key for key in missing_keys])
    if not needs_remapping:
        return state_dict

    log.critical("Old checkpoint, requires remapping of gate_and_up_projs")

    new_state_dict = state_dict.copy()
    for key, v in state_dict.items():
        moe_pattern = r"^([\w.]*)model\.layers\.(\d+)\.mlp\.experts\.gate_and_up_projs$"
        match = re.match(moe_pattern, key)
        if match:
            prefix = match.group(1)
            layer_num = match.group(2)
            log.info(f"Remapping {key} to {prefix}model.layers.{layer_num}.mlp.experts.xxx_projs")

            v_1, v_2 = torch.chunk(v, 2, -1)
            new_state_dict[f"{prefix}model.layers.{layer_num}.mlp.experts.gate_projs"] = v_1.transpose(
                1, 2
            )  # [num_experts,out,in]
            new_state_dict[f"{prefix}model.layers.{layer_num}.mlp.experts.up_projs"] = v_2.transpose(
                1, 2
            )  # [num_experts,out,in]
            del new_state_dict[key]
        elif "mlp.experts.down_projs" in key:
            new_state_dict[key] = v.transpose(1, 2)  # [num_experts,out,in]

    return new_state_dict
