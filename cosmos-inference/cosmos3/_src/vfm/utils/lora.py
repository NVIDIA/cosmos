# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
# -----------------------------------------------------------------------------

"""Custom in-place LoRA injection for MoT-style models.

The key design choice is to subclass ``nn.Linear`` (``LoraInjectedLinear``)
rather than wrap it (as PEFT's ``LoraLayer`` does). The wrapped weight stays
at ``<path>.weight`` so checkpoints saved from a LoRA-trained model load
cleanly into either a LoRA or non-LoRA model — no key rename and no loader
alias needed. ``lora_A`` and ``lora_B`` are sibling submodules, producing
the state-dict keys ``<path>.lora_A.weight`` and ``<path>.lora_B.weight``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos3._src.imaginaire.utils import log


class LoraInjectedLinear(nn.Linear):
    """nn.Linear with sibling lora_A and lora_B that preserves the original ``.weight`` key.

    State-dict keys for a module at ``<path>``:
      ``<path>.weight``        — original Linear weight (unchanged key)
      ``<path>.bias``          — original Linear bias (if present)
      ``<path>.lora_A.weight`` — low-rank down-projection
      ``<path>.lora_B.weight`` — low-rank up-projection

    Forward computes ``y = base(x) + (alpha / r) * lora_B(lora_A(x))``.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: int) -> None:
        # Reuse base's geometry. Inherit nn.Linear so ``super().forward(x)``
        # dispatches to the standard F.linear path.
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device="meta",
        )
        # Replace nn.Linear's freshly-allocated parameters with the base's
        # existing ones. On meta device this is a no-op for memory; on a
        # real device this preserves the pretrained weight identity.
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        # Sibling submodules. Using bias=False nn.Linear gives us a single
        # ``weight`` parameter and a clean ``.lora_A.weight`` state-dict key.
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, device="meta")
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, device="meta")
        self._lora_rank = int(rank)
        self._lora_alpha = float(alpha)

    @property
    def _lora_scale(self) -> float:
        return self._lora_alpha / self._lora_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = self.lora_B(self.lora_A(x))
        return base_out + self._lora_scale * lora_out


def _inject_lora_inplace(
    network: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: int,
) -> int:
    """Replace each ``<target>`` ``nn.Linear`` child in-place with ``LoraInjectedLinear``.

    Match is by exact child name (e.g., ``q_proj_moe_gen``), not substring.
    Snapshots ``named_modules()`` before mutating the tree so newly-inserted
    LoRA submodules are not re-visited.
    """
    target_set = set(target_modules)
    replaced = 0
    for _parent_name, parent in list(network.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in target_set and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoraInjectedLinear(child, rank, alpha))
                replaced += 1
    return replaced


def inject_lora_pre_fsdp(
    network: torch.nn.Module,
    *,
    lora_rank: int,
    lora_alpha: int,
    lora_target_modules: str,
) -> torch.nn.Module:
    """Inject LoRA adapters into ``network`` BEFORE FSDP wrap on meta device.

    Must be called on the meta-device network (pre-FSDP) so the injector sees
    unsharded weight shapes; injecting after FSDP causes ``lora_B`` to be
    constructed with the per-rank shard size (e.g., 8192/8=1024) and triggers
    a shape mismatch at forward time.

    ``lora_A`` and ``lora_B`` parameters are left uninitialized on meta;
    the caller must initialize them AFTER
    ``to_empty(device="cuda") + init_weights(buffer_device="cuda")``
    via ``init_lora_weights_post_materialization``.

    Also freezes every non-LoRA parameter so the optimizer's
    ``keys_to_select=["lora_"]`` filter trains adapters only.
    """
    assert network is not None, "Network is not initialized"

    if lora_rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {lora_rank}")
    if lora_alpha <= 0:
        raise ValueError(f"LoRA alpha must be positive, got {lora_alpha}")

    target_modules_list = [m.strip() for m in lora_target_modules.split(",") if m.strip()]
    if not target_modules_list:
        raise ValueError("LoRA target_modules cannot be empty")

    model_module_names = {name.split(".")[-1] for name, _ in network.named_modules()}
    invalid_modules = [t for t in target_modules_list if t not in model_module_names]
    if invalid_modules:
        log.warning(f"LoRA target modules not found in model: {invalid_modules}")

    log.info(f"Injecting LoRA on meta device: rank={lora_rank}, alpha={lora_alpha}, targets={target_modules_list}")

    try:
        replaced = _inject_lora_inplace(network, target_modules_list, lora_rank, lora_alpha)
    except Exception as e:
        raise RuntimeError(f"Failed to inject LoRA adapters into model: {e}") from e

    if replaced == 0:
        log.warning(f"LoRA injection replaced 0 modules — check lora_target_modules={lora_target_modules!r}")

    lora_params = 0
    frozen_params = 0
    for name, param in network.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
            lora_params += param.numel()
        else:
            param.requires_grad_(False)
            frozen_params += param.numel()

    log.info(
        f"LoRA injection successful: {replaced} modules wrapped, "
        f"{lora_params:,} trainable LoRA params, "
        f"{frozen_params:,} frozen base params "
        f"({100 * lora_params / max(1, lora_params + frozen_params):.3f}% trainable)"
    )
    return network


def init_lora_weights_post_materialization(network: torch.nn.Module) -> None:
    """Initialize LoRA params after ``to_empty + init_weights`` materializes them.

    The custom injector leaves lora_A/lora_B as uninitialized meta-device
    parameters. After ``to_empty(device=DEVICE)``, they have allocated but
    uninitialized memory. Manually init: ``lora_A ~ kaiming_uniform_(a=sqrt(5))``,
    ``lora_B = zeros``, both upcast to fp32 for stable training.
    """
    for name, param in network.named_parameters():
        if "lora_A" in name:
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            param.data = param.data.to(torch.float32)
        elif "lora_B" in name:
            torch.nn.init.zeros_(param)
            param.data = param.data.to(torch.float32)
