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

"""MFU (Model FLOPs Utilization) callback for OmniMoT training.

Computes and logs MFU metrics for specified hardware targets (e.g. H100, GB200)
by calculating the actual training FLOPs per step and comparing against
theoretical peak throughput.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal

import torch
import wandb

from cosmos.model.attention.utils import is_blackwell_dc
from cosmos.callbacks.every_n import EveryN
from cosmos.tools.flops import (
    OmniMoTModelDescriptor,
    compute_omni_mot_flops_per_batch,
    compute_wan_vae_encoder_flops,
    get_omni_mot_model_descriptor,
)
from cosmos.model._base import ImaginaireModel
from cosmos.trainer import ImaginaireTrainer
from cosmos.utils import log
from cosmos.utils.distributed import rank0_only


@dataclass
class HardwareTarget:
    """Specification of a hardware target for MFU computation.

    Attributes:
        name: Human-readable name (used as W&B tag, e.g. "H100").
        peak_tflops: Theoretical peak throughput in TFLOPS (e.g. 989 for H100 BF16).
    """

    name: str
    peak_tflops: float


# Pre-defined hardware targets
H100 = HardwareTarget(name="H100", peak_tflops=989.0)
GB200 = HardwareTarget(name="GB200", peak_tflops=2250.0)


class MFUCallback(EveryN):
    """Callback that computes and logs Model FLOPs Utilization (MFU) to W&B.

    MFU is defined as:
        MFU = achieved_tflops_per_gpu / peak_tflops_per_gpu

    where achieved_tflops_per_gpu is computed from the model's theoretical
    training FLOPs (forward + backward) divided by the measured wall-clock
    time per step.

    This callback accumulates per-step FLOPs between logging intervals and
    reports the average MFU over that window.

    Args:
        backwardpass_ratio: Ratio of backward-to-forward FLOPs (default 2.0).
        hit_thres: Number of warm-up iterations before logging begins.
        include_vae_encoder: If True (default), include the Wan 2.2 VAE encoder
            forward-pass FLOPs in the per-step total.  The VAE is frozen during
            training so only forward FLOPs are counted.
        include_padding: If True, include FLOPs spent on padding tokens (the
            causal split appended by sequence-packing finalize()).  Gives a
            ``total GPU FLOPs`` view instead of ``useful FLOPs`` only.
        grad_accum_iter: Number of gradient accumulation steps per optimizer
            update (default 1).  When > 1, ``on_training_step_end`` is called
            once per optimizer step but the wall-clock time covers all
            micro-batches, so per-step FLOPs are multiplied by this count.
    """

    def __init__(
        self,
        *args,
        backwardpass_ratio: float = 2.0,
        hit_thres: int = 5,
        include_vae_encoder: bool = True,
        include_padding: bool = True,
        grad_accum_iter: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hardware_target = GB200 if is_blackwell_dc() else H100
        self.backwardpass_ratio = backwardpass_ratio
        self.hit_thres = hit_thres
        self.include_vae_encoder = include_vae_encoder
        self.include_padding = include_padding
        self.grad_accum_iter = grad_accum_iter

        # Lazily initialised from model config on first call
        self._model_descriptor: OmniMoTModelDescriptor | None = None
        self._freeze_und: bool = False
        self._vision_gen: bool = True
        self._action_gen: bool = False
        self._sound_gen: bool = False
        self._world_size: int = 1
        self._use_activation_checkpointing: bool = False

        # Accumulation state between every_n windows
        self._accumulated_flops = Decimal(0)
        self._accumulated_flops_vae = Decimal(0)
        self._steps_in_window: int = 0
        self._window_start_time: float | None = None

        # Warm-up counter
        self._hit_counter: int = 0

    # ------------------------------------------------------------------ #
    # Lazy initialisation from model
    # ------------------------------------------------------------------ #

    def _ensure_initialised(self, model: ImaginaireModel) -> None:
        """Build the ``OmniMoTModelDescriptor`` from the live model config."""
        if self._model_descriptor is not None:
            return

        # Access VLM config from the language model inside the network
        vlm_cfg = model.net.language_model.config  # type: ignore[attr-defined]
        net_cfg = model.net.config  # type: ignore[attr-defined]

        self._freeze_und = getattr(vlm_cfg, "freeze_und", False)
        self._vision_gen = getattr(net_cfg, "vision_gen", True)
        self._action_gen = getattr(net_cfg, "action_gen", False)
        self._sound_gen = getattr(net_cfg, "sound_gen", False)

        # Read activation checkpointing from the model's parallelism config
        model_cfg = getattr(model, "config", None)
        parallelism_cfg = getattr(model_cfg, "parallelism", None)
        self._use_activation_checkpointing = getattr(parallelism_cfg, "use_activation_checkpointing", False)

        # MoE fields (may not exist for dense-only configs)
        use_moe = getattr(vlm_cfg, "use_moe", False)
        num_experts = getattr(vlm_cfg, "num_experts", 0)
        num_experts_per_tok = getattr(vlm_cfg, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(vlm_cfg, "moe_intermediate_size", 0)
        decoder_sparse_step = getattr(vlm_cfg, "decoder_sparse_step", 1)
        mlp_only_layers = list(getattr(vlm_cfg, "mlp_only_layers", []))

        self._model_descriptor = get_omni_mot_model_descriptor(
            hidden_size=vlm_cfg.hidden_size,
            num_hidden_layers=vlm_cfg.num_hidden_layers,
            num_attention_heads=vlm_cfg.num_attention_heads,
            num_key_value_heads=vlm_cfg.num_key_value_heads,
            head_dim=getattr(vlm_cfg, "head_dim", None),
            intermediate_size=vlm_cfg.intermediate_size,
            vocab_size=vlm_cfg.vocab_size,
            use_moe=use_moe,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            decoder_sparse_step=decoder_sparse_step,
            mlp_only_layers=mlp_only_layers,
            latent_patch_size=getattr(net_cfg, "latent_patch_size", 2),
            latent_channel_size=getattr(net_cfg, "latent_channel_size", 48),
            action_dim=getattr(net_cfg, "action_dim", 32),
            sound_dim=getattr(net_cfg, "sound_dim", 64),
            frequency_embedding_size=getattr(net_cfg, "frequency_embedding_size", 256),
            predict_text_tokens=getattr(net_cfg, "predict_text_tokens", False),
        )

        self._world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # ------------------------------------------------------------------ #
    # Per-step accumulation
    # ------------------------------------------------------------------ #

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # Warm-up: skip first few iterations (compilation, allocation, etc.)
        if self._hit_counter < self.hit_thres:
            self._hit_counter += 1
            return

        self._ensure_initialised(model)

        # Start the timing window on the first post-warmup step
        if self._window_start_time is None:
            self._window_start_time = time.monotonic()

        # Extract per-modality token counts from output_batch
        und_token_length = output_batch.get("und_token_length")
        if und_token_length is None:
            return

        und_tokens = int(und_token_length)
        vision_tokens = int(output_batch.get("vision_token_length", 0))
        action_tokens = int(output_batch.get("action_token_length", 0))
        sound_tokens = int(output_batch.get("sound_token_length", 0))

        # Per-split attention metadata for packed sequences
        split_lens: list[int] | None = output_batch.get("split_lens")
        attn_modes_list: list[str] | None = output_batch.get("attn_modes")

        # Compute FLOPs for this per-device micro-batch.
        # B = 1 because token counts are already summed across all samples in
        # the packed sequence on this device.
        assert self._model_descriptor is not None
        step_flops = compute_omni_mot_flops_per_batch(
            cfg=self._model_descriptor,
            B=1,
            text_tokens=und_tokens,
            vision_tokens=vision_tokens,
            action_tokens=action_tokens,
            sound_tokens=sound_tokens,
            freeze_und=self._freeze_und,
            vision_gen=self._vision_gen,
            action_gen=self._action_gen,
            sound_gen=self._sound_gen,
            backwardpass_ratio=self.backwardpass_ratio,
            split_lens=split_lens,
            attn_modes=attn_modes_list,
            include_padding=self.include_padding,
            use_activation_checkpointing=self._use_activation_checkpointing,
        )

        # VAE encoder forward-pass FLOPs (frozen, no backward).
        if self.include_vae_encoder:
            vae_pixel_shapes = output_batch.get("vae_pixel_shapes")
            if vae_pixel_shapes:
                for pT, pH, pW in vae_pixel_shapes:
                    vae_flops = compute_wan_vae_encoder_flops(B=1, T=pT, H=pH, W=pW)
                    self._accumulated_flops_vae += vae_flops
                    step_flops += vae_flops

        # When gradient accumulation is used, on_training_step_end is called
        # once per optimizer step (not per micro-batch).  Multiply by the
        # accumulation count so the FLOPs cover all micro-batches in the step.
        # For VAE with gradient accumulation we assume all micro-batches have the same FLOP count
        if self.grad_accum_iter > 1:
            step_flops *= self.grad_accum_iter

        self._accumulated_flops += step_flops
        self._steps_in_window += 1

        # Delegate to EveryN for the periodic reporting logic
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    # ------------------------------------------------------------------ #
    # Periodic reporting
    # ------------------------------------------------------------------ #

    @rank0_only
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        if self._window_start_time is None or self._steps_in_window == 0:
            return

        elapsed = time.monotonic() - self._window_start_time
        if elapsed <= 0:
            return

        if self._accumulated_flops <= 0:
            log.warning(
                f"Number of calculated FLOPs must be more than 0, got  {self._accumulated_flops} at iteration {iteration} for {self._steps_in_window} steps."
            )

        # Achieved TFLOPS *per GPU* over the window
        # accumulated_flops is the total per-device FLOPs over all steps in window
        achieved_tflops_per_gpu = float(self._accumulated_flops) / elapsed / 1e12

        avg_flops_per_step = float(self._accumulated_flops) / self._steps_in_window
        avg_time_per_step = elapsed / self._steps_in_window

        log_info: dict[str, float] = {
            "mfu/achieved_tflops_per_gpu": achieved_tflops_per_gpu,
            "mfu/avg_flops_per_step": avg_flops_per_step,
            "mfu/avg_time_per_step_s": avg_time_per_step,
            "mfu/steps_in_window": float(self._steps_in_window),
            "mfu/vae_flops_percentage": float(self._accumulated_flops_vae / self._accumulated_flops) * 100.0,
        }

        mfu = (
            achieved_tflops_per_gpu / self.hardware_target.peak_tflops if self.hardware_target.peak_tflops > 0 else 0.0
        )
        log_info[f"mfu/{self.hardware_target.name}"] = mfu

        # W&B log
        if wandb.run is not None:
            wandb.log(log_info, step=iteration)

        # Reset accumulation window
        self._accumulated_flops = Decimal(0)
        self._accumulated_flops_vae = Decimal(0)
        self._steps_in_window = 0
        self._window_start_time = time.monotonic()
