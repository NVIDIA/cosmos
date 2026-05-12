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

import os
from typing import Optional, Union

import torch
import torch.distributed as dist
import wandb
from torch import nn
from torch.distributed.tensor import DTensor

from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.utils import distributed, log, misc
from cosmos3._src.imaginaire.utils.callback import Callback
from cosmos3._src.imaginaire.utils.distributed import DistributedDataParallel
from cosmos3._src.imaginaire.utils.easy_io import easy_io
from cosmos3._src.vfm.datasets.sequence_packing import get_gen_seq

try:
    from apex.contrib.layer_norm import FastLayerNorm
except ImportError:
    FastLayerNorm = None


class NormMonitor(Callback):
    def __init__(
        self,
        every_n: Optional[int] = None,
        step_size: int = 1,
        layer_norm_only: bool = False,
        model_key: Optional[str] = None,
        log_stat_wandb: bool = False,
        save_s3: bool = False,
        track_activations: bool = False,
    ):
        """Monitor and log parameter/gradient/activation norms during training.

        Args:
            every_n: Log statistics every N global steps. If None, logging is disabled.
            step_size: Number of micro-steps per global step (for gradient accumulation).
            layer_norm_only: If True, only track LayerNorm and Embedding parameters.
                If False, track all parameters.
            model_key: Attribute name to access the model (e.g., "diffusion_model").
                If None, use the model directly.
            log_stat_wandb: If True, log per-parameter statistics to wandb.
                If False, only log aggregate norms.
            save_s3: If True, save statistics to S3 bucket.
            track_activations: If True, track activation norms
                and gradients of activations at each transformer block. If set to False, only
                weight norms and weight gradient norms will be tracked.
        """
        self.every_n = every_n
        self.step_size = step_size
        self.model_key = model_key
        self.layer_norm_only = layer_norm_only
        self.log_stat_wandb = log_stat_wandb
        self.save_s3 = save_s3
        self.track_activations = track_activations
        self.name = self.__class__.__name__

        # Storage for activation statistics (populated by hooks)
        self._activation_stats: dict[str, dict[str, torch.Tensor]] = {}
        self._activation_grad_stats: dict[str, dict[str, torch.Tensor]] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._should_record = False

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/norm_monitor"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"{self.__class__.__name__} callback: local_dir: {self.local_dir}")

        # Register activation hooks if enabled
        if self.track_activations:
            self._register_activation_hooks(model)

    def _register_activation_hooks(self, model: ImaginaireModel) -> None:
        """Register forward and backward hooks on transformer blocks to capture activation statistics.

        Hooks are registered at the block level (on model.model.layers children) rather than
        on individual modules inside blocks. This is compatible with torch.compile since
        compile is applied per-block, and hooks on the outer block fire outside the compiled graph.
        """
        if self.model_key is not None:
            model = getattr(model, self.model_key)

        # Get the transformer layers - hooks are registered on each block
        if not hasattr(model.net.language_model.model, "layers"):
            log.warning(
                f"{self.__class__.__name__}: Could not find model.net.language_model.model.layers. "
                "Activation tracking requires model structure with model.net.language_model.model.layers."
            )
            return

        layers = model.net.language_model.model.layers

        for layer_id, block in layers.named_children():
            block_name = f"blocks.{layer_id}"

            # Forward hook to capture activation norms (block output)
            # Also registers a tensor hook for gradient tracking
            def make_forward_hook(name: str):
                def forward_hook(
                    mod: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor | tuple[torch.Tensor, ...]
                ) -> None:
                    if not self._should_record:
                        return
                    # We track activation norms of only generation sequences.
                    activation = get_gen_seq(out[0])

                    # Certain algorithms do more than one pass through the model.
                    # (E.g. teacher forcing).  We merge stats in that case.
                    new_stats = self._compute_l2_stats(activation)
                    existing = self._activation_stats.get(name)
                    if existing is not None:
                        existing["sq_sum"] += new_stats["sq_sum"]
                        existing["max"] = torch.max(existing["max"], new_stats["max"])
                    else:
                        self._activation_stats[name] = new_stats

                    # Register tensor hook for gradient tracking.
                    # This works with activation checkpointing (unlike module backward hooks).
                    def make_tensor_grad_hook(hook_name: str):
                        def tensor_grad_hook(grad: torch.Tensor | None) -> None:
                            # The block may get gradients internally via attention,
                            # even if the output is unused.
                            if grad is None:
                                return

                            # If there is more than one pass through the model
                            # (e.g. teacher forcing), then merge the stats.
                            new_stats = self._compute_l2_stats(grad)
                            existing = self._activation_grad_stats.get(hook_name)
                            if existing is not None:
                                existing["sq_sum"] += new_stats["sq_sum"]
                                existing["max"] = torch.max(existing["max"], new_stats["max"])
                            else:
                                self._activation_grad_stats[hook_name] = new_stats

                        return tensor_grad_hook

                    if activation.requires_grad:
                        activation.register_hook(make_tensor_grad_hook(name))

                return forward_hook

            forward_handle = block.register_forward_hook(make_forward_hook(block_name))
            self._hooks.append(forward_handle)

        if distributed.is_rank0():
            num_blocks = len(list(layers.named_children()))
            log.info(f"{self.__class__.__name__}: Registered activation hooks on {num_blocks} transformer blocks")

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        """Clean up hooks when training ends."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def on_before_forward(
        self,
        iteration: int = 0,
    ) -> None:
        """Enable activation recording before forward pass if this iteration should be logged."""
        if not self.track_activations:
            return
        global_step = iteration // self.step_size
        should_run = global_step % self.every_n == 0
        self._should_record = should_run
        if should_run:
            # Clear previous activation stats
            self._activation_stats.clear()
            self._activation_grad_stats.clear()

    def on_before_optimizer_step(
        self,
        model_ddp: Union[DistributedDataParallel, ImaginaireModel],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        global_step = iteration // self.step_size
        should_run = global_step % self.every_n == 0
        if not should_run:
            return

        if isinstance(model_ddp, DistributedDataParallel):
            model = model_ddp.module
        else:
            model = model_ddp
        if self.model_key is not None:
            model = getattr(model, self.model_key)

        self._compute_and_log_stats(model, iteration)

        # Disable recording after logging
        self._should_record = False

    def _get_named_parameters(self, model: nn.Module) -> dict[str, nn.Parameter]:
        """Get named parameters, optionally filtered to layer norm only."""
        named_parameters = {}
        if self.layer_norm_only:
            ln_modules = (nn.LayerNorm, nn.Embedding)
            if FastLayerNorm is not None:
                ln_modules += (FastLayerNorm,)
            for mn, m in model.named_modules():
                if isinstance(m, ln_modules):
                    for pn, p in m.named_parameters():
                        fpn = f"{mn}.{pn}" if mn else pn
                        named_parameters[fpn] = p
        else:
            named_parameters = dict(model.named_parameters())
        return named_parameters

    def _should_track_param(self, param_name: str) -> bool:
        """Check if parameter should be tracked based on naming conventions."""
        # Track only generation tower params, exclude EMA params
        return "moe_gen" in param_name and "net_ema" not in param_name

    def _compute_l2_stats(self, tensor: torch.Tensor, detach: bool = True) -> dict[str, torch.Tensor]:
        """Compute statistics (squared sum and max) for a tensor.

        Args:
            tensor: Input tensor to compute statistics for.
            detach: If True, detach the tensor before computing stats.

        Returns:
            Dictionary with "sq_sum" (squared sum for L2 norm) and "max" (absolute max).
        """
        data = tensor.detach() if detach else tensor
        if isinstance(data, DTensor):
            data = data.to_local()

        return {
            "sq_sum": (data.float() ** 2).sum(),
            "max": data.abs().max(),
        }

    @misc.timer("norm_monitor")
    def _compute_and_log_stats(self, model: nn.Module, iteration: int = 0) -> None:
        """FSDP-efficient implementation using local shards + all_reduce.

        Instead of gathering full parameters with summon_full_params (expensive),
        we compute local statistics on each rank's shard and use all_reduce to
        aggregate them across all ranks.
        """
        named_parameters = self._get_named_parameters(model)

        # Accumulators for local shard statistics (squared sum for L2 norm)
        local_param_sq_sum = torch.tensor(0.0, device="cuda", dtype=torch.float32)
        local_grad_sq_sum = torch.tensor(0.0, device="cuda", dtype=torch.float32)

        # Per-parameter stats: {param_name: [local_sq_sum, local_max]}
        per_param_stats: dict[str, dict[str, torch.Tensor]] = {}
        per_grad_stats: dict[str, dict[str, torch.Tensor]] = {}

        for param_name, param in named_parameters.items():
            if not self._should_track_param(param_name):
                continue

            # Compute local statistics on this rank's shard
            per_param_stats[param_name] = self._compute_l2_stats(param)
            local_param_sq_sum += per_param_stats[param_name]["sq_sum"]

            if param.grad is not None:
                per_grad_stats[param_name] = self._compute_l2_stats(param.grad, detach=False)
                local_grad_sq_sum += per_grad_stats[param_name]["sq_sum"]

        # All-reduce to aggregate statistics across all FSDP ranks
        dist.all_reduce(local_param_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_grad_sq_sum, op=dist.ReduceOp.SUM)

        # All-reduce per-parameter stats
        for param_name, stats_dict in per_param_stats.items():
            dist.all_reduce(stats_dict["sq_sum"], op=dist.ReduceOp.SUM)
            dist.all_reduce(stats_dict["max"], op=dist.ReduceOp.MAX)

        for param_name, stats_dict in per_grad_stats.items():
            dist.all_reduce(stats_dict["sq_sum"], op=dist.ReduceOp.SUM)
            dist.all_reduce(stats_dict["max"], op=dist.ReduceOp.MAX)

        # All-reduce activation stats (activations are replicated, so reduce across all ranks for consistency)
        for module_name, stats_dict in self._activation_stats.items():
            dist.all_reduce(stats_dict["sq_sum"], op=dist.ReduceOp.SUM)
            dist.all_reduce(stats_dict["max"], op=dist.ReduceOp.MAX)

        for module_name, stats_dict in self._activation_grad_stats.items():
            dist.all_reduce(stats_dict["sq_sum"], op=dist.ReduceOp.SUM)
            dist.all_reduce(stats_dict["max"], op=dist.ReduceOp.MAX)

        # Only rank 0 logs the results
        if distributed.is_rank0():
            important_info = {
                "trainer/global_step": iteration,
                "sample_counter": getattr(self.trainer, "sample_counter", iteration),
                "total_param_l2_norm": local_param_sq_sum.sqrt().item(),
            }
            if local_grad_sq_sum > 0:
                important_info["total_grad_l2_norm"] = local_grad_sq_sum.sqrt().item()

            stats = {}
            for param_name, stats_dict in per_param_stats.items():
                l2_norm = stats_dict["sq_sum"].sqrt()
                stats[f"stats/weight_norm/{param_name}"] = l2_norm.item()
                stats[f"stats/weight_max/{param_name}"] = stats_dict["max"].item()

            for param_name, stats_dict in per_grad_stats.items():
                l2_norm = stats_dict["sq_sum"].sqrt()
                stats[f"stats/grad_norm/{param_name}"] = l2_norm.item()
                stats[f"stats/grad_max/{param_name}"] = stats_dict["max"].item()

            # Add activation stats
            for module_name, stats_dict in self._activation_stats.items():
                l2_norm = stats_dict["sq_sum"].sqrt()
                stats[f"stats/act_norm/{module_name}"] = l2_norm.item()
                stats[f"stats/act_max/{module_name}"] = stats_dict["max"].item()

            for module_name, stats_dict in self._activation_grad_stats.items():
                l2_norm = stats_dict["sq_sum"].sqrt()
                stats[f"stats/act_grad_norm/{module_name}"] = l2_norm.item()
                stats[f"stats/act_grad_max/{module_name}"] = stats_dict["max"].item()

            if wandb.run is not None:
                if self.log_stat_wandb:
                    wandb.log({**stats, **important_info}, step=iteration)
                else:
                    wandb.log(important_info, step=iteration)

            if self.save_s3:
                easy_io.dump({**stats, **important_info}, f"s3://rundir/{self.name}/stats_{iteration:09d}.pt")
