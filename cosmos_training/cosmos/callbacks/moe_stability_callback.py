# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""
MoE Stability Callback
======================
Monitors whether the MoE router is staying healthy over the course of training.
A healthy router distributes tokens reasonably evenly, keeps all experts alive,
and remains uncertain enough (high entropy) that it is still learning to route.

Three metrics are tracked per layer, per tower (und / gen):

  Dead Expert Rate
  ----------------
  Fraction of experts receiving fewer than 10% of their fair-share of tokens
  (i.e. load fraction f_i < 0.1 / N). A dead expert has been effectively shut
  out by the router — it gets no gradient signal and its capacity is wasted.
  Ideal = 0. A rising dead-expert rate in the gen tower during early training
  is a common failure mode.

  Load Imbalance Factor (LIF)
  ---------------------------
  N * max(f_i), where f_i is the fraction of tokens routed to expert i.
  Measures how much the busiest expert is overloaded relative to uniform.
  LIF = 1.0 is perfect balance; <= 1.3 is healthy; > 3.0 indicates severe
  collapse onto a small set of experts. This is the same quantity watched by
  the load-balancing loss, but measured empirically rather than from the loss
  objective.

  Router Entropy (normalized)
  ---------------------------
  Mean per-token Shannon entropy of the full routing distribution, divided by
  log(N) to put it on a [0, 1] scale. H = 1 means the router is maximally
  uncertain (uniform over all experts); H = 0 means it always picks the same
  expert. Early in training entropy is high; we want it to stay reasonably
  high (> ~0.7) so the router continues to explore. A sudden drop signals
  routing collapse.

Buffer ownership
----------------
  This callback is fully self-contained: it reads and resets its own dedicated
  buffers (stability_tokens_per_expert, stability_total_tokens, sum_token_entropy).
  It does not depend on ExpertHeatmap's reset cycle.
"""

import math

# Fraction of uniform fair-share below which an expert is considered "dead" (e.g. 0.1 → < 10% of K/N).
DEAD_EXPERT_THRESHOLD_MULTIPLIER = 0.1

import torch
import wandb
from torch.distributed.tensor import DTensor, Partial

from cosmos.callbacks.every_n import EveryN
from cosmos.model._base import ImaginaireModel
from cosmos.trainer import ImaginaireTrainer
from cosmos.utils import distributed
from cosmos.model.vfm.vlm.qwen3_vl_moe.qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock


def compute_moe_stability_metrics(vfm: torch.nn.Module) -> dict[str, dict]:
    """
    Compute per-layer MoE stability metrics for both towers.

    Iterates over all model layers, skipping any that do not use
    Qwen3VLMoeTextSparseMoeBlock (e.g. dense layers when decoder_sparse_step > 1).
    Actual model layer indices are preserved so W&B keys (layer_000, layer_042, ...)
    always refer to the correct transformer layer regardless of MoE sparsity pattern.

    Returns a dict: tower -> {
        "layer_indices":            list[int]          — actual model layer positions
        "dead_expert_rate":         Tensor[num_moe_layers]
        "lif":                      Tensor[num_moe_layers]
        "router_entropy_normalized":Tensor[num_moe_layers]
    }
    """
    with torch.no_grad():
        num_layers = len(vfm.language_model.model.layers)

        example_weight = vfm.language_model.model.layers[0].self_attn.q_proj.weight
        device_mesh = example_weight.device_mesh if isinstance(example_weight, DTensor) else None

        if device_mesh is None:
            return {}

        def _allreduce(t: torch.Tensor) -> torch.Tensor:
            return DTensor.from_local(
                t,
                device_mesh=device_mesh,
                placements=[Partial()] * device_mesh.ndim,
            ).full_tensor()

        results: dict[str, dict] = {}
        for tower in ["und", "gen"]:
            layer_indices, dead_rates, lifs, entropies = [], [], [], []

            for layer_idx in range(num_layers):
                layer_module = vfm.language_model.model.layers[layer_idx]
                # "und" tower uses layer.mlp; "gen" tower uses layer.mlp_moe_gen.
                # Both attributes exist on every layer (set in unified_mot.py), but only
                # layers where (layer_idx+1) % decoder_sparse_step == 0 are MoE blocks.
                mlp_module = layer_module.mlp if tower == "und" else getattr(layer_module, "mlp_moe_gen", None)
                if not isinstance(mlp_module, Qwen3VLMoeTextSparseMoeBlock):
                    continue

                total_tokens_per_expert = _allreduce(mlp_module.get_stability_tokens_per_expert(reset=True))
                total_tokens = _allreduce(mlp_module.get_stability_total_tokens(reset=True))
                sum_token_entropy = _allreduce(mlp_module.get_sum_token_entropy(reset=True))

                n = mlp_module.num_experts
                total = total_tokens.float().clamp(min=1)
                f_i = total_tokens_per_expert.float() / total  # [N] load fraction per expert

                k = mlp_module.top_k

                layer_indices.append(layer_idx)
                # Uniform fair share per expert is K/N.  "Dead" = below 10% of that.
                dead_rates.append((f_i < DEAD_EXPERT_THRESHOLD_MULTIPLIER * k / n).float().mean())
                # LIF = max(f_i) * N / K.  Interpretation:
                #   1.0 = perfectly balanced (every expert gets its fair share)
                #   2.0 = busiest expert handles 2x its fair share
                #   >3.0 = severe imbalance, consider tuning load-balancing loss
                lifs.append(f_i.max() * n / k)
                # Mean per-token entropy, normalized to [0, 1] by log(N).
                # squeeze() collapses the [1] buffer shape to a 0-d scalar.
                entropies.append((sum_token_entropy.float() / total / math.log(n)).squeeze())

            if layer_indices:
                results[tower] = {
                    "layer_indices": layer_indices,
                    "dead_expert_rate": torch.stack(dead_rates),
                    "lif": torch.stack(lifs),
                    "router_entropy_normalized": torch.stack(entropies),
                }

    return results


class MoEStabilityCallback(EveryN):
    """
    Logs per-layer MoE stability metrics to W&B every N training steps.

    What it captures
    ----------------
    Whether the MoE router remains in a healthy, balanced state over training.
    The three metrics collectively answer: are all experts still being used
    (dead_expert_rate), is load spread evenly (lif), and is the router still
    making uncertain, exploratory decisions (router_entropy_normalized)?

    W&B layout
    ----------
    For each metric and each tower, two kinds of series are logged:
      - moe_stability/<metric>/<tower>/layer_NNN  — per model layer time series
      - moe_stability/<metric>/<tower>/mean|max   — summary across all MoE layers

    Typical healthy ranges:
      dead_expert_rate  → 0 (any sustained non-zero value is a concern)
      lif               → <= 1.3 (alarm at > 3.0)
      router_entropy_normalized → > 0.7 (collapse if it drops sharply)

    Args:
        every_n (int): Logging interval in training steps.
    """

    def __init__(self, every_n: int = 100):
        super().__init__(every_n=every_n)

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        metrics = compute_moe_stability_metrics(model.net)

        if not (distributed.is_rank0() and wandb.run):
            return

        log_dict: dict[str, float] = {}
        for tower, tower_metrics in metrics.items():
            layer_indices = tower_metrics.pop("layer_indices")
            for metric_name, values in tower_metrics.items():
                for layer_idx, val in zip(layer_indices, values):
                    log_dict[f"moe_stability/{metric_name}/{tower}/layer_{layer_idx:03d}"] = val.item()
                log_dict[f"moe_stability/{metric_name}/{tower}/mean"] = values.mean().item()
                log_dict[f"moe_stability/{metric_name}/{tower}/max"] = values.max().item()

        wandb.log(log_dict, step=iteration)
