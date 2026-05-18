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

Five metrics are tracked per layer, per tower (und / gen):

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

  Soft-vs-Hard Effective Experts (normalized)
  -------------------------------------------
  Soft and hard effective experts separate what the router *considers* (full
  probability distribution, before dispatch) from what top-k dispatch *actually
  uses* (empirical token-to-expert assignment, after dispatch). Both are
  expressed as a fraction of N, so they sit on the same axis as
  router_entropy_normalized. Their lower bounds differ slightly:
    soft_eff_normalized is bounded in [1/N, 1].
    hard_eff_normalized is bounded in [K/N, 1] — top-K dispatch always engages
      at least K experts in aggregate (the floor case is when every token
      picks the same K-expert subset).

      soft_eff_normalized = mean_t exp(H(p_t)) / N
        Average per-token router perplexity, divided by N. Asks: what fraction
        of experts is the router *considering* on a typical token? Computed
        as sum_per_token_soft_eff / total_tokens / N. Note: the unnormalized
        numerator is NOT exp of the mean entropy — by Jensen,
        mean_t exp(H_t) >= exp(mean_t H_t), and the gap matters when
        per-token entropies are heterogeneous.

      hard_eff_normalized = exp(H(f)) / N
        where f_i is the empirical fraction of *expert assignments* (not
        tokens) that went to expert i: f_i = tokens_per_expert_i / (T * K).
        Perplexity of the buffer-wide dispatch distribution, divided by N.
        Asks: what fraction of experts is top-k *actually* engaging across the
        buffer? A smoother sibling of LIF: where LIF watches the busiest
        expert, hard_eff watches the spread of the whole load distribution.

  Interpretation (high/low refer to values close to 1 vs close to 1/N):

      high soft_eff, high hard_eff
        Router considers many experts; top-k dispatch also uses many experts.
        Broadly healthy routing.
      low  soft_eff, low  hard_eff
        Router is confident or collapsed in probability space; dispatch is
        also concentrated. Entropy, LIF, and hard usage all agree that
        routing is narrow.
      high soft_eff, low  hard_eff
        Router distribution is broad, but top-k dispatch is concentrated —
        the "hidden top-k concentration" case where entropy can look healthy
        while LIF and co-activation are high.
      low  soft_eff, high hard_eff
        Less common: each token has a sharp router distribution, but
        different tokens choose different experts. Per-token confidence with
        buffer-wide diversity.

Buffer ownership
----------------
  This callback is fully self-contained: it reads and resets its own dedicated
  buffers (stability_tokens_per_expert, stability_total_tokens, sum_token_entropy,
  sum_per_token_soft_eff). It does not depend on ExpertHeatmap's reset cycle.
"""

import math

# Fraction of uniform fair-share below which an expert is considered "dead" (e.g. 0.1 → < 10% of K/N).
DEAD_EXPERT_THRESHOLD_MULTIPLIER = 0.1

# Smoothing added inside log() to avoid log(0) for experts that received zero
# tokens in the current buffer window. Matches the constant used inside the
# MoE block when accumulating router entropy.
ENTROPY_EPSILON = 1e-9

import torch
import wandb
from torch.distributed.tensor import DTensor, Partial

from cosmos3._src.imaginaire.callbacks.every_n import EveryN
from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.trainer import ImaginaireTrainer
from cosmos3._src.imaginaire.utils import distributed
from cosmos3._src.vfm.models.vlm.qwen3_vl_moe.qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock


def _effective_experts(
    sum_per_token_soft_eff: torch.Tensor,
    total_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (soft_eff, hard_eff) from already-reduced stability buffers.

    Extracted as a pure-tensor function so it can be unit-tested without
    instantiating any MoE module or distributed state.

    Args:
        sum_per_token_soft_eff: 0-d or [1] tensor holding sum_t exp(H(p_t))
            accumulated across the buffer window.
        total_tokens: 0-d or [1] tensor holding the number of tokens seen
            since the last reset.
        tokens_per_expert: [N] tensor of per-expert token counts over the
            same buffer window.

    Returns:
        soft_eff: scalar tensor, mean_t exp(H(p_t)) in [1, N].
        hard_eff: scalar tensor, exp(H(f)) over the empirical dispatch
            distribution f_i = tokens_per_expert_i / sum_i tokens_per_expert_i.
            Bounded in [K, N] (not [1, N]) because top-K dispatch always
            engages at least K experts in aggregate.

    Note on hard_eff normalization:
        tokens_per_expert is a histogram over the K top-k slots per token, so
        it sums to T * K rather than T. We must divide by its own sum (== T*K)
        to get a true probability distribution before taking entropy.
        Dividing by total_tokens (== T) instead would give a vector summing to
        K, producing exp(H) values up to (N/K)^K — orders of magnitude beyond
        the intended [K, N] range.
    """
    total = total_tokens.float().clamp(min=1)
    soft_eff = (sum_per_token_soft_eff.float() / total).squeeze()

    total_assignments = tokens_per_expert.sum().float().clamp(min=1)
    f_i = (tokens_per_expert.float() / total_assignments).clamp(min=ENTROPY_EPSILON)
    hard_entropy = -(f_i * f_i.log()).sum()
    hard_eff = hard_entropy.exp()

    return soft_eff, hard_eff


def compute_moe_stability_metrics(vfm: torch.nn.Module) -> dict[str, dict]:
    """
    Compute per-layer MoE stability metrics for both towers.

    Iterates over all model layers, skipping any that do not use
    Qwen3VLMoeTextSparseMoeBlock (e.g. dense layers when decoder_sparse_step > 1).
    Actual model layer indices are preserved so W&B keys (layer_000, layer_042, ...)
    always refer to the correct transformer layer regardless of MoE sparsity pattern.

    Returns a dict: tower -> {
        "layer_indices":             list[int]                — actual model layer positions
        "dead_expert_rate":          Tensor[num_moe_layers]
        "lif":                       Tensor[num_moe_layers]
        "router_entropy_normalized": Tensor[num_moe_layers]
        "soft_eff_normalized":       Tensor[num_moe_layers]   — mean_t exp(H(p_t)) / N,  in [1/N, 1]
        "hard_eff_normalized":       Tensor[num_moe_layers]   — exp(H(f)) / N,           in [1/N, 1]
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
            layer_indices: list[int] = []
            dead_rates: list[torch.Tensor] = []
            lifs: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            soft_effs_norm: list[torch.Tensor] = []
            hard_effs_norm: list[torch.Tensor] = []

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
                sum_per_token_soft_eff = _allreduce(mlp_module.get_sum_per_token_soft_eff(reset=True))

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

                soft_eff, hard_eff = _effective_experts(
                    sum_per_token_soft_eff=sum_per_token_soft_eff,
                    total_tokens=total_tokens,
                    tokens_per_expert=total_tokens_per_expert,
                )
                soft_effs_norm.append(soft_eff / n)
                hard_effs_norm.append(hard_eff / n)

            if layer_indices:
                results[tower] = {
                    "layer_indices": layer_indices,
                    "dead_expert_rate": torch.stack(dead_rates),
                    "lif": torch.stack(lifs),
                    "router_entropy_normalized": torch.stack(entropies),
                    "soft_eff_normalized": torch.stack(soft_effs_norm),
                    "hard_eff_normalized": torch.stack(hard_effs_norm),
                }

    return results


class MoEStabilityCallback(EveryN):
    """
    Logs per-layer MoE stability metrics to W&B every N training steps.

    What it captures
    ----------------
    Whether the MoE router remains in a healthy, balanced state over training.
    The metrics collectively answer: are all experts still being used
    (dead_expert_rate), is load spread evenly (lif), is the router still
    making uncertain, exploratory decisions (router_entropy_normalized), and
    do the experts the router considers (soft_eff) match the experts top-k
    dispatch actually engages (hard_eff)?

    W&B layout
    ----------
    For each metric and each tower, two kinds of series are logged:
      - moe_stability/<metric>/<tower>/layer_NNN  — per model layer time series
      - moe_stability/<metric>/<tower>/mean|max   — summary across all MoE layers

    Metrics logged: dead_expert_rate, lif, router_entropy_normalized,
    soft_eff_normalized, hard_eff_normalized.

    Typical healthy ranges:
      dead_expert_rate  → 0 (any sustained non-zero value is a concern)
      lif               → <= 1.3 (alarm at > 3.0)
      router_entropy_normalized → > 0.7 (collapse if it drops sharply)
      soft_eff_normalized, hard_eff_normalized → high; a large gap between
        them (e.g. soft high, hard low) indicates hidden top-k concentration

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
