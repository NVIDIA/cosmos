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
MoE Specialization Callback
============================
Monitors whether MoE experts are developing distinct, stable roles over training.
A well-trained MoE should have experts that specialize — each processing a different
kind of input — rather than a few generalist experts doing everything while the rest
idle.

  Expert Co-activation Rate
  -------------------------
  If two experts frequently fire together on the same token (both in the top-K
  selected), they are likely learning redundant representations. Ideally experts
  specialize on non-overlapping token types, so co-activation should stay close
  to the chance baseline of K/N (e.g. 8/128 ≈ 0.0625 for the 235B model).

  For each layer and each unique expert pair (i, j), we compute:
      CoAct(i, j) = N_{i,j} / N_i
  where N_{i,j} = number of tokens where both i and j were selected, and N_i =
  total tokens routed to expert i. We then summarize across all pairs as max and
  mean. A rising mean_coact, especially well above the chance baseline, signals
  that the router is collapsing onto a small correlated cluster of experts.

Buffer ownership
----------------
  coactivation_counts is reset here (in compute_moe_coactivation_metrics).
  Per-expert token counts are derived from coactivation_counts itself
  (row_sum + col_sum) / (K-1), so this callback is fully independent of
  ExpertHeatmap's reset cycle for total_tokens_per_expert.
"""

import torch
import wandb
from torch.distributed.tensor import DTensor, Partial

from cosmos3._src.imaginaire.callbacks.every_n import EveryN
from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.trainer import ImaginaireTrainer
from cosmos3._src.imaginaire.utils import distributed
from cosmos3._src.vfm.models.vlm.qwen3_vl_moe.qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock


def _get_device_mesh(vfm: torch.nn.Module):
    weight = vfm.language_model.model.layers[0].self_attn.q_proj.weight
    return weight.device_mesh if isinstance(weight, DTensor) else None


def _allreduce_dtensor(t: torch.Tensor, device_mesh) -> torch.Tensor:
    """Sum-reduce a local tensor across all FSDP ranks and return the global tensor."""
    return DTensor.from_local(
        t,
        device_mesh=device_mesh,
        placements=[Partial()] * device_mesh.ndim,
    ).full_tensor()


def compute_moe_coactivation_metrics(vfm: torch.nn.Module) -> dict[str, dict]:
    """
    Compute per-layer Expert Co-activation metrics for both towers.

    For each unique expert pair (i < j) in the upper triangle of the N×N
    coactivation matrix, computes:
        CoAct(i, j) = N_{i,j} / N_i
    where N_{i,j} is the count of tokens where both i and j were in the top-K,
    and N_i is the total token count for expert i (the row expert, i.e. the
    lower-indexed expert in the pair).

    N_i is derived directly from the co-activation matrix rather than from
    the shared total_tokens_per_expert buffer, so this metric is independent
    of ExpertHeatmap's reset cycle.  Each token routed to expert i contributes
    to (K-1) co-activation pairs, so N_i = (row_sum_i + col_sum_i) / (K-1).

    High co-activation relative to the chance baseline (K/N) indicates that
    certain expert pairs are systematically selected together — a sign of
    redundancy rather than specialization.

    Returns a dict: tower -> {
        "layer_indices":  list[int]          — actual model layer positions
        "max_coact":      Tensor[num_moe_layers]  — worst pair per layer
        "mean_coact":     Tensor[num_moe_layers]  — average over all pairs
        "chance_baseline": float             — K/N, same for all layers (reference)
    }
    """
    with torch.no_grad():
        device_mesh = _get_device_mesh(vfm)
        if device_mesh is None:
            return {}

        results: dict[str, dict] = {}
        for tower in ["und", "gen"]:
            layer_indices, max_coacts, mean_coacts, chance_baselines = [], [], [], []

            num_layers = len(vfm.language_model.model.layers)
            for layer_idx in range(num_layers):
                layer = vfm.language_model.model.layers[layer_idx]
                mlp = layer.mlp if tower == "und" else getattr(layer, "mlp_moe_gen", None)
                if not isinstance(mlp, Qwen3VLMoeTextSparseMoeBlock):
                    continue

                coact_counts = _allreduce_dtensor(mlp.get_coactivation_counts(reset=True), device_mesh)  # [N, N]

                n = mlp.num_experts
                k = mlp.top_k

                # Derive per-expert token counts directly from the co-activation
                # matrix so we don't depend on ExpertHeatmap's reset cycle.
                # Each token that routes to expert i contributes (K-1) entries
                # across row i and column i of the upper-triangle matrix.
                tokens_per_expert = (coact_counts.sum(dim=1) + coact_counts.sum(dim=0)).float() / (k - 1)

                mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=coact_counts.device), diagonal=1)
                # CoAct(i, j) = N_{i,j} / N_i — normalise by how often expert i fires overall.
                denom = tokens_per_expert.unsqueeze(1).clamp(min=1)  # [N, 1]
                coact_rates = (coact_counts.float() / denom)[mask]  # [N*(N-1)/2]

                layer_indices.append(layer_idx)
                max_coacts.append(coact_rates.max())
                mean_coacts.append(coact_rates.mean())
                # Chance baseline = probability two randomly-chosen top-K slots land on the
                # same pair under uniform routing = K/N. Constant across layers and steps,
                # logged once per tower as a reference line.
                chance_baselines.append(k / n)

            if layer_indices:
                results[tower] = {
                    "layer_indices": layer_indices,
                    "max_coact": torch.stack(max_coacts),
                    "mean_coact": torch.stack(mean_coacts),
                    "chance_baseline": chance_baselines[0],  # same value for all layers
                }

    return results


class MoESpecializationCallback(EveryN):
    """
    Logs per-layer MoE specialization metrics to W&B every N training steps.

    What it captures
    ----------------
    Whether MoE experts are developing distinct routing identities:

    Expert Co-activation (logged every N steps)
      - mean_coact / max_coact per layer: how often expert pairs fire together
        relative to the chance_baseline (K/N). Values well above the baseline
        suggest the router is selecting a redundant cluster of experts rather
        than a diverse set.

    W&B layout
    ----------
    moe_specialization/coact_chance_baseline/<tower>         — flat reference (K/N)
    moe_specialization/max_coact/<tower>/layer_NNN|mean|max
    moe_specialization/mean_coact/<tower>/layer_NNN|mean|max

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
        vfm = model.net

        coact_results = compute_moe_coactivation_metrics(vfm)

        if not (distributed.is_rank0() and wandb.run):
            return

        log_dict: dict[str, float] = {}

        for tower, tower_metrics in coact_results.items():
            layer_indices = tower_metrics.pop("layer_indices")
            chance_baseline = tower_metrics.pop("chance_baseline")
            log_dict[f"moe_specialization/coact_chance_baseline/{tower}"] = chance_baseline
            for metric_name, values in tower_metrics.items():
                for layer_idx, val in zip(layer_indices, values):
                    log_dict[f"moe_specialization/{metric_name}/{tower}/layer_{layer_idx:03d}"] = val.item()
                log_dict[f"moe_specialization/{metric_name}/{tower}/mean"] = values.mean().item()
                log_dict[f"moe_specialization/{metric_name}/{tower}/max"] = values.max().item()

        wandb.log(log_dict, step=iteration)
