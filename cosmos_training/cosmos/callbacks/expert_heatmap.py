import matplotlib.pyplot as plt
import torch
import wandb
from torch.distributed.tensor import DTensor, Partial

from cosmos.callbacks.every_n import EveryN
from cosmos.model._base import ImaginaireModel
from cosmos.trainer import ImaginaireTrainer
from cosmos.utils import distributed
from cosmos.model.vfm.vlm.qwen3_vl_moe.qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock


def compute_expert_heatmap(vfm: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Compute the heatmap for the MoE blocks in the language model.

    The heatmap is a dictionary with keys set to ["und", "gen"] and values set to
    a tensor of shape (num_layers, num_experts).

    Each element of the tensor is the average number of tokens routed to each expert for a
    given layer. The sum of the elements in each row should be equal to the average number
    of experts per token for the MoE model (config.num_experts_per_tok).

    For dense models, the heatmap is an empty dictionary.
    """
    with torch.no_grad():
        num_layers = len(vfm.language_model.model.layers)

        example_dtensor = vfm.language_model.model.layers[0].self_attn.q_proj.weight
        if isinstance(example_dtensor, DTensor):
            assert hasattr(example_dtensor, "device_mesh")
            device_mesh = example_dtensor.device_mesh
        else:
            device_mesh = None

        expert_heatmaps = {}
        for tower in ["und", "gen"]:
            expert_heatmaps_per_layer = []

            for layer_idx in range(num_layers):
                layer_module = vfm.language_model.model.layers[layer_idx]
                mlp_module = layer_module.mlp if tower == "und" else layer_module.mlp_moe_gen
                if isinstance(mlp_module, Qwen3VLMoeTextSparseMoeBlock):
                    # This is accumulated across all iterations.
                    total_tokens_per_expert = mlp_module.get_total_tokens_per_expert()
                    total_tokens = mlp_module.get_total_tokens()

                    # Compute the average across all ranks.
                    assert device_mesh is not None, "MoE models require multiple GPUs."
                    total_tokens_per_expert = DTensor.from_local(
                        total_tokens_per_expert,
                        device_mesh=device_mesh,
                        placements=[Partial()] * device_mesh.ndim,
                    ).full_tensor()
                    total_tokens = DTensor.from_local(
                        total_tokens,
                        device_mesh=device_mesh,
                        placements=[Partial()] * device_mesh.ndim,
                    ).full_tensor()

                    mean_tokens_per_expert = total_tokens_per_expert.float() / total_tokens.float()  # [num_experts]
                    expert_heatmaps_per_layer.append(mean_tokens_per_expert)

            if len(expert_heatmaps_per_layer) > 0:
                expert_heatmaps[tower] = torch.stack(expert_heatmaps_per_layer, dim=0)  # [num_layers,num_experts]

        return expert_heatmaps


class ExpertHeatmap(EveryN):
    """
    Plots the expert heatmap for the MoE blocks in the language model.

    Args:
        every_n (int): Number of iterations to log the expert heatmap.
    """

    def __init__(self, every_n: int = 1000):
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
        expert_heatmaps = compute_expert_heatmap(model.net)

        if distributed.is_rank0() and wandb.run:
            for tower, heatmap in expert_heatmaps.items():
                fig, ax = plt.subplots()
                im = ax.imshow(heatmap.cpu().numpy())
                ax.set_xlabel("Experts")
                ax.set_ylabel("Layers")
                plt.colorbar(im, ax=ax)
                wandb.log(
                    {
                        f"expert_heatmap/{tower}": fig,
                    },
                    step=iteration,
                )
                plt.close(fig)
