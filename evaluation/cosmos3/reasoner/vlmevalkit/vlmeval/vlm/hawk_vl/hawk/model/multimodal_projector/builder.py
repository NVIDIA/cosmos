import re

import torch
import torch.nn as nn


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


# copied from modeling_qwen2_vl.py
# similar to the pixel unshuffle operation in InternVL
class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class PoolingMerger(nn.Module):
    def __init__(self, pool_type, kernel_size):
        super().__init__()

        self.pooling_layer = nn.AvgPool1d(kernel_size) if pool_type == 'avgpool' else nn.MaxPool1d(kernel_size)

    def forward(self, x):
        return self.pooling_layer(x)


def build_vision_projector(config, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'pixel_unshuffle')
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    # add pooling & pixel unshuffle.
    if projector_type == 'pixel_unshuffle':
        return PatchMerger(config.lm_hidden_size, config.embed_dim, config.spatial_merge_size)

    if projector_type in ['avgpool', 'maxpool']:
        return PoolingMerger(projector_type, config.spatial_merge_size)

    raise ValueError(f'Unknown projector type: {projector_type}')
