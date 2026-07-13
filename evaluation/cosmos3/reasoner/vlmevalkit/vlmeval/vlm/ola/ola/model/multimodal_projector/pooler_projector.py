import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.clip.modeling_clip import CLIPVisionModel


class PoolerProjector(nn.Module):
    def __init__(self, config, vision_cfg):
        super().__init__()
        self._config = config
        self.hw = vision_cfg.image_size // vision_cfg.patch_size

        self.conv_pool = nn.Conv2d(
            config.mm_hidden_size, config.hidden_size,
            kernel_size=2, stride=2
        )

        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        height = width = self.hw
        assert height * width == x.shape[1]
        x = x.view(x.shape[0], height, width, -1).permute(0, 3, 1, 2)
        x = self.conv_pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'pooler'}


class NormalizedDwPooler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.predictor = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, forward_type='2x'):
        B, H, W, C = x.shape

        if forward_type == '2x':
            new_x = x.reshape(B, H//2, 2, W//2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H//2, W//2, 4, C)
            pooled_x = new_x.mean(-2, keepdim=True).expand(-1, -1, -1, 4, -1)
            fused_x = torch.cat([new_x, pooled_x], dim=-1)
        elif forward_type == '1x':
            new_x = x.reshape(B, H, W, 1, C)
            fused_x = torch.cat([new_x, new_x], dim=-1)
        elif forward_type == '4x':
            new_x = x.reshape(B, H//4, 4, W//4, 4, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H//4, W//4, 16, C)
            pooled_x = new_x.mean(-2, keepdim=True).expand(-1, -1, -1, 16, -1)
            fused_x = torch.cat([new_x, pooled_x], dim=-1)

        score = self.predictor(fused_x)
        normalized_score = F.softmax(score, dim=-2)
        new_x = (new_x * normalized_score).sum(dim=-2)
        return new_x
