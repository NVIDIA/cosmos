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

"""Nemotron-H style text modules for Nemotron 3 Dense VL (ReLU^2 MLP, partial RoPE helper, mRoPE)."""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

if "relu2" not in ACT2FN:
    ACT2FN["relu2"] = lambda x: F.relu(x).square()
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel

from cosmos.model.vfm.vlm.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_partial(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to the first rot_dim channels; remainder passes through (rot_dim == head_dim for 2B Dense)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat((q_embed, q_pass), dim=-1), torch.cat((k_embed, k_pass), dim=-1)


class Nemotron3DenseVLRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Nemotron3DenseVLMLP(nn.Module):
    def __init__(self, config: Nemotron3DenseVLTextConfig, layer_idx: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        if config.mlp_hidden_act in ACT2FN:
            self.act_fn = ACT2FN[config.mlp_hidden_act]
        else:
            self.act_fn = lambda x: F.relu(x).square()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class MultiModalRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Nemotron3DenseVLTextConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.mrope_section = getattr(config, "mrope_section", [24, 20, 20])
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Nemotron3DenseVLTextConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        rope_theta = config.rope_theta
        dim = config.head_dim
        attention_factor = 1.0
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).to(dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    def apply_interleaved_mrope(self, freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(self.config, buffer_device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)


class Nemotron3DenseVLPreTrainedModel(PreTrainedModel):
    config_class = Nemotron3DenseVLTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module, buffer_device: torch.device | None) -> None:
        super()._init_weights(module)
        if isinstance(module, MultiModalRotaryEmbedding):
            module.init_weights(buffer_device=buffer_device)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        self.apply(functools.partial(self._init_weights, buffer_device=buffer_device))
