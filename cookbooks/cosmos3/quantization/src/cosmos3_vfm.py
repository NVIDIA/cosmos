# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Cosmos3 video foundation model (VFM) — self-contained PyTorch implementation.

The generation (DiT) transformer used for FP8 post-training quantization: a Qwen3-VL
understanding (UND) tower + a generation (GEN) tower with per-tower Linear projections,
QK-norm, 3D mRoPE and a dense ``[B, C, T, H, W]`` forward. ``Cosmos3VFMTransformer.load_weights``
loads the Diffusers-format ``transformer/`` weights (8B / 32B variants). Depends only on torch.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_MODEL_CONFIGS = {
    "8b": dict(
        hidden_size=4096,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=12288,
    ),
    "32b": dict(
        hidden_size=5120,
        num_hidden_layers=64,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=25600,
    ),
}


class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16) -> None:
        """
        Qwen3VLTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.dtype = dtype

    def post_load_weights(self):
        self.weight.data = self.weight.data.to(self.dtype)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * hidden_states.to(input_dtype)
        return output


def compute_mrope_position_ids_text(
    num_tokens: int,
    temporal_offset: int,
) -> tuple[torch.Tensor, int]:
    """Generate 3D mRoPE position IDs for text tokens.

    Text tokens: all three axes (T, H, W) share the same monotonically
    increasing position IDs: (0,0,0), (1,1,1), (2,2,2), ...

    Returns:
        (position_ids [3, num_tokens], next_temporal_offset)
    """
    ids = torch.arange(num_tokens, dtype=torch.long) + temporal_offset
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    return mrope_ids, temporal_offset + num_tokens


def compute_mrope_position_ids_vision(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
) -> tuple[torch.Tensor, int | float]:
    """Generate 3D mRoPE position IDs for vision tokens.

    Creates a (T, H, W) position grid. Spatial indices reset to 0
    per vision segment (Qwen3VL-style, reset_spatial_indices=True).
    Flattened in T-major order.

    When ``fps`` is provided and ``grid_t > 1``, temporal positions are scaled
    to reflect real time so that videos at different frame rates get comparable
    temporal embeddings.

    Returns:
        (position_ids [3, grid_t * grid_h * grid_w], next_temporal_offset)
    """
    fps_modulation = fps is not None and grid_t > 1

    if fps_modulation:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / temporal_compression_factor
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        t_index = (frame_indices / tps * base_tps + temporal_offset).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            .flatten()
            + int(temporal_offset)
        )

    h_index = (
        torch.arange(grid_h, dtype=torch.long)
        .view(1, -1, 1)
        .expand(grid_t, -1, grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(grid_w, dtype=torch.long)
        .view(1, 1, -1)
        .expand(grid_t, grid_h, -1)
        .flatten()
    )

    if fps_modulation:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    next_offset = math.ceil(mrope_ids.max().item()) + 1
    return mrope_ids, next_offset


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000, target_dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size

        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=target_dtype) / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.frequency_embedding_size)
        torch.nn.init.trunc_normal_(self.mlp[0].weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.mlp[2].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, t):
        # use .float() here if acc loss
        args = t[:, None] * self.freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_emb = self.mlp(t_freq)
        return t_emb


def qwen3_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen3/Llama-style rotate_half: split first/second half of head_dim."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def qwen3_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Qwen3-style RoPE: (x * cos) + (rotate_half(x) * sin).

    Args:
        q: [B, S, H, D]
        k: [B, S, H_kv, D]
        cos: [1, S, 1, D] or broadcastable
        sin: [1, S, 1, D] or broadcastable
    """
    q_embed = (q * cos) + (qwen3_rotate_half(q) * sin)
    k_embed = (k * cos) + (qwen3_rotate_half(k) * sin)
    return q_embed, k_embed


class Cosmos3CausalAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.q_dim = self.num_attention_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.norm_q = Qwen3VLTextRMSNorm(hidden_size=head_dim, dtype=torch.bfloat16)
        self.norm_k = Qwen3VLTextRMSNorm(hidden_size=head_dim, dtype=torch.bfloat16)

        self.to_q = nn.Linear(hidden_size, self.q_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(self.q_dim, hidden_size, bias=False)])

    def apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head RMSNorm on 4D tensors [B, S, H, D]."""
        q = F.rms_norm(q, (q.shape[-1],), self.norm_q.weight, self.norm_q.variance_epsilon)
        k = F.rms_norm(k, (k.shape[-1],), self.norm_k.weight, self.norm_k.variance_epsilon)
        return q, k

    def forward(
        self, 
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor: 
        batch_size, seq_len = hidden_states.shape[:2]

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        q, k = self.apply_qk_norm(q, k)
        q, k = qwen3_apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        k_out, v_out = k, v

        q = q.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True, scale=self.scale)
        out = out.transpose(1, 2).flatten(2)

        return self.to_out[0](out), k_out, v_out
        


class Cosmos3CrossAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.q_dim = self.num_attention_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.scale = 1.0 / math.sqrt(head_dim)


        self.to_q = nn.Linear(hidden_size, self.q_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(self.q_dim, hidden_size, bias=False)])

        self.norm_q = Qwen3VLTextRMSNorm(hidden_size=head_dim, dtype=torch.bfloat16)
        self.norm_k = Qwen3VLTextRMSNorm(hidden_size=head_dim, dtype=torch.bfloat16)

    def apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head RMSNorm on 4D tensors [B, S, H, D]."""
        q = F.rms_norm(q, (q.shape[-1],), self.norm_q.weight, self.norm_q.variance_epsilon)
        k = F.rms_norm(k, (k.shape[-1],), self.norm_k.weight, self.norm_k.variance_epsilon)
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, S_gen, hidden_size] visual tokens
            k_und: [B, S_und, H_kv, D] pre-computed und keys (post-norm, post-RoPE)
            v_und: [B, S_und, H_kv, D] pre-computed und values
            freqs_cos: [B, S_gen, 1, D] cosine part of RoPE
            freqs_sin: [B, S_gen, 1, D] sine part of RoPE

        Returns:
            [B, S_gen, hidden_size] cross-attention output
        """
        batch_size, seq_len_gen = hidden_states.shape[:2]

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        q = q.view(batch_size, seq_len_gen, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len_gen, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len_gen, self.num_key_value_heads, self.head_dim)

        q, k = self.apply_qk_norm(q, k)
        q, k = qwen3_apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        k_all = torch.cat([k_und, k], dim=1).contiguous()
        v_all = torch.cat([v_und, v], dim=1).contiguous()

        q = q.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k_all = k_all.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v_all = v_all.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=False, scale=self.scale, 
                                enable_gqa=True)
        out = out.transpose(1, 2).flatten(2)

        return self.to_out[0](out)


class GatedMLP(nn.Module):
    """SwiGLU MLP with separate gate_proj and up_proj.

    gate_proj and up_proj are kept as independent Linears so quantization
    (FP8/NVFP4) assigns each its own weight_scale / input_scale. Fusing them
    into a single Linear(hidden, 2*inter) would force one shared scale = the
    max amax over both halves, coarsening the smaller-amax half. The vllm-omni
    Cosmos3 serving target (`Cosmos3GatedMLP`) also runs them as two separate
    ColumnParallelLinear GEMMs and loads separate per-projection scales, so
    keeping them split matches the deployment exactly.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Cosmos3UndDecoderLayer(nn.Module):
    """Understanding pathway decoder layer: causal self-attention + MLP."""

    def __init__(self, layer_idx: int, cfg: dict):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = cfg["hidden_size"]
        intermediate_size = cfg["intermediate_size"]

        self.self_attn = Cosmos3CausalAttention(
            hidden_size=hidden_size,
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=128,
            layer_idx=layer_idx,
        )
        self.input_layernorm = Qwen3VLTextRMSNorm(hidden_size=hidden_size, eps=1e-6, dtype=torch.bfloat16)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(hidden_size=hidden_size, eps=1e-6, dtype=torch.bfloat16)
        self.mlp = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=False,
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            (hidden_states, K, V) where K/V are post-QKnorm, post-RoPE
            for consumption by the GEN cross-attention.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        cos, sin = freqs
        attn_out, k, v = self.self_attn(hidden_states, cos, sin)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        B, S, D = hidden_states.shape
        hidden_states = self.mlp(hidden_states.view(-1, D)).view(B, S, D)
        hidden_states = residual + hidden_states

        return hidden_states, k, v


class Cosmos3GenDecoderLayer(nn.Module):
    """Generation pathway decoder layer: cross-attention (to UND K/V) + MLP."""

    def __init__(self, layer_idx: int, cfg: dict):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = cfg["hidden_size"]
        intermediate_size = cfg["intermediate_size"]

        self.cross_attention = Cosmos3CrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=128,
            layer_idx=layer_idx,
        )
        self.input_layernorm = Qwen3VLTextRMSNorm(hidden_size=hidden_size, eps=1e-6, dtype=torch.bfloat16)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(hidden_size=hidden_size, eps=1e-6, dtype=torch.bfloat16)
        self.mlp = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=False,
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        freqs: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        cos, sin = freqs
        hidden_states = self.cross_attention(
            hidden_states, k_und=k_und, v_und=v_und,
            freqs_cos=cos, freqs_sin=sin,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        B, S, D = hidden_states.shape
        hidden_states = self.mlp(hidden_states.view(-1, D)).view(B, S, D)
        hidden_states = residual + hidden_states

        return hidden_states


def _compute_default_rope_parameters() -> tuple["torch.Tensor", float]:
    base = 5000000
    partial_rotary_factor = 1
    head_dim = 128
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))
    return inv_freq, attention_factor


class Qwen3VLTextRotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.rope_type = "default"
        self.max_seq_len_cached = 262144
        self.original_max_seq_len = 262144

        self.mrope_section = [24, 20, 20]

        inv_freq, self.attention_scaling = _compute_default_rope_parameters()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, x, position_ids):
        assert self.inv_freq.dtype == torch.float32, f"inv_freq must be float32, but got {self.inv_freq.dtype}"

        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :]  # shape (3, bs, 1, positions)

        freqs = (inv_freq_expanded @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Cosmos3LanguageModel(nn.Module):
    """Understanding pathway: a standard causal LM that processes text tokens.

    Returns per-layer K/V tensors for the generation pathway's cross-attention.
    The UND pathway is independent of the denoising step, so its K/V can be
    computed once and reused across all sampling steps.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        hidden_size = cfg["hidden_size"]
        num_hidden_layers = cfg["num_hidden_layers"]

        self.embed_tokens = nn.Embedding(151936, hidden_size)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding()
        self.layers = nn.ModuleList([
            Cosmos3UndDecoderLayer(layer_idx=i, cfg=cfg)
            for i in range(num_hidden_layers)
        ])
        # self.norm = Qwen3VLTextRMSNorm(hidden_size=hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        freqs: Tuple[torch.Tensor, torch.Tensor],
    ) -> list[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            text_ids: [B, S] token IDs
            text_mask: [B, S] float mask (1=real, 0=pad)
            freqs: (cos, sin) each [B, S, 1, D] — precomputed UND RoPE

        Returns:
            List of (K, V) per layer. K/V are [B, S, H_kv, D], post-QKnorm
            and post-RoPE, ready for GEN cross-attention.
        """
        hidden = self.embed_tokens(text_ids)
        mask_3d = text_mask.unsqueeze(-1).to(hidden.dtype)  # [B, S, 1]

        cached_kv: list[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            hidden = hidden * mask_3d
            hidden, k, v = layer(hidden, freqs)
            cached_kv.append((k, v))

        return cached_kv


class Cosmos3VFMTransformer(nn.Module):

    def __init__(self, variant: str = "8b"):
        super().__init__()

        if variant not in _MODEL_CONFIGS:
            raise ValueError(f"Unknown variant {variant!r}. Choose from {list(_MODEL_CONFIGS)}")
        cfg = _MODEL_CONFIGS[variant]

        self.hidden_size = cfg["hidden_size"]
        self.num_hidden_layers = cfg["num_hidden_layers"]
        self.latent_patch_size = 2
        self.latent_channel_size = 48
        self.patch_latent_dim = (self.latent_patch_size ** 2) * self.latent_channel_size
        self.timestep_scale = 0.001
        self.base_fps = 24.0
        self.temporal_compression_factor = 4
        self.unified_3d_mrope_temporal_modality_margin = 15000

        self.language_model = Cosmos3LanguageModel(cfg=cfg)

        self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
        self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)

        self.time_embedder = TimestepEmbedder(self.hidden_size, target_dtype=torch.bfloat16)

        self.gen_layers = nn.ModuleList([
            Cosmos3GenDecoderLayer(layer_idx=i, cfg=cfg)
            for i in range(self.num_hidden_layers)
        ])

        self.norm_moe_gen = Qwen3VLTextRMSNorm(
            hidden_size=self.hidden_size, eps=1e-6,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def _pad_to_patch_size(self, H: int, W: int) -> Tuple[int, int, int, int]:
        """Compute padded spatial dims aligned to patch_size.

        Returns (Hp, Wp, H_padded, W_padded) where Hp/Wp are the patch grid
        dimensions and H_padded/W_padded are the padded latent dimensions.
        """
        p = self.latent_patch_size
        H_padded = ((H + p - 1) // p) * p
        W_padded = ((W + p - 1) // p) * p
        return H_padded // p, W_padded // p, H_padded, W_padded

    def patchify(self, latents: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """[B, C, T, H, W] -> [B, T*Hp*Wp, p*p*C], padding H/W if needed."""
        B = latents.shape[0]
        p = self.latent_patch_size
        C = self.latent_channel_size
        Hp, Wp, H_padded, W_padded = self._pad_to_patch_size(H, W)

        if H_padded != H or W_padded != W:
            latents = F.pad(latents, (0, W_padded - W, 0, H_padded - H))

        x = latents.reshape(B, C, T, Hp, p, Wp, p)
        x = x.permute(0, 2, 3, 5, 4, 6, 1)  # [B, T, Hp, Wp, p, p, C]
        return x.reshape(B, T * Hp * Wp, p * p * C)

    def unpatchify(self, tokens: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """[B, T*Hp*Wp, p*p*C] -> [B, C, T, H, W], cropping padding if needed."""
        B = tokens.shape[0]
        p = self.latent_patch_size
        C = self.latent_channel_size
        Hp, Wp, H_padded, W_padded = self._pad_to_patch_size(H, W)

        x = tokens.reshape(B, T, Hp, Wp, p, p, C)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)  # [B, C, T, Hp, p, Wp, p]
        x = x.reshape(B, C, T, H_padded, W_padded)

        if H_padded != H or W_padded != W:
            x = x[:, :, :, :H, :W]
        return x

    def _compute_rope_freqs(
        self,
        text_mask: torch.Tensor,
        T: int, Hp: int, Wp: int,
        fps: float | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Compute mRoPE cos/sin for UND (text) and GEN (visual) pathways."""
        B = text_mask.shape[0]
        S_text = text_mask.shape[1]
        text_lengths = text_mask.sum(dim=1).long()
        effective_fps = fps if fps is not None and T > 1 else None

        text_pos_list = []
        vis_pos_list = []
        for b in range(B):
            real_len = int(text_lengths[b].item())
            t_pos, t_offset = compute_mrope_position_ids_text(real_len, temporal_offset=0)
            v_pos, _ = compute_mrope_position_ids_vision(
                T, Hp, Wp, temporal_offset=t_offset+self.unified_3d_mrope_temporal_modality_margin,
                fps=effective_fps,
                base_fps=self.base_fps,
                temporal_compression_factor=self.temporal_compression_factor,
            )
            if real_len < S_text:
                t_pos = torch.cat([t_pos, torch.zeros(3, S_text - real_len, dtype=t_pos.dtype)], dim=1)
            text_pos_list.append(t_pos)
            vis_pos_list.append(v_pos)

        text_pos_ids = torch.stack(text_pos_list, dim=1).to(device)  # [3, B, S_text]
        vis_pos_ids = torch.stack(vis_pos_list, dim=1).to(device)    # [3, B, S_vis]

        rotary_emb = self.language_model.rotary_emb
        _dummy = torch.tensor([], dtype=dtype, device=device)
        cos_und, sin_und = rotary_emb(_dummy, position_ids=text_pos_ids)
        cos_gen, sin_gen = rotary_emb(_dummy, position_ids=vis_pos_ids)

        freqs_und = (cos_und.unsqueeze(2), sin_und.unsqueeze(2))  # (B, S, 1, 128)
        freqs_gen = (cos_gen.unsqueeze(2), sin_gen.unsqueeze(2))
        return freqs_und, freqs_gen

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        video_shape: Tuple[int, int, int],
        fps: float | None = None,
        noisy_frame_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for parallel denoising.

        Args:
            hidden_states: [B, C, T, H, W] noisy latents
            timestep: [B] diffusion timestep per sample
            text_ids: [B, S_text] tokenized text input
            text_mask: [B, S_text] attention mask for text (1=real, 0=pad)
            video_shape: (T, H, W) in latent space
            fps: video frame rate; when provided, temporal mRoPE positions are
                 scaled to reflect real time (FPS modulation).

        Returns:
            [B, C, T, H, W] velocity prediction
        """
        T, H, W = video_shape
        Hp, Wp, _, _ = self._pad_to_patch_size(H, W)
        max_real_len = text_mask.sum(dim=1).max().item()

        hidden_gen = self.vae2llm(self.patchify(hidden_states, T, H, W))
        
        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            time_embed = self.time_embedder((timestep * self.timestep_scale))

        time_embed = time_embed.to(hidden_states.dtype)
        if noisy_frame_mask is not None:
            # I2V: add the timestep embedding only to noisy tokens; conditioning
            # frames (mask==0, e.g. the clean first frame) get none. Mirrors the
            # production transformer (transformer_cosmos3.py). Token order is
            # frame-major [B, T*Hp*Wp, ...] (see patchify), so expand the per-frame
            # mask over the Hp*Wp patches of each frame.
            token_noisy_mask = (
                noisy_frame_mask[:, 0, :, 0, 0]          # [B, T]
                .unsqueeze(-1)                            # [B, T, 1]
                .expand(-1, -1, Hp * Wp)                  # [B, T, Hp*Wp]
                .reshape(hidden_gen.shape[0], -1, 1)      # [B, T*Hp*Wp, 1]
                .to(hidden_gen.dtype)
            )
            hidden_gen = hidden_gen + time_embed.unsqueeze(1) * token_noisy_mask
        else:
            hidden_gen = hidden_gen + time_embed.unsqueeze(1)

        freqs_und, freqs_gen = self._compute_rope_freqs(
            text_mask, T, Hp, Wp, fps, hidden_states.device, hidden_states.dtype,
        )
        kv_und = self.language_model(text_ids, text_mask, freqs_und)

        for i, layer in enumerate(self.gen_layers):
            k_und, v_und = kv_und[i]
            hidden_gen = layer(hidden_gen, k_und, v_und, freqs_gen)
        
        hidden_gen = self.norm_moe_gen(hidden_gen)
        return self.unpatchify(self.llm2vae(hidden_gen), T, H, W)

    def load_weights(self, weights: dict) -> None:
        """Load weights from a final new-format Cosmos3 checkpoint.

        Source format: ``Cosmos3OmniTransformer`` Diffusers JointAttention export
        (``_diffusers_version`` ``0.37.1``, no ``model.`` prefix on keys). Same
        layout ships from both BYOC mounts and NGC snapshots going forward.

        Legacy ``model.``-prefixed and intermediate exports are NOT supported here
        — pull the prior implementation from git history if needed.

        Maps:
        * Top-level: ``embed_tokens.*``, ``norm.*``, ``norm_moe_gen.*``,
          ``proj_in.*`` / ``proj_out.*`` (-> ``vae2llm.*`` / ``llm2vae.*``),
          ``time_embedder.linear_{1,2}.*`` (-> this model's ``nn.Sequential``
          indices ``time_embedder.mlp.{0,2}.*``).
        * Per-layer (GEN-side patterns listed first so ``*_moe_gen`` /
          ``add_*_proj`` win the ``startswith`` race against the UND substring).
        * Silently dropped: ``lm_head.*``, ``action_*``, ``audio_*``,
          ``*_modality_embed`` (modules not instantiated on this model).
        """
        remapped = {}

        _TOP_MAP = {
            "embed_tokens.":           "language_model.embed_tokens.",
            "norm_moe_gen.":           "norm_moe_gen.",
            "norm.":                   "language_model.norm.",
            "proj_in.":                "vae2llm.",
            "proj_out.":               "llm2vae.",
            # time_embedder: Diffusers naming -> this model's Sequential indices
            # (TimestepEmbedder.mlp is nn.Sequential, NOT named submodules).
            "time_embedder.linear_1.": "time_embedder.mlp.0.",
            "time_embedder.linear_2.": "time_embedder.mlp.2.",
        }

        for key, value in weights.items():
            k = key

            matched_top = False
            for src, dst in _TOP_MAP.items():
                if k.startswith(src):
                    remapped[dst + k[len(src):]] = value
                    matched_top = True
                    break
            if matched_top:
                continue

            if not k.startswith("layers."):
                continue

            parts = k.split(".", 2)  # ['layers', '{i}', '{rest}']
            layer_idx = parts[1]
            rest = parts[2]
            und = f"language_model.layers.{layer_idx}"
            gen = f"gen_layers.{layer_idx}"

            _LAYER_MAP = (
                # GEN attention (Diffusers "added" QKV) -> cross_attention
                ("self_attn.add_q_proj.",             f"{gen}.cross_attention.to_q."),
                ("self_attn.add_k_proj.",             f"{gen}.cross_attention.to_k."),
                ("self_attn.add_v_proj.",             f"{gen}.cross_attention.to_v."),
                ("self_attn.to_add_out.",             f"{gen}.cross_attention.to_out.0."),
                ("self_attn.norm_added_q.",           f"{gen}.cross_attention.norm_q."),
                ("self_attn.norm_added_k.",           f"{gen}.cross_attention.norm_k."),
                # UND attention (Diffusers "to_*") -> self_attn
                ("self_attn.to_q.",                   f"{und}.self_attn.to_q."),
                ("self_attn.to_k.",                   f"{und}.self_attn.to_k."),
                ("self_attn.to_v.",                   f"{und}.self_attn.to_v."),
                ("self_attn.to_out.",                 f"{und}.self_attn.to_out.0."),
                ("self_attn.norm_q.",                 f"{und}.self_attn.norm_q."),
                ("self_attn.norm_k.",                 f"{und}.self_attn.norm_k."),
                # Norms (GEN before UND so *_moe_gen wins)
                ("input_layernorm_moe_gen.",          f"{gen}.input_layernorm."),
                ("post_attention_layernorm_moe_gen.", f"{gen}.post_attention_layernorm."),
                ("input_layernorm.",                  f"{und}.input_layernorm."),
                ("post_attention_layernorm.",         f"{und}.post_attention_layernorm."),
                # MLPs (GEN before UND so mlp_moe_gen wins)
                ("mlp_moe_gen.gate_proj.",            f"{gen}.mlp.gate_proj."),
                ("mlp_moe_gen.up_proj.",              f"{gen}.mlp.up_proj."),
                ("mlp_moe_gen.down_proj.",            f"{gen}.mlp.down_proj."),
                ("mlp.gate_proj.",                    f"{und}.mlp.gate_proj."),
                ("mlp.up_proj.",                      f"{und}.mlp.up_proj."),
                ("mlp.down_proj.",                    f"{und}.mlp.down_proj."),
            )
            for pat, repl in _LAYER_MAP:
                if rest.startswith(pat):
                    remapped[repl + rest[len(pat):]] = value
                    break

        # `GatedMLP` keeps gate_proj and up_proj as separate Linears, so the
        # remapped keys land directly on their own params — no fusion needed.
        for param_name, param in self.named_parameters():
            if param_name in remapped:
                param.data.copy_(remapped[param_name].to(param.dtype))
            else:
                print(f"Warning: no checkpoint weight for {param_name}")

    def post_load_weights(self) -> None:
        """Post-load processing: dtype conversion and Linear finalization."""
        target_dtype = torch.bfloat16

        self.time_embedder.to(torch.float32)
        self.language_model.rotary_emb.to(torch.float32)
        self.language_model.embed_tokens.to(target_dtype)
        self.vae2llm.to(target_dtype)
        self.llm2vae.to(target_dtype)

        for _, module in self.named_modules():
            if isinstance(module, Qwen3VLTextRMSNorm):
                module.post_load_weights()