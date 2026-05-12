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

"""FLOPs estimation for the OmniMoT (Mixture-of-Tokens) dual-pathway transformer."""

from decimal import Decimal
from typing import NamedTuple

from cosmos3._src.imaginaire.utils import log


class OmniMoTModelDescriptor(NamedTuple):
    """
    Holds information about the OmniMoT model architecture needed for the custom flops formula.

    This captures the dual-pathway (MoT) transformer with support for vision, action, and sound
    modalities, and optional Mixture of Experts (MoE) layers.
    """

    # LLM / Transformer core
    hidden_size: int  # D: hidden dimension of the transformer (e.g. 2048, 3584)
    num_hidden_layers: int  # number of transformer decoder layers
    num_attention_heads: int  # number of Q heads
    num_key_value_heads: int  # number of K/V heads (GQA when < num_attention_heads)
    head_dim: int  # dimension per head
    intermediate_size: int  # dense MLP intermediate size (gate_proj / up_proj output dim)
    vocab_size: int  # vocabulary size for embed_tokens and lm_head

    # MoE parameters
    use_moe: bool  # whether MoE layers are used
    num_experts: int  # total number of experts per MoE layer
    num_experts_per_tok: int  # top-k experts activated per token
    moe_intermediate_size: int  # intermediate size inside each expert
    decoder_sparse_step: int  # every `decoder_sparse_step`-th layer is MoE
    mlp_only_layers: list[int]  # layers forced to use dense MLP even in MoE config

    # Vision modality
    latent_patch_size: int  # spatial patch size for latent patchification (default 2)
    latent_channel_size: int  # number of channels in the VAE latent (default 16)

    # Action modality
    action_dim: int  # action token dimension (default 32)

    # Sound modality
    sound_dim: int  # sound token dimension

    # TimestepEmbedder
    frequency_embedding_size: int  # sinusoidal frequency embedding dim (default 256)

    # Text prediction
    predict_text_tokens: bool  # whether lm_head is applied for text CE loss


def get_omni_mot_model_descriptor(
    hidden_size: int = 2048,
    num_hidden_layers: int = 24,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 16,
    head_dim: int | None = None,
    intermediate_size: int = 5632,
    vocab_size: int = 151936,
    use_moe: bool = True,
    num_experts: int = 60,
    num_experts_per_tok: int = 4,
    moe_intermediate_size: int = 1408,
    decoder_sparse_step: int = 1,
    mlp_only_layers: list[int] | None = None,
    latent_patch_size: int = 2,
    latent_channel_size: int = 16,
    action_dim: int = 32,
    sound_dim: int = 64,
    frequency_embedding_size: int = 256,
    predict_text_tokens: bool = False,
) -> OmniMoTModelDescriptor:
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads
    if mlp_only_layers is None:
        mlp_only_layers = []
    return OmniMoTModelDescriptor(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        use_moe=use_moe,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        decoder_sparse_step=decoder_sparse_step,
        mlp_only_layers=mlp_only_layers,
        latent_patch_size=latent_patch_size,
        latent_channel_size=latent_channel_size,
        action_dim=action_dim,
        sound_dim=sound_dim,
        frequency_embedding_size=frequency_embedding_size,
        predict_text_tokens=predict_text_tokens,
    )


def _pct(part: Decimal, whole: Decimal) -> str:
    """Return percentage string, guarding against division by zero."""
    if whole == 0:
        return "0"
    return str(round(part / whole * 100, 1))


def _extract_padding_tokens(
    split_lens: list[int],
    attn_modes: list[str],
) -> int:
    """Return the total number of padding tokens in a packed sequence.

    Padding splits are lone ``"causal"`` entries that do not form a
    ``(causal, full)`` pair with the next split.  In practice, finalize()
    appends at most one such split at the end.
    """
    padding = 0
    i = 0
    while i < len(split_lens):
        if i + 1 < len(split_lens) and attn_modes[i] == "causal" and attn_modes[i + 1] == "full":
            i += 2
        else:
            if attn_modes[i] == "causal":
                padding += split_lens[i]
            i += 1
    return padding


def _compute_per_sample_attn_flops(
    n_heads: int,
    d_head: int,
    B: int | Decimal,
    S_und: int | Decimal,
    S_gen: int | Decimal,
    split_lens: list[int] | None = None,
    attn_modes: list[str] | None = None,
    include_padding: bool = False,
) -> tuple[Decimal, Decimal]:
    """Compute per-layer attention dot-product FLOPs (QK^T + Attn*V).

    The MoT attention pattern is:
      - Und tokens (causal): each sample's text tokens self-attend causally.
      - Gen tokens (full): each sample's gen tokens attend to ALL tokens in
        that sample (und + gen) with full (non-causal) attention.

    When ``split_lens``/``attn_modes`` are provided (packed-sequence mode),
    per-sample lengths are extracted from the alternating (causal, full) pairs.
    Otherwise, ``B`` uniform samples each with ``S_und`` and ``S_gen`` tokens
    are assumed.

    Args:
        include_padding: If True, lone ``"causal"`` splits (padding tokens
            appended by finalize()) are counted as additional causal
            self-attention windows.

    Returns:
        (und_attn_flops, gen_attn_flops) for a single layer (QK^T + Attn*V).
    """
    if split_lens is not None and attn_modes is not None:
        und_attn = Decimal(0)
        gen_attn = Decimal(0)
        i = 0
        while i < len(split_lens):
            if i + 1 < len(split_lens) and attn_modes[i] == "causal" and attn_modes[i + 1] == "full":
                s_und_i = split_lens[i]
                s_gen_i = split_lens[i + 1]
                und_attn += 4 * n_heads * d_head * s_und_i * s_und_i
                gen_attn += 4 * n_heads * d_head * s_gen_i * (s_und_i + s_gen_i)
                i += 2
            else:
                if include_padding and attn_modes[i] == "causal":
                    s_pad = split_lens[i]
                    und_attn += 4 * n_heads * d_head * s_pad * s_pad
                i += 1
        return und_attn, gen_attn

    und_attn = Decimal(4 * B * n_heads * d_head * S_und * S_und)
    gen_attn = Decimal(4 * B * n_heads * d_head * S_gen * (S_und + S_gen))
    return und_attn, gen_attn


def compute_omni_mot_flops_per_batch(
    cfg: OmniMoTModelDescriptor,
    B: int | Decimal,
    text_tokens: int = 512,
    vision_tokens: int = 0,
    action_tokens: int = 0,
    sound_tokens: int = 0,
    freeze_und: bool = False,
    vision_gen: bool = True,
    action_gen: bool = False,
    sound_gen: bool = False,
    backwardpass_ratio: float = 2.0,
    split_lens: list[int] | None = None,
    attn_modes: list[str] | None = None,
    include_padding: bool = False,
    use_activation_checkpointing: bool = False,
) -> Decimal:
    """Compute training FLOPs for a single batch of the OmniMoT model.

    This is a standalone function that can be called from calculators or callbacks.
    It accounts for all parts of the dual-pathway (MoT) transformer, including:
      - Modality-specific embedding/projection layers (vae2llm, llm2vae, action2llm,
        llm2action, sound2llm, llm2sound).
      - TimestepEmbedder MLPs.
      - lm_head for text prediction.
      - Transformer blocks with dual-pathway attention (separate Q/K/V/O projections
        for und and gen pathways).
      - Per-sample attention: und tokens self-attend causally, gen tokens attend to
        all tokens in their sample with full attention.
      - Attention softmax FLOPs (~5 ops per element of the attention matrix).
      - Dual-pathway MLPs (dense SwiGLU or MoE per layer).
      - RMSNorm at all positions (4 per layer + Q/K norms + 2 final norms).
      - Backward pass with special handling for freeze_und.
      - Activation checkpointing forward recomputation during backward.

    Args:
        cfg: Model architecture descriptor.
        B: Batch size.  For the packed-sequence path (``split_lens`` provided),
            set ``B=1`` and let ``text_tokens``/``vision_tokens`` be the totals
            across all packed samples.
        text_tokens: Total number of text (understanding) tokens across all samples.
        vision_tokens: Total number of vision generation tokens (after patchification)
            across all samples.
        action_tokens: Total number of action tokens across all samples.
        sound_tokens: Total number of sound tokens across all samples.
        freeze_und: If True, understanding pathway is frozen (no backward FLOPs for und).
        vision_gen: Whether vision generation is active.
        action_gen: Whether action generation is active.
        sound_gen: Whether sound generation is active.
        backwardpass_ratio: Multiplier for backward pass FLOPs relative to forward
            (default 2.0).
        split_lens: Per-split token lengths from the packed sequence.  Alternating
            ``[und_0, gen_0, und_1, gen_1, ...]`` with matching ``attn_modes``.
            When provided, per-sample attention FLOPs are computed correctly
            instead of assuming one big attention window.
        attn_modes: Attention mode for each split (``"causal"`` or ``"full"``).
            Must have the same length as ``split_lens``.
        include_padding: If True, padding tokens (lone ``"causal"`` splits at
            the end of ``split_lens``) are included in FLOPs for attention,
            projections, MLP, and norms.  Useful for measuring total GPU FLOPs
            including wasted work on padding.
        use_activation_checkpointing: If True, add FLOPs for the forward
            recomputation of each transformer layer during the backward pass.
            Activation checkpointing discards intermediate activations and
            recomputes them on-the-fly, adding ~1x layer forward FLOPs.

    Returns:
        Total training FLOPs (forward + backward) as a Decimal.
    """
    bp_ratio = Decimal(backwardpass_ratio)
    D = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    d_head = cfg.head_dim
    n_layers = cfg.num_hidden_layers

    # ===================================================================
    # Token counts
    # ===================================================================
    L_vision = vision_tokens if vision_gen else 0

    S_und = text_tokens
    S_gen = L_vision + (action_tokens if action_gen else 0) + (sound_tokens if sound_gen else 0)

    # Padding tokens follow the causal (und) path.  When include_padding is
    # set, add them to S_und so projections, MLP, and norms account for the
    # extra work the GPU performs on padding.
    S_pad = 0
    if include_padding and split_lens is not None and attn_modes is not None:
        S_pad = _extract_padding_tokens(split_lens, attn_modes)
        S_und = S_und + S_pad

    # ===================================================================
    # 1. Embedding / Projection Layers (outside transformer blocks)
    # ===================================================================
    embedding_flops = Decimal(0)

    if vision_gen and L_vision > 0:
        patch_latent_dim = cfg.latent_patch_size**2 * cfg.latent_channel_size
        embedding_flops += 2 * B * L_vision * patch_latent_dim * D

    if vision_gen and L_vision > 0:
        embedding_flops += 2 * B * L_vision * D * patch_latent_dim

    if action_gen and action_tokens > 0:
        embedding_flops += 2 * B * action_tokens * cfg.action_dim * D

    if action_gen and action_tokens > 0:
        embedding_flops += 2 * B * action_tokens * D * cfg.action_dim

    if sound_gen and sound_tokens > 0 and cfg.sound_dim is not None:
        embedding_flops += 2 * B * sound_tokens * cfg.sound_dim * D

    if sound_gen and sound_tokens > 0 and cfg.sound_dim is not None:
        embedding_flops += 2 * B * sound_tokens * D * cfg.sound_dim

    # TimestepEmbedder MLP: Linear(freq_dim, D) -> SiLU -> Linear(D, D)
    freq_dim = cfg.frequency_embedding_size
    timestep_mlp_flops_per_call = 2 * freq_dim * D + 2 * D * D
    n_timestep_calls = 0
    if vision_gen and L_vision > 0:
        n_timestep_calls += 1
    if action_gen and action_tokens > 0:
        n_timestep_calls += 1
    if sound_gen and sound_tokens > 0:
        n_timestep_calls += 1
    embedding_flops += n_timestep_calls * B * timestep_mlp_flops_per_call

    if cfg.predict_text_tokens:
        embedding_flops += 2 * B * text_tokens * D * cfg.vocab_size

    log.debug(f"embedding_flops: {embedding_flops}")

    # ===================================================================
    # Pre-compute per-sample attention dot-product FLOPs (shared by
    # forward and backward).  Und tokens self-attend causally,
    # gen tokens attend to all tokens in their sample.
    # ===================================================================
    und_attn_dot, gen_attn_dot = _compute_per_sample_attn_flops(
        n_heads,
        d_head,
        B,
        S_und,
        S_gen,
        split_lens,
        attn_modes,
        include_padding=include_padding,
    )

    # Softmax FLOPs: ~5 ops per element of the S_q x S_k attention matrix
    # (subtract max, exp, sum, divide, plus the mask/scale).
    # Same sequence-length dependency as dot product but with coefficient
    # 5 * n_heads instead of 4 * n_heads * d_head.
    softmax_ratio = Decimal(5) / Decimal(4 * d_head)
    und_softmax = und_attn_dot * softmax_ratio
    gen_softmax = gen_attn_dot * softmax_ratio

    # ===================================================================
    # 2. Transformer Blocks
    # ===================================================================
    total_block_flops = Decimal(0)
    total_attn_dot_fwd = Decimal(0)
    total_softmax_fwd = Decimal(0)
    q_dim = n_heads * d_head
    kv_dim = n_kv_heads * d_head

    def _dense_mlp_flops(seq_len: int | Decimal) -> Decimal:
        return Decimal(6 * B * seq_len * D * cfg.intermediate_size)

    def _moe_mlp_flops(seq_len: int | Decimal) -> Decimal:
        gate_flops = 2 * B * seq_len * D * cfg.num_experts
        expert_flops = cfg.num_experts_per_tok * 6 * B * seq_len * D * cfg.moe_intermediate_size
        return Decimal(gate_flops + expert_flops)

    for layer_idx in range(n_layers):
        is_moe_layer = (
            cfg.use_moe
            and cfg.num_experts > 0
            and layer_idx not in cfg.mlp_only_layers
            and (layer_idx + 1) % cfg.decoder_sparse_step == 0
        )

        # 2a. Attention (PackedAttentionMoT)
        attn_und_proj = 2 * B * S_und * D * q_dim + 2 * B * S_und * D * kv_dim + 2 * B * S_und * D * kv_dim
        attn_gen_proj = 2 * B * S_gen * D * q_dim + 2 * B * S_gen * D * kv_dim + 2 * B * S_gen * D * kv_dim
        attn_dot = und_attn_dot + gen_attn_dot
        attn_o_proj = 2 * B * S_und * q_dim * D + 2 * B * S_gen * q_dim * D
        attn_qk_norm = (
            5 * B * S_und * n_heads * d_head
            + 5 * B * S_und * n_kv_heads * d_head
            + 5 * B * S_gen * n_heads * d_head
            + 5 * B * S_gen * n_kv_heads * d_head
        )
        layer_attn_flops = attn_und_proj + attn_gen_proj + attn_qk_norm + attn_dot + attn_o_proj

        # 2b. MLP (separate for und and gen pathways)
        mlp_und_flops = _moe_mlp_flops(S_und) if is_moe_layer else _dense_mlp_flops(S_und)
        mlp_gen_flops = _moe_mlp_flops(S_gen) if is_moe_layer else _dense_mlp_flops(S_gen)
        layer_mlp_flops = mlp_und_flops + mlp_gen_flops

        # 2c. RMSNorm (4 layer norms per decoder layer, dimension D)
        layer_norm_flops = 5 * B * S_und * D + 5 * B * S_gen * D + 5 * B * S_und * D + 5 * B * S_gen * D

        # 2d. Attention softmax
        layer_softmax_flops = und_softmax + gen_softmax

        layer_flops = layer_attn_flops + layer_mlp_flops + layer_norm_flops + layer_softmax_flops
        total_block_flops += layer_flops
        total_attn_dot_fwd += attn_dot
        total_softmax_fwd += layer_softmax_flops

        if layer_idx == 0:
            log.debug(f"Layer 0 breakdown (MoE={is_moe_layer}):")
            log.debug(f"  attn_und_proj: {attn_und_proj}")
            log.debug(f"  attn_gen_proj: {attn_gen_proj}")
            log.debug(f"  attn_qk_norm:  {attn_qk_norm}")
            log.debug(f"  attn_dot:      {attn_dot}")
            log.debug(f"  attn_softmax:  {layer_softmax_flops}")
            log.debug(f"  attn_o_proj:   {attn_o_proj}")
            log.debug(f"  mlp_und:       {mlp_und_flops}")
            log.debug(f"  mlp_gen:       {mlp_gen_flops}")
            log.debug(f"  layer_norms:   {layer_norm_flops}")
            log.debug(f"  total layer:   {layer_flops}")

    # ===================================================================
    # 3. Final norms (applied to und and gen separately after all layers)
    # ===================================================================
    final_norm_flops = Decimal(5 * B * S_und * D + 5 * B * S_gen * D)

    log.debug(f"final_norm_flops: {final_norm_flops}")

    # ===================================================================
    # 4. Forward pass total
    # ===================================================================
    fp = embedding_flops + total_block_flops + final_norm_flops

    log.debug(f"Forward pass FLOPs: {fp}")
    log.debug(f"  embedding_flops:    {embedding_flops} ({_pct(embedding_flops, fp)}%)")
    log.debug(f"  transformer_blocks: {total_block_flops} ({_pct(total_block_flops, fp)}%)")
    log.debug(f"  final_norms:        {final_norm_flops} ({_pct(final_norm_flops, fp)}%)")

    # ===================================================================
    # 5. Backward pass
    # ===================================================================

    if freeze_und:
        # When freeze_und is True, the understanding pathway gradients are detached.
        # Backward cost: gen-pathway projections/MLPs, gen-side attention (gen Q
        # attends to the full sample), gen norms, and gen embedding layers.
        # Causal (und) attention has zero backward cost.
        gen_proj_mlp_flops = Decimal(0)
        gen_norm_flops = Decimal(0)
        for layer_idx in range(n_layers):
            is_moe_layer = (
                cfg.use_moe
                and cfg.num_experts > 0
                and layer_idx not in cfg.mlp_only_layers
                and (layer_idx + 1) % cfg.decoder_sparse_step == 0
            )
            gen_proj_mlp_flops += (
                2 * B * S_gen * D * q_dim
                + 2 * B * S_gen * D * kv_dim
                + 2 * B * S_gen * D * kv_dim
                + 2 * B * S_gen * q_dim * D
            )
            gen_proj_mlp_flops += _moe_mlp_flops(S_gen) if is_moe_layer else _dense_mlp_flops(S_gen)

            gen_norm_flops += 5 * B * S_gen * D * 2
            gen_norm_flops += 5 * B * S_gen * n_heads * d_head + 5 * B * S_gen * n_kv_heads * d_head

        gen_norm_flops += 5 * B * S_gen * D

        gen_embedding_flops = embedding_flops  # conservative: count all embedding flops

        backward_attn_flops = gen_attn_dot * n_layers
        backward_softmax_flops = gen_softmax * n_layers

        bp = (
            gen_proj_mlp_flops + backward_attn_flops + backward_softmax_flops + gen_norm_flops + gen_embedding_flops
        ) * bp_ratio

    else:
        bp = fp * bp_ratio

    # ===================================================================
    # 6. Activation checkpointing recomputation
    # ===================================================================
    # When activation checkpointing is enabled, each transformer layer's
    # forward pass is fully recomputed during the backward pass.  This adds
    # ~1x of the transformer-block forward FLOPs (projections, attention
    # dot products, softmax, MLP, and norms — everything inside the layer).
    ac_recomp = Decimal(0)
    if use_activation_checkpointing:
        ac_recomp = total_block_flops

    total = fp + bp + ac_recomp

    log.debug(f"Backward pass FLOPs: {bp}")
    if use_activation_checkpointing:
        log.debug(f"Activation checkpointing recomp FLOPs: {ac_recomp}")
    log.debug(f"Total FLOPs: {total}")

    return total
