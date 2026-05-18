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

"""
Compute FLOPs for Qwen3VL model.

This script provides utilities to calculate the theoretical FLOPs for a Qwen3VL model
given the model configuration and input specifications (total tokens, visual tokens, etc.).

Usage:
    from projects.cosmos3.vlm.scripts.compute_qwen3vl_flops import compute_qwen3vl_flops

    flops = compute_qwen3vl_flops(
        num_text_layers=32,
        num_vision_layers=27,
        hidden_size=4096,
        intermediate_size=22016,
        num_attention_heads=32,
        num_key_value_heads=32,
        vision_hidden_size=1152,
        vision_intermediate_size=4304,
        vision_num_heads=16,
        total_tokens=2048,
        visual_tokens=512,
        pixel_values_shape=(512, 3, 16, 16)  # (seq_len, channels, height, width)
    )

    print(f"Total FLOPs: {flops:.2e}")
"""

from typing import Optional


def compute_linear_flops(in_features: int, out_features: int, seq_len: int, has_bias: bool = True) -> int:
    """Compute FLOPs for a linear layer.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        seq_len: Sequence length
        has_bias: Whether the layer has bias

    Returns:
        Total FLOPs for the linear layer
    """
    # Matrix multiplication: 2 * seq_len * in_features * out_features
    # (2 accounts for multiply-add operations)
    matmul_flops = 2 * seq_len * in_features * out_features

    # Bias addition if present
    bias_flops = seq_len * out_features if has_bias else 0

    return matmul_flops + bias_flops


def compute_attention_flops(
    seq_len: int, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: Optional[int] = None
) -> int:
    """Compute FLOPs for attention mechanism.

    Args:
        seq_len: Sequence length
        hidden_size: Hidden dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        head_dim: Dimension per head (defaults to hidden_size // num_heads)

    Returns:
        Total FLOPs for attention
    """
    if head_dim is None:
        head_dim = hidden_size // num_heads

    # QKV projection: 3 linear layers (but KV uses num_kv_heads)
    q_proj_flops = compute_linear_flops(hidden_size, num_heads * head_dim, seq_len, has_bias=True)
    k_proj_flops = compute_linear_flops(hidden_size, num_kv_heads * head_dim, seq_len, has_bias=True)
    v_proj_flops = compute_linear_flops(hidden_size, num_kv_heads * head_dim, seq_len, has_bias=True)

    # Q @ K^T: (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
    # = (batch, num_heads, seq_len, seq_len)
    qk_matmul_flops = 2 * num_heads * seq_len * seq_len * head_dim

    # Softmax: approximately 3 * num_heads * seq_len * seq_len (exp, sum, divide)
    softmax_flops = 3 * num_heads * seq_len * seq_len

    # Attention @ V: (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
    # = (batch, num_heads, seq_len, head_dim)
    attn_v_matmul_flops = 2 * num_heads * seq_len * seq_len * head_dim

    # Output projection
    o_proj_flops = compute_linear_flops(num_heads * head_dim, hidden_size, seq_len, has_bias=True)

    total_flops = (
        q_proj_flops
        + k_proj_flops
        + v_proj_flops
        + qk_matmul_flops
        + softmax_flops
        + attn_v_matmul_flops
        + o_proj_flops
    )

    return total_flops


def compute_mlp_flops(seq_len: int, hidden_size: int, intermediate_size: int, use_swiglu: bool = True) -> int:
    """Compute FLOPs for MLP layer.

    Args:
        seq_len: Sequence length
        hidden_size: Hidden dimension
        intermediate_size: Intermediate dimension
        use_swiglu: Whether using SwiGLU activation (requires gate and up projections)

    Returns:
        Total FLOPs for MLP
    """
    if use_swiglu:
        # SwiGLU: gate_proj, up_proj, down_proj
        gate_proj_flops = compute_linear_flops(hidden_size, intermediate_size, seq_len, has_bias=False)
        up_proj_flops = compute_linear_flops(hidden_size, intermediate_size, seq_len, has_bias=False)
        down_proj_flops = compute_linear_flops(intermediate_size, hidden_size, seq_len, has_bias=False)

        # Activation (SiLU) + element-wise multiply: ~2 ops per element
        activation_flops = 2 * seq_len * intermediate_size
        multiply_flops = seq_len * intermediate_size

        total_flops = gate_proj_flops + up_proj_flops + down_proj_flops + activation_flops + multiply_flops
    else:
        # Standard MLP: fc1, activation, fc2
        fc1_flops = compute_linear_flops(hidden_size, intermediate_size, seq_len, has_bias=True)
        fc2_flops = compute_linear_flops(intermediate_size, hidden_size, seq_len, has_bias=True)
        activation_flops = seq_len * intermediate_size

        total_flops = fc1_flops + fc2_flops + activation_flops

    return total_flops


def compute_moe_flops(
    seq_len: int,
    hidden_size: int,
    moe_intermediate_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    use_swiglu: bool = True,
) -> int:
    """Compute FLOPs for Mixture of Experts (MoE) layer.

    Args:
        seq_len: Sequence length
        hidden_size: Hidden dimension
        moe_intermediate_size: Intermediate dimension for each expert
        num_experts: Total number of experts
        num_experts_per_tok: Number of experts activated per token (top-k)
        use_swiglu: Whether using SwiGLU activation (requires gate and up projections)

    Returns:
        Total FLOPs for MoE layer

    Note:
        MoE uses sparse computation - only num_experts_per_tok out of num_experts
        are activated per token. Each expert has its own gate_proj, up_proj, down_proj.
    """
    # Router FLOPs: linear projection to select experts
    router_flops = compute_linear_flops(hidden_size, num_experts, seq_len, has_bias=False)

    # Softmax over experts: ~3 ops per element (exp, sum, divide)
    softmax_flops = 3 * seq_len * num_experts

    # Top-k selection: approximate as O(k * log(n)) operations per token
    # Using conservative estimate: num_experts_per_tok * log2(num_experts) ops per token
    import math

    topk_flops = seq_len * num_experts_per_tok * int(math.log2(num_experts))

    # Expert computation FLOPs
    # Only num_experts_per_tok experts are active per token (sparse computation)
    if use_swiglu:
        # Each active expert: gate_proj, up_proj, down_proj with moe_intermediate_size
        # Note: gate_up_proj is fused in implementation but we count separately for clarity
        gate_proj_flops = compute_linear_flops(hidden_size, moe_intermediate_size, seq_len, has_bias=False)
        up_proj_flops = compute_linear_flops(hidden_size, moe_intermediate_size, seq_len, has_bias=False)
        down_proj_flops = compute_linear_flops(moe_intermediate_size, hidden_size, seq_len, has_bias=False)

        # Activation (SiLU) + element-wise multiply
        activation_flops = 2 * seq_len * moe_intermediate_size
        multiply_flops = seq_len * moe_intermediate_size

        # Total per expert, scaled by number of active experts per token
        expert_flops_per_token = (
            gate_proj_flops + up_proj_flops + down_proj_flops + activation_flops + multiply_flops
        ) / seq_len
        total_expert_flops = seq_len * num_experts_per_tok * expert_flops_per_token
    else:
        # Standard MLP-style expert
        fc1_flops = compute_linear_flops(hidden_size, moe_intermediate_size, seq_len, has_bias=True)
        fc2_flops = compute_linear_flops(moe_intermediate_size, hidden_size, seq_len, has_bias=True)
        activation_flops = seq_len * moe_intermediate_size

        expert_flops_per_token = (fc1_flops + fc2_flops + activation_flops) / seq_len
        total_expert_flops = seq_len * num_experts_per_tok * expert_flops_per_token

    # Weighted sum of expert outputs (element-wise multiply + sum)
    # Each token combines num_experts_per_tok expert outputs
    weighted_sum_flops = seq_len * num_experts_per_tok * hidden_size

    total_flops = router_flops + softmax_flops + topk_flops + total_expert_flops + weighted_sum_flops

    return int(total_flops)


def compute_layernorm_flops(seq_len: int, hidden_size: int) -> int:
    """Compute FLOPs for layer normalization.

    Args:
        seq_len: Sequence length
        hidden_size: Hidden dimension

    Returns:
        Total FLOPs for LayerNorm
    """
    # Mean: sum + divide
    # Variance: (x - mean)^2, sum, divide
    # Normalize: (x - mean) / sqrt(var + eps)
    # Scale and shift: x * weight + bias
    # Approximately 5 operations per element
    return 5 * seq_len * hidden_size


def compute_vision_encoder_flops(
    num_patches: int,
    vision_hidden_size: int,
    vision_intermediate_size: int,
    vision_num_heads: int,
    num_vision_layers: int,
    out_hidden_size: int,  # Text decoder hidden size
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    spatial_merge_size: int = 2,
    in_channels: int = 3,
) -> int:
    """Compute FLOPs for vision encoder.

    Args:
        num_patches: Number of patches (from pixel_values.shape[0] or image_grid_thw.prod(-1))
        vision_hidden_size: Vision encoder hidden size
        vision_intermediate_size: Vision encoder intermediate size
        vision_num_heads: Number of vision attention heads
        num_vision_layers: Number of vision encoder layers
        out_hidden_size: Output hidden size (text decoder hidden size) for patch merger
        patch_size: Spatial patch size
        temporal_patch_size: Temporal patch size
        spatial_merge_size: Spatial merge size for patch merger
        in_channels: Input channels (typically 3 for RGB)

    Returns:
        Total FLOPs for vision encoder

    Note:
        pixel_values from processor has shape [num_patches, patch_dim] where:
        - num_patches = t * h * w (from image_grid_thw)
        - patch_dim = in_channels * temporal_patch_size * patch_size * patch_size

        Example: pixel_values [11008, 1536] with image_grid_thw [[1, 86, 128]]
        - num_patches = 1 * 86 * 128 = 11008
        - patch_dim = 3 * 2 * 16 * 16 = 1536
        - num_visual_tokens = num_patches // (spatial_merge_size ** 2) = 11008 // 4 = 2752
    """

    # Conv3D FLOPs: 2 * num_output_elements * kernel_volume * in_channels * out_channels
    kernel_volume = temporal_patch_size * patch_size * patch_size
    patch_embed_flops = 2 * num_patches * kernel_volume * in_channels * vision_hidden_size

    # Add bias
    patch_embed_flops += num_patches * vision_hidden_size

    # Vision transformer blocks
    vision_block_flops = 0
    for _ in range(num_vision_layers):
        # Attention (vision uses different pattern, but similar complexity)
        attn_flops = compute_attention_flops(num_patches, vision_hidden_size, vision_num_heads, vision_num_heads)

        # MLP
        mlp_flops = compute_mlp_flops(num_patches, vision_hidden_size, vision_intermediate_size, use_swiglu=False)

        # Layer norms (2 per block)
        ln_flops = 2 * compute_layernorm_flops(num_patches, vision_hidden_size)

        vision_block_flops += attn_flops + mlp_flops + ln_flops

    # Patch merger: projects merged patches to output hidden size (text decoder hidden size)
    merged_patches = num_patches // (spatial_merge_size * spatial_merge_size)
    merged_hidden = vision_hidden_size * (spatial_merge_size * spatial_merge_size)

    # Merger: LayerNorm + FC1 + GELU + FC2
    # FC2 projects to out_hidden_size (text decoder hidden size)
    merger_flops = compute_layernorm_flops(merged_patches, merged_hidden)
    merger_flops += compute_linear_flops(merged_hidden, merged_hidden, merged_patches, has_bias=True)
    merger_flops += merged_patches * merged_hidden  # GELU activation
    merger_flops += compute_linear_flops(merged_hidden, out_hidden_size, merged_patches, has_bias=True)

    total_vision_flops = patch_embed_flops + vision_block_flops + merger_flops

    return total_vision_flops


def compute_text_decoder_flops(
    total_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    num_text_layers: int,
    head_dim: int = 128,
    # MoE parameters
    num_experts: int = 0,
    num_experts_per_tok: int = 0,
    moe_intermediate_size: Optional[int] = None,
    decoder_sparse_step: int = 1,
    mlp_only_layers: Optional[list] = None,
) -> int:
    """Compute FLOPs for text decoder with optional MoE support.

    Args:
        total_tokens: Total number of tokens (text + visual)
        hidden_size: Hidden dimension
        intermediate_size: Intermediate dimension for standard MLP
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
        num_text_layers: Number of decoder layers
        head_dim: Dimension per attention head
        num_experts: Total number of experts in MoE (0 means no MoE)
        num_experts_per_tok: Number of experts activated per token (top-k)
        moe_intermediate_size: Intermediate dimension for each MoE expert
        decoder_sparse_step: MoE is used every nth layer (default 1 = every layer)
        mlp_only_layers: List of layer indices that should use standard MLP instead of MoE

    Returns:
        Total FLOPs for text decoder

    Note:
        A layer uses MoE if:
        1. layer_idx is NOT in mlp_only_layers, AND
        2. num_experts > 0, AND
        3. (layer_idx + 1) % decoder_sparse_step == 0
    """
    if mlp_only_layers is None:
        mlp_only_layers = []

    decoder_flops = 0
    use_moe = num_experts > 0 and moe_intermediate_size is not None

    for layer_idx in range(num_text_layers):
        # Self-attention
        attn_flops = compute_attention_flops(
            total_tokens, hidden_size, num_attention_heads, num_key_value_heads, head_dim
        )

        # MLP or MoE (Qwen uses SwiGLU)
        # Check if this layer uses MoE
        is_moe_layer = use_moe and (layer_idx not in mlp_only_layers) and ((layer_idx + 1) % decoder_sparse_step == 0)

        if is_moe_layer:
            mlp_flops = compute_moe_flops(
                total_tokens,
                hidden_size,
                moe_intermediate_size,
                num_experts,
                num_experts_per_tok,
                use_swiglu=True,
            )
        else:
            mlp_flops = compute_mlp_flops(total_tokens, hidden_size, intermediate_size, use_swiglu=True)

        # Layer norms (2 per layer: input_layernorm and post_attention_layernorm)
        ln_flops = 2 * compute_layernorm_flops(total_tokens, hidden_size)

        # RMSNorm for Q and K (applied to each head dimension)
        qk_norm_flops = 2 * compute_layernorm_flops(total_tokens * num_attention_heads, head_dim)

        decoder_flops += attn_flops + mlp_flops + ln_flops + qk_norm_flops

    return decoder_flops


def compute_qwen3vl_flops(
    num_text_layers: int,
    num_vision_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    vision_hidden_size: int,
    vision_intermediate_size: int,
    vision_num_heads: int,
    vocab_size: int,
    total_tokens: int,
    visual_tokens: int,
    num_patches: Optional[int] = None,
    head_dim: int = 128,
    include_embeddings: bool = True,
    include_lm_head: bool = True,
    spatial_merge_size: int = 2,
    # MoE parameters
    num_experts: int = 0,
    num_experts_per_tok: int = 0,
    moe_intermediate_size: Optional[int] = None,
    decoder_sparse_step: int = 1,
    mlp_only_layers: Optional[list] = None,
) -> dict:
    """Compute total FLOPs for Qwen3VL model forward pass (supports MoE).

    Args:
        num_text_layers: Number of text decoder layers
        num_vision_layers: Number of vision encoder layers
        hidden_size: Text decoder hidden size
        intermediate_size: Text decoder intermediate size (for standard MLP layers)
        num_attention_heads: Number of attention heads in text decoder
        num_key_value_heads: Number of key-value heads in text decoder (for GQA)
        vision_hidden_size: Vision encoder hidden size
        vision_intermediate_size: Vision encoder intermediate size
        vision_num_heads: Number of attention heads in vision encoder
        vocab_size: Vocabulary size
        total_tokens: Total number of tokens in sequence
        visual_tokens: Number of visual tokens (after vision encoder processing)
        num_patches: Number of patches (pixel_values.shape[0] or image_grid_thw.prod(-1))
                    Required if computing vision encoder FLOPs
        head_dim: Dimension per attention head
        include_embeddings: Whether to include embedding layer FLOPs
        include_lm_head: Whether to include LM head FLOPs
        spatial_merge_size: Spatial merge size (default 2)
        num_experts: Total number of experts in MoE (0 means no MoE)
        num_experts_per_tok: Number of experts activated per token (top-k)
        moe_intermediate_size: Intermediate dimension for each MoE expert
        decoder_sparse_step: MoE is used every nth layer (default 1 = every layer)
        mlp_only_layers: List of layer indices that should use standard MLP instead of MoE

    Returns:
        Dictionary containing:
            - total_flops: Total FLOPs for the model
            - vision_encoder_flops: FLOPs for vision encoder
            - text_decoder_flops: FLOPs for text decoder
            - embedding_flops: FLOPs for embedding layer
            - lm_head_flops: FLOPs for LM head
            - breakdown: Detailed breakdown of FLOPs

    Note:
        From processor output:
        - pixel_values.shape = [num_patches, patch_dim] where patch_dim = C * T_patch * H_patch * W_patch
        - image_grid_thw.shape = [num_images, 3] with values [t, h, w]
        - num_patches = image_grid_thw.prod(-1).sum() for all images
        - visual_tokens = num_patches // (spatial_merge_size ** 2)

        Example: pixel_values [11008, 1536], image_grid_thw [[1, 86, 128]]
        - num_patches = 1 * 86 * 128 = 11008
        - visual_tokens = 11008 // 4 = 2752

        For MoE models (e.g., Qwen3-VL-30B-A3B):
        - A layer uses MoE if: (layer_idx not in mlp_only_layers) and (num_experts > 0)
          and ((layer_idx + 1) % decoder_sparse_step == 0)
        - MoE uses sparse computation: only num_experts_per_tok experts are active per token
    """
    flops_breakdown = {}

    # Vision encoder FLOPs
    if num_patches is not None:
        vision_flops = compute_vision_encoder_flops(
            num_patches=num_patches,
            vision_hidden_size=vision_hidden_size,
            vision_intermediate_size=vision_intermediate_size,
            vision_num_heads=vision_num_heads,
            num_vision_layers=num_vision_layers,
            out_hidden_size=hidden_size,  # Projects to text decoder hidden size
            spatial_merge_size=spatial_merge_size,
        )
        flops_breakdown["vision_encoder"] = vision_flops
    else:
        vision_flops = 0
        flops_breakdown["vision_encoder"] = 0

    # Embedding layer FLOPs

    text_tokens = total_tokens - visual_tokens
    if include_embeddings:
        # Embedding lookup: typically counted as 0 or hidden_size operations per token
        # We'll use hidden_size operations per token as it's memory read + indexing
        embedding_flops = text_tokens * hidden_size
        flops_breakdown["embeddings"] = embedding_flops
    else:
        embedding_flops = 0
        flops_breakdown["embeddings"] = 0

    # Text decoder FLOPs
    # IMPORTANT: The decoder processes ALL tokens (text + visual).
    # Visual tokens from the vision encoder are concatenated with text embeddings
    # before being fed to the decoder, so total_tokens = text_tokens + visual_tokens
    decoder_flops = compute_text_decoder_flops(
        total_tokens=total_tokens,  # Includes both text and visual tokens
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_text_layers=num_text_layers,
        head_dim=head_dim,
        # MoE parameters
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        decoder_sparse_step=decoder_sparse_step,
        mlp_only_layers=mlp_only_layers,
    )
    flops_breakdown["text_decoder"] = decoder_flops

    # LM head FLOPs
    if include_lm_head:
        # Final layer norm + linear projection to vocabulary
        lm_head_flops = compute_layernorm_flops(total_tokens, hidden_size)
        lm_head_flops += compute_linear_flops(hidden_size, vocab_size, total_tokens, has_bias=False)
        flops_breakdown["lm_head"] = lm_head_flops
    else:
        lm_head_flops = 0
        flops_breakdown["lm_head"] = 0

    total_flops = vision_flops + embedding_flops + decoder_flops + lm_head_flops

    return {
        "total_flops": total_flops,
        "vision_encoder_flops": vision_flops,
        "text_decoder_flops": decoder_flops,
        "embedding_flops": embedding_flops,
        "lm_head_flops": lm_head_flops,
        "breakdown": flops_breakdown,
    }


def compute_qwen3vl_flops_from_config(
    config,
    total_tokens: int,
    visual_tokens: int,
    num_patches: Optional[int] = None,
) -> dict:
    """Compute FLOPs using Qwen3VL config object (supports MoE).

    Args:
        config: Qwen3VLConfig or Qwen3VLMoeConfig object
        total_tokens: Total number of tokens in sequence
        visual_tokens: Number of visual tokens (after vision encoder processing)
        num_patches: Number of patches (pixel_values.shape[0] or image_grid_thw.prod(-1))

    Returns:
        Dictionary with FLOPs breakdown (same as compute_qwen3vl_flops)

    Note:
        visual_tokens = num_patches // (config.vision_config.spatial_merge_size ** 2)

        For MoE models, the config.text_config should contain:
        - num_experts: Total number of experts
        - num_experts_per_tok: Number of active experts per token
        - moe_intermediate_size: Intermediate size for each expert
        - decoder_sparse_step: MoE frequency (default 1 = every layer)
        - mlp_only_layers: List of layers that use standard MLP
    """
    # Extract MoE parameters if available (MoE models)
    text_config = config.text_config
    num_experts = getattr(text_config, "num_experts", 0)
    num_experts_per_tok = getattr(text_config, "num_experts_per_tok", 0)
    moe_intermediate_size = getattr(text_config, "moe_intermediate_size", None)
    decoder_sparse_step = getattr(text_config, "decoder_sparse_step", 1)
    mlp_only_layers = getattr(text_config, "mlp_only_layers", [])

    return compute_qwen3vl_flops(
        num_text_layers=text_config.num_hidden_layers,
        num_vision_layers=config.vision_config.depth,
        hidden_size=text_config.hidden_size,
        intermediate_size=text_config.intermediate_size,
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        vision_hidden_size=config.vision_config.hidden_size,
        vision_intermediate_size=config.vision_config.intermediate_size,
        vision_num_heads=config.vision_config.num_heads,
        vocab_size=text_config.vocab_size,
        total_tokens=total_tokens,
        visual_tokens=visual_tokens,
        num_patches=num_patches,
        head_dim=text_config.head_dim,
        spatial_merge_size=config.vision_config.spatial_merge_size,
        # MoE parameters
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        decoder_sparse_step=decoder_sparse_step,
        mlp_only_layers=mlp_only_layers,
    )


def format_flops(flops: float) -> str:
    """Format FLOPs in human-readable format.

    Args:
        flops: Number of FLOPs

    Returns:
        Formatted string (e.g., "1.23 TFLOPS")
    """
    if flops >= 1e15:
        return f"{flops / 1e15:.2f} PFLOPS"
    elif flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPS"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPS"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPS"
    else:
        return f"{flops:.2f} FLOPS"


if __name__ == "__main__":
    # Example usage with Qwen3-VL-2B configuration
    print("=" * 80)
    print("Qwen3-VL-2B FLOP Estimation")
    print("=" * 80)

    # Qwen3-VL-2B configuration (from actual model config)
    config_2b = {
        "num_text_layers": 28,
        "num_vision_layers": 27,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "vision_hidden_size": 1152,
        "vision_intermediate_size": 4304,
        "vision_num_heads": 16,
        "vocab_size": 151936,
        "head_dim": 128,
    }

    # Example input specifications (matching processor output format)
    # From processor: pixel_values.shape = [11008, 1536], image_grid_thw = [[1, 86, 128]]
    # num_patches = 1 * 86 * 128 = 11008
    # visual_tokens = num_patches // 4 = 2752
    total_tokens = 2048
    num_patches = 1024  # Example: 1 * 32 * 32 = 1024 patches
    visual_tokens = num_patches // 4  # 256 visual tokens after spatial merge

    result = compute_qwen3vl_flops(
        **config_2b,
        total_tokens=total_tokens,
        visual_tokens=visual_tokens,
        num_patches=num_patches,
    )

    print(f"\nInput specifications:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Visual tokens: {visual_tokens}")
    print(f"  Text tokens: {total_tokens - visual_tokens}")
    print(f"  Num patches: {num_patches}")
    print(f"  Spatial merge: 2x2 -> {num_patches} patches -> {visual_tokens} tokens")

    print(f"\nFLOPs breakdown:")
    print(f"  Vision encoder: {format_flops(result['vision_encoder_flops'])}")
    print(f"  Text decoder: {format_flops(result['text_decoder_flops'])}")
    print(f"  Embeddings: {format_flops(result['embedding_flops'])}")
    print(f"  LM head: {format_flops(result['lm_head_flops'])}")
    print(f"  {'=' * 50}")
    print(f"  Total: {format_flops(result['total_flops'])}")

    print(f"\nTotal FLOPs: {result['total_flops']:.2e}")
