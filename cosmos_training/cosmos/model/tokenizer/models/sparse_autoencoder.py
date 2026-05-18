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

"""Sparse Autoencoder for image/video tokenization.

This module provides:
    - DiagonalGaussianDistribution: VAE latent distribution
    - SparseTransformerBase: Base transformer class for encoder/decoder
    - Encoder: Vision transformer encoder with pretrained weight loading
    - Decoder: Transformer decoder with multiscale support
    - AutoencoderKLConfig: Configuration dataclass for the autoencoder
    - AutoencoderKL: Base VAE model with KL loss for encoding/decoding
"""

import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from loguru import logger as logging
from transformers import Siglip2Model

from cosmos.model.tokenizer.models.dense_backends import (
    resolve_dense_backend,
    run_batched_block_stack,
    run_varlen_block_stack,
)
from cosmos.model.tokenizer.models.modules import (
    AbsolutePositionEmbedder,
    LayerNorm32,
    LearnedPositionEmbedder,
    LearnedPositionEmbedder4D,
    RMSNorm32,
    SparseLinear,
    SparseMultiheadAttentionPoolingHead,
    SparseTensor,
    SparseTransformerBlock,
)
from cosmos.model.tokenizer.models.modules.quantizers import (
    FSQ,
    LFQ,
    RQBottleneck,
    levels_from_codebook_size,
)
from cosmos.model.tokenizer.models.utils import (
    batch_tensor_to_sparse,
    reconstruct_from_temporal_slices,
    sparse_to_batched_tensor,
    sparse_to_img_list,
)

# =============================================================================
# Helper Functions
# =============================================================================


def multiply_all_factors(config: dict[int, dict[str, Any]]) -> list[int]:
    """Multiply all factors from a multiscale config.

    Args:
        config: Dict mapping layer indices to config dicts with 'factor' keys.

    Returns:
        List of multiplied factors.
    """
    config_values = list(config.values())

    if not config_values:
        return []

    first_factor = config_values[0]["factor"]
    result = [1] * len(first_factor)

    for value_dict in config_values:
        factor_list = value_dict["factor"]
        for i in range(len(result)):
            result[i] *= factor_list[i]
    return result


def expand_multiscale_cfg(config: dict[int, dict[str, Any]], num_layer: int) -> dict[int, dict[str, Any]]:
    """Expand multiscale config to cover layers that have corresponding configs.

    Args:
        config: Dict with layer numbers as keys and config dicts as values.
        num_layer: Total number of layers (0 to num_layer-1).

    Returns:
        Dict with config for layers that have a corresponding config key.
    """
    expanded_config = {}
    config_keys = sorted(config.keys())

    for layer in range(num_layer):
        best_key = None
        for key in config_keys:
            if key <= layer:
                best_key = key
            else:
                break

        if best_key is not None:
            if layer == best_key:
                expanded_config[layer] = config[best_key].copy()
            else:
                layer_config = {}
                for key, value in config[best_key].items():
                    if key not in ["factor", "channel_duplication"]:
                        layer_config[key] = value
                expanded_config[layer] = layer_config

    return expanded_config


def _sparse_tensor_to_dense_batch_tokens(sparse_tensor: SparseTensor) -> tuple[torch.Tensor, bool] | None:
    """Convert a uniform sparse batch into dense `[B, S, D]` token features.

    Returns:
        Tuple of (`dense_feats`, `used_reshape_fast_path`) when every batch has
        the same token count, otherwise `None`.
    """
    q_seqlen = sparse_tensor.get_batch_seq_lens()
    if not q_seqlen:
        return None
    seq_len = q_seqlen[0]
    if any(current_seq_len != seq_len for current_seq_len in q_seqlen):
        return None

    batch_size = sparse_tensor.shape[0]
    feature_shape = sparse_tensor.feats.shape[1:]
    contiguous_layout = all(
        batch_slice.start == batch_idx * seq_len and batch_slice.stop == (batch_idx + 1) * seq_len
        for batch_idx, batch_slice in enumerate(sparse_tensor.layout)
    )
    if contiguous_layout:
        dense_feats = sparse_tensor.feats.reshape(batch_size, seq_len, *feature_shape)
        return dense_feats, True

    dense_feats = torch.stack([sparse_tensor.feats[batch_slice] for batch_slice in sparse_tensor.layout], dim=0)
    return dense_feats, False


def _dense_batch_tokens_to_flat_features(dense_feats: torch.Tensor, used_reshape_fast_path: bool) -> torch.Tensor:
    """Restore dense `[B, S, D]` features to the sparse flat token order."""
    if used_reshape_fast_path:
        return dense_feats.reshape(-1, *dense_feats.shape[2:])
    return torch.cat(list(dense_feats.unbind(dim=0)), dim=0)


def _crop_temporal_slices_to_ownership(
    slices: list[SparseTensor],
    *,
    frame_batch_size: int,
    frame_batch_strides: int,
) -> list[SparseTensor]:
    """Crop overlapping temporal slices to non-overlapping ownership regions.

    Adjacent slices split their actual overlap at the midpoint, with odd-length
    overlaps assigning the extra timestep to the later slice. This keeps one
    owner per reconstructed timestep even when the final slice is truncated.
    """
    if frame_batch_size <= frame_batch_strides or len(slices) <= 1:
        return slices

    slice_bounds: list[tuple[int, int, int | None] | None] = []
    owned_starts: list[int] = []
    owned_ends: list[int] = []

    for slice_tensor in slices:
        min_frame, max_frame = slice_tensor.get_temporal_range()
        temporal_offset = slice_tensor.get_spatial_cache("temporal_offset")
        if min_frame is None or max_frame is None:
            slice_bounds.append(None)
            owned_starts.append(0)
            owned_ends.append(0)
            continue

        global_start = min_frame + (0 if temporal_offset is None else temporal_offset)
        global_end = max_frame + 1 + (0 if temporal_offset is None else temporal_offset)
        slice_bounds.append((global_start, global_end, temporal_offset))
        owned_starts.append(global_start)
        owned_ends.append(global_end)

    for slice_index in range(len(slices) - 1):
        current_bounds = slice_bounds[slice_index]
        next_bounds = slice_bounds[slice_index + 1]
        if current_bounds is None or next_bounds is None:
            continue

        current_start, current_end, _ = current_bounds
        next_start, next_end, _ = next_bounds
        overlap_start = max(current_start, next_start)
        overlap_end = min(current_end, next_end)
        if overlap_start >= overlap_end:
            continue

        boundary = overlap_start + (overlap_end - overlap_start) // 2
        owned_ends[slice_index] = min(owned_ends[slice_index], boundary)
        owned_starts[slice_index + 1] = max(owned_starts[slice_index + 1], boundary)

    cropped_slices: list[SparseTensor] = []
    for slice_tensor, bounds, owned_start, owned_end in zip(slices, slice_bounds, owned_starts, owned_ends):
        if bounds is None:
            cropped_slices.append(slice_tensor)
            continue

        _, _, temporal_offset = bounds
        offset_value = 0 if temporal_offset is None else temporal_offset
        local_start = owned_start - offset_value
        local_end = owned_end - offset_value
        cropped_slice = slice_tensor.slice_temporal_range(local_start, local_end, adjust_temporal=False)
        if temporal_offset is not None:
            cropped_slice.register_spatial_cache("temporal_offset", temporal_offset)
        cropped_slices.append(cropped_slice)

    return cropped_slices


def _validate_no_legacy_attention_block_cfg(
    cfg: dict[int, dict[str, Any]],
    num_blocks: int,
) -> None:
    """Reject removed sparse-attention overrides inside per-block configs."""
    for block_index in range(num_blocks):
        block_cfg = cfg.get(block_index, {})
        block_attn_mode = block_cfg.get("attn_mode")
        if block_attn_mode is not None and block_attn_mode != "full":
            raise ValueError(
                "Tokenizer sparse attention only supports full attention. "
                f"Got attn_mode={block_attn_mode!r} for block {block_index}."
            )
        if block_cfg.get("window_size") is not None:
            raise ValueError(
                "Tokenizer sparse attention no longer supports windowed attention. "
                f"Got window_size={block_cfg['window_size']!r} for block {block_index}."
            )
        for legacy_key in ("shift_sequence", "shift_window", "serialize_mode"):
            if block_cfg.get(legacy_key) is not None:
                raise ValueError(
                    "Tokenizer sparse attention only supports full attention. "
                    f"Got legacy override {legacy_key}={block_cfg[legacy_key]!r} for block {block_index}."
                )


# =============================================================================
# Distribution Classes
# =============================================================================


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution for VAE latent space.

    Parameterized by mean and log-variance, supports sampling and KL divergence.
    """

    def __init__(
        self,
        parameters: torch.Tensor | SparseTensor,
        deterministic: bool = False,
    ):
        """Initialize distribution from parameters.

        Args:
            parameters: Tensor containing concatenated [mean, logvar] along dim 1.
            deterministic: If True, sampling returns the mean.
        """
        if isinstance(parameters, SparseTensor):
            self.raw_parameters = parameters
            self.parameters = parameters.feats
        else:
            self.raw_parameters = None
            self.parameters = parameters

        self.original_dtype = self.parameters.dtype
        self.parameters = self.parameters.to(torch.float32)
        self.mean, self.logvar = torch.chunk(self.parameters, 2, dim=1)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample from the distribution.

        Args:
            generator: Optional random generator for reproducibility.

        Returns:
            Sampled tensor.
        """
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x.to(self.original_dtype)

    def kl(self, other: "DiagonalGaussianDistribution | None" = None) -> torch.Tensor:
        """Compute KL divergence.

        Args:
            other: Other distribution to compare against (defaults to N(0,1)).

        Returns:
            KL divergence tensor.
        """
        reduce_dims = tuple(range(1, self.mean.ndim))
        if self.deterministic:
            return torch.zeros(
                self.mean.shape[0],
                device=self.parameters.device,
                dtype=self.parameters.dtype,
            )
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=reduce_dims,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=reduce_dims,
                )

    def nll(self, sample: torch.Tensor, dims: tuple[int, ...] = (1, 2, 3)) -> torch.Tensor:
        """Compute negative log-likelihood.

        Args:
            sample: Sample to evaluate.
            dims: Dimensions to sum over.

        Returns:
            NLL tensor.
        """
        if self.deterministic:
            return torch.zeros(
                self.mean.shape[0],
                device=self.parameters.device,
                dtype=self.parameters.dtype,
            )
        logtwopi = np.log(2.0 * np.pi)
        reduce_dims = dims if sample.ndim > max(dims, default=0) else tuple(range(1, sample.ndim))
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=reduce_dims,
        )

    def mode(self) -> torch.Tensor:
        """Return the mode (mean) of the distribution."""
        return self.mean.to(self.original_dtype)


# =============================================================================
# Transformer Base Classes
# =============================================================================


class SparseTransformerBase(nn.Module):
    """Base class for sparse transformer encoder and decoder."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: int | None = None,
        num_head_channels: int | None = 64,
        mlp_channels: int = 2048,
        pe_mode: Literal["ape", "rope", "learned", "learned4d", "joint"] = "rope",
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        use_bias: bool = False,
        use_rms_norm: bool = True,
        ln_affine: bool = True,
        multiscale: dict[int, dict[str, Any]] | None = None,
        dense_train_backend: Literal["disabled", "varlen", "batched", "auto"] = "disabled",
    ):
        """Initialize SparseTransformerBase.

        Args:
            in_channels: Number of input channels.
            model_channels: Hidden dimension size.
            num_blocks: Number of transformer blocks.
            num_heads: Number of attention heads.
            num_head_channels: Channels per head (used if num_heads is None).
            mlp_channels: MLP hidden size.
            pe_mode: Position embedding mode.
            use_checkpoint: Whether to use gradient checkpointing.
            qk_rms_norm: Whether to apply RMS norm to Q/K.
            use_bias: Whether to use bias in linear layers.
            use_rms_norm: Whether to use RMSNorm (vs LayerNorm).
            ln_affine: Whether to use affine parameters in LayerNorm.
            multiscale: Multiscale configuration.
            dense_train_backend: Optional dense tensor backend to use for the
                supported fixed-shape train-time subset.
        """
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_channels = mlp_channels
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        self.use_bias = use_bias
        self.use_rms_norm = use_rms_norm
        self.dense_train_backend = dense_train_backend

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        elif pe_mode == "learned4d":
            self.pos_embedder = LearnedPositionEmbedder4D(model_channels)
        elif pe_mode in ["joint", "learned"]:
            self.pos_embedder = LearnedPositionEmbedder(model_channels)

        self.input_layer = SparseLinear(in_channels, model_channels, bias=use_bias)

        if multiscale is not None:
            cfg = expand_multiscale_cfg(multiscale, num_blocks)
        else:
            cfg = {}

        _validate_no_legacy_attention_block_cfg(cfg=cfg, num_blocks=num_blocks)

        self.blocks = nn.ModuleList(
            [
                SparseTransformerBlock(
                    cfg[i]["model_channels"] if i in cfg else model_channels,
                    num_heads=cfg[i]["num_heads"] if i in cfg else self.num_heads,
                    mlp_channels=cfg[i]["mlp_channels"] if i in cfg else mlp_channels,
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode in ["rope", "joint"]),
                    qk_rms_norm=self.qk_rms_norm,
                    use_bias=self.use_bias,
                    use_rms_norm=self.use_rms_norm,
                    multiscale=cfg.get(i, None),
                    ln_affine=ln_affine,
                    layer_idx=i,
                )
                for i in range(num_blocks)
            ]
        )

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    def _resolve_dense_train_backend(self) -> Literal["varlen", "batched"] | None:
        """Resolve the optional dense-train backend for the current execution mode."""
        if self.dense_train_backend == "disabled":
            return None
        return resolve_dense_backend(
            self.dense_train_backend,
            use_compile=torch.compiler.is_compiling(),
        )

    def forward(
        self,
        x: SparseTensor,
        kv_cache: dict[str, SparseTensor] | None = None,
        kv_cache_size: int | None = None,
        kv_cache_detach: bool = True,
        temporal_causal_mask: bool = False,
        collect_hidden_states: bool = True,
    ) -> tuple[SparseTensor, list[SparseTensor] | None, dict[str, SparseTensor]]:
        """Forward pass through transformer blocks.

        Args:
            x: Input SparseTensor.
            kv_cache: Optional KV cache for autoregressive generation.
            kv_cache_size: Maximum KV cache size.
            kv_cache_detach: Whether tensors stored in the KV cache should be detached.
            temporal_causal_mask: Whether to apply temporal-causal, same-timestep
                bidirectional self-attention in decoder blocks.
            collect_hidden_states: Whether to retain per-block hidden states for
                downstream consumers such as concat_latent or multiscale outputs.

        Returns:
            Tuple of (output, hidden_states, updated_kv_cache).
        """
        hs: list[SparseTensor] | None = [] if collect_hidden_states else None

        input_dtype = next(self.input_layer.parameters()).dtype
        if x.dtype != input_dtype:
            x = x.to(input_dtype)

        h = self.input_layer(x)

        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:]).to(h.dtype)
        elif self.pe_mode in ["learned", "learned4d", "joint"]:
            h = h + self.pos_embedder(x)

        block_dtype = next(self.blocks.parameters()).dtype
        if h.dtype != block_dtype:
            h = h.to(block_dtype)

        if hs is not None:
            hs.append(h)
        dense_train_backend = self._resolve_dense_train_backend() if self.training else None
        can_use_dense_train_path = dense_train_backend is not None and all(
            (
                hs is None,
                kv_cache_size is None,
                not temporal_causal_mask,
                not h.has_special_tokens(),
                all(getattr(block, "multiscale", None) is None for block in self.blocks),
                all(getattr(block.attn, "_type", None) == "self" for block in self.blocks),
            )
        )
        if can_use_dense_train_path:
            q_freqs_cis: torch.Tensor | None = None
            blocks_with_rope = [block for block in self.blocks if getattr(block.attn, "use_rope", False)]
            if blocks_with_rope:
                rope_configs = {
                    (
                        block.attn.rope.head_dim,
                        block.attn.rope.pos_cls_token,
                    )
                    for block in blocks_with_rope
                }
                if len(rope_configs) != 1:
                    can_use_dense_train_path = False
                else:
                    q_freqs_cis = blocks_with_rope[0].attn.rope.compute_freqs_cis(
                        h.coords[:, 1:],
                        has_special_tokens=False,
                    )
            dense_batch_tokens = _sparse_tensor_to_dense_batch_tokens(h) if can_use_dense_train_path else None
            if dense_batch_tokens is None:
                can_use_dense_train_path = False
            if can_use_dense_train_path and dense_batch_tokens is not None:
                dense_feats, used_reshape_fast_path = dense_batch_tokens
                q_seqlen = h.get_batch_seq_lens()
                max_q_seqlen = q_seqlen[0] if q_seqlen else 0
                cu_seqlens_q = h.get_cu_seqlens(device=h.device)
                if dense_train_backend == "varlen":
                    dense_feats = run_varlen_block_stack(
                        self.blocks,
                        dense_feats,
                        q_seqlen=q_seqlen,
                        cu_seqlens_q=cu_seqlens_q,
                        max_q_seqlen=max_q_seqlen,
                        q_freqs_cis=q_freqs_cis,
                    )
                else:
                    dense_feats = run_batched_block_stack(
                        self.blocks,
                        dense_feats,
                        q_freqs_cis=q_freqs_cis,
                    )
                h = h.replace(_dense_batch_tokens_to_flat_features(dense_feats, used_reshape_fast_path))
                empty_kv_cache: dict[str, SparseTensor]
                if kv_cache is None:
                    empty_kv_cache = {}
                else:
                    kv_cache.clear()
                    empty_kv_cache = kv_cache
                return h, hs, empty_kv_cache
        can_use_tensor_no_cache_fast_path = (
            not self.training
            and hs is None
            and kv_cache_size is None
            and not temporal_causal_mask
            and all(callable(getattr(block, "forward_tensor_no_cache", None)) for block in self.blocks)
            and all(getattr(block, "multiscale", None) is None for block in self.blocks)
            and all(getattr(block.attn, "_type", None) == "self" for block in self.blocks)
        )
        if can_use_tensor_no_cache_fast_path:
            q_freqs_cis: torch.Tensor | None = None
            if (
                any(getattr(block.attn, "_debug_capture_enabled", False) for block in self.blocks)
                and h.has_special_tokens()
            ):
                raise AssertionError(
                    "Debug attention capture for the encoder tensor fast path requires image-like tokens without "
                    "special sparse tokens."
                )
            blocks_with_rope = [block for block in self.blocks if getattr(block.attn, "use_rope", False)]
            if blocks_with_rope:
                rope_configs = {
                    (
                        block.attn.rope.head_dim,
                        block.attn.rope.pos_cls_token,
                    )
                    for block in blocks_with_rope
                }
                if len(rope_configs) != 1:
                    can_use_tensor_no_cache_fast_path = False
                else:
                    q_freqs_cis = blocks_with_rope[0].attn.rope.compute_freqs_cis(
                        h.coords[:, 1:],
                        has_special_tokens=h.has_special_tokens(),
                    )
            if can_use_tensor_no_cache_fast_path:
                q_seqlen = h.get_batch_seq_lens()
                max_q_seqlen = max(q_seqlen) if q_seqlen else 0
                cu_seqlens_q = h.get_cu_seqlens(device=h.device)
                feats = h.feats
                for block in self.blocks:
                    feats = block.forward_tensor_no_cache(
                        feats,
                        q_seqlen=q_seqlen,
                        cu_seqlens_q=cu_seqlens_q,
                        max_q_seqlen=max_q_seqlen,
                        q_freqs_cis=q_freqs_cis,
                    )
                h = h.replace(feats)
                empty_kv_cache: dict[str, SparseTensor]
                if kv_cache is None:
                    empty_kv_cache = {}
                else:
                    kv_cache.clear()
                    empty_kv_cache = kv_cache
                return h, hs, empty_kv_cache

        can_use_tensor_flat_kv_fast_path = (
            not self.training
            and hs is None
            and kv_cache_size is not None
            and kv_cache_size > 0
            and not temporal_causal_mask
            and h.shape[0] == 1
            and not h.has_special_tokens()
            and all(callable(getattr(block, "forward_tensor_flat_kv", None)) for block in self.blocks)
            and all(getattr(block, "multiscale", None) is None for block in self.blocks)
            and all(getattr(block.attn, "_type", None) == "self" for block in self.blocks)
        )
        if can_use_tensor_flat_kv_fast_path:
            q_seqlen = h.get_batch_seq_lens()
            max_q_seqlen = max(q_seqlen) if q_seqlen else 0
            cu_seqlens_q = h.get_cu_seqlens(device=h.device)
            current_times = h.coords[:, 1].contiguous()
            q_freqs_cis: torch.Tensor | None = None
            blocks_with_rope = [block for block in self.blocks if getattr(block.attn, "use_rope", False)]
            if blocks_with_rope:
                rope_configs = {
                    (
                        block.attn.rope.head_dim,
                        block.attn.rope.pos_cls_token,
                    )
                    for block in blocks_with_rope
                }
                if len(rope_configs) != 1:
                    can_use_tensor_flat_kv_fast_path = False
                else:
                    q_freqs_cis = blocks_with_rope[0].attn.rope.compute_freqs_cis(
                        h.coords[:, 1:],
                        has_special_tokens=h.has_special_tokens(),
                    )
            if kv_cache is not None:
                for block_idx, block in enumerate(self.blocks):
                    block_cache = kv_cache.get(f"block_{block_idx}", {})
                    if not block.attn._has_compatible_flat_kv_cache_state(block_cache):
                        can_use_tensor_flat_kv_fast_path = False
                        break
            if can_use_tensor_flat_kv_fast_path:
                if kv_cache is None:
                    kv_cache = {}
                feats = h.feats
                for block_idx, block in enumerate(self.blocks):
                    block_cache_key = f"block_{block_idx}"
                    if block_cache_key not in kv_cache:
                        kv_cache[block_cache_key] = {}
                    block_cache = kv_cache[block_cache_key]
                    feats, updated_block_cache = block.forward_tensor_flat_kv(
                        feats,
                        current_times=current_times,
                        q_seqlen=q_seqlen,
                        cu_seqlens_q=cu_seqlens_q,
                        max_q_seqlen=max_q_seqlen,
                        kv_cache=block_cache,
                        kv_cache_size=kv_cache_size,
                        kv_cache_detach=kv_cache_detach,
                        q_freqs_cis=q_freqs_cis,
                    )
                    kv_cache[block_cache_key] = updated_block_cache
                h = h.replace(feats)
                return h, hs, kv_cache

        if kv_cache_size is None and not temporal_causal_mask:
            for block in self.blocks:
                forward_no_cache = getattr(block, "forward_no_cache", None)
                if callable(forward_no_cache):
                    h = forward_no_cache(h)
                else:
                    h, _ = block(
                        h,
                        kv_cache=None,
                        kv_cache_size=None,
                        kv_cache_detach=kv_cache_detach,
                        temporal_causal_mask=False,
                    )

                if hs is not None:
                    hs.append(h)

            empty_kv_cache: dict[str, SparseTensor]
            if kv_cache is None:
                empty_kv_cache = {}
            else:
                kv_cache.clear()
                empty_kv_cache = kv_cache
            return h, hs, empty_kv_cache

        if kv_cache is None:
            kv_cache = {}

        for block_idx, block in enumerate(self.blocks):
            block_cache_key = f"block_{block_idx}"
            if block_cache_key not in kv_cache:
                kv_cache[block_cache_key] = {}

            block_cache = kv_cache[block_cache_key]

            h, updated_block_cache = block(
                h,
                kv_cache=block_cache,
                kv_cache_size=kv_cache_size,
                kv_cache_detach=kv_cache_detach,
                temporal_causal_mask=temporal_causal_mask,
            )

            kv_cache[block_cache_key] = updated_block_cache

            if hs is not None:
                hs.append(h)

        return h, hs, kv_cache


class Encoder(SparseTransformerBase):
    """Sparse Transformer Encoder with optional pretrained weight loading."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: int | None = None,
        num_head_channels: int = 64,
        mlp_channels: int = 2048,
        pe_mode: Literal["ape", "rope", "learned", "learned4d", "joint"] = "ape",
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        use_bias: bool = False,
        use_rms_norm: bool = True,
        pretrained_model: nn.Module | None = None,
        concat_latent: list[int] | None = None,
        use_head: bool = False,
        dense_train_backend: Literal["disabled", "varlen", "batched", "auto"] = "disabled",
    ):
        """Initialize Encoder.

        Args:
            in_channels: Number of input channels.
            model_channels: Hidden dimension size.
            num_blocks: Number of transformer blocks.
            num_heads: Number of attention heads.
            num_head_channels: Channels per head.
            mlp_channels: MLP hidden size.
            pe_mode: Position embedding mode.
            use_checkpoint: Whether to use gradient checkpointing.
            qk_rms_norm: Whether to apply RMS norm to Q/K.
            use_bias: Whether to use bias in linear layers.
            use_rms_norm: Whether to use RMSNorm.
            pretrained_model: Optional pretrained vision model for weight init.
            concat_latent: List of layer indices to concatenate for output.
            use_head: Whether to apply pooling head for image features.
            dense_train_backend: Optional dense tensor backend for the
                supported fixed-shape train-time subset.
        """
        super().__init__(
            in_channels,
            model_channels,
            num_blocks,
            num_heads,
            num_head_channels,
            mlp_channels,
            pe_mode,
            use_checkpoint,
            qk_rms_norm,
            use_bias=use_bias,
            use_rms_norm=use_rms_norm,
            dense_train_backend=dense_train_backend,
        )
        self.concat_latent = concat_latent
        self.use_head = use_head

        if use_rms_norm:
            self.post_layernorm = RMSNorm32(model_channels, eps=1e-6)
        else:
            self.post_layernorm = LayerNorm32(model_channels, elementwise_affine=True, eps=1e-6)

        self.head = SparseMultiheadAttentionPoolingHead(
            hidden_size=model_channels,
            num_attention_heads=self.num_heads,
            intermediate_size=mlp_channels,
            use_bias=use_bias,
            use_rms_norm=use_rms_norm,
            qk_rms_norm=qk_rms_norm,
        )

        if pretrained_model is not None:
            self._initialize_weights(pretrained_model=pretrained_model)

    def _initialize_weights(self, pretrained_model: nn.Module) -> None:
        """Initialize weights from pretrained vision model.

        Args:
            pretrained_model: Pretrained vision model (e.g., SigLIP2).
        """
        pretrained_weight = pretrained_model.state_dict()

        # Copy patch embedding weights
        if "embeddings.patch_embedding.weight" in pretrained_weight:
            s1 = pretrained_weight["embeddings.patch_embedding.weight"].shape[1]
            self.input_layer.weight.data[:, :s1].copy_(pretrained_weight["embeddings.patch_embedding.weight"])
            if hasattr(self.input_layer, "bias") and self.input_layer.bias is not None:
                self.input_layer.bias.data.copy_(pretrained_weight["embeddings.patch_embedding.bias"])

        # Copy position embedding
        if (
            "embeddings.position_embedding.weight" in pretrained_weight
            and getattr(self, "pos_embedder", None) is not None
        ):
            self.pos_embedder.position_embedding.weight.data.copy_(
                pretrained_weight["embeddings.position_embedding.weight"]
            )

        # Copy transformer block weights
        for block_idx, block in enumerate(self.blocks):
            prefix = f"encoder.layers.{block_idx}"

            # Layer norms
            if f"{prefix}.layer_norm1.weight" in pretrained_weight:
                block.norm1.weight.data.copy_(pretrained_weight[f"{prefix}.layer_norm1.weight"])
            if f"{prefix}.layer_norm1.bias" in pretrained_weight:
                block.norm1.bias.data.copy_(pretrained_weight[f"{prefix}.layer_norm1.bias"])
            if f"{prefix}.layer_norm2.weight" in pretrained_weight:
                block.norm2.weight.data.copy_(pretrained_weight[f"{prefix}.layer_norm2.weight"])
            if f"{prefix}.layer_norm2.bias" in pretrained_weight:
                block.norm2.bias.data.copy_(pretrained_weight[f"{prefix}.layer_norm2.bias"])

            # Attention QKV
            if hasattr(block.attn, "to_qkv"):
                q_weight = pretrained_weight.get(f"{prefix}.self_attn.q_proj.weight")
                k_weight = pretrained_weight.get(f"{prefix}.self_attn.k_proj.weight")
                v_weight = pretrained_weight.get(f"{prefix}.self_attn.v_proj.weight")

                if q_weight is not None and k_weight is not None and v_weight is not None:
                    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    block.attn.to_qkv.weight.data.copy_(qkv_weight)

                q_bias = pretrained_weight.get(f"{prefix}.self_attn.q_proj.bias")
                k_bias = pretrained_weight.get(f"{prefix}.self_attn.k_proj.bias")
                v_bias = pretrained_weight.get(f"{prefix}.self_attn.v_proj.bias")

                if q_bias is not None and k_bias is not None and v_bias is not None:
                    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                    block.attn.to_qkv.bias.data.copy_(qkv_bias)

            # Attention output projection
            if hasattr(block.attn, "to_out"):
                if f"{prefix}.self_attn.out_proj.weight" in pretrained_weight:
                    block.attn.to_out.weight.data.copy_(pretrained_weight[f"{prefix}.self_attn.out_proj.weight"])
                if f"{prefix}.self_attn.out_proj.bias" in pretrained_weight:
                    block.attn.to_out.bias.data.copy_(pretrained_weight[f"{prefix}.self_attn.out_proj.bias"])

            # MLP weights
            if hasattr(block.mlp, "mlp") and len(block.mlp.mlp) > 2:
                if f"{prefix}.mlp.fc1.weight" in pretrained_weight:
                    block.mlp.mlp[0].weight.data.copy_(pretrained_weight[f"{prefix}.mlp.fc1.weight"])
                if f"{prefix}.mlp.fc1.bias" in pretrained_weight:
                    block.mlp.mlp[0].bias.data.copy_(pretrained_weight[f"{prefix}.mlp.fc1.bias"])
                if f"{prefix}.mlp.fc2.weight" in pretrained_weight:
                    block.mlp.mlp[2].weight.data.copy_(pretrained_weight[f"{prefix}.mlp.fc2.weight"])
                if f"{prefix}.mlp.fc2.bias" in pretrained_weight:
                    block.mlp.mlp[2].bias.data.copy_(pretrained_weight[f"{prefix}.mlp.fc2.bias"])

        # Post layer norm
        if hasattr(self, "post_layernorm"):
            if "post_layernorm.weight" in pretrained_weight:
                self.post_layernorm.weight.data.copy_(pretrained_weight["post_layernorm.weight"])
            if "post_layernorm.bias" in pretrained_weight:
                self.post_layernorm.bias.data.copy_(pretrained_weight["post_layernorm.bias"])

        # Head weights
        if hasattr(self, "head"):
            self._initialize_head_weights(pretrained_weight)

        logging.info("Successfully initialized encoder weights from pretrained model")

    def _initialize_head_weights(self, pretrained_weight: dict[str, torch.Tensor]) -> None:
        """Initialize pooling head weights from pretrained model.

        Args:
            pretrained_weight: State dict from pretrained model.
        """
        head_prefix = "head"

        if f"{head_prefix}.probe" in pretrained_weight:
            probe_weight = pretrained_weight[f"{head_prefix}.probe"].squeeze()
            self.head.probe.data.copy_(probe_weight)

        # Cross-attention weights
        if hasattr(self.head.attention, "to_q") and hasattr(self.head.attention, "to_kv"):
            if f"{head_prefix}.attention.in_proj_weight" in pretrained_weight:
                in_proj_weight = pretrained_weight[f"{head_prefix}.attention.in_proj_weight"]
                q_weight, k_weight, v_weight = in_proj_weight.chunk(3, dim=0)
                self.head.attention.to_q.weight.data.copy_(q_weight)
                kv_weight = torch.cat([k_weight, v_weight], dim=0)
                self.head.attention.to_kv.weight.data.copy_(kv_weight)

            if f"{head_prefix}.attention.in_proj_bias" in pretrained_weight:
                in_proj_bias = pretrained_weight[f"{head_prefix}.attention.in_proj_bias"]
                q_bias, k_bias, v_bias = in_proj_bias.chunk(3, dim=0)
                self.head.attention.to_q.bias.data.copy_(q_bias)
                kv_bias = torch.cat([k_bias, v_bias], dim=0)
                self.head.attention.to_kv.bias.data.copy_(kv_bias)

        # Attention output projection
        if f"{head_prefix}.attention.out_proj.weight" in pretrained_weight:
            self.head.attention.to_out.weight.data.copy_(pretrained_weight[f"{head_prefix}.attention.out_proj.weight"])
        if f"{head_prefix}.attention.out_proj.bias" in pretrained_weight:
            self.head.attention.to_out.bias.data.copy_(pretrained_weight[f"{head_prefix}.attention.out_proj.bias"])

        # Head layer norm
        if f"{head_prefix}.layernorm.weight" in pretrained_weight:
            self.head.layernorm.weight.data.copy_(pretrained_weight[f"{head_prefix}.layernorm.weight"])
        if f"{head_prefix}.layernorm.bias" in pretrained_weight:
            self.head.layernorm.bias.data.copy_(pretrained_weight[f"{head_prefix}.layernorm.bias"])

        # Head MLP
        if hasattr(self.head.mlp, "mlp") and len(self.head.mlp.mlp) > 2:
            if f"{head_prefix}.mlp.fc1.weight" in pretrained_weight:
                self.head.mlp.mlp[0].weight.data.copy_(pretrained_weight[f"{head_prefix}.mlp.fc1.weight"])
            if f"{head_prefix}.mlp.fc1.bias" in pretrained_weight:
                self.head.mlp.mlp[0].bias.data.copy_(pretrained_weight[f"{head_prefix}.mlp.fc1.bias"])
            if f"{head_prefix}.mlp.fc2.weight" in pretrained_weight:
                self.head.mlp.mlp[2].weight.data.copy_(pretrained_weight[f"{head_prefix}.mlp.fc2.weight"])
            if f"{head_prefix}.mlp.fc2.bias" in pretrained_weight:
                self.head.mlp.mlp[2].bias.data.copy_(pretrained_weight[f"{head_prefix}.mlp.fc2.bias"])

    def forward(
        self,
        x: SparseTensor,
        kv_cache: dict[str, SparseTensor] | None = None,
    ) -> tuple[SparseTensor, dict[str, SparseTensor]] | tuple[SparseTensor, torch.Tensor, dict[str, SparseTensor]]:
        """Forward pass through encoder.

        Args:
            x: Input SparseTensor.
            kv_cache: Optional KV cache.

        Returns:
            If use_head: (encoded, pooled_features, kv_cache).
            Otherwise: (encoded, kv_cache).
        """
        collect_hidden_states = self.concat_latent is not None
        h, hs, kv_cache = super().forward(
            x,
            kv_cache=kv_cache,
            collect_hidden_states=collect_hidden_states,
        )
        h = h.replace(self.post_layernorm(h.feats))

        if self.concat_latent is not None:
            assert hs is not None, "concat_latent requires hidden states to be collected"
            assert all(0 <= idx < len(hs) for idx in self.concat_latent), (
                f"concat_latent indices {self.concat_latent} must be within [0, {len(hs) - 1}]"
            )
            hs_concat = [hs[i].feats for i in self.concat_latent]
            hs_concat = torch.cat(hs_concat, dim=1)
            h = h.replace(hs_concat)

        if self.use_head:
            pooler_output = self.head(h)
            return h, pooler_output.feats, kv_cache
        else:
            return h, kv_cache


class Decoder(SparseTransformerBase):
    """Sparse Transformer Decoder with multiscale output support."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: int | None = None,
        num_head_channels: int = 64,
        mlp_channels: int = 2048,
        pe_mode: Literal["ape", "rope", "learned", "learned4d", "joint"] = "ape",
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        use_bias: bool = False,
        use_rms_norm: bool = True,
        multiscale: dict[int, dict[str, Any]] | None = None,
        multiscale_outputs: dict[int, dict[str, Any]] | None = None,
        dense_train_backend: Literal["disabled", "varlen", "batched", "auto"] = "disabled",
    ):
        """Initialize Decoder.

        Args:
            in_channels: Number of input channels (latent dimension).
            out_channels: Number of output channels.
            model_channels: Hidden dimension size.
            num_blocks: Number of transformer blocks.
            num_heads: Number of attention heads.
            num_head_channels: Channels per head.
            mlp_channels: MLP hidden size.
            pe_mode: Position embedding mode.
            use_checkpoint: Whether to use gradient checkpointing.
            qk_rms_norm: Whether to apply RMS norm to Q/K.
            use_bias: Whether to use bias in linear layers.
            use_rms_norm: Whether to use RMSNorm.
            multiscale: Multiscale expansion configuration.
            multiscale_outputs: Multiscale output layer configuration.
            dense_train_backend: Optional dense tensor backend for the
                supported fixed-shape train-time subset.
        """
        super().__init__(
            in_channels,
            model_channels,
            num_blocks,
            num_heads,
            num_head_channels,
            mlp_channels,
            pe_mode,
            use_checkpoint,
            qk_rms_norm,
            use_bias=use_bias,
            use_rms_norm=use_rms_norm,
            multiscale=multiscale,
            dense_train_backend=dense_train_backend,
        )
        self.multiscale = multiscale
        self.multiscale_outputs = multiscale_outputs

        # Select last multiscale channel configuration
        if multiscale is not None and multiscale_outputs is None:
            last_layer = max(multiscale.keys())
            model_channels = multiscale[last_layer]["model_channels"]

        if use_rms_norm:
            self.out_norm = RMSNorm32(model_channels, eps=1e-6)
        else:
            self.out_norm = LayerNorm32(model_channels, elementwise_affine=False, eps=1e-6)

        self.out_layer = SparseLinear(
            model_channels,
            out_channels,
            bias=False,
        )

        if multiscale is not None:
            self.recover_factor = multiply_all_factors(multiscale)

        if multiscale_outputs is not None:
            self.multiscale_out_layers = nn.ModuleList()
            self.multiscale_out_norms = nn.ModuleList()
            self.recover_factors = []

            for cfg in multiscale_outputs[1:]:
                if use_rms_norm:
                    out_norm = RMSNorm32(cfg["model_channels"], eps=1e-6)
                else:
                    out_norm = LayerNorm32(cfg["model_channels"], elementwise_affine=False, eps=1e-6)

                out_layer = SparseLinear(
                    cfg["model_channels"],
                    cfg["out_channels"],
                    bias=False,
                )

                scale = {k: v for k, v in multiscale.items() if k < cfg["layer_id"]}
                recover_factor = multiply_all_factors(scale)

                self.multiscale_out_norms.append(out_norm)
                self.multiscale_out_layers.append(out_layer)
                self.recover_factors.append(recover_factor)

    def forward(
        self,
        x: SparseTensor,
        kv_cache: dict[str, SparseTensor] | None = None,
        kv_cache_size: int | None = None,
        kv_cache_detach: bool = True,
        temporal_causal_mask: bool = False,
    ) -> tuple[SparseTensor, dict[str, SparseTensor]]:
        """Forward pass through decoder.

        Args:
            x: Input SparseTensor (latent representation).
            kv_cache: Optional KV cache.
            kv_cache_size: Maximum KV cache size.
            kv_cache_detach: Whether tensors stored in the KV cache should be detached.
            temporal_causal_mask: Whether to apply temporal-causal, same-timestep
                bidirectional self-attention in decoder blocks.

        Returns:
            Tuple of (decoded output, updated kv_cache).
        """
        collect_hidden_states = self.multiscale_outputs is not None
        h, hs, kv_cache = super().forward(
            x,
            kv_cache=kv_cache,
            kv_cache_size=kv_cache_size,
            kv_cache_detach=kv_cache_detach,
            temporal_causal_mask=temporal_causal_mask,
            collect_hidden_states=collect_hidden_states,
        )

        if self.multiscale_outputs is not None:
            assert hs is not None, "multiscale_outputs requires hidden states to be collected"
            requested_layer_ids = [cfg["layer_id"] for cfg in self.multiscale_outputs]
            assert all(0 <= layer_id < len(hs) for layer_id in requested_layer_ids), (
                f"multiscale layer_ids {requested_layer_ids} must be within [0, {len(hs) - 1}]"
            )
            h = hs[self.multiscale_outputs[0]["layer_id"]]
            h = h.replace(self.out_norm(h.feats))
            h = self.out_layer(h)

            for i, cfg in enumerate(self.multiscale_outputs[1:]):
                h_i = hs[cfg["layer_id"]]
                h_i = h_i.replace(self.multiscale_out_norms[i](h_i.feats))
                h_i = self.multiscale_out_layers[i](h_i)
                if self.multiscale is not None:
                    h_i = h_i.shrink_by_factors(self.recover_factors[i])
                h = h + h_i

            h = h.replace(h.feats / len(self.multiscale_outputs))

        else:
            h = h.replace(self.out_norm(h.feats))
            h = self.out_layer(h)

            if self.multiscale is not None:
                h = h.shrink_by_factors(self.recover_factor)

        return h, kv_cache


@dataclass
class AutoencoderKLConfig:
    """Configuration for AutoencoderKL model."""

    patch_size: tuple[int, int, int] = (1, 8, 8)
    in_channels: int = 192
    out_channels: int = 192
    latent_channels: int = 8
    encoder_model_channels: int = 768
    encoder_num_blocks: int = 12
    encoder_num_heads: int | None = None
    encoder_mlp_channels: float = 2048
    encoder_pe_mode: str = "rope"
    encoder_qk_rms_norm: bool = True
    encoder_use_bias: bool = True
    encoder_use_rms_norm: bool = False
    decoder_model_channels: int = 768
    decoder_num_blocks: int = 12
    decoder_num_heads: int | None = None
    decoder_mlp_channels: float = 2048
    decoder_pe_mode: str = "rope"
    decoder_qk_rms_norm: bool = True
    decoder_use_bias: bool = True
    decoder_use_rms_norm: bool = False
    decoder_multiscale: dict[int, dict[str, Any]] | None = None
    decoder_multiscale_outputs: dict[int, dict[str, Any]] | None = None
    use_decoder: bool = True
    use_quantizer: bool = False
    quantizer_type: Literal["fsq", "lfq", "rq"] = "rq"
    quantizer_codebook_size: int = 65536
    quantizer_num_codebooks: int = 1
    quantizer_feature_dim: int = 48
    quantizer_chunk_size: int = 1
    use_text_alignment: bool = False
    use_post_text_alignment: bool = False
    use_text_decoder: bool = False
    use_post_text_decoder: bool = False
    spatial_pool_size: int = 1
    text_decoder_model_name: str | None = None  # e.g., "Qwen/Qwen3-0.6B" or local path
    text_decoder_family: str = "qwen3"
    text_decoder_gradient_checkpointing: bool = True
    encoder_use_checkpoint: bool | None = None
    decoder_use_checkpoint: bool | None = None
    encoder_dense_train_backend: Literal["disabled", "varlen", "batched", "auto"] = "disabled"
    decoder_dense_train_backend: Literal["disabled", "varlen", "batched", "auto"] = "disabled"
    decoder_temporal_mode: Literal["bidirectional", "causal_mask", "causal"] = "bidirectional"
    decoder_temporal_query_latent_steps: int = 1
    decoder_temporal_cache_latent_steps: int | None = None
    decoder_temporal_detach_cache: bool = True
    inference_num_sample_frames_batch_size: int = 16
    inference_num_sample_frames_stride: int = 16
    inference_kv_cache_size: int = 0
    task_decode_runtime_settings: dict[str, dict[str, int]] | None = None
    use_vf_loss: bool = False
    freeze_encoder: bool = False
    pretrained_model_name: str | None = None
    concat_latent: list | None = None
    random_num_sample_frames_batch_sizes: list[int] | None = None
    task_random_num_sample_frames_batch_sizes: dict[str, list[int]] | None = None
    use_dual_latent: bool = False


class AutoencoderKL(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """Sparse Autoencoder with KL loss for image/video encoding.

    This model encodes images/videos into latent representations using
    a sparse transformer architecture with SigLIP2 pretrained features.

    Args:
        patch_size: Patch size for tokenization (T, H, W).
        in_channels: Number of input channels per patch.
        out_channels: Number of output channels per patch.
        latent_channels: Number of latent channels.
        encoder_*: Encoder configuration parameters.
        decoder_*: Decoder configuration parameters.
        use_quantizer: Whether to use quantization.
        quantizer_type: Type of quantizer ("fsq", "lfq", "rq").
        use_text_alignment: Whether to enable text alignment.
        pretrained_model_name: HuggingFace model name for SigLIP2.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D"]

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 8, 8),
        in_channels: int = 192,
        out_channels: int = 192,
        latent_channels: int = 8,
        encoder_model_channels: int = 768,
        encoder_num_blocks: int = 12,
        encoder_num_heads: int | None = None,
        encoder_mlp_channels: float = 2048,
        encoder_pe_mode: str = "rope",
        encoder_qk_rms_norm: bool = True,
        encoder_use_bias: bool = False,
        encoder_use_rms_norm: bool = True,
        decoder_model_channels: int = 768,
        decoder_num_blocks: int = 12,
        decoder_num_heads: int | None = None,
        decoder_mlp_channels: float = 2048,
        decoder_pe_mode: str = "rope",
        decoder_qk_rms_norm: bool = True,
        decoder_use_bias: bool = False,
        decoder_use_rms_norm: bool = True,
        decoder_multiscale: dict[int, dict[str, Any]] | None = None,
        decoder_multiscale_outputs: dict[int, dict[str, Any]] | None = None,
        use_decoder: bool = True,
        use_quantizer: bool = False,
        quantizer_type: Literal["fsq", "lfq", "rq"] = "rq",
        quantizer_codebook_size: int = 16384,
        quantizer_num_codebooks: int = 1,
        quantizer_feature_dim: int = 48,
        quantizer_chunk_size: int = 1,
        use_text_alignment: bool = False,
        use_post_text_alignment: bool = False,
        use_text_decoder: bool = False,
        use_post_text_decoder: bool = False,
        spatial_pool_size: int = 1,
        text_decoder_model_name: str | None = None,
        text_decoder_family: str = "qwen3",
        text_decoder_gradient_checkpointing: bool = True,
        encoder_use_checkpoint: bool | None = None,
        decoder_use_checkpoint: bool | None = None,
        encoder_dense_train_backend: Literal["disabled", "varlen", "batched", "auto"] = "disabled",
        decoder_dense_train_backend: Literal["disabled", "varlen", "batched", "auto"] = "disabled",
        decoder_temporal_mode: Literal["bidirectional", "causal_mask", "causal"] = "bidirectional",
        decoder_temporal_query_latent_steps: int = 1,
        decoder_temporal_cache_latent_steps: int | None = None,
        decoder_temporal_detach_cache: bool = True,
        inference_num_sample_frames_batch_size: int = 16,
        inference_num_sample_frames_stride: int = 16,
        inference_kv_cache_size: int = 0,
        task_decode_runtime_settings: dict[str, dict[str, int]] | None = None,
        use_vf_loss: bool = False,
        freeze_encoder: bool = False,
        pretrained_model_name: str | None = None,
        concat_latent: list | None = None,
        random_num_sample_frames_batch_sizes: list[int] | None = None,
        task_random_num_sample_frames_batch_sizes: dict[str, list[int]] | None = None,
        use_dual_latent: bool = False,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.encoder_use_checkpoint = use_checkpoint if encoder_use_checkpoint is None else encoder_use_checkpoint
        self.decoder_use_checkpoint = use_checkpoint if decoder_use_checkpoint is None else decoder_use_checkpoint
        self.encoder_dense_train_backend = encoder_dense_train_backend
        self.decoder_dense_train_backend = decoder_dense_train_backend
        self.patch_size = patch_size
        self.use_text_alignment = use_text_alignment
        self.use_post_text_alignment = use_post_text_alignment
        self.use_text_decoder = use_text_decoder
        self.use_post_text_decoder = use_post_text_decoder
        self.spatial_pool_size = spatial_pool_size
        self.text_decoder_family = text_decoder_family
        self.decoder_temporal_mode = decoder_temporal_mode
        self.decoder_temporal_query_latent_steps = decoder_temporal_query_latent_steps
        self.decoder_temporal_cache_latent_steps = decoder_temporal_cache_latent_steps
        self.decoder_temporal_detach_cache = decoder_temporal_detach_cache
        self.inference_num_sample_frames_batch_size = inference_num_sample_frames_batch_size
        self.inference_num_sample_frames_stride = inference_num_sample_frames_stride
        self.inference_kv_cache_size = inference_kv_cache_size
        self.task_decode_runtime_settings = task_decode_runtime_settings
        self.use_quantizer = use_quantizer
        self.quantizer_type = quantizer_type
        self.quantizer_codebook_size = quantizer_codebook_size
        self.quantizer_num_codebooks = quantizer_num_codebooks
        self.quantizer_feature_dim = quantizer_feature_dim
        self.use_vf_loss = use_vf_loss
        self.freeze_encoder = freeze_encoder
        self.quantizer_chunk_size = quantizer_chunk_size
        self.use_dual_latent = use_dual_latent

        self.latent_channels = latent_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.random_num_sample_frames_batch_sizes = random_num_sample_frames_batch_sizes
        self.task_random_num_sample_frames_batch_sizes = task_random_num_sample_frames_batch_sizes
        self.num_sample_frames_batch_size = 16
        self.num_sample_frames_stride = 12
        self.kv_cache_size = 4
        self._logged_decoder_temporal_plan = False

        # Load SigLIP2 pretrained model (text encoder always needed for text alignment)
        # Use HF_HOME env var for cache, with local_files_only to avoid network downloads
        hf_cache_dir = os.environ.get("HF_HOME")
        local_files_only = hf_cache_dir is not None
        pretrained_model = None
        pretrained_vision_model = None
        need_pretrained_for_text = use_text_alignment or use_post_text_alignment
        if pretrained_model_name is not None or need_pretrained_for_text:
            model_name = pretrained_model_name or "google/siglip2-so400m-patch16-naflex"
            pretrained_model = Siglip2Model.from_pretrained(
                model_name,
                cache_dir=hf_cache_dir,
                local_files_only=local_files_only,
            )
            if pretrained_model_name is not None:
                pretrained_vision_model = pretrained_model.vision_model

        # Initialize encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            model_channels=encoder_model_channels,
            num_blocks=encoder_num_blocks,
            num_heads=encoder_num_heads,
            mlp_channels=encoder_mlp_channels,
            pe_mode=encoder_pe_mode,
            qk_rms_norm=encoder_qk_rms_norm,
            use_bias=encoder_use_bias,
            use_rms_norm=encoder_use_rms_norm,
            pretrained_model=pretrained_vision_model,
            concat_latent=concat_latent,
            use_checkpoint=self.encoder_use_checkpoint,
            dense_train_backend=self.encoder_dense_train_backend,
        )

        # Initialize teacher encoder (frozen) — only needed for ITD loss
        # which requires use_text_alignment=True. Skip when only using text decoder (ITG)
        # to save ~3GB GPU memory.
        if use_text_alignment:
            self.teacher_encoder = Encoder(
                in_channels=in_channels,
                model_channels=encoder_model_channels,
                num_blocks=encoder_num_blocks,
                num_heads=encoder_num_heads,
                mlp_channels=encoder_mlp_channels,
                pe_mode="learned",
                qk_rms_norm=False,
                use_bias=True,
                use_rms_norm=False,
                pretrained_model=pretrained_vision_model,
                use_head=True,
                dense_train_backend="disabled",
            )
            self.teacher_encoder.requires_grad_(False)

        # Projection layer
        if concat_latent is not None:
            proj_in_channels = encoder_model_channels * len(concat_latent)
        else:
            proj_in_channels = encoder_model_channels

        self.proj = SparseLinear(proj_in_channels, 2 * latent_channels)

        # Initialize quantizer
        if use_quantizer:
            if self.quantizer_type == "lfq":
                self.quantizer = LFQ(
                    dim=latent_channels,
                    codebook_size=self.quantizer_codebook_size // self.quantizer_chunk_size,
                    num_codebooks=self.quantizer_num_codebooks,
                    sample_minimization_weight=1.0,
                    batch_maximization_weight=1.0,
                    token_factorization=False,
                    factorized_bits=[9, 9],
                )
            elif self.quantizer_type == "rq":
                self.quantizer = RQBottleneck(
                    latent_shape=(16, 16, latent_channels),
                    code_shape=(16, 16, 4),
                    n_embed=self.quantizer_codebook_size,
                    decay=0.99,
                    shared_codebook=True,
                    restart_unused_codes=True,
                )
            else:
                levels, _ = levels_from_codebook_size(self.quantizer_codebook_size)
                self.quantizer = FSQ(
                    levels=levels,
                    dim=self.quantizer_feature_dim // self.quantizer_chunk_size,
                    num_codebooks=self.quantizer_num_codebooks,
                )

        # Initialize decoder
        if use_decoder:
            self.decoder = Decoder(
                in_channels=latent_channels,
                out_channels=out_channels,
                model_channels=decoder_model_channels,
                num_blocks=decoder_num_blocks,
                num_heads=decoder_num_heads,
                mlp_channels=decoder_mlp_channels,
                qk_rms_norm=decoder_qk_rms_norm,
                use_bias=decoder_use_bias,
                use_rms_norm=decoder_use_rms_norm,
                pe_mode=decoder_pe_mode,
                use_checkpoint=self.decoder_use_checkpoint,
                multiscale=decoder_multiscale,
                multiscale_outputs=decoder_multiscale_outputs,
                dense_train_backend=self.decoder_dense_train_backend,
            )

        if use_decoder and use_dual_latent:
            self.discrete_decoder = Decoder(
                in_channels=latent_channels,
                out_channels=out_channels,
                model_channels=decoder_model_channels,
                num_blocks=decoder_num_blocks,
                num_heads=decoder_num_heads,
                mlp_channels=decoder_mlp_channels,
                qk_rms_norm=decoder_qk_rms_norm,
                use_bias=decoder_use_bias,
                use_rms_norm=decoder_use_rms_norm,
                pe_mode=decoder_pe_mode,
                use_checkpoint=self.decoder_use_checkpoint,
                multiscale=decoder_multiscale,
                multiscale_outputs=decoder_multiscale_outputs,
                dense_train_backend=self.decoder_dense_train_backend,
            )

        self.use_slicing = False
        self.logit_bias = None
        self.logit_scale = None

        # Text alignment
        if self.use_text_alignment:
            self.text_encoder = pretrained_model.text_model
            self.text_encoder.requires_grad_(False)

            self.logit_scale = nn.Parameter(pretrained_model.logit_scale.data.clone())
            self.logit_bias = nn.Parameter(pretrained_model.logit_bias.data.clone())

            self.teacher_logit_scale = nn.Parameter(pretrained_model.logit_scale.data.clone())
            self.teacher_logit_bias = nn.Parameter(pretrained_model.logit_bias.data.clone())
            self.teacher_logit_scale.requires_grad = False
            self.teacher_logit_bias.requires_grad = False
        else:
            self.teacher_logit_bias = None
            self.teacher_logit_scale = None

        if self.use_post_text_alignment:
            self.post_logit_scale = nn.Parameter(pretrained_model.logit_scale.data.clone())
            self.post_logit_bias = nn.Parameter(pretrained_model.logit_bias.data.clone())
            self.post_alignment_proj = SparseLinear(latent_channels, proj_in_channels)
            self.post_alignment_head = SparseMultiheadAttentionPoolingHead(
                hidden_size=encoder_model_channels,
                num_attention_heads=encoder_num_heads,
                intermediate_size=encoder_mlp_channels,
                use_bias=encoder_use_bias,
                use_rms_norm=encoder_use_rms_norm,
                qk_rms_norm=False,
            )
        else:
            self.post_logit_scale = None
            self.post_logit_bias = None

        # Text decoder (Qwen3-based causal LM for image-to-text generation)
        if self.use_text_decoder and text_decoder_model_name is not None:
            from cosmos.model.tokenizer.models.text_decoder import (
                TextDecoderWrapper,
                get_text_decoder_family_spec,
            )

            self.text_decoder_wrapper = TextDecoderWrapper(
                model_name=text_decoder_model_name,
                image_hidden_size=encoder_model_channels,
                spatial_pool_size=spatial_pool_size,
                gradient_checkpointing=text_decoder_gradient_checkpointing,
                family_spec=get_text_decoder_family_spec(
                    family=text_decoder_family,
                    model_name=text_decoder_model_name,
                ),
            )
        else:
            self.text_decoder_wrapper = None

    def decode_text(
        self,
        input_ids: torch.Tensor,
        image_feats: "SparseTensor",
        image_patch_indices: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
    ):
        """Decode text from image features using the configured text decoder.

        Passes encoder output (x_no_proj) through spatial merger + causal LM.

        Args:
            input_ids: [B, S] text token IDs with vision placeholders.
            image_feats: SparseTensor from encoder output (x_no_proj).
            image_patch_indices: [N_pooled] flat indices into [B*S].
            segment_ids: [B, S] segment IDs for packed sequences. Enables
                segment-isolated attention with per-segment position reset.

        Returns:
            Tuple of (lm_logits [B, S, vocab_size], num_pooled_tokens int).
        """
        if self.text_decoder_wrapper is None:
            raise RuntimeError("Text decoder not initialized. Set use_text_decoder=True and text_decoder_model_name.")
        return self.text_decoder_wrapper(
            input_ids=input_ids,
            image_feats_tensor=image_feats.feats,
            image_coords=image_feats.coords,
            image_patch_indices=image_patch_indices,
            image_layout=image_feats.layout if hasattr(image_feats, "layout") else None,
            segment_ids=segment_ids,
        )

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Encode text using the text encoder.

        Args:
            text: Input text token IDs.
            normalize: Whether to L2 normalize the output.

        Returns:
            Text embeddings.
        """
        text_outputs = self.text_encoder(text)
        pooled_output = text_outputs.pooler_output
        return F.normalize(pooled_output, dim=-1) if normalize else pooled_output

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def enable_slicing(self):
        """Enable sliced VAE decoding for memory efficiency."""
        self.use_slicing = True

    def disable_slicing(self):
        """Disable sliced VAE decoding."""
        self.use_slicing = False

    def _encode(
        self,
        x: SparseTensor,
        normalize: bool = False,
        compute_image_feat: bool = True,
    ) -> tuple[SparseTensor, torch.Tensor | None, SparseTensor]:
        """Internal encode method with temporal batching.

        Args:
            x: Input SparseTensor.
            normalize: Whether to normalize image features.
            compute_image_feat: Whether to compute image features via encoder.head.
                Set to False for reconstruction-only tasks to save computation.

        Returns:
            Tuple of (projected latent, image features or None, unprojected encoder output).
        """
        if self.training and self.random_num_sample_frames_batch_sizes is not None:
            num_sample_frames_batch_size = np.random.choice(self.random_num_sample_frames_batch_sizes)
        else:
            num_sample_frames_batch_size = self.num_sample_frames_batch_size

        frame_batch_size = num_sample_frames_batch_size // self.patch_size[0]

        temporal_slices = x.split_by_temporal_batches(frame_batch_size, adjust_temporal=True)
        processed_slices = []

        for x_slice in temporal_slices:
            if x_slice.coords.shape[0] > 0:
                if self.freeze_encoder:
                    with torch.no_grad():
                        enc_slice, _ = self.encoder(x_slice)
                else:
                    enc_slice, _ = self.encoder(x_slice)
                processed_slices.append(enc_slice)
            else:
                processed_slices.append(x_slice)

        enc_full = reconstruct_from_temporal_slices(processed_slices, target_coords=x.coords, use_cached_offsets=True)

        # Only compute image features if needed (e.g., for image-text alignment tasks)
        # Skip for reconstruction-only tasks to save computation
        image_feat = None
        if compute_image_feat:
            image_feat = self.encoder.head(enc_full).feats
            if normalize:
                image_feat = F.normalize(image_feat, dim=-1)

        enc_proj = self.proj(enc_full)

        return enc_proj, image_feat, enc_full

    @apply_forward_hook
    def encode(
        self,
        x: SparseTensor | torch.Tensor,
        normalize: bool = False,
        return_dict: bool = False,
        compute_image_feat: bool = True,
    ) -> tuple[SparseTensor, torch.Tensor | None, SparseTensor]:
        """Encode input into latent representation.

        Args:
            x: Input tensor or SparseTensor.
            normalize: Whether to normalize image features.
            return_dict: Unused, kept for compatibility.
            compute_image_feat: Whether to compute image features via encoder.head.
                Set to False for reconstruction-only tasks to save computation.

        Returns:
            Tuple of (latent, image_features or None, encoder_output).
        """
        del return_dict
        if self.use_slicing and isinstance(x, torch.Tensor) and x.shape[0] > 1:
            raise ValueError("Legacy tensor slicing not implemented yet")

        if isinstance(x, torch.Tensor):
            x = batch_tensor_to_sparse(x, self.patch_size)

        x, image_feat, x_no_proj = self._encode(x, normalize=normalize, compute_image_feat=compute_image_feat)
        return x, image_feat, x_no_proj

    def _decode(
        self,
        z: SparseTensor,
        return_dict: bool = True,
        training: bool = True,
        discrete_decoder: bool = False,
    ) -> DecoderOutput | SparseTensor:
        """Internal decode method with temporal batching or causal-mask decoding.

        Args:
            z: Latent SparseTensor.
            return_dict: Whether to return DecoderOutput.
            training: Whether in training mode.
            discrete_decoder: Whether to use discrete decoder.

        Returns:
            Decoded output.
        """
        decoder_temporal_mode = "causal_mask" if self.decoder_temporal_mode == "causal" else self.decoder_temporal_mode

        if decoder_temporal_mode == "causal_mask" and training:
            if not self._logged_decoder_temporal_plan:
                logging.info(
                    "Decoder temporal plan: mode=causal_mask, full-clip masked training, inference uses KV cache."
                )
                self._logged_decoder_temporal_plan = True

            decoder_module = self.discrete_decoder if discrete_decoder else self.decoder
            dec, _ = decoder_module(
                z,
                kv_cache=None,
                kv_cache_size=None,
                kv_cache_detach=True,
                temporal_causal_mask=True,
            )

            if not return_dict:
                return (dec,)

            return DecoderOutput(sample=dec)

        frame_batch_size, frame_batch_strides, kv_cache_size, kv_cache_detach = self._get_decode_temporal_plan(
            z=z,
            training=training,
        )

        decoder_kv_cache_size: int | None = None if kv_cache_size == 0 else kv_cache_size
        kv_cache = None

        temporal_slices = z.split_by_temporal_batches(
            frame_batch_size,
            frame_batch_strides,
            adjust_temporal=True,
            offset=kv_cache_size,
        )

        processed_slices = []
        for z_slice in temporal_slices:
            if z_slice.coords.shape[0] > 0:
                if discrete_decoder:
                    dec_slice, updated_kv_cache = self.discrete_decoder(
                        z_slice,
                        kv_cache if decoder_kv_cache_size is not None else None,
                        decoder_kv_cache_size,
                        kv_cache_detach=kv_cache_detach,
                        temporal_causal_mask=False,
                    )
                else:
                    dec_slice, updated_kv_cache = self.decoder(
                        z_slice,
                        kv_cache if decoder_kv_cache_size is not None else None,
                        decoder_kv_cache_size,
                        kv_cache_detach=kv_cache_detach,
                        temporal_causal_mask=False,
                    )
                if decoder_kv_cache_size is not None:
                    kv_cache = updated_kv_cache
                processed_slices.append(dec_slice)
            else:
                processed_slices.append(z_slice)

        if not training and frame_batch_size > frame_batch_strides:
            processed_slices = _crop_temporal_slices_to_ownership(
                processed_slices,
                frame_batch_size=frame_batch_size,
                frame_batch_strides=frame_batch_strides,
            )

        dec = reconstruct_from_temporal_slices(processed_slices, target_coords=z.coords, use_cached_offsets=True)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def _get_decode_temporal_plan(
        self,
        z: SparseTensor,
        training: bool,
    ) -> tuple[int, int, int, bool]:
        """Resolve temporal decode scheduling.

        Returns:
            Tuple of (query_latent_steps, stride_latent_steps, cache_latent_steps, detach_cache).
        """
        decoder_temporal_mode = "causal_mask" if self.decoder_temporal_mode == "causal" else self.decoder_temporal_mode

        if decoder_temporal_mode == "causal_mask":
            query_latent_steps = self.decoder_temporal_query_latent_steps
            if query_latent_steps != 1:
                raise ValueError(
                    "Causal-mask decoder currently requires decoder_temporal_query_latent_steps=1; "
                    f"got {query_latent_steps}. Multi-step causal windows are not implemented yet."
                )

            auto_cache_resolution = self.decoder_temporal_cache_latent_steps is None
            if self.decoder_temporal_cache_latent_steps is None:
                min_t, max_t = z.get_temporal_range()
                cache_latent_steps = 0 if min_t is None or max_t is None else max_t - min_t + 1
            else:
                cache_latent_steps = self.decoder_temporal_cache_latent_steps

            if cache_latent_steps < 0:
                raise ValueError(
                    f"decoder_temporal_cache_latent_steps must be non-negative or None, got {cache_latent_steps}"
                )

            if not self._logged_decoder_temporal_plan:
                logging.info(
                    "Decoder temporal plan: mode=causal_mask, "
                    f"query_latent_steps={query_latent_steps}, "
                    f"cache_latent_steps={cache_latent_steps}"
                    f"{' (auto)' if auto_cache_resolution else ''}, "
                    f"detach_cache={self.decoder_temporal_detach_cache}"
                )
                self._logged_decoder_temporal_plan = True

            return (
                query_latent_steps,
                query_latent_steps,
                cache_latent_steps,
                self.decoder_temporal_detach_cache,
            )

        if training and self.random_num_sample_frames_batch_sizes is not None:
            num_sample_frames_batch_size = np.random.choice(self.random_num_sample_frames_batch_sizes)
        else:
            num_sample_frames_batch_size = self.num_sample_frames_batch_size

        if training:
            frame_batch_size = num_sample_frames_batch_size // self.patch_size[0]
            frame_batch_strides = frame_batch_size
            kv_cache_size = 0
        else:
            frame_batch_size = self.inference_num_sample_frames_batch_size // self.patch_size[0]
            frame_batch_strides = self.inference_num_sample_frames_stride // self.patch_size[0]
            kv_cache_size = self.inference_kv_cache_size // self.patch_size[0]
            if frame_batch_size < frame_batch_strides:
                raise ValueError(
                    "Non-causal inference requires inference_num_sample_frames_batch_size >= "
                    "inference_num_sample_frames_stride; "
                    f"got batch={self.inference_num_sample_frames_batch_size}, "
                    f"stride={self.inference_num_sample_frames_stride}."
                )

        return frame_batch_size, frame_batch_strides, kv_cache_size, True

    @apply_forward_hook
    def decode(
        self,
        z: SparseTensor | torch.Tensor,
        return_dict: bool = True,
        return_batched_tensor: bool = False,
        training: bool = True,
        discrete_decoder: bool = False,
    ) -> DecoderOutput | SparseTensor:
        """Decode latent representation.

        Args:
            z: Latent tensor or SparseTensor.
            return_dict: Whether to return DecoderOutput.
            return_batched_tensor: Whether to return as batched tensor.
            training: Whether in training mode.
            discrete_decoder: Whether to use discrete decoder.

        Returns:
            Decoded output.
        """
        if self.use_slicing and isinstance(z, torch.Tensor) and z.shape[0] > 1:
            raise ValueError("Legacy tensor slicing not implemented yet")
        else:
            decoded = self._decode(z, training=training, discrete_decoder=discrete_decoder).sample

        if return_batched_tensor and isinstance(decoded, SparseTensor):
            patch_volume = int(np.prod(self.patch_size))
            channels = self.out_channels // patch_volume if self.out_channels % patch_volume == 0 else 3
            decoded_batched = sparse_to_batched_tensor(decoded, self.patch_size, channels=channels)
            if decoded_batched is not None:
                decoded = rearrange(decoded_batched, "b t c h w -> b t h w c")
                if decoded.shape[1] == 1:
                    decoded = decoded.squeeze(1)
            else:
                decoded_list = sparse_to_img_list(decoded, self.patch_size)
                if len(set(x.shape for x in decoded_list)) > 1:
                    logging.warning(f"Decoded shapes are not the same: {[x.shape for x in decoded_list]}")
                    decoded = decoded_list
                else:
                    decoded = torch.stack(decoded_list, dim=0)
                    decoded = rearrange(decoded, "b t c h w -> b t h w c")
                    if decoded.shape[1] == 1:
                        decoded = decoded.squeeze(1)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
