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

"""Modulated sparse transformer blocks with adaptive layer norm conditioning.

This module provides:
    - ModulatedSparseTransformerBlock: Self-attention block with adaLN modulation
    - ModulatedSparseTransformerCrossBlock: Self + cross attention with adaLN modulation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from cosmos.model.tokenizer.models.modules.sparse_ops import LayerNorm32
from cosmos.model.tokenizer.models.modules.transformer.blocks import SparseFeedForwardNet

if TYPE_CHECKING:
    from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

__all__ = [
    "ModulatedSparseTransformerBlock",
    "ModulatedSparseTransformerCrossBlock",
]


class ModulatedSparseTransformerBlock(nn.Module):
    """Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.

    Uses adaLN (adaptive layer norm) to modulate the features based on a
    conditioning signal, enabling conditional generation.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        pos_cls_token: int = 0,
    ):
        """Initialize ModulatedSparseTransformerBlock.

        Args:
            channels: Number of input/output channels.
            num_heads: Number of attention heads.
            mlp_ratio: MLP expansion ratio.
            use_checkpoint: Whether to use gradient checkpointing.
            use_rope: Whether to use rotary position embeddings.
            qk_rms_norm: Whether to apply RMS normalization to Q and K.
            qkv_bias: Whether to use bias in QKV projections.
            share_mod: Whether modulation is shared (computed externally).
            pos_cls_token: Position of CLS token in attention.
        """
        super().__init__()
        # Import here to avoid circular imports
        from cosmos.model.tokenizer.models.modules.attention.modules import SparseMultiHeadAttention

        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            use_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
            pos_cls_token=pos_cls_token,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_channels=int(channels * mlp_ratio),
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True))

    def _forward(self, x: "SparseTensor", mod: torch.Tensor) -> "SparseTensor":
        """Internal forward pass with adaLN modulation.

        Args:
            x: Input SparseTensor.
            mod: Conditioning tensor for modulation.

        Returns:
            Output SparseTensor.
        """
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h, _ = self.attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: "SparseTensor", mod: torch.Tensor) -> "SparseTensor":
        """Forward pass with optional gradient checkpointing.

        Args:
            x: Input SparseTensor.
            mod: Conditioning tensor for modulation.

        Returns:
            Output SparseTensor.
        """
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaLN conditioning.

    Combines self-attention, cross-attention with context, and MLP,
    all modulated by adaptive layer norm.
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        """Initialize ModulatedSparseTransformerCrossBlock.

        Args:
            channels: Number of input/output channels.
            ctx_channels: Number of context channels for cross-attention.
            num_heads: Number of attention heads.
            mlp_ratio: MLP expansion ratio.
            use_checkpoint: Whether to use gradient checkpointing.
            use_rope: Whether to use rotary position embeddings.
            qk_rms_norm: Whether to apply RMS norm to Q/K in self-attention.
            qk_rms_norm_cross: Whether to apply RMS norm to Q/K in cross-attention.
            qkv_bias: Whether to use bias in QKV projections.
            share_mod: Whether modulation is shared (computed externally).
        """
        super().__init__()
        # Import here to avoid circular imports
        from cosmos.model.tokenizer.models.modules.attention.modules import SparseMultiHeadAttention

        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            use_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            use_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_channels=int(channels * mlp_ratio),
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True))

    def _forward(
        self,
        x: "SparseTensor",
        mod: torch.Tensor,
        context: "SparseTensor" | torch.Tensor,
    ) -> "SparseTensor":
        """Internal forward pass with self-attention, cross-attention, and MLP.

        Args:
            x: Input SparseTensor.
            mod: Conditioning tensor for adaLN modulation.
            context: Context tensor for cross-attention.

        Returns:
            Output SparseTensor.
        """
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h, _ = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h, _ = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(
        self,
        x: "SparseTensor",
        mod: torch.Tensor,
        context: "SparseTensor" | torch.Tensor,
    ) -> "SparseTensor":
        """Forward pass with optional gradient checkpointing.

        Args:
            x: Input SparseTensor.
            mod: Conditioning tensor for adaLN modulation.
            context: Context tensor for cross-attention.

        Returns:
            Output SparseTensor.
        """
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)
