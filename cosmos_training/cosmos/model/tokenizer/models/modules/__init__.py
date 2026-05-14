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

"""Core neural network components for tokenizers.

This module contains building blocks for tokenizer networks:
    - sparse_tensor: SparseTensor data structure and operations
    - sparse_ops: Linear, normalization, and activation layers for sparse tensors
    - attention: Attention mechanisms for sparse tensors
    - quantizers: Vector quantization (FSQ, LFQ)
"""

from __future__ import annotations

import importlib
import os
from typing import Literal

from loguru import logger as logging

# Backend configuration
BACKEND: str = "pytorch"
DEBUG: bool = False

# Valid backend options
_VALID_BACKENDS = ["pytorch", "spconv", "torchsparse"]


def _init_from_env() -> None:
    """Initialize backend settings from environment variables."""
    global BACKEND, DEBUG

    env_sparse_backend = os.environ.get("SPARSE_BACKEND")
    env_sparse_debug = os.environ.get("SPARSE_DEBUG")
    env_sparse_attn = os.environ.get("SPARSE_ATTN_BACKEND")
    if env_sparse_attn is None:
        env_sparse_attn = os.environ.get("ATTN_BACKEND")

    if env_sparse_backend is not None and env_sparse_backend in _VALID_BACKENDS:
        BACKEND = env_sparse_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == "1"
    if env_sparse_attn is not None:
        logging.warning(
            f"Ignoring sparse tokenizer attention backend override {env_sparse_attn!r}. "
            "Tokenizer sparse attention now defers backend selection to cosmos.model.attention. "
            "If you need to filter i4 backend choices, use I4_ATTN_BACKENDS instead."
        )


_init_from_env()


def set_backend(backend: Literal["pytorch", "spconv", "torchsparse"]) -> None:
    """Set the sparse tensor backend.

    Args:
        backend: Backend to use.
            - "pytorch": Pure PyTorch implementation (default, no external dependencies)
            - "spconv": Uses spconv.pytorch.SparseConvTensor
            - "torchsparse": Uses torchsparse.SparseTensor
    """
    global BACKEND
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Invalid backend: {backend}. Must be one of {_VALID_BACKENDS}")
    BACKEND = backend


def set_debug(debug: bool) -> None:
    """Enable or disable debug mode.

    Args:
        debug: Whether to enable debug mode.
    """
    global DEBUG
    DEBUG = debug


# Lazy loading attribute mapping
_ATTRIBUTES = {
    # SparseTensor and operations
    "SparseTensor": "sparse_tensor",
    "PureTorchSparseTensor": "sparse_tensor",
    "sparse_batch_broadcast": "sparse_tensor",
    "sparse_batch_op": "sparse_tensor",
    "sparse_cat": "sparse_tensor",
    "sparse_unbind": "sparse_tensor",
    "reconstruct_from_temporal_slices": "sparse_tensor",
    # Linear layers
    "SparseLinear": "sparse_ops",
    # Normalization layers
    "SparseGroupNorm": "sparse_ops",
    "SparseLayerNorm": "sparse_ops",
    "SparseGroupNorm32": "sparse_ops",
    "SparseLayerNorm32": "sparse_ops",
    "SparseRMSNorm32": "sparse_ops",
    "LayerNorm32": "sparse_ops",
    "GroupNorm32": "sparse_ops",
    "ChannelLayerNorm32": "sparse_ops",
    "RMSNorm": "sparse_ops",
    "RMSNorm32": "sparse_ops",
    # Activation functions
    "SparseReLU": "sparse_ops",
    "SparseSiLU": "sparse_ops",
    "SparseGELU": "sparse_ops",
    "SparseActivation": "sparse_ops",
    # Spatial operations
    "SparseDownsample": "sparse_ops",
    "SparseDownsampleKeepCoords": "sparse_ops",
    "SparseUpsample": "sparse_ops",
    "SparseUpsampleTokenSplit": "sparse_ops",
    "SparseSubdivide": "sparse_ops",
    "SparseUpsampleNoCache": "sparse_ops",
    # Attention modules
    "sparse_scaled_dot_product_attention": "attention.full_attn",
    "RotaryPositionEmbedder": "attention.modules",
    "SparseMultiHeadRMSNorm": "attention.modules",
    "SparseMultiHeadAttention": "attention.modules",
    # Quantizers
    "FSQ": "quantizers.fsq",
    "levels_from_codebook_size": "quantizers.fsq",
    "LFQ": "quantizers.lfq",
    "LossBreakdown": "quantizers.lfq",
    "RQBottleneck": "quantizers.residual_vq",
    "VQEmbedding": "quantizers.residual_vq",
    # Transformer blocks
    "AbsolutePositionEmbedder": "transformer.blocks",
    "LearnedPositionEmbedder": "transformer.blocks",
    "LearnedPositionEmbedder4D": "transformer.blocks",
    "SparseFeedForwardNet": "transformer.blocks",
    "SparseMultiheadAttentionPoolingHead": "transformer.blocks",
    "SparseTransformerBlock": "transformer.blocks",
    "ModulatedSparseTransformerBlock": "transformer.modulated",
    "ModulatedSparseTransformerCrossBlock": "transformer.modulated",
}

__all__ = list(_ATTRIBUTES.keys())


def __getattr__(name: str):
    """Lazy import of module attributes."""
    if name not in globals():
        if name in _ATTRIBUTES:
            module_name = _ATTRIBUTES[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]
