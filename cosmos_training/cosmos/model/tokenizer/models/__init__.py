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

"""Cosmos3 tokenizer models.

This module provides:
    - modules: Low-level building blocks (SparseTensor, quantizers, attention)
    - utils: Generic utilities (image processing, metrics, logging)
    - sparse_autoencoder: AutoencoderKL model for image/video tokenization
"""

# Generic utilities
# Metrics (moved from utils to metrics module for consolidation)

# Dense runtime
from cosmos.model.tokenizer.models.dense_runtime import (
    DenseAutoencoderRuntime,
    DenseGridMetadata,
    DenseTemporalChunkSpec,
)

# Quantizer utilities
from cosmos.model.tokenizer.models.modules.quantizers import levels_from_codebook_size

# Model classes (from sparse_autoencoder)
from cosmos.model.tokenizer.models.sparse_autoencoder import (
    AutoencoderKL,
    AutoencoderKLConfig,
    Decoder,
    DiagonalGaussianDistribution,
    Encoder,
    SparseTransformerBase,
)
from cosmos.model.tokenizer.models.utils import (
    SampleLogger,
    average_with_scatter_add,
    batch_tensor_to_sparse,
    crop_tensors_to_match,
    reconstruct_from_temporal_slices,
    resize_and_crop,
    restore_original_shape,
    sparse_to_img_list,
    split_temporal_dimension,
)

__all__ = [
    # Utils
    "average_with_scatter_add",
    "batch_tensor_to_sparse",
    "crop_tensors_to_match",
    "reconstruct_from_temporal_slices",
    "resize_and_crop",
    "restore_original_shape",
    "SampleLogger",
    "sparse_to_img_list",
    "split_temporal_dimension",
    "DenseAutoencoderRuntime",
    "DenseGridMetadata",
    "DenseTemporalChunkSpec",
    # Quantizer utilities
    "levels_from_codebook_size",
    # Model classes
    "AutoencoderKL",
    "AutoencoderKLConfig",
    "Decoder",
    "DiagonalGaussianDistribution",
    "Encoder",
    "SparseTransformerBase",
]
