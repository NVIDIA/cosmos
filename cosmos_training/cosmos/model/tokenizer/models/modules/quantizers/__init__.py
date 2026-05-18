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

"""Vector quantization modules for tokenizers.

This module provides quantization implementations:
    - fsq: Finite Scalar Quantization
    - lfq: Lookup-Free Quantization
    - residual_vq: Residual Quantization (RQ)
"""

from cosmos.model.tokenizer.models.modules.quantizers.fsq import FSQ, levels_from_codebook_size
from cosmos.model.tokenizer.models.modules.quantizers.lfq import LFQ, LossBreakdown
from cosmos.model.tokenizer.models.modules.quantizers.residual_vq import RQBottleneck, VQEmbedding

__all__ = [
    "FSQ",
    "levels_from_codebook_size",
    "LFQ",
    "LossBreakdown",
    "RQBottleneck",
    "VQEmbedding",
]
