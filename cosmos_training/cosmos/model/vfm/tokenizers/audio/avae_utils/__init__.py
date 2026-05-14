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
AVAE Tokenizer

This module provides the AVAE tokenizer with spec_convnext encoder,
oobleck decoder, and VAE bottleneck configuration.
"""

from cosmos.model.vfm.tokenizers.audio.avae_utils.env import AttrDict
from cosmos.model.vfm.tokenizers.audio.avae_utils.models import LatentAutoEncoderV2, load_generator

__all__ = [
    "LatentAutoEncoderV2",
    "AttrDict",
    "load_generator",
]
