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

from enum import Enum


class EmbeddingConcatStrategy(str, Enum):
    FULL_CONCAT = "full_concat"  # Concatenate embeddings all layers
    MEAN_POOLING = "mean_pooling"  # Average pool embeddings all layers
    POOL_EVERY_N_LAYERS_AND_CONCAT = "pool_every_n_layers_and_concat"  # Pool every n layers and concatenatenate

    def __str__(self) -> str:
        return self.value
