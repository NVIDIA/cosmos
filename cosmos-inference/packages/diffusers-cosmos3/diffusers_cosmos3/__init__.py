# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import diffusers

from .pipeline import Cosmos3OmniDiffusersPipeline
from .transformer import Cosmos3OmniTransformer

# Spoof the eventual upstream module paths so save_pretrained writes
# "diffusers" as the library in model_index.json.
Cosmos3OmniTransformer.__module__ = "diffusers.models.transformers.transformer_cosmos3"
Cosmos3OmniDiffusersPipeline.__module__ = "diffusers.pipelines.cosmos.pipeline_cosmos3_omni"

# Loader does `getattr(importlib.import_module(library), class_name)`,
# so the class must be reachable as `diffusers.<ClassName>`.
diffusers.Cosmos3OmniTransformer = Cosmos3OmniTransformer
diffusers.Cosmos3OmniDiffusersPipeline = Cosmos3OmniDiffusersPipeline
