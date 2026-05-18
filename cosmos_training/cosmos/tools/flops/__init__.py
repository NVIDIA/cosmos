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

"""Reusable FLOPs estimation utilities for model architectures."""

from cosmos.tools.flops.omni_mot import (
    OmniMoTModelDescriptor,
    compute_omni_mot_flops_per_batch,
    get_omni_mot_model_descriptor,
)
from cosmos.tools.flops.qwen3_vl import (
    compute_qwen3vl_flops,
    compute_qwen3vl_flops_from_config,
)
from cosmos.tools.flops.wan_vae import compute_wan_vae_encoder_flops

__all__ = [
    "OmniMoTModelDescriptor",
    "compute_omni_mot_flops_per_batch",
    "compute_qwen3vl_flops",
    "compute_qwen3vl_flops_from_config",
    "compute_wan_vae_encoder_flops",
    "get_omni_mot_model_descriptor",
]
