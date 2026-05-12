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

from __future__ import annotations

from typing import Any

import torch

from cosmos3._src.imaginaire.utils.misc import get_local_tensor_if_DTensor as dt2lt


class DTensorFastEmaModelUpdater:
    """
    Similar as FastEmaModelUpdater
    """

    def __init__(self):
        # Flag to indicate whether the cache is taken or not. Useful to avoid cache overwrite
        self.is_cached = False

    def copy_to(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module) -> None:
        with torch.no_grad():
            for tgt_params, src_params in zip(tgt_model.parameters(), src_model.parameters()):
                dt2lt(tgt_params).data.copy_(dt2lt(src_params).data)

    @torch.no_grad()
    def update_average(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module, beta: float = 0.9999) -> None:
        target_list = []
        source_list = []
        for tgt_params, src_params in zip(tgt_model.parameters(), src_model.parameters()):
            assert tgt_params.dtype == torch.float32, (
                f"EMA model only works in FP32 dtype, got {tgt_params.dtype} instead."
            )
            target_list.append(dt2lt(tgt_params))
            source_list.append(dt2lt(src_params).data)
        torch._foreach_mul_(target_list, beta)
        torch._foreach_add_(target_list, source_list, alpha=1.0 - beta)

    @torch.no_grad()
    def cache(self, parameters: Any, is_cpu: bool = False) -> None:
        assert self.is_cached is False, "EMA cache is already taken. Did you forget to restore it?"
        device = "cpu" if is_cpu else "cuda"
        self.collected_params = [dt2lt(param).clone().to(device) for param in parameters]
        self.is_cached = True

    @torch.no_grad()
    def restore(self, parameters: Any) -> None:
        assert self.is_cached, "EMA cache is not taken yet."
        for c_param, param in zip(self.collected_params, parameters, strict=False):
            dt2lt(param).copy_(c_param.data.type_as(param.data))
        self.collected_params = []
        # Release the cache after we call restore
        self.is_cached = False
