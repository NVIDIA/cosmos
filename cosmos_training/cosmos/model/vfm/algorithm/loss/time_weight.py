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

import torch


class TrainTimeWeight:
    def __init__(
        self,
        noise_scheduler,
        weight: str = "uniform",
    ):
        # Map reweighting -> uniform to support inference for existing checkpoints.
        if weight == "reweighting":
            weight = "uniform"

        self.weight = weight
        self.noise_scheduler = noise_scheduler

        assert self.weight == "uniform", "Only uniform loss weight is supported in RF"

    def __call__(self, t, tensor_kwargs) -> torch.Tensor:  # t: [B], returns [B]
        if self.weight == "uniform":
            wts = torch.ones_like(t)  # [B]
        else:
            raise NotImplementedError(f"Time weight '{self.weight}' is not implemented.")

        return wts
