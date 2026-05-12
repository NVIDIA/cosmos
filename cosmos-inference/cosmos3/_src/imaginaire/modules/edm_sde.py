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

from statistics import NormalDist

import numpy as np
import torch


class EDMSDE:
    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
    ):
        self.gaussian_dist = NormalDist(mu=p_mean, sigma=p_std)
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def sample_t(self, batch_size: int) -> torch.Tensor:
        cdf_vals = np.random.uniform(size=(batch_size))
        samples_interval_gaussian = [self.gaussian_dist.inv_cdf(cdf_val) for cdf_val in cdf_vals]

        log_sigma = torch.tensor(samples_interval_gaussian, device="cuda")
        return torch.exp(log_sigma)

    def marginal_prob(self, x0: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """This is trivial in the base class, but may be used by derived classes in a more interesting way"""
        return x0, sigma
