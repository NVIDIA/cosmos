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
# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

if "sinc" in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc under the MIT License
    # https://adefossez.github.io/julius/julius/core.html
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(
            x == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), torch.sin(math.pi * x) / math.pi / x
        )


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):  # return filter [1,1,kernel_size]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)  # [kernel_size]

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5  # [kernel_size]
    else:
        time = torch.arange(kernel_size) - half_size  # [kernel_size]
    if cutoff == 0:
        filter_ = torch.zeros_like(time)  # [kernel_size]
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)  # [kernel_size]
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
    filter = filter_.view(1, 1, kernel_size)  # [1,1,kernel_size]

    return filter  # [1,1,kernel_size]


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        # kernel_size should be even number for stylegan3 setup,
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    # input [B,C,T]
    def forward(self, x):  # x: [B,C,T]
        _, C, _ = x.shape

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)  # [B,C,T+pad_left+pad_right]
        out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)  # [B,C,T//stride]

        return out  # [B,C,T//stride]
