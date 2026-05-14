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
# Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.

from typing import List

import torch
from torch import nn, pow, sin
from torch.nn import Parameter


# https://github.com/jaywalnut310/vits/blob/main/commons.py
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: List[int]
) -> torch.Tensor:
    n_channels_int = n_channels[0]
    in_act = input_a + input_b  # [B,2*C,T]
    t_act = torch.tanh(in_act[:, :n_channels_int, :])  # [B,C,T]
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])  # [B,C,T]
    acts = t_act * s_act  # [B,C,T]
    return acts  # [B,C,T]


# about 10% faster training. no_div_by_zero (1e-9) baked in
@torch.jit.script
def fused_snake(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return x + (1.0 / (beta + 1e-9)) * pow(sin(x * alpha), 2)


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = True
    ) -> None:
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        """
        super(Snake, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self: "Snake", x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake := x + 1/a * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # [1,C,1]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)  # [1,C,1]

        return fused_snake(x, alpha, alpha)  # [B,C,T]
        # x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)
        # return x


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = True
    ) -> None:
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self: "SnakeBeta", x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta := x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # [1,C,1]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)  # [1,C,1]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)  # [1,C,1]
            beta = torch.exp(beta)  # [1,C,1]

        return fused_snake(x, alpha, beta)  # [B,C,T]
