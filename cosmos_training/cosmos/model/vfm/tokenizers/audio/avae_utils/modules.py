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
# Adapted from https://github.com/jik876/hifi-gan under the MIT license.

"""AVAE Modules.

This file contains only the modules needed for the spec_convnext encoder +
oobleck decoder + vae configuration.
"""

import math
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda import amp
from torch.nn.utils import weight_norm

from . import activations
from .alias_free_torch.act import Activation1d as TorchActivation1d

# for causal models we use encodec modules
from .modules_encodec import SConvTranspose1d


def WNConv1d(*args: Any, **kwargs: Any) -> nn.Conv1d:
    """Weight-normalized 1D convolution."""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args: Any, **kwargs: Any) -> nn.ConvTranspose1d:
    """Weight-normalized 1D transpose convolution."""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    Used for identity initialization in ConvNeXt blocks.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def may_mask(
    x: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Apply mask to tensor if provided.

    Args:
        x: Input tensor
        mask: Optional mask tensor

    Returns:
        Masked tensor if mask is provided, otherwise original tensor
    """
    if mask is not None:
        x = x * mask
    return x


class LayerNorm(nn.Module):
    """
    LayerNorm with optional bias.
    PyTorch doesn't support bias=False natively.
    """

    def __init__(self, size: int, gamma0: float = 1, eps: float = 1e-5, use_bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if use_bias else None
        self.eps = eps
        self.size = size

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            tensor: Input tensor of shape (B, T, C)

        Returns:
            Normalized tensor
        """
        dtype = tensor.dtype
        # fp32 to avoid numerical issues
        with amp.autocast(enabled=True, dtype=torch.float32):
            tensor = F.layer_norm(tensor, self.weight.shape, self.weight, self.bias, self.eps)
        return tensor.to(dtype)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt 1D Block adapted from https://github.com/charactr-platform/vocos
    which is adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.
    Supports causal and non-causal mode.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        identity_init (bool): If True, initializes the 1x1 conv in residual paths to zero (identity-friendly).
        use_snake (bool): If True, uses SnakeBeta activation; otherwise, GELU.
        causal (bool): If True, applies causal padding; otherwise, applies symmetric padding for non-causal.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        identity_init: bool = False,
        use_snake: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal

        if causal:
            # Causal padding: Only pad on the left
            self.dwconv = nn.Sequential(
                nn.ConstantPad1d((6, 0), 0),  # causal padding
                nn.Conv1d(dim, dim, kernel_size=7, groups=dim),
            )
        else:
            # Non-causal padding: Symmetric padding
            self.dwconv = nn.Sequential(
                nn.ConstantPad1d((3, 3), 0),  # symmetric padding (kernel_size // 2 on both sides)
                nn.Conv1d(dim, dim, kernel_size=7, groups=dim),
            )

        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, 1)  # pointwise/1x1 convs
        self.act = activations.SnakeBeta(intermediate_dim) if use_snake else nn.GELU()

        if identity_init:
            self.pwconv2 = zero_module(nn.Conv1d(intermediate_dim, dim, 1))
        else:
            self.pwconv2 = nn.Conv1d(intermediate_dim, dim, 1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, T)
            mask: Optional mask tensor

        Returns:
            Output tensor of shape (B, C, T)
        """
        residual = x  # [B,C,T]
        x = self.dwconv(may_mask(x, mask))  # [B,C,T]
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B,C,T] -> [B,T,C] -> [B,C,T]
        x = self.pwconv1(x)  # [B,intermediate_dim,T]
        x = self.act(x)  # [B,intermediate_dim,T]
        x = self.pwconv2(x)  # [B,C,T]
        x = residual + x  # [B,C,T]
        return may_mask(x, mask)  # [B,C,T]

    def remove_weight_norm(self) -> None:
        """No weight norm is applied in ConvNeXtBlock."""
        pass


def get_activation(
    activation: Literal["elu", "snake", "none"],
    antialias: bool = False,
    channels: Optional[int] = None,
    use_cuda_kernel: bool = False,
) -> nn.Module:
    """
    Get activation module by name.

    Args:
        activation: Activation type ('elu', 'snake', or 'none')
        antialias: Whether to wrap with anti-aliasing
        channels: Number of channels (required for snake activation)
        use_cuda_kernel: Whether to use CUDA kernel (not supported)

    Returns:
        Activation module
    """
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = activations.SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")

    if antialias:
        # select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if use_cuda_kernel:
            raise NotImplementedError("CUDA kernels not supported in this port")
        else:
            Activation1d = TorchActivation1d

        act = Activation1d(act)

    return act


class ResidualUnit(nn.Module):
    """
    Residual unit with dilated convolutions.
    Used in OobleckDecoderBlock.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dilation: Dilation rate
        kernel_size: Convolution kernel size (default: 7)
        use_snake: Whether to use Snake activation (default: False)
        antialias_activation: Whether to use anti-aliasing (default: False)
        causal: Whether to use causal convolutions (default: False)
        padding_mode: Padding mode for convolutions (default: 'zeros')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_size: int = 7,
        use_snake: bool = False,
        antialias_activation: bool = False,
        causal: bool = False,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.dilation = dilation
        self.causal = causal
        self.kernel_size = kernel_size

        if causal:
            self.padding = dilation * (kernel_size - 1)
        else:
            self.padding = (dilation * (kernel_size - 1)) // 2

        # original non-causal impl used zero padding (DAC, SAVAE)
        # reflect padding may be better to reduce edge artifacts (EnCodec's default), but it increases VRAM usage during training
        self.padding_mode = padding_mode

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=self.padding,
                padding_mode=self.padding_mode,
            ),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, T)

        Returns:
            Output tensor of shape (B, C, T)
        """
        res = x  # [B,C,T]

        # apply conv layers
        x = self.layers(x)  # [B,C,T] (padded if causal)

        if self.causal:
            # Trim right padding to get the causal output
            x = x[:, :, : -self.padding]  # [B,C,T]

        return x + res  # [B,C,T]


class OobleckDecoderBlock(nn.Module):
    """
    Oobleck decoder block with upsampling and residual units.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Upsampling stride
        use_snake: Whether to use Snake activation (default: False)
        antialias_activation: Whether to use anti-aliasing (default: False)
        use_nearest_upsample: Whether to use nearest neighbor upsampling (default: False)
        causal: Whether to use causal convolutions (default: False)
        padding_mode: Padding mode for convolutions (default: 'zeros')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        use_snake: bool = False,
        antialias_activation: bool = False,
        use_nearest_upsample: bool = False,
        causal: bool = False,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.causal = causal

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            self._create_upsample_layer(in_channels, out_channels, stride, use_nearest_upsample, causal, padding_mode),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=1,
                use_snake=use_snake,
                causal=causal,
                padding_mode=padding_mode,
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=3,
                use_snake=use_snake,
                causal=causal,
                padding_mode=padding_mode,
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=9,
                use_snake=use_snake,
                causal=causal,
                padding_mode=padding_mode,
            ),
        )

    def _create_upsample_layer(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        use_nearest_upsample: bool,
        causal: bool,
        padding_mode: str,
    ) -> nn.Module:
        """
        Create upsampling layer based on configuration.

        Note: padding_mode parameter is not used in this function.
        """

        if causal:  # use EnCodec's SConvTransposed1d for convenience. padding_mode is reflect by default
            assert not use_nearest_upsample, "use_nearest_upsample is not implemented for causal mode!"
            upsample_layer = SConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                causal=True,
                norm="weight_norm",
            )
        else:
            if use_nearest_upsample:
                upsample_layer = nn.Sequential(
                    nn.Upsample(scale_factor=stride, mode="nearest"),
                    WNConv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=2 * stride,
                        stride=1,
                        bias=False,
                        padding="same",
                    ),
                )
            else:
                # WNConvTranspose1d only supports zeros padding mode so it's hardcoded
                upsample_layer = WNConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                    output_padding=stride % 2,
                    padding_mode="zeros",
                )

        return upsample_layer

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, T)

        Returns:
            Output tensor of shape (B, C, T_upsampled)
        """
        return self.layers(x)

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from all layers."""
        from torch.nn.utils import remove_weight_norm

        for l in self.layers:
            try:
                remove_weight_norm(l)
            except (ValueError, AttributeError):
                # Layer doesn't have weight norm or is not a module with weight norm
                pass
