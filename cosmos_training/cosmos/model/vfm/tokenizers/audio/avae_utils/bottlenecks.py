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
"""Bottleneck modules for AVAE tokenizer.

This cleaned-up version only includes VAEBottleneck which is used
by the spec_convnext encoder + oobleck decoder + vae configuration.
"""

from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn


# Base class
class Bottleneck(nn.Module):
    """Base class for bottleneck modules."""

    def __init__(self: "Bottleneck", is_discrete: bool = False) -> None:
        super().__init__()
        self.is_discrete = is_discrete

    def encode(
        self: "Bottleneck", x: Tensor, return_info: bool = False, **kwargs: Any
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def decode(self: "Bottleneck", x: Tensor, return_info: bool = False) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        raise NotImplementedError


def vae_sample(mean: Tensor, scale: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Sample from VAE latent distribution.

    Args:
        mean: Mean of the latent distribution
        scale: Scale parameter (will be passed through softplus)

    Returns:
        latents: Sampled latents
        kl: KL divergence loss
    """
    stdev = nn.functional.softplus(scale) + 1e-4  # [B,C,T]
    var = stdev * stdev  # [B,C,T]
    logvar = torch.log(var)  # [B,C,T]
    latents = torch.randn_like(mean) * stdev + mean  # [B,C,T]

    kl = (mean * mean + var - logvar - 1).sum(1).mean()  # scalar

    return latents, kl


class VAEBottleneck(Bottleneck):
    """
    Variational Autoencoder (VAE) bottleneck.

    Applies VAE reparameterization trick during encoding.
    """

    def __init__(self: "VAEBottleneck") -> None:
        super().__init__(is_discrete=False)

    def encode(
        self: "VAEBottleneck", x: Tensor, return_info: bool = False, **kwargs: Any
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        """
        Encode input through VAE bottleneck.

        Args:
            x: Input tensor with shape [B, C*2, T] where C*2 contains
               concatenated mean and scale parameters
            return_info: Whether to return additional info dict

        Returns:
            Sampled latents (and optionally info dict with KL divergence)
        """
        info: Dict[str, Any] = {}

        mean, scale = x.chunk(2, dim=1)  # mean,scale: [B,C,T]
        x, kl = vae_sample(mean, scale)  # x: [B,C,T]

        info["kl"] = kl

        if return_info:
            return x, info
        else:
            return x

    def decode(self: "VAEBottleneck", x: Tensor, return_info: bool = False) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        """
        Decode from latents (identity operation for VAE).

        Args:
            x: Latent tensor
            return_info: Whether to return additional info dict

        Returns:
            Latents (and optionally empty info dict)
        """
        info: Dict[str, Any] = {}
        if return_info:
            return x, info
        else:
            return x


def create_bottleneck_from_config(bottleneck_config: Dict[str, Any]) -> Bottleneck:
    """
    Create a bottleneck module from configuration.

    Args:
        bottleneck_config: Dictionary with 'type' key specifying bottleneck type

    Returns:
        Bottleneck module instance

    Note:
        This cleaned version only supports 'vae' bottleneck type.
    """
    bottleneck_type = bottleneck_config.get("type", None)

    assert bottleneck_type is not None, "type must be specified in bottleneck config"

    if bottleneck_type == "vae":
        bottleneck = VAEBottleneck()
    else:
        raise NotImplementedError(
            f"Bottleneck type '{bottleneck_type}' not supported in cleaned AVAE. "
            f"Only 'vae' is supported for the spec_convnext + oobleck + vae configuration."
        )

    return bottleneck
