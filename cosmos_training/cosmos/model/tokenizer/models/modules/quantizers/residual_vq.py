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

"""Residual Quantization (RQ).

Vector quantization with residual learning, where each codebook quantizes
the residual from previous codebooks. This enables hierarchical discrete
representations with improved reconstruction quality.

Reference:
    - SoundStream: https://arxiv.org/abs/2107.03312
    - RQ-VAE: https://arxiv.org/abs/2203.01941
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

__all__ = [
    "VQEmbedding",
    "RQBottleneck",
]


class VQEmbedding(nn.Embedding):
    """VQ embedding module with exponential moving average (EMA) update.

    This module extends nn.Embedding to support vector quantization with
    optional EMA-based codebook updates and unused code restart.

    Args:
        n_embed: Number of embeddings in the codebook.
        embed_dim: Dimension of each embedding vector.
        ema: Whether to use EMA for codebook updates.
        decay: EMA decay rate.
        restart_unused_codes: Whether to restart unused codebook entries.
        eps: Small epsilon for numerical stability.
    """

    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
        ema: bool = True,
        decay: float = 0.99,
        restart_unused_codes: bool = True,
        eps: float = 1e-5,
    ):
        """Initialize VQEmbedding."""
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]

            # Padding index is not updated by EMA
            self.register_buffer("cluster_size_ema", torch.zeros(n_embed))
            self.register_buffer("embed_ema", self.weight[:-1, :].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute squared L2 distances to codebook entries.

        Args:
            inputs: Input tensor of shape (..., embed_dim).

        Returns:
            Distance tensor of shape (..., n_embed).
        """
        codebook_t = self.weight[:-1, :].t()
        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.0).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.0).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        """Find nearest codebook entry for each input.

        Args:
            inputs: Input tensor of shape (..., embed_dim).

        Returns:
            Index tensor of shape (...,).
        """
        distances = self.compute_distances(inputs)
        embed_idxs = distances.argmin(dim=-1)
        return embed_idxs

    @torch.no_grad()
    def _tile_with_noise(self, x: torch.Tensor, target_n: int) -> torch.Tensor:
        """Tile tensor with noise to reach target size.

        Args:
            x: Input tensor of shape (B, embed_dim).
            target_n: Target number of rows.

        Returns:
            Tiled tensor of shape (target_n, embed_dim).
        """
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _update_buffers(self, vectors: torch.Tensor, idxs: torch.Tensor) -> None:
        """Update EMA buffers with current batch.

        Args:
            vectors: Feature vectors of shape (..., embed_dim).
            idxs: Codebook indices of shape (...,).
        """
        n_embed, embed_dim = self.weight.shape[0] - 1, self.weight.shape[-1]

        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)

        n_vectors = vectors.shape[0]
        n_total_embed = n_embed

        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(dim=0, index=idxs.unsqueeze(0), src=vectors.new_ones(1, n_vectors))

        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)

        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][:n_embed]

            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)

            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1 - usage).view(-1))

    @torch.no_grad()
    def _update_embedding(self) -> None:
        """Update codebook embeddings from EMA buffers."""
        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: quantize inputs to nearest codebook entries.

        Args:
            inputs: Input tensor of shape (..., embed_dim).

        Returns:
            Tuple of (quantized embeddings, codebook indices).
        """
        embed_idxs = self.find_nearest_embedding(inputs)

        if self.training:
            if self.ema:
                self._update_buffers(inputs, embed_idxs)

        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs

    def embed(self, idxs: torch.Tensor) -> torch.Tensor:
        """Look up embeddings by indices.

        Args:
            idxs: Index tensor.

        Returns:
            Embedding tensor.
        """
        embeds = super().forward(idxs)
        return embeds


class RQBottleneck(nn.Module):
    """Residual Quantization bottleneck.

    Quantization bottleneck via Residual Quantization. Each codebook
    quantizes the residual from previous codebooks, enabling hierarchical
    discrete representations.

    Args:
        latent_shape: Shape of latent features (H, W, D).
        code_shape: Shape of codes (h, w, d).
        n_embed: Number of embeddings per codebook (int or list).
        decay: EMA decay rate (float or list).
        shared_codebook: Whether to share codebook across depth.
        restart_unused_codes: Whether to restart unused codes.
        commitment_loss: Type of commitment loss ("cumsum").
    """

    def __init__(
        self,
        latent_shape: tuple[int, int, int],
        code_shape: tuple[int, int, int],
        n_embed: int | list[int],
        decay: float | list[float] = 0.99,
        shared_codebook: bool = False,
        restart_unused_codes: bool = True,
        commitment_loss: str = "cumsum",
    ):
        """Initialize RQBottleneck."""
        super().__init__()

        if not len(code_shape) == len(latent_shape) == 3:
            raise ValueError("incompatible code shape or latent shape")
        if any([y % x != 0 for x, y in zip(code_shape[:2], latent_shape[:2])]):
            raise ValueError("incompatible code shape or latent shape")

        # Residual quantization does not divide feature dims for quantization
        embed_dim = np.prod(latent_shape[:2]) // np.prod(code_shape[:2]) * latent_shape[2]

        self.latent_shape = torch.Size(latent_shape)
        self.code_shape = torch.Size(code_shape)
        self.shape_divisor = torch.Size([latent_shape[i] // code_shape[i] for i in range(len(latent_shape))])

        self.shared_codebook = shared_codebook
        if self.shared_codebook:
            if isinstance(n_embed, Iterable) or isinstance(decay, Iterable):
                raise ValueError(
                    "Shared codebooks are incompatible with list types of momentums or sizes: Change it into int"
                )

        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed if isinstance(n_embed, Iterable) else [n_embed for _ in range(self.code_shape[-1])]
        self.decay = decay if isinstance(decay, Iterable) else [decay for _ in range(self.code_shape[-1])]
        assert len(self.n_embed) == self.code_shape[-1]
        assert len(self.decay) == self.code_shape[-1]

        if self.shared_codebook:
            codebook0 = VQEmbedding(
                self.n_embed[0],
                embed_dim,
                decay=self.decay[0],
                restart_unused_codes=restart_unused_codes,
            )
            self.codebooks = nn.ModuleList([codebook0 for _ in range(self.code_shape[-1])])
        else:
            codebooks = [
                VQEmbedding(
                    self.n_embed[idx],
                    embed_dim,
                    decay=self.decay[idx],
                    restart_unused_codes=restart_unused_codes,
                )
                for idx in range(self.code_shape[-1])
            ]
            self.codebooks = nn.ModuleList(codebooks)

        self.commitment_loss = commitment_loss

    def quantize(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Quantize input using residual quantization.

        The code is selected by the residuals between x and quantized
        features by the previous codebooks.

        Args:
            x: Bottleneck feature maps to quantize, shape (N, embed_dim).

        Returns:
            Tuple of:
                - quant_list: List of sequentially aggregated quantized features.
                - codes: Codeword indices, shape (N, d).
        """
        N, embed_dim = x.shape

        residual_feature = x.detach().clone()

        quant_list = []
        code_list = []
        aggregated_quants = torch.zeros_like(x)
        for i in range(self.code_shape[-1]):
            quant, code = self.codebooks[i](residual_feature)
            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)

            quant_list.append(aggregated_quants.clone())
            code_list.append(code.unsqueeze(-1))

        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for RQ bottleneck.

        Args:
            x: Input features, shape (N, embed_dim).

        Returns:
            Tuple of (quantized output, codes, commitment loss).
        """
        quant_list, codes = self.quantize(x)
        commitment_loss = self.compute_commitment_loss(x, quant_list)
        quants_trunc = quant_list[-1]
        quants_trunc = x + (quants_trunc - x).detach()

        return quants_trunc, codes, commitment_loss

    def compute_commitment_loss(self, x: torch.Tensor, quant_list: list[torch.Tensor]) -> torch.Tensor:
        """Compute commitment loss for residual quantization.

        The loss is iteratively computed by aggregating quantized features.

        Args:
            x: Original input features.
            quant_list: List of quantized features at each depth.

        Returns:
            Commitment loss scalar.
        """
        loss_list = []

        for idx, quant in enumerate(quant_list):
            partial_loss = (x - quant.detach()).pow(2.0).mean()
            loss_list.append(partial_loss)

        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss

    @torch.no_grad()
    def embed_code(self, code: torch.Tensor) -> torch.Tensor:
        """Decode codes to embeddings.

        Args:
            code: Code tensor of shape (B, h, w, d).

        Returns:
            Embedded features of shape (B, H, W, embed_dim).
        """
        assert code.shape[1:] == self.code_shape

        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]

        embeds = torch.cat(embeds, dim=-2).sum(-2)
        embeds = self.to_latent_shape(embeds)

        return embeds

    @torch.no_grad()
    def embed_code_with_depth(self, code: torch.Tensor, to_latent_shape: bool = False) -> tuple[torch.Tensor, None]:
        """Decode codes without reducing over depth dimension.

        Does not reduce the code embedding over the axis of code-depth.

        Args:
            code: Code tensor.
            to_latent_shape: Whether to reshape to latent shape.

        Returns:
            Tuple of (embedded features, None).
        """
        assert code.shape[-1] == self.code_shape[-1]

        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]

        if to_latent_shape:
            embeds = [self.to_latent_shape(embed.squeeze(-2)).unsqueeze(-2) for embed in embeds]
        embeds = torch.cat(embeds, dim=-2)

        return embeds, None

    @torch.no_grad()
    def embed_partial_code(
        self,
        code: torch.Tensor,
        code_idx: int,
        decode_type: str = "select",
    ) -> torch.Tensor:
        """Decode input codes using subset of codebooks.

        Args:
            code: Codes of input, shape (B, h, w, d).
            code_idx: Index of the last selected codebook for decoding.
            decode_type: "select" for single codebook, "add" for cumulative.

        Returns:
            Quantized feature map.
        """
        assert code.shape[1:] == self.code_shape
        assert code_idx < code.shape[-1]

        B, h, w, _ = code.shape

        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)
        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]

        if decode_type == "select":
            embeds = embeds[code_idx].view(B, h, w, -1)
        elif decode_type == "add":
            embeds = torch.cat(embeds[: code_idx + 1], dim=-2).sum(-2)
        else:
            raise NotImplementedError(f"{decode_type} is not implemented in partial decoding")

        embeds = self.to_latent_shape(embeds)

        return embeds

    @torch.no_grad()
    def get_soft_codes(
        self,
        x: torch.Tensor,
        temp: float = 1.0,
        stochastic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get soft codes (probabilities) from input.

        Args:
            x: Input features.
            temp: Temperature for softmax.
            stochastic: Whether to sample from distribution.

        Returns:
            Tuple of (soft codes, hard codes).
        """
        x = self.to_code_shape(x)

        residual_feature = x.detach().clone()
        soft_code_list = []
        code_list = []

        n_codebooks = self.code_shape[-1]
        for i in range(n_codebooks):
            codebook = self.codebooks[i]
            distances = codebook.compute_distances(residual_feature)
            soft_code = F.softmax(-distances / temp, dim=-1)

            if stochastic:
                soft_code_flat = soft_code.reshape(-1, soft_code.shape[-1])
                code = torch.multinomial(soft_code_flat, 1)
                code = code.reshape(*soft_code.shape[:-1])
            else:
                code = distances.argmin(dim=-1)
            quants = codebook.embed(code)
            residual_feature -= quants

            code_list.append(code.unsqueeze(-1))
            soft_code_list.append(soft_code.unsqueeze(-2))

        code = torch.cat(code_list, dim=-1)
        soft_code = torch.cat(soft_code_list, dim=-2)
        return soft_code, code
