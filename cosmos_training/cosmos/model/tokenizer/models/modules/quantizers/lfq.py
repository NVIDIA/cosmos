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

"""Lookup-Free Quantization (LFQ).

Proposed in https://arxiv.org/abs/2310.05737

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.

References:
    - https://github.com/lucidrains/vector-quantize-pytorch
    - https://github.com/theAdamColton/ijepa-enhanced
"""

from __future__ import annotations

from collections import namedtuple
from math import ceil, log2
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import pack, rearrange, reduce, unpack
from torch.nn import Module

if TYPE_CHECKING:
    from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

__all__ = [
    "LFQ",
    "LossBreakdown",
]


LossBreakdown = namedtuple(
    "LossBreakdown",
    ["per_sample_entropy", "codebook_entropy", "commitment", "avg_probs"],
)


# Helper functions


def exists(v):
    """Check if value exists (is not None)."""
    return v is not None


def default(*args):
    """Return first non-None value from args (calling if callable)."""
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def pack_one(t, pattern):
    """Pack single tensor."""
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    """Unpack single tensor."""
    return unpack(t, ps, pattern)[0]


def entropy(prob: torch.Tensor) -> torch.Tensor:
    """Compute entropy of probability distribution."""
    return (-prob * torch.log(prob + 1e-5)).sum(dim=-1)


def mult_along_first_dims(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multiply x by y along leading dimensions of y."""
    ndim_to_expand = x.ndim - y.ndim
    for _ in range(ndim_to_expand):
        y = y.unsqueeze(-1)
    return x * y


def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Take mean of x elements not masked by m.

    The mean is taken along the shared leading dims of m.
    This is faster for torch-compile on batches than tensor indexing.
    """
    x = mult_along_first_dims(x, m)
    x = x / m.sum()
    return x.sum(tuple(range(m.ndim)))


def entropy_loss(
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 0.01,
    sample_minimization_weight: float = 1.0,
    batch_maximization_weight: float = 1.0,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute entropy loss for LFQ.

    The loss encourages:
        - LOW per-sample entropy (focused code usage per sample)
        - HIGH batch-level entropy (diverse code usage across batch)

    Formula: loss = sample_entropy - batch_entropy

    NEGATIVE loss is GOOD: means batch entropy > sample entropy,
    indicating good codebook utilization with focused per-sample usage.

    Reference:
        https://github.com/google-research/magvit/
        LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)

    Args:
        logits: Affinities over the last dimension.
        mask: Optional mask for selective processing.
        temperature: Softmax temperature.
        sample_minimization_weight: Weight for per-sample entropy term.
        batch_maximization_weight: Weight for batch entropy term.
        eps: Small epsilon for numerical stability.

    Returns:
        Tuple of (per_sample_entropy, batch_entropy, loss).
    """
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)

    if mask is not None:
        avg_probs = masked_mean(probs, mask)
    else:
        avg_probs = reduce(probs, "... D -> D", "mean")

    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    if mask is not None:
        sample_entropy = masked_mean(sample_entropy, mask).mean()
    else:
        sample_entropy = torch.mean(sample_entropy)

    # Key insight: loss = sample_entropy - batch_entropy
    # Negative loss is DESIRABLE (batch entropy > sample entropy)
    loss = (sample_minimization_weight * sample_entropy) - (batch_maximization_weight * avg_entropy)

    return sample_entropy, avg_entropy, loss


class LFQ(Module):
    """Lookup-Free Quantization module.

    Quantizes continuous values to binary codes {-1, 1} per dimension,
    with entropy loss to encourage codebook utilization.
    """

    def __init__(
        self,
        *,
        dim: int | None = None,
        codebook_size: int | None = None,
        num_codebooks: int = 1,
        sample_minimization_weight: float = 1.0,
        batch_maximization_weight: float = 1.0,
        token_factorization: bool = False,
        factorized_bits: list[int] = [9, 9],
    ):
        """Initialize LFQ.

        Args:
            dim: Input feature dimension (defaults to log2(codebook_size)).
            codebook_size: Size of codebook (must be power of 2).
            num_codebooks: Number of codebooks to use.
            sample_minimization_weight: Weight for per-sample entropy in loss.
            batch_maximization_weight: Weight for batch entropy in loss.
            token_factorization: Whether to use factorized tokens.
            factorized_bits: Bit split for factorized tokens.
        """
        super().__init__()

        # Validation
        assert exists(dim) or exists(codebook_size), "either dim or codebook_size must be specified for LFQ"
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), (
            f"codebook size must be power of 2 (suggested {2 ** ceil(log2(codebook_size))})"
        )

        self.codebook_size = default(codebook_size, lambda: 2**dim)
        self.codebook_dim = int(log2(codebook_size))

        codebook_dims = self.codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = dim != codebook_dims
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = self.codebook_dim
        self.num_codebooks = num_codebooks

        # For entropy loss
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

        # For token factorization
        self.token_factorization = token_factorization
        if not self.token_factorization:
            self.register_buffer("mask", 2 ** torch.arange(self.codebook_dim), persistent=False)
        else:
            self.factorized_bits = factorized_bits
            self.register_buffer("pre_mask", 2 ** torch.arange(factorized_bits[0]), persistent=False)
            self.register_buffer("post_mask", 2 ** torch.arange(factorized_bits[1]), persistent=False)

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # Build codebook
        all_codes = torch.arange(codebook_size)
        bits = self.indices_to_bits(all_codes)
        codebook = bits * 2.0 - 1.0

        self.register_buffer("codebook", codebook, persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        """Return dtype of codebook."""
        return self.codebook.dtype

    def indices_to_bits(self, x: torch.Tensor) -> torch.Tensor:
        """Convert indices to big-endian bits.

        Args:
            x: Long tensor of indices.

        Returns:
            Boolean tensor of bits.
        """
        mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
        x = (x.unsqueeze(-1) & mask) != 0
        return x

    def get_codebook_entry(
        self,
        x: torch.Tensor,
        bhwc: tuple[int, int, int, int],
        order: str,
    ) -> torch.Tensor:
        """Get codebook entry for given indices.

        Args:
            x: Index tensor.
            bhwc: Tuple of (batch, height, width, channels).
            order: 'pre' or 'post' for factorized tokens.

        Returns:
            Decoded tensor.
        """
        if self.token_factorization:
            if order == "pre":
                mask = 2 ** torch.arange(self.factorized_bits[0], device=x.device, dtype=torch.long)
            else:
                mask = 2 ** torch.arange(self.factorized_bits[1], device=x.device, dtype=torch.long)
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)

        x = (x.unsqueeze(-1) & mask) != 0
        x = x * 2.0 - 1.0  # back to float
        b, h, w, c = bhwc
        x = rearrange(x, "b (h w) c -> b h w c", h=h, w=w, c=c)
        x = rearrange(x, "b h w c -> b c h w")
        return x

    def bits_to_indices(self, bits: torch.Tensor) -> torch.Tensor:
        """Convert big-endian bits to indices.

        Args:
            bits: Boolean tensor with bit dimension last.

        Returns:
            Long integer indices from 0 to codebook_size.
        """
        assert bits.shape[-1] == self.codebook_dim
        indices = 2 ** torch.arange(
            0,
            self.codebook_dim,
            1,
            dtype=torch.long,
            device=bits.device,
        )
        return (bits * indices).sum(-1)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode indices to continuous values.

        Args:
            x: Long tensor of codebook indices (..., NH) where NH is num_codebooks.

        Returns:
            Decoded tensor with values in {-1, 1}.
        """
        x = self.indices_to_bits(x)
        x = x.to(self.dtype)
        x = x * 2 - 1
        x = rearrange(x, "... NC Z-> ... (NC Z)")
        return x

    def forward(
        self,
        x: "SparseTensor",
        inv_temperature: float = 100.0,
        return_loss_breakdown: bool = False,
        mask: torch.Tensor | None = None,
        return_loss: bool = True,
        fp32_loss_computation: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], LossBreakdown]
    ):
        """Forward pass for LFQ on SparseTensor.

        Args:
            x: SparseTensor input with .feats, .coords, .shape, .layout.
            inv_temperature: Inverse temperature for entropy loss softmax.
            return_loss_breakdown: Whether to return detailed loss components.
            mask: Optional mask for selective processing.
            return_loss: Whether to compute training losses.
            fp32_loss_computation: Whether to compute losses in fp32.

        Returns:
            Tuple of (quantized_feats, entropy_loss, indices).
            If return_loss_breakdown, also returns LossBreakdown.
        """
        # Extract features from sparse tensor
        N, feature_dim = x.shape

        # Validate feature dimension
        expected_dim = self.num_codebooks * self.codebook_dim
        if feature_dim != expected_dim:
            raise ValueError(
                f"Feature dimension {feature_dim} doesn't match expected {expected_dim} "
                f"(num_codebooks={self.num_codebooks} * codebook_dim={self.codebook_dim})"
            )

        # Reshape for codebook processing: [N, num_codebooks, codebook_dim]
        features_reshaped = x.view(N, self.num_codebooks, self.codebook_dim)

        # Quantization step
        codebook_value = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        quantized_values = torch.where(features_reshaped > 0, codebook_value, -codebook_value)

        # Index calculation
        if self.token_factorization:
            pre_bits = quantized_values[..., : self.factorized_bits[0]]
            post_bits = quantized_values[..., self.factorized_bits[0] :]

            indices_pre = ((pre_bits > 0).int() * self.pre_mask.int()).sum(-1)
            indices_post = ((post_bits > 0).int() * self.post_mask.int()).sum(-1)

            indices_pre_flat = indices_pre.flatten()
            indices_post_flat = indices_post.flatten()
            sparse_indices_quantized = (indices_pre_flat, indices_post_flat)
        else:
            indices = ((quantized_values > 0).int() * self.mask.int()).sum(-1)
            sparse_indices_quantized = indices.flatten()

        # Entropy loss (training only)
        if self.training and return_loss:
            if fp32_loss_computation:
                features_flat_fp32 = features_reshaped.view(-1, self.codebook_dim).float()
                codebook_fp32 = self.codebook.float()
            else:
                features_flat_fp32 = features_reshaped.view(-1, self.codebook_dim)
                codebook_fp32 = self.codebook

            logits = 2 * torch.mm(features_flat_fp32, codebook_fp32.T)

            if mask is not None:
                if mask.shape[0] != N:
                    raise ValueError(f"Mask shape {mask.shape} doesn't match number of features {N}")
                mask_expanded = mask.unsqueeze(1).repeat(1, self.num_codebooks).view(-1)
            else:
                mask_expanded = None

            temperature = 1.0 / inv_temperature if inv_temperature > 0 else 0.01
            per_sample_entropy, codebook_entropy, entropy_aux_loss = entropy_loss(
                logits=logits,
                mask=mask_expanded,
                temperature=temperature,
                sample_minimization_weight=self.sample_minimization_weight,
                batch_maximization_weight=self.batch_maximization_weight,
            )
        else:
            dtype = torch.float32 if fp32_loss_computation else x.dtype
            per_sample_entropy = torch.tensor(0.0, dtype=dtype, device=x.device)
            codebook_entropy = torch.tensor(0.0, dtype=dtype, device=x.device)
            entropy_aux_loss = torch.tensor(0.0, dtype=dtype, device=x.device)

        # Commitment loss
        if self.training:
            if fp32_loss_computation:
                features_fp32 = features_reshaped.float()
                quantized_fp32 = quantized_values.float()
            else:
                features_fp32 = features_reshaped
                quantized_fp32 = quantized_values

            commit_loss = F.mse_loss(features_fp32, quantized_fp32.detach(), reduction="none")

            if mask is not None:
                mask_expanded = mask.view(N, 1, 1).expand_as(commit_loss)
                commit_loss = commit_loss[mask_expanded].mean()
            else:
                commit_loss = commit_loss.mean()
        else:
            dtype = torch.float32 if fp32_loss_computation else x.dtype
            commit_loss = torch.tensor(0.0, dtype=dtype, device=x.device)

        # Straight-through estimator
        quantized_values_ste = features_reshaped + (quantized_values - features_reshaped).detach()

        # Output construction
        quantized_feats = quantized_values_ste.view(N, feature_dim)

        # Ensure fp32 losses if requested
        if self.training and return_loss and fp32_loss_computation:
            entropy_aux_loss = entropy_aux_loss.float()
            per_sample_entropy = per_sample_entropy.float()
            codebook_entropy = codebook_entropy.float()
            commit_loss = commit_loss.float()

        ret = (quantized_feats, entropy_aux_loss, sparse_indices_quantized)

        if not return_loss_breakdown:
            return ret

        placeholder_dtype = torch.float32 if fp32_loss_computation else x.dtype
        return ret, LossBreakdown(
            per_sample_entropy,
            codebook_entropy,
            commit_loss,
            torch.tensor(0.0, dtype=placeholder_dtype, device=x.device),
        )
