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

"""Sparse tensor operations for neural network layers.

This module provides neural network layers that operate on SparseTensor objects,
including linear layers, normalization layers, activation functions, and
spatial resampling operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from cosmos.model.tokenizer.models.modules import DEBUG

if TYPE_CHECKING:
    from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

__all__ = [
    # Linear
    "SparseLinear",
    # Normalization
    "SparseGroupNorm",
    "SparseLayerNorm",
    "SparseGroupNorm32",
    "SparseLayerNorm32",
    "SparseRMSNorm32",
    "LayerNorm32",
    "GroupNorm32",
    "ChannelLayerNorm32",
    "RMSNorm",
    "RMSNorm32",
    # Activation
    "SparseReLU",
    "SparseSiLU",
    "SparseGELU",
    "SparseActivation",
    # Spatial
    "SparseDownsample",
    "SparseDownsampleKeepCoords",
    "SparseUpsample",
    "SparseUpsampleTokenSplit",
    "SparseSubdivide",
    "SparseUpsampleNoCache",
]


# =============================================================================
# Linear Layers
# =============================================================================


class SparseLinear(nn.Linear):
    """Linear layer for SparseTensor inputs.

    Applies a linear transformation to the features of a SparseTensor
    while preserving coordinates and layout.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        """Initialize SparseLinear.

        Args:
            in_features: Size of input features.
            out_features: Size of output features.
            bias: Whether to include a bias term.
        """
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input: "SparseTensor") -> "SparseTensor":
        """Apply linear transformation to sparse tensor features.

        Args:
            input: SparseTensor with features to transform.

        Returns:
            SparseTensor with transformed features.
        """
        return input.replace(super().forward(input.feats))


# =============================================================================
# Normalization Layers
# =============================================================================


class SparseGroupNorm(nn.GroupNorm):
    """Group normalization for SparseTensor inputs.

    Applies group normalization per batch element.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """Initialize SparseGroupNorm.

        Args:
            num_groups: Number of groups to separate channels into.
            num_channels: Number of channels in input.
            eps: Small constant for numerical stability.
            affine: Whether to include learnable affine parameters.
        """
        super(SparseGroupNorm, self).__init__(num_groups, num_channels, eps, affine)

    def forward(self, input: "SparseTensor") -> "SparseTensor":
        """Apply group normalization to sparse tensor.

        Args:
            input: SparseTensor to normalize.

        Returns:
            Normalized SparseTensor.
        """
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            if DEBUG:
                assert (input.coords[input.layout[k], 0] == k).all(), "SparseGroupNorm: batch index mismatch"
            bfeats = input.feats[input.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = nn.functional.group_norm(
                bfeats,
                self.num_groups,
                self.weight.to(bfeats.dtype) if self.weight is not None else None,
                self.bias.to(bfeats.dtype) if self.bias is not None else None,
                self.eps,
            )
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)


class SparseLayerNorm(nn.LayerNorm):
    """Layer normalization for SparseTensor inputs.

    Applies layer normalization per batch element.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        """Initialize SparseLayerNorm.

        Args:
            normalized_shape: Input shape from expected input dimensions.
            eps: Small constant for numerical stability.
            elementwise_affine: Whether to include learnable affine parameters.
        """
        super(SparseLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input: "SparseTensor") -> "SparseTensor":
        """Apply layer normalization to sparse tensor.

        Args:
            input: SparseTensor to normalize.

        Returns:
            Normalized SparseTensor.
        """
        nfeats = nn.functional.layer_norm(
            input.feats,
            self.normalized_shape,
            self.weight.to(input.feats.dtype) if self.weight is not None else None,
            self.bias.to(input.feats.dtype) if self.bias is not None else None,
            self.eps,
        )
        return input.replace(nfeats)


class SparseGroupNorm32(SparseGroupNorm):
    """GroupNorm layer that converts to float32 before the forward pass."""

    def forward(self, x: "SparseTensor") -> "SparseTensor":
        """Apply group normalization in float32.

        Args:
            x: SparseTensor to normalize.

        Returns:
            Normalized SparseTensor in original dtype.
        """
        return super().forward(x.float()).type(x.dtype)


class SparseLayerNorm32(SparseLayerNorm):
    """LayerNorm layer that converts to float32 before the forward pass."""

    def forward(self, x: "SparseTensor") -> "SparseTensor":
        """Apply layer normalization in float32.

        Args:
            x: SparseTensor to normalize.

        Returns:
            Normalized SparseTensor in original dtype.
        """
        return super().forward(x.float()).type(x.dtype)


class LayerNorm32(nn.LayerNorm):
    """Standard LayerNorm that operates in float32."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization in float32.

        Args:
            x: Tensor to normalize.

        Returns:
            Normalized tensor in original dtype.
        """
        output = nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.to(x.dtype)


class GroupNorm32(nn.GroupNorm):
    """GroupNorm layer that converts to float32 before the forward pass."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply group normalization in float32.

        Args:
            x: Tensor to normalize.

        Returns:
            Normalized tensor in original dtype.
        """
        return super().forward(x.float()).type(x.dtype)


class ChannelLayerNorm32(LayerNorm32):
    """LayerNorm applied to channel dimension (NCHW format)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise layer normalization.

        Args:
            x: Tensor in NCHW format.

        Returns:
            Normalized tensor.
        """
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM - 1, *range(1, DIM - 1)).contiguous()
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            hidden_size: Size of hidden dimension.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class RMSNorm32(RMSNorm):
    """RMSNorm that operates in float32."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm in float32.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor in original dtype.
        """
        if not self.training and x.dtype != torch.float32 and self.weight.dtype == x.dtype:
            output = self._norm(x.float()).to(x.dtype)
            return output * self.weight

        output = self._norm(x.float()).type_as(x)
        output = output * self.weight.float()
        return output.to(x.dtype)


class SparseRMSNorm32(RMSNorm):
    """RMSNorm for SparseTensor inputs operating in float32."""

    def forward(self, input: "SparseTensor") -> "SparseTensor":
        """Apply RMSNorm to sparse tensor.

        Args:
            input: SparseTensor to normalize.

        Returns:
            Normalized SparseTensor.
        """
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]][None, :, :]
            bfeats = self._norm(bfeats.float())
            bfeats = bfeats * self.weight.float()
            nfeats[input.layout[k]] = bfeats.to(nfeats.dtype)

        return input.replace(nfeats)


# =============================================================================
# Activation Functions
# =============================================================================


class SparseReLU(nn.ReLU):
    """ReLU activation for SparseTensor inputs."""

    def forward(self, input: "SparseTensor") -> "SparseTensor":
        """Apply ReLU to sparse tensor features.

        Args:
            input: SparseTensor with features to activate.

        Returns:
            SparseTensor with activated features.
        """
        return input.replace(super().forward(input.feats))


class SparseSiLU(nn.SiLU):
    """SiLU (Swish) activation for SparseTensor inputs."""

    def forward(self, input: "SparseTensor") -> "SparseTensor":
        """Apply SiLU to sparse tensor features.

        Args:
            input: SparseTensor with features to activate.

        Returns:
            SparseTensor with activated features.
        """
        return input.replace(super().forward(input.feats))


class SparseGELU(nn.GELU):
    """GELU activation for SparseTensor inputs."""

    def forward(self, input: "SparseTensor") -> "SparseTensor":
        """Apply GELU to sparse tensor features.

        Args:
            input: SparseTensor with features to activate.

        Returns:
            SparseTensor with activated features.
        """
        return input.replace(super().forward(input.feats))


class SparseActivation(nn.Module):
    """Wrapper to apply any activation function to SparseTensor inputs."""

    def __init__(self, activation: nn.Module):
        """Initialize SparseActivation.

        Args:
            activation: Activation module to wrap.
        """
        super().__init__()
        self.activation = activation

    def forward(self, input: "SparseTensor") -> "SparseTensor":
        """Apply wrapped activation to sparse tensor features.

        Args:
            input: SparseTensor with features to activate.

        Returns:
            SparseTensor with activated features.
        """
        return input.replace(self.activation(input.feats))


# =============================================================================
# Spatial Operations
# =============================================================================


class SparseDownsample(nn.Module):
    """Downsample a sparse tensor by a factor using average pooling."""

    def __init__(
        self,
        factor: int | tuple[int, ...] | list[int],
        identifier: str = "",
    ):
        """Initialize SparseDownsample.

        Args:
            factor: Downsampling factor (int or tuple for each dimension).
            identifier: String identifier for caching upsample coordinates.
        """
        super(SparseDownsample, self).__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor
        self.identifier = identifier

    def forward(self, input: "SparseTensor") -> "SparseTensor":
        """Downsample sparse tensor.

        Args:
            input: SparseTensor to downsample.

        Returns:
            Downsampled SparseTensor.
        """
        from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert DIM == len(factor), "Input coordinates must have the same dimension as the downsample factor."

        is_special = (input.coords[:, 1:] == -1).all(dim=1)
        normal_mask = ~is_special

        if is_special.any():
            special_coords = input.coords[is_special]
            special_feats = input.feats[is_special]
        else:
            special_coords = None
            special_feats = None

        coords = input.coords[normal_mask]
        feats = input.feats[normal_mask]

        if coords.shape[0] == 0:
            out = SparseTensor(special_feats, special_coords, input.shape)
            out._scale = tuple([s / f for s, f in zip(input._scale, factor)])
            out._spatial_cache = input._spatial_cache
            out.register_spatial_cache(f"upsample_{self.identifier}_{factor}_coords", input.coords)
            out.register_spatial_cache(f"upsample_{self.identifier}_{factor}_layout", input.layout)
            out.register_spatial_cache(
                f"upsample_{self.identifier}_{factor}_idx",
                torch.arange(input.coords.shape[0], device=input.coords.device, dtype=torch.long),
            )
            return out

        coord = list(coords.unbind(dim=-1))
        for i, f in enumerate(factor):
            coord[i + 1] = coord[i + 1] // f

        MAX = [coord[i + 1].max().item() + 1 for i in range(DIM)]
        OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
        code = sum([c * o for c, o in zip(coord, OFFSET)])
        code, idx = code.unique(return_inverse=True)

        new_feats = torch.scatter_reduce(
            torch.zeros(
                code.shape[0],
                input.feats.shape[1],
                device=input.feats.device,
                dtype=input.feats.dtype,
            ),
            dim=0,
            index=idx.unsqueeze(1).expand(-1, input.feats.shape[1]),
            src=feats,
            reduce="mean",
        )
        new_coords = torch.stack(
            [code // OFFSET[0]] + [(code // OFFSET[i + 1]) % MAX[i] for i in range(DIM)],
            dim=-1,
        )

        num_special = 0
        if special_coords is not None:
            num_special = special_coords.shape[0]
            new_coords = torch.cat([special_coords, new_coords], dim=0)
            new_feats = torch.cat([special_feats, new_feats], dim=0)

        row_order = torch.arange(new_coords.shape[0], device=new_coords.device, dtype=torch.long)
        batch_sort_key = new_coords[:, 0].to(torch.long) * (new_coords.shape[0] + 1) + row_order
        batch_sort_idx = torch.argsort(batch_sort_key)
        new_coords = new_coords[batch_sort_idx]
        new_feats = new_feats[batch_sort_idx]

        perm_inv = torch.empty_like(batch_sort_idx).to(new_coords)
        perm_inv[batch_sort_idx] = torch.arange(batch_sort_idx.size(0)).to(new_coords)

        full_idx = torch.empty(
            input.coords.shape[0],
            dtype=idx.dtype,
            device=idx.device,
        )
        full_idx[normal_mask] = idx + num_special
        if special_coords is not None:
            full_idx[is_special] = torch.arange(
                num_special,
                dtype=idx.dtype,
                device=idx.device,
            )
        idx = perm_inv[full_idx]

        out = SparseTensor(
            new_feats,
            new_coords,
            input.shape,
        )
        out._scale = tuple([s / f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache

        out.register_spatial_cache(f"upsample_{self.identifier}_{factor}_coords", input.coords)
        out.register_spatial_cache(f"upsample_{self.identifier}_{factor}_layout", input.layout)
        out.register_spatial_cache(f"upsample_{self.identifier}_{factor}_idx", idx)

        return out


class SparseDownsampleKeepCoords(nn.Module):
    """Optimized sparse tensor downsampling with support for mean pooling and indexing.

    Coordinates start from (0, 0, 0, 0) with (-1, -1, -1, -1) as a special index.
    """

    def __init__(
        self,
        factor: int | tuple[int, ...] | list[int],
        mode: str = "index",
        identifier: str = "",
    ):
        """Initialize SparseDownsampleKeepCoords.

        Args:
            factor: Downsampling factor.
            mode: Either 'mean' for average pooling or 'index' to keep first point.
            identifier: String identifier for caching.
        """
        super(SparseDownsampleKeepCoords, self).__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor
        assert mode in ["mean", "index"], "Mode must be either 'mean' or 'index'"
        self.mode = mode
        self.identifier = identifier

    def forward(self, input: "SparseTensor") -> "SparseTensor":
        """Downsample sparse tensor while preserving coordinate structure.

        Args:
            input: SparseTensor to downsample.

        Returns:
            Downsampled SparseTensor.
        """
        from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

        # Quick dimension check
        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert DIM == len(factor)

        # Separate special case (-1, -1, -1, -1) from other coordinates
        is_special = (input.coords[:, 1:] == -1).all(dim=1)
        normal_mask = ~is_special

        # Handle special case
        if is_special.any():
            special_coords = input.coords[is_special]
            special_feats = input.feats[is_special]
        else:
            special_coords = None
            special_feats = None

        # Process normal coordinates
        coords = input.coords[normal_mask]
        feats = input.feats[normal_mask]

        if len(coords) == 0:
            # If only special case exists
            return SparseTensor(special_feats, special_coords, input.shape)

        # Get current scale and compute new scale
        current_scale = torch.tensor(input._scale, device=coords.device)
        new_scale = current_scale / torch.tensor(factor, device=coords.device)

        # Create scaling factor including batch (no scaling for batch dimension)
        full_new_scale = torch.ones(DIM + 1, device=coords.device)
        full_new_scale[1:] = new_scale  # No scaling for batch dimension

        # Group points based on their actual positions
        grouping_coords = (coords.float() * full_new_scale[None, :]).floor().long()
        grouping_coords_residual = (coords.float() * full_new_scale[None, :]) - grouping_coords.float()

        # Single MAX_BITS for all dimensions including batch
        MAX_BITS = 10

        # Verify dimensions don't exceed limits
        max_val = (1 << MAX_BITS) - 1
        if (grouping_coords > max_val).any():
            raise ValueError(f"Coordinates exceed {max_val}. Increase MAX_BITS.")

        # Compute hash treating all dimensions equally
        shifts = torch.arange(0, (DIM + 1) * MAX_BITS, MAX_BITS, device=coords.device, dtype=torch.long)
        final_hash = (grouping_coords * (1 << shifts[None, :])).sum(dim=1)

        # Get unique downsampled coordinates and inverse indices
        unique_hash, inverse_indices = torch.unique(final_hash, return_inverse=True)

        # Give each residual value type a unique index
        residual_hash = (grouping_coords_residual * (1 << shifts[None, :])).sum(dim=1)
        residual_indices = torch.unique(residual_hash, return_inverse=True)[1]

        # Find first occurrence of each unique hash
        first_indices = torch.full_like(unique_hash, len(final_hash), dtype=torch.long)
        first_indices = first_indices.scatter_reduce(
            0,
            inverse_indices,
            torch.arange(len(final_hash), device=final_hash.device),
            reduce="min",
        )

        if self.mode == "mean":
            # Initialize tensors for sum and count
            sum_feats = torch.zeros(len(unique_hash), feats.shape[1], device=feats.device, dtype=feats.dtype)

            # Use scatter_add_ to sum features
            sum_feats.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, feats.shape[1]), feats)

            # Count number of elements per group
            ones = torch.ones_like(inverse_indices, dtype=feats.dtype)
            group_counts = torch.zeros(len(unique_hash), device=feats.device, dtype=feats.dtype)
            group_counts.scatter_add_(0, inverse_indices, ones)

            # Compute mean
            new_feats = sum_feats / group_counts.unsqueeze(1)
        else:  # mode == 'index'
            # Keep features from first occurrences
            new_feats = feats[first_indices]

        # Get coordinates for first occurrence of each unique hash
        new_coords = coords[first_indices]

        num_special = 0
        if special_coords is not None:
            # Keep special tokens in the same cache/index space as regular tokens.
            num_special = special_coords.shape[0]
            new_coords = torch.cat([special_coords, new_coords], dim=0)
            new_feats = torch.cat([special_feats, new_feats], dim=0)

        row_order = torch.arange(new_coords.shape[0], device=new_coords.device, dtype=torch.long)
        batch_sort_key = new_coords[:, 0].to(torch.long) * (new_coords.shape[0] + 1) + row_order
        batch_sort_idx = torch.argsort(batch_sort_key)
        new_coords = new_coords[batch_sort_idx]
        new_feats = new_feats[batch_sort_idx]

        perm_inv = torch.empty_like(batch_sort_idx).to(new_coords)
        perm_inv[batch_sort_idx] = torch.arange(batch_sort_idx.size(0)).to(new_coords)

        full_inverse_indices = torch.empty(
            input.coords.shape[0],
            dtype=inverse_indices.dtype,
            device=inverse_indices.device,
        )
        full_inverse_indices[normal_mask] = inverse_indices + num_special

        full_residual_indices = torch.zeros(
            input.coords.shape[0],
            dtype=residual_indices.dtype,
            device=residual_indices.device,
        )
        full_residual_indices[normal_mask] = residual_indices

        if special_coords is not None:
            full_inverse_indices[is_special] = torch.arange(
                num_special,
                dtype=inverse_indices.dtype,
                device=inverse_indices.device,
            )

        inverse_indices = perm_inv[full_inverse_indices]
        residual_indices = full_residual_indices

        # Create output tensor
        out = SparseTensor(
            new_feats.to(input.feats.dtype),
            new_coords.to(input.coords.dtype),
            input.shape,
        )
        out._scale = tuple([s / f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache

        out.register_spatial_cache(f"upsample_{self.identifier}_{factor}_coords", input.coords)
        out.register_spatial_cache(f"upsample_{self.identifier}_{factor}_layout", input.layout)
        out.register_spatial_cache(f"upsample_{self.identifier}_{factor}_idx", inverse_indices)
        out.register_spatial_cache(f"upsample_{self.identifier}_{factor}_residual_idx", residual_indices)

        return out


class SparseUpsample(nn.Module):
    """Upsample a sparse tensor using nearest neighbor interpolation."""

    def __init__(
        self,
        factor: int | tuple[int, int, int] | list[int],
        identifier: str = "",
    ):
        """Initialize SparseUpsample.

        Args:
            factor: Upsampling factor.
            identifier: String identifier matching the downsample cache key.
        """
        super(SparseUpsample, self).__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor
        self.identifier = identifier

    def forward(self, input: "SparseTensor") -> "SparseTensor":
        """Upsample sparse tensor using cached coordinates.

        Args:
            input: SparseTensor to upsample.

        Returns:
            Upsampled SparseTensor.
        """
        from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert DIM == len(factor), "Input coordinates must have the same dimension as the upsample factor."

        new_coords = input.get_spatial_cache(f"upsample_{self.identifier}_{factor}_coords")
        new_layout = input.get_spatial_cache(f"upsample_{self.identifier}_{factor}_layout")
        idx = input.get_spatial_cache(f"upsample_{self.identifier}_{factor}_idx")
        if any([x is None for x in [new_coords, new_layout, idx]]):
            raise ValueError(f"Got None for {new_coords}, {new_layout}, {idx}")
        new_feats = input.feats[idx]
        out = SparseTensor(new_feats, new_coords, input.shape, new_layout)
        out._scale = tuple([s * f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache
        return out


class SparseUpsampleTokenSplit(nn.Module):
    """Upsample sparse tensor with token splitting for learned upsampling."""

    def __init__(
        self,
        factor: int | tuple[int, int, int] | list[int],
        input_dim: int,
        identifier: str = "",
    ):
        """Initialize SparseUpsampleTokenSplit.

        Args:
            factor: Upsampling factor.
            input_dim: Input feature dimension.
            identifier: String identifier matching the downsample cache key.
        """
        super().__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor
        self.total_upsample_factor = 1
        for f in factor:
            self.total_upsample_factor *= f

        self.linear_up = nn.Linear(input_dim, input_dim * self.total_upsample_factor)
        self.act = nn.GELU(approximate="tanh")
        self.norm = LayerNorm32(input_dim, elementwise_affine=False, eps=1e-6)
        self.linear_out = nn.Linear(input_dim, input_dim)
        self.identifier = identifier

    def forward(self, input: "SparseTensor") -> "SparseTensor":
        """Upsample with learned token splitting.

        Args:
            input: SparseTensor to upsample.

        Returns:
            Upsampled SparseTensor.
        """
        from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert DIM == len(factor), "Input coordinates must have the same dimension as the upsample factor."

        new_coords = input.get_spatial_cache(f"upsample_{self.identifier}_{factor}_coords")
        new_layout = input.get_spatial_cache(f"upsample_{self.identifier}_{factor}_layout")
        idx = input.get_spatial_cache(f"upsample_{self.identifier}_{factor}_idx")
        if any([x is None for x in [new_coords, new_layout, idx]]):
            raise ValueError(f"Got None for {new_coords}, {new_layout}, {idx}")
        new_feats = input.feats[idx]

        # split the new feats tokens
        new_feats = self.linear_up(new_feats)
        new_feats = self.act(new_feats)

        assert new_feats.shape[1] % self.total_upsample_factor == 0, "The new feats cannot be split into tokens."
        new_feats = new_feats.view(new_feats.shape[0], -1, self.total_upsample_factor)
        residual_idx = input.get_spatial_cache(f"upsample_{self.identifier}_{factor}_residual_idx")
        new_feats = torch.gather(
            new_feats,
            2,
            residual_idx.unsqueeze(1).unsqueeze(1).expand(-1, new_feats.shape[1], -1),
        ).squeeze(-1)

        new_feats = self.norm(new_feats)
        new_feats = self.linear_out(new_feats)

        out = SparseTensor(new_feats, new_coords, input.shape, new_layout)
        out._scale = tuple([s * f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache
        return out


class SparseSubdivide(nn.Module):
    """Upsample sparse tensor by subdividing each point into a grid."""

    def __init__(self, factor: tuple[int, ...] | list[int]):
        """Initialize SparseSubdivide.

        Args:
            factor: Subdivision factor as tuple/list of integers.
        """
        super(SparseSubdivide, self).__init__()
        assert isinstance(factor, (tuple, list)), "factor must be a list/tuple of integers."
        self.factor = factor
        grid_shapes = [torch.arange(fac) for fac in factor]
        grids = torch.meshgrid(*grid_shapes, indexing="ij")
        grid_offsets = torch.stack([g.flatten() for g in grids], dim=1)
        self.register_buffer("grid_offsets", grid_offsets)

    def forward(self, x: "SparseTensor") -> "SparseTensor":
        """Subdivide each point into a grid of points.

        Args:
            x: SparseTensor to subdivide.

        Returns:
            Subdivided SparseTensor.
        """
        from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

        num_dim = x.coords.shape[-1]
        assert len(self.factor) == num_dim - 1, "factor must have the same number of dimensions as the input tensor."
        coords = x.coords
        feats = x.feats

        upsampled_size = self.grid_offsets.shape[0]
        repeated_coords = coords.repeat_interleave(upsampled_size, dim=0)
        expanded_offsets = self.grid_offsets.repeat(coords.shape[0], 1)
        for dim in range(1, num_dim):
            repeated_coords[:, dim] = repeated_coords[:, dim] * self.factor[dim - 1] + expanded_offsets[:, dim - 1]

        feats = feats.repeat_interleave(upsampled_size, dim=0)
        out = SparseTensor(feats=feats, coords=repeated_coords)
        return out


class SparseUpsampleNoCache(nn.Module):
    """Upsample sparse tensor without using cached coordinates.

    For each input point, creates a grid of points scaled by the factor.
    """

    def __init__(
        self,
        factor: int | tuple[int, ...] | list[int],
        identifier: str = "",
    ):
        """Initialize SparseUpsampleNoCache.

        Args:
            factor: Upsampling factor.
            identifier: String identifier (unused, for API compatibility).
        """
        super(SparseUpsampleNoCache, self).__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor
        self.identifier = identifier

    def forward(self, input: "SparseTensor") -> "SparseTensor":
        """Upsample sparse tensor by creating grid points.

        Args:
            input: SparseTensor to upsample.

        Returns:
            Upsampled SparseTensor.
        """
        from cosmos.model.tokenizer.models.modules.sparse_tensor import SparseTensor

        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert DIM == len(factor), "Input coordinates must have the same dimension as the upsample factor."

        # Extract batch and spatial coordinates
        batch_coords = input.coords[:, 0:1]  # Keep dimension
        spatial_coords = input.coords[:, 1:]

        # Create all possible offsets within the factor
        offset_ranges = [torch.arange(f, device=input.coords.device) for f in factor]
        offset_meshgrid = torch.meshgrid(*offset_ranges, indexing="ij")
        offsets = torch.stack([grid.flatten() for grid in offset_meshgrid], dim=-1)

        # Ensure offsets are int32
        offsets = offsets.to(torch.int32)

        num_offsets = offsets.shape[0]

        # Scale the original coordinates by the factor and ensure they're int32
        scaled_coords = (
            spatial_coords * torch.tensor(factor, device=input.coords.device, dtype=torch.int32).unsqueeze(0)
        ).to(torch.int32)

        # For each input point, create a grid of upsampled points
        # First, repeat each coordinate for all offsets
        expanded_coords = scaled_coords.repeat_interleave(num_offsets, dim=0)
        expanded_batch = batch_coords.repeat_interleave(num_offsets, dim=0).to(torch.int32)

        # Then, add all offsets to create the upsampled grid
        repeated_offsets = offsets.repeat(input.coords.shape[0], 1)
        upsampled_spatial_coords = expanded_coords + repeated_offsets

        # Combine batch dimension and spatial coordinates
        upsampled_coords = torch.cat([expanded_batch, upsampled_spatial_coords], dim=1)

        # Ensure the final coordinates are int32
        upsampled_coords = upsampled_coords.to(torch.int32)

        # Repeat features for all upsampled points
        upsampled_feats = input.feats.repeat_interleave(num_offsets, dim=0)

        # Create the output sparse tensor
        out = SparseTensor(
            upsampled_feats,
            upsampled_coords,
            input.shape,
        )

        # Update scale
        out._scale = tuple([s * f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache

        return out
