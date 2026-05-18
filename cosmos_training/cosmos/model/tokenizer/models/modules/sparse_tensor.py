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

"""SparseTensor data structure and operations.

This module provides the SparseTensor class for efficient sparse tensor operations
with support for both torchsparse and spconv backends.
"""

# ruff: noqa
# pylint: skip-file
from __future__ import annotations

from typing import overload

import numpy as np
import torch

from cosmos.model.tokenizer.models.modules import BACKEND, DEBUG

SparseTensorData = None  # Lazy import


def _assert_temporal_coords_sorted(batch_times: torch.Tensor) -> None:
    """Validate the searchsorted temporal-order invariant in debug mode."""
    if not DEBUG or batch_times.numel() < 2:
        return
    if not bool(torch.all(batch_times[1:] >= batch_times[:-1]).item()):
        raise AssertionError("temporal coords must be sorted within each batch")


class PureTorchSparseTensor:
    """Pure PyTorch sparse tensor container.

    This is a lightweight container that stores features and coordinates
    without requiring spconv or torchsparse. It provides the same interface
    as the external backends for compatibility.

    Args:
        feats: Feature tensor of shape [N, *feature_dims]
        coords: Coordinate tensor of shape [N, coord_dims]
        spatial_shape: Optional spatial shape (for spconv compatibility)
        batch_size: Optional batch size (for spconv compatibility)
    """

    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        spatial_shape=None,
        batch_size=None,
        **kwargs,
    ):
        # Store features and coordinates directly
        self._feats = feats
        self._coords = coords
        self._spatial_shape = spatial_shape
        self._batch_size = batch_size

        # Store any extra kwargs for compatibility
        self._kwargs = kwargs

    # torchsparse-style accessors
    @property
    def F(self) -> torch.Tensor:
        """Features (torchsparse-style accessor)."""
        return self._feats

    @F.setter
    def F(self, value: torch.Tensor):
        self._feats = value

    @property
    def C(self) -> torch.Tensor:
        """Coordinates (torchsparse-style accessor)."""
        return self._coords

    @C.setter
    def C(self, value: torch.Tensor):
        self._coords = value

    # spconv-style accessors
    @property
    def features(self) -> torch.Tensor:
        """Features (spconv-style accessor)."""
        return self._feats

    @features.setter
    def features(self, value: torch.Tensor):
        self._feats = value

    @property
    def indices(self) -> torch.Tensor:
        """Indices/coordinates (spconv-style accessor)."""
        return self._coords

    @indices.setter
    def indices(self, value: torch.Tensor):
        self._coords = value

    @property
    def spatial_shape(self):
        """Spatial shape (spconv compatibility)."""
        return self._spatial_shape

    @property
    def batch_size(self):
        """Batch size (spconv compatibility)."""
        return self._batch_size

    def dense(self) -> torch.Tensor:
        """Convert to dense tensor.

        Returns a dense tensor by placing features at their coordinate positions.
        This is a simplified implementation that works for our use case.
        """
        if self._coords.shape[0] == 0:
            # Empty tensor case
            return torch.zeros(
                (1,) + tuple(self._feats.shape[1:]),
                dtype=self._feats.dtype,
                device=self._feats.device,
            )

        # Get dimensions from coordinates
        # coords format: [batch, T, H, W, Z] or similar
        batch_size = self._coords[:, 0].max().item() + 1 if self._coords.shape[0] > 0 else 1

        # For T, H, W, Z dimensions
        max_coords = self._coords.max(dim=0)[0]
        spatial_dims = [max_coords[i].item() + 1 for i in range(1, self._coords.shape[1])]

        # Feature dimensions
        feat_dims = list(self._feats.shape[1:])

        # Create output tensor
        output_shape = [batch_size] + spatial_dims + feat_dims
        output = torch.zeros(output_shape, dtype=self._feats.dtype, device=self._feats.device)

        # Place features at coordinates
        # This handles the general case
        if self._coords.shape[1] == 5:  # [batch, T, H, W, Z]
            for i in range(self._coords.shape[0]):
                b, t, h, w, z = self._coords[i].tolist()
                output[b, t, h, w, z] = self._feats[i]
        elif self._coords.shape[1] == 4:  # [batch, T, H, W]
            for i in range(self._coords.shape[0]):
                b, t, h, w = self._coords[i].tolist()
                output[b, t, h, w] = self._feats[i]

        return output


__all__ = [
    "SparseTensor",
    "PureTorchSparseTensor",
    "sparse_batch_broadcast",
    "sparse_batch_op",
    "sparse_cat",
    "sparse_unbind",
    "reconstruct_from_temporal_slices",
]


class SparseTensor:
    """Sparse tensor with support for both torchsparse and spconv backends.

    A sparse tensor representation that stores only non-zero elements efficiently.
    Supports batched operations with per-batch layout tracking.

    Args:
        feats: Features of the sparse tensor with shape [N, *feature_dims].
        coords: Coordinates of the sparse tensor with shape [N, 5] as [batch, T, H, W, Z].
        shape: Shape of the sparse tensor (batch_size, *feature_dims).
        layout: Layout slices for each batch element.
        data: Backend-specific sparse tensor data (torchsparse or spconv).

    Note:
        - Data corresponding to the same batch should be contiguous.
        - Coords should be in [0, 1023] range for each dimension.
    """

    @overload
    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size | None = None,
        layout: list[slice] | None = None,
        **kwargs,
    ): ...

    @overload
    def __init__(
        self,
        data,
        shape: torch.Size | None = None,
        layout: list[slice] | None = None,
        **kwargs,
    ): ...

    def __init__(self, *args, **kwargs):
        # Lazy import of sparse tensor backend
        global SparseTensorData
        if SparseTensorData is None:
            import importlib

            if BACKEND == "pytorch":
                SparseTensorData = PureTorchSparseTensor
            elif BACKEND == "torchsparse":
                SparseTensorData = importlib.import_module("torchsparse").SparseTensor
            elif BACKEND == "spconv":
                SparseTensorData = importlib.import_module("spconv.pytorch").SparseConvTensor

        scale = kwargs.pop("scale", (1, 1, 1, 1))
        spatial_cache = kwargs.pop("spatial_cache", {})
        has_special_tokens = kwargs.pop("has_special_tokens", None)

        method_id = 0
        if len(args) != 0:
            method_id = 0 if isinstance(args[0], torch.Tensor) else 1
        else:
            method_id = 1 if "data" in kwargs else 0

        if method_id == 0:
            feats, coords, shape, layout = args + (None,) * (4 - len(args))
            if "feats" in kwargs:
                feats = kwargs["feats"]
                del kwargs["feats"]
            if "coords" in kwargs:
                coords = kwargs["coords"]
                del kwargs["coords"]
            if "shape" in kwargs:
                shape = kwargs["shape"]
                del kwargs["shape"]
            if "layout" in kwargs:
                layout = kwargs["layout"]
                del kwargs["layout"]

            if shape is None:
                shape = self.__cal_shape(feats, coords)
            if layout is None:
                layout = self.__cal_layout(coords, shape[0])
            if BACKEND == "pytorch":
                self.data = SparseTensorData(feats, coords, **kwargs)
            elif BACKEND == "torchsparse":
                self.data = SparseTensorData(feats, coords, **kwargs)
            elif BACKEND == "spconv":
                # Handle empty tensor case for spconv
                if coords.shape[0] == 0:
                    # For empty tensors, calculate ndim from coordinate shape
                    ndim = coords.shape[1] - 1  # subtract 1 for batch dimension

                    # Create spatial_shape that matches ndim
                    if ndim <= 0:
                        spatial_shape = [1]  # Minimum 1D
                    else:
                        spatial_shape = [1] * ndim  # Create ndim-dimensional spatial shape

                    batch_size = max(shape[0], 1)  # Ensure at least 1 for batch_size

                    # Create properly shaped empty features - avoid reshape issues
                    if len(feats.shape) == 1:
                        # If feats is 1D, we can't reshape it safely when empty
                        feature_dim = 1 if feats.numel() == 0 else feats.shape[0]
                        reshaped_feats = torch.empty((0, feature_dim), dtype=feats.dtype, device=feats.device)
                    else:
                        # Multi-dimensional features
                        feature_dim = feats.shape[1:].numel() if feats.numel() > 0 else 1
                        reshaped_feats = torch.empty((0, feature_dim), dtype=feats.dtype, device=feats.device)
                else:
                    # Non-empty tensor handling
                    max_coords = coords.max(0)[0]
                    ndim = coords.shape[1] - 1  # subtract 1 for batch dimension

                    # Create spatial_shape based on actual coordinate dimensions
                    if ndim <= 0:
                        spatial_shape = [1]
                    else:
                        # Use actual coordinate bounds for each spatial dimension
                        spatial_shape = []
                        for i in range(1, min(len(max_coords), ndim + 1)):
                            spatial_shape.append(max_coords[i].item() + 1)

                        # Pad with 1s if we don't have enough dimensions
                        while len(spatial_shape) < ndim:
                            spatial_shape.append(1)

                        # Truncate if we have too many dimensions
                        spatial_shape = spatial_shape[:ndim]

                    batch_size = shape[0]

                    # Handle feature reshaping more carefully
                    if feats.numel() == 0:
                        reshaped_feats = torch.empty((0, 1), dtype=feats.dtype, device=feats.device)
                    else:
                        # Calculate expected feature dimension
                        feature_dims = feats.shape[1:]
                        if len(feature_dims) == 0:
                            # 1D feature vector
                            reshaped_feats = feats.reshape(feats.shape[0], 1)
                        else:
                            # Multi-dimensional features - flatten
                            reshaped_feats = feats.reshape(feats.shape[0], -1)

                self.data = SparseTensorData(
                    reshaped_feats,
                    coords,
                    spatial_shape,
                    batch_size,
                    **kwargs,
                )
                self.data._features = feats
        elif method_id == 1:
            data, shape, layout = args + (None,) * (3 - len(args))
            if "data" in kwargs:
                data = kwargs["data"]
                del kwargs["data"]
            if "shape" in kwargs:
                shape = kwargs["shape"]
                del kwargs["shape"]
            if "layout" in kwargs:
                layout = kwargs["layout"]
                del kwargs["layout"]

            self.data = data
            if shape is None:
                shape = self.__cal_shape(self.feats, self.coords)
            if layout is None:
                layout = self.__cal_layout(self.coords, shape[0])

        self._shape = shape
        self._layout = layout
        self._scale = scale
        self._spatial_cache = spatial_cache
        self._has_special_tokens = has_special_tokens

        if DEBUG:
            try:
                assert self.feats.shape[0] == self.coords.shape[0], (
                    f"Invalid feats shape: {self.feats.shape}, coords shape: {self.coords.shape}"
                )
                assert self.shape == self.__cal_shape(self.feats, self.coords), f"Invalid shape: {self.shape}"
                assert self.layout == self.__cal_layout(self.coords, self.shape[0]), f"Invalid layout: {self.layout}"
                for i in range(self.shape[0]):
                    batch_slice = self.layout[i]
                    if batch_slice.start < batch_slice.stop:  # Only check non-empty batches
                        assert torch.all(self.coords[batch_slice, 0] == i), f"The data of batch {i} is not contiguous"
            except Exception as e:
                print("Debugging information:")
                print(f"- Shape: {self.shape}")
                print(f"- Layout: {self.layout}")
                print(f"- Scale: {self._scale}")
                print(f"- Coords shape: {self.coords.shape}")
                print(f"- Feats shape: {self.feats.shape}")
                raise e

    def __cal_shape(self, feats, coords):
        shape = []

        # Handle empty tensor case
        if coords.shape[0] == 0:
            shape.append(0)  # batch size 0
            shape.extend([*feats.shape[1:]])  # feature dimensions
            return torch.Size(shape)

        shape.append(coords[:, 0].max().item() + 1)
        shape.extend([*feats.shape[1:]])
        return torch.Size(shape)

    def __cal_layout(self, coords, batch_size):
        # Handle empty tensor case
        if coords.shape[0] == 0:
            return [slice(0, 0) for _ in range(batch_size)]

        seq_len = torch.bincount(coords[:, 0], minlength=batch_size)
        offset = torch.cumsum(seq_len, dim=0)

        # Single CPU transfer instead of O(batch_size) .item() calls
        # This reduces from 2*batch_size CPU-GPU syncs to just 2 syncs
        offset_list = offset.tolist()
        seq_len_list = seq_len.tolist()

        layout = [slice(offset_list[i] - seq_len_list[i], offset_list[i]) for i in range(batch_size)]
        return layout

    @property
    def shape(self) -> torch.Size:
        return self._shape

    def dim(self) -> int:
        return len(self.shape)

    @property
    def layout(self) -> list[slice]:
        return self._layout

    @staticmethod
    def _build_layout_from_batch_indices(batch_indices: torch.Tensor, batch_size: int) -> list[slice]:
        """Build batch-contiguous layout slices from ordered batch indices."""
        if batch_size <= 0:
            return []
        if batch_indices.numel() == 0:
            return [slice(0, 0) for _ in range(batch_size)]

        seq_len = torch.bincount(batch_indices.to(torch.long), minlength=batch_size)
        offset = torch.cumsum(seq_len, dim=0)
        offset_list = offset.tolist()
        seq_len_list = seq_len.tolist()
        return [slice(offset_list[i] - seq_len_list[i], offset_list[i]) for i in range(batch_size)]

    def _select_contiguous_batch_ranges(
        self,
        batch_ranges: list[tuple[int, int]],
        temporal_shift: int | torch.Tensor | None = None,
        has_special_tokens: bool | None = None,
    ) -> "SparseTensor":
        """Build a sparse tensor from contiguous per-batch row ranges.

        Assumes each `(start, end)` pair indexes a contiguous subrange inside the
        corresponding batch slice. This avoids boolean masking and layout
        recomputation on the common no-special-token decode path.
        """
        selected_coords_parts = []
        selected_feats_parts = []
        new_layout: list[slice] = []
        offset = 0

        temporal_shift_value = None
        if temporal_shift is not None:
            temporal_shift_value = (
                temporal_shift.to(dtype=self.coords.dtype)
                if isinstance(temporal_shift, torch.Tensor)
                else temporal_shift
            )

        for batch_idx, (relative_start, relative_end) in enumerate(batch_ranges):
            batch_slice = self.layout[batch_idx]
            absolute_start = batch_slice.start + relative_start
            absolute_end = batch_slice.start + relative_end
            batch_len = max(0, absolute_end - absolute_start)

            if batch_len > 0:
                batch_coords = self.coords[absolute_start:absolute_end]
                if temporal_shift_value is not None:
                    batch_coords = batch_coords.clone()
                    batch_coords[:, 1] -= temporal_shift_value
                selected_coords_parts.append(batch_coords)
                selected_feats_parts.append(self.feats[absolute_start:absolute_end])

            new_layout.append(slice(offset, offset + batch_len))
            offset += batch_len

        if not selected_coords_parts:
            empty_coords = torch.empty(
                (0, self.coords.shape[1]),
                dtype=self.coords.dtype,
                device=self.coords.device,
            )
            empty_feats = torch.empty(
                (0,) + self.feats.shape[1:],
                dtype=self.feats.dtype,
                device=self.feats.device,
            )
            return SparseTensor(
                feats=empty_feats,
                coords=empty_coords,
                shape=torch.Size([self.shape[0]] + list(self.feats.shape[1:])),
                layout=new_layout,
                has_special_tokens=has_special_tokens,
            )

        return SparseTensor(
            feats=torch.cat(selected_feats_parts, dim=0),
            coords=torch.cat(selected_coords_parts, dim=0),
            shape=torch.Size([self.shape[0]] + list(self.feats.shape[1:])),
            layout=new_layout,
            has_special_tokens=has_special_tokens,
        )

    def get_batch_seq_lens(self) -> list[int]:
        """Return per-batch sequence lengths, cached per sparse layout."""
        cache_key = "batch_seq_lens"
        seq_lens = self.get_spatial_cache(cache_key)
        if seq_lens is None:
            seq_lens = [layout_slice.stop - layout_slice.start for layout_slice in self.layout]
            self.register_spatial_cache(cache_key, seq_lens)
        return seq_lens

    def get_cu_seqlens(self, device: str | torch.device | None = None) -> torch.Tensor:
        """Return cached cumulative sequence lengths for varlen attention backends."""
        target_device = self.device if device is None else torch.device(device)
        cache_key = ("cu_seqlens", str(target_device))
        cu_seqlens = self.get_spatial_cache(cache_key)
        if cu_seqlens is None:
            seq_lens = self.get_batch_seq_lens()
            cu_seqlens = torch.tensor([0, *seq_lens], dtype=torch.int32, device=target_device).cumsum(
                dim=0, dtype=torch.int32
            )
            self.register_spatial_cache(cache_key, cu_seqlens)
        return cu_seqlens

    @property
    def feats(self) -> torch.Tensor:
        if BACKEND == "pytorch":
            return self.data.F
        elif BACKEND == "torchsparse":
            return self.data.F
        elif BACKEND == "spconv":
            return self.data.features

    @feats.setter
    def feats(self, value: torch.Tensor):
        if BACKEND == "pytorch":
            self.data.F = value
        elif BACKEND == "torchsparse":
            self.data.F = value
        elif BACKEND == "spconv":
            self.data.features = value

    @property
    def coords(self) -> torch.Tensor:
        if BACKEND == "pytorch":
            return self.data.C
        elif BACKEND == "torchsparse":
            return self.data.C
        elif BACKEND == "spconv":
            return self.data.indices

    @coords.setter
    def coords(self, value: torch.Tensor):
        if BACKEND == "pytorch":
            self.data.C = value
        elif BACKEND == "torchsparse":
            self.data.C = value
        elif BACKEND == "spconv":
            self.data.indices = value

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def device(self):
        return self.feats.device

    @overload
    def to(self, dtype: torch.dtype) -> "SparseTensor": ...

    @overload
    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SparseTensor": ...

    def to(self, *args, **kwargs) -> "SparseTensor":
        device = None
        dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if "dtype" in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs["dtype"]
        if "device" in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs["device"]

        target_device: torch.device | None
        if device is None:
            target_device = None
        elif isinstance(device, torch.device):
            target_device = device
        elif isinstance(device, int):
            target_device = torch.device("cuda", device)
        else:
            target_device = torch.device(device)

        if (target_device is None or target_device == self.feats.device) and (
            dtype is None or dtype == self.feats.dtype
        ):
            return self

        new_feats = self.feats.to(device=target_device, dtype=dtype)
        if target_device is None or target_device == self.coords.device:
            return self.replace(new_feats)

        new_coords = self.coords.to(device=target_device)
        return SparseTensor(
            feats=new_feats,
            coords=new_coords,
            shape=self.shape,
            layout=self.layout,
            scale=self._scale,
            has_special_tokens=self._has_special_tokens,
        )

    def type(self, dtype):
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats)

    def cpu(self) -> "SparseTensor":
        return self.to(device=torch.device("cpu"))

    def cuda(self) -> "SparseTensor":
        return self.to(device=torch.device("cuda"))

    def half(self) -> "SparseTensor":
        new_feats = self.feats.half()
        return self.replace(new_feats)

    def float(self) -> "SparseTensor":
        new_feats = self.feats.float()
        return self.replace(new_feats)

    def detach(self) -> "SparseTensor":
        new_feats = self.feats.detach()
        return self.replace(new_feats)

    def dense(self) -> torch.Tensor:
        if BACKEND == "pytorch":
            return self.data.dense()
        elif BACKEND == "torchsparse":
            return self.data.dense()
        elif BACKEND == "spconv":
            return self.data.dense()

    def reshape(self, *shape) -> "SparseTensor":
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats)

    def unbind(self, dim: int) -> list["SparseTensor"]:
        return sparse_unbind(self, dim)

    def add_first_token_(self, feats):
        """Add features to the first token of each batch in-place."""
        indices = [layout_slice.start for layout_slice in self.layout]
        self.feats[indices] += feats

    def extract_first_n_tokens(self, n=1) -> tuple["SparseTensor", "SparseTensor"]:
        """Extract the first n tokens from each batch and remove from tensor.

        Args:
            n: Number of tokens to extract from the beginning of each batch.

        Returns:
            Tuple of:
                - SparseTensor with only the first n tokens from each batch
                - SparseTensor with the remaining features
        """
        batch_size = len(self.layout)

        # Collect all indices to extract and to keep
        extract_indices = []
        for batch_idx in range(batch_size):
            batch_slice = self.layout[batch_idx]
            if batch_slice.stop - batch_slice.start < n:
                raise ValueError(f"Batch {batch_idx} has fewer than {n} tokens")
            extract_indices.extend(range(batch_slice.start, batch_slice.start + n))

        # Create masks for extracted and remaining indices
        all_indices = torch.arange(self.feats.shape[0], device=self.device)
        extract_mask = torch.zeros_like(all_indices, dtype=torch.bool)
        extract_mask[extract_indices] = True

        # Extract features and coords for both parts
        extracted_feats = self.feats[extract_mask]
        extracted_coords = self.coords[extract_mask]
        remaining_feats = self.feats[~extract_mask]
        remaining_coords = self.coords[~extract_mask]

        # Create new sparse tensors with proper metadata
        extracted = SparseTensor(
            feats=extracted_feats, coords=extracted_coords, has_special_tokens=self.has_special_tokens()
        )
        remaining = SparseTensor(
            feats=remaining_feats,
            coords=remaining_coords,
            has_special_tokens=self._has_special_tokens,
        )
        remaining._scale = self._scale
        remaining._spatial_cache = self._spatial_cache

        return extracted, remaining

    def extract_first_token(self) -> tuple["SparseTensor", "SparseTensor"]:
        """Extract features at layout start indices and remove from tensor.

        This is a special case of extract_first_n_tokens with n=1.

        Returns:
            Tuple of:
                - SparseTensor with only the features at start indices
                - SparseTensor with the remaining features
        """
        return self.extract_first_n_tokens(1)

    def has_special_tokens(self) -> bool:
        """Detect whether tensor contains special sparse tokens.

        CLS token has coords of [-1, -1, -1, -1] (excluding batch index).

        Returns:
            bool: True if any batch has a CLS token, False otherwise.
        """
        if self._has_special_tokens is None:
            self._has_special_tokens = (
                bool((self.coords[:, 1:] == -1).all(dim=1).any().item()) if self.coords.shape[0] > 0 else False
            )
        return self._has_special_tokens

    def has_cls_token(self) -> bool:
        """Backward-compatible alias for special sparse token detection."""
        return self.has_special_tokens()

    def replace(self, feats: torch.Tensor, coords: torch.Tensor | None = None) -> "SparseTensor":
        new_shape = [self.shape[0]]
        new_shape.extend(feats.shape[1:])
        if coords is not None:
            # Coordinate changes invalidate cached layout- and coord-derived
            # metadata, including the special-token flag. Rebuild from scratch.
            return SparseTensor(
                feats=feats,
                coords=coords,
                shape=torch.Size(new_shape),
                scale=self._scale,
            )

        if BACKEND == "pytorch":
            new_data = SparseTensorData(
                feats=feats,
                coords=self.data.C,
            )
        elif BACKEND == "torchsparse":
            new_data = SparseTensorData(
                feats=feats,
                coords=self.data.coords,
                stride=self.data.stride,
                spatial_range=self.data.spatial_range,
            )
            new_data._caches = self.data._caches
        elif BACKEND == "spconv":
            new_data = SparseTensorData(
                self.data.features.reshape(self.data.features.shape[0], -1),
                self.data.indices,
                self.data.spatial_shape,
                self.data.batch_size,
                self.data.grid,
                self.data.voxel_num,
                self.data.indice_dict,
            )
            new_data._features = feats
            new_data.benchmark = self.data.benchmark
            new_data.benchmark_record = self.data.benchmark_record
            new_data.thrust_allocator = self.data.thrust_allocator
            new_data._timer = self.data._timer
            new_data.force_algo = self.data.force_algo
            new_data.int8_scale = self.data.int8_scale
        new_tensor = SparseTensor(
            new_data,
            shape=torch.Size(new_shape),
            layout=self.layout,
            scale=self._scale,
            spatial_cache=self._spatial_cache,
            has_special_tokens=self._has_special_tokens,
        )
        return new_tensor

    @staticmethod
    def full(aabb, dim, value, dtype=torch.float32, device=None) -> "SparseTensor":
        N, C = dim
        x = torch.arange(aabb[0], aabb[3] + 1)
        y = torch.arange(aabb[1], aabb[4] + 1)
        z = torch.arange(aabb[2], aabb[5] + 1)
        coords = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1).reshape(-1, 3)
        coords = torch.cat(
            [
                torch.arange(N).view(-1, 1).repeat(1, coords.shape[0]).view(-1, 1),
                coords.repeat(N, 1),
            ],
            dim=1,
        ).to(dtype=torch.int32, device=device)
        feats = torch.full((coords.shape[0], C), value, dtype=dtype, device=device)
        return SparseTensor(feats=feats, coords=coords, has_special_tokens=False)

    def __merge_sparse_cache(self, other: "SparseTensor") -> dict:
        new_cache = {}
        for k in set(list(self._spatial_cache.keys()) + list(other._spatial_cache.keys())):
            if k in self._spatial_cache:
                new_cache[k] = self._spatial_cache[k].copy()
            if k in other._spatial_cache:
                if k not in new_cache:
                    new_cache[k] = other._spatial_cache[k].copy()
                else:
                    new_cache[k].update(other._spatial_cache[k])
        return new_cache

    def __neg__(self) -> "SparseTensor":
        return self.replace(-self.feats)

    def __elemwise__(self, other: torch.Tensor | "SparseTensor", op: callable) -> "SparseTensor":
        other_tensor = other
        if isinstance(other, torch.Tensor):
            try:
                other_tensor = torch.broadcast_to(other, self.shape)
                other_tensor = sparse_batch_broadcast(self, other_tensor)
            except Exception:
                pass
        elif isinstance(other, SparseTensor):
            other_tensor = other.feats

        new_feats = op(self.feats, other_tensor)
        new_tensor = self.replace(new_feats)
        if isinstance(other, SparseTensor):
            new_tensor._spatial_cache = self.__merge_sparse_cache(other)
        return new_tensor

    def __add__(self, other: torch.Tensor | "SparseTensor" | float) -> "SparseTensor":
        return self.__elemwise__(other, torch.add)

    def __radd__(self, other: torch.Tensor | "SparseTensor" | float) -> "SparseTensor":
        return self.__elemwise__(other, torch.add)

    def __sub__(self, other: torch.Tensor | "SparseTensor" | float) -> "SparseTensor":
        return self.__elemwise__(other, torch.sub)

    def __rsub__(self, other: torch.Tensor | "SparseTensor" | float) -> "SparseTensor":
        return self.__elemwise__(other, lambda x, y: torch.sub(y, x))

    def __mul__(self, other: torch.Tensor | "SparseTensor" | float) -> "SparseTensor":
        return self.__elemwise__(other, torch.mul)

    def __rmul__(self, other: torch.Tensor | "SparseTensor" | float) -> "SparseTensor":
        return self.__elemwise__(other, torch.mul)

    def __truediv__(self, other: torch.Tensor | "SparseTensor" | float) -> "SparseTensor":
        return self.__elemwise__(other, torch.div)

    def __rtruediv__(self, other: torch.Tensor | "SparseTensor" | float) -> "SparseTensor":
        return self.__elemwise__(other, lambda x, y: torch.div(y, x))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (self.shape[0],), f"Invalid index shape: {idx.shape}"
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")

        coords = []
        feats = []
        for new_idx, old_idx in enumerate(idx):
            # Check if old_idx is within bounds
            if old_idx >= len(self.layout):
                # Return empty tensors for out-of-bounds batch indices
                empty_coords = torch.empty(
                    (0, self.coords.shape[1]),
                    dtype=self.coords.dtype,
                    device=self.coords.device,
                )
                empty_feats = torch.empty(
                    (0,) + self.feats.shape[1:],
                    dtype=self.feats.dtype,
                    device=self.feats.device,
                )
                coords.append(empty_coords)
                feats.append(empty_feats)
            else:
                batch_coords = self.coords[self.layout[old_idx]]
                batch_feats = self.feats[self.layout[old_idx]]

                # Handle empty batches (can happen due to temporal slicing)
                if batch_coords.shape[0] > 0:
                    batch_coords = batch_coords.clone()
                    batch_coords[:, 0] = new_idx
                    coords.append(batch_coords)
                    feats.append(batch_feats)
                else:
                    # Create empty tensors for empty batches
                    empty_coords = torch.empty(
                        (0, self.coords.shape[1]),
                        dtype=self.coords.dtype,
                        device=self.coords.device,
                    )
                    empty_feats = torch.empty(
                        (0,) + self.feats.shape[1:],
                        dtype=self.feats.dtype,
                        device=self.feats.device,
                    )
                    coords.append(empty_coords)
                    feats.append(empty_feats)

        # Concatenate all coordinates and features (including empty ones)
        if len(coords) > 0:
            coords = torch.cat(coords, dim=0).contiguous()
            feats = torch.cat(feats, dim=0).contiguous()
        else:
            # Handle case where idx is empty
            coords = torch.empty(
                (0, self.coords.shape[1]),
                dtype=self.coords.dtype,
                device=self.coords.device,
            )
            feats = torch.empty(
                (0,) + self.feats.shape[1:],
                dtype=self.feats.dtype,
                device=self.feats.device,
            )

        return SparseTensor(feats=feats, coords=coords)

    def register_spatial_cache(self, key, value) -> None:
        """Register a spatial cache.

        The spatial cache can be any value you want to cache.
        The registry and retrieval of the cache is based on current scale.
        """
        scale_key = str(self._scale)
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        self._spatial_cache[scale_key][key] = value

    def get_spatial_cache(self, key=None):
        """Get a spatial cache."""
        scale_key = str(self._scale)
        cur_scale_cache = self._spatial_cache.get(scale_key, {})
        if key is None:
            return cur_scale_cache
        return cur_scale_cache.get(key, None)

    def to_dense_padded(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert sparse tensor to dense padded representation.

        Returns:
            Tuple containing:
                - dense_features: Tensor with shape [batch, seq_len, feat_dim]
                - dense_coords: Tensor with shape [batch, seq_len, coord_dim]
                - mask: Boolean mask with shape [batch, seq_len] (True for valid positions)
        """
        batch_size = len(self.layout)

        # Calculate max sequence length across all batches
        max_seq_len = max(layout_slice.stop - layout_slice.start for layout_slice in self.layout)

        # Get feature and coordinate dimensions
        feat_dims = self.feats.shape[1:]
        coord_dims = self.coords.shape[1:]

        # Create output tensors
        feat_shape = (batch_size, max_seq_len) + feat_dims
        coord_shape = (batch_size, max_seq_len) + coord_dims

        dense_feats = torch.zeros(feat_shape, dtype=self.dtype, device=self.device)
        dense_coords = torch.full(coord_shape, -1, dtype=self.coords.dtype, device=self.coords.device)
        mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=self.device)

        # Fill in the tensors with data from each batch
        for batch_idx in range(batch_size):
            batch_slice = self.layout[batch_idx]
            seq_len = batch_slice.stop - batch_slice.start

            # Copy features and coordinates from the sparse tensor
            dense_feats[batch_idx, :seq_len] = self.feats[batch_slice]
            dense_coords[batch_idx, :seq_len] = self.coords[batch_slice]

            # Mark valid positions in the mask
            mask[batch_idx, :seq_len] = True

        return dense_feats, dense_coords, mask

    def get_batch_mean(self) -> "SparseTensor":
        """Compute the mean of features for each batch.

        Returns:
            SparseTensor: A sparse tensor with a single point per batch containing
                        the mean feature values for that batch.
        """
        batch_size = len(self.layout)

        # Get the feature dimensions
        feature_dims = self.feats.shape[1:]

        # Create output tensor with shape (batch_size, feature_dim)
        batch_means = torch.zeros((batch_size,) + feature_dims, dtype=self.dtype, device=self.device)

        # Create the coordinates for the mean features
        # Using [-1, -1, -1, -1] for spatial coordinates to indicate special tokens
        coords = torch.full(
            (batch_size, self.coords.shape[1]),
            -1,
            dtype=self.coords.dtype,
            device=self.coords.device,
        )

        # Set batch indices
        coords[:, 0] = torch.arange(batch_size, device=self.coords.device)

        # Compute mean for each batch
        for batch_idx in range(batch_size):
            # Get the batch slice
            batch_slice = self.layout[batch_idx]

            # Compute mean of features in this batch
            batch_features = self.feats[batch_slice]
            if len(batch_features) > 0:  # Check to avoid division by zero
                batch_means[batch_idx] = torch.mean(batch_features, dim=0)

        return SparseTensor(feats=batch_means, coords=coords)

    def expand_by_factors(self, factors, channel_duplicate=1):
        """Convert sparse representation to original locations.

        Args:
            factors: Tuple specifying patch dimensions (T, X, Y, Z) or (X, Y) or (T, X, Y)
            channel_duplicate: Factor to duplicate channels after expansion (default: 1)
        """
        if len(factors) == 2:
            X, Y = factors
            T, Z = 1, 1
        elif len(factors) == 3:
            T, X, Y = factors
            Z = 1
        elif len(factors) == 4:
            T, X, Y, Z = factors
        else:
            raise ValueError(f"Invalid patch size: {factors}")

        patch_product = T * X * Y * Z

        N, C = self.feats.shape
        channels = C // patch_product

        assert C == patch_product * channels, (
            f"Feature dimension {C} doesn't match patch_product ({patch_product}) * channels ({channels})"
        )

        feats_reshaped = self.feats.reshape(N, T, X, Y, Z, channels)

        # Create mesh grids for offsets within each patch dimension
        t_offsets, y_offsets, x_offsets, z_offsets = torch.meshgrid(
            torch.arange(T, device=self.device),
            torch.arange(X, device=self.device),
            torch.arange(Y, device=self.device),
            torch.arange(Z, device=self.device),
            indexing="ij",
        )

        # Flatten the offsets
        t_offsets = t_offsets.reshape(-1)
        x_offsets = x_offsets.reshape(-1)
        y_offsets = y_offsets.reshape(-1)
        z_offsets = z_offsets.reshape(-1)

        # Repeat each coordinate patch_product times
        expanded_coords = self.coords.repeat_interleave(patch_product, dim=0)

        tiled_t_offsets = t_offsets.repeat(N)
        tiled_x_offsets = x_offsets.repeat(N)
        tiled_y_offsets = y_offsets.repeat(N)
        tiled_z_offsets = z_offsets.repeat(N)

        # Update T, X, Y, and Z coordinates (indices 1, 2, 3, 4)
        if T > 1:
            expanded_coords[:, 1] = expanded_coords[:, 1] * T + tiled_t_offsets

        expanded_coords[:, 2] = expanded_coords[:, 2] * X + tiled_x_offsets
        expanded_coords[:, 3] = expanded_coords[:, 3] * Y + tiled_y_offsets

        if Z > 1:
            expanded_coords[:, 4] = expanded_coords[:, 4] * Z + tiled_z_offsets

        # Reshape features to match the expanded coordinates
        expanded_feats = feats_reshaped.reshape(-1, channels)

        # Apply channel duplication if requested
        if channel_duplicate > 1:
            expanded_feats = (
                expanded_feats.unsqueeze(-1)
                .repeat(1, 1, channel_duplicate)
                .reshape(N * patch_product, channels * channel_duplicate)
            )

        return SparseTensor(feats=expanded_feats, coords=expanded_coords)

    def shrink_by_factors(self, factors, channel_average=1):
        """Convert sparse representation to coarser resolution.

        This method is the inverse of expand_by_factors, converting from smaller patches
        to larger patches by grouping points and concatenating their features.

        Args:
            factors: Tuple specifying the target patch dimensions.
            channel_average: Factor to average channels after shrinking.

        Returns:
            SparseTensor: A new sparse tensor with merged patches.
        """
        # Parse target patch size
        if len(factors) == 2:
            X, Y = factors
            T, Z = 1, 1
        elif len(factors) == 3:
            T, X, Y = factors
            Z = 1
        elif len(factors) == 4:
            T, X, Y, Z = factors
        else:
            raise ValueError(f"Invalid target patch size: {factors}")

        patch_product = T * X * Y * Z
        N, C = self.feats.shape
        new_C = C * patch_product

        if channel_average > 1:
            assert new_C % channel_average == 0, (
                f"Feature dimension {new_C} must be divisible by channel_average {channel_average}"
            )
            final_C = new_C // channel_average
        else:
            final_C = new_C

        # Calculate new coordinates by integer division
        new_coords = self.coords.clone()

        if T > 1:
            new_coords[:, 1] = new_coords[:, 1] // T
        new_coords[:, 2] = new_coords[:, 2] // X
        new_coords[:, 3] = new_coords[:, 3] // Y
        if Z > 1:
            new_coords[:, 4] = new_coords[:, 4] // Z

        # Calculate position within the patch (local offsets)
        local_t = self.coords[:, 1] % T if T > 1 else torch.zeros_like(self.coords[:, 1])
        local_x = self.coords[:, 2] % X
        local_y = self.coords[:, 3] % Y
        local_z = self.coords[:, 4] % Z if Z > 1 else torch.zeros_like(self.coords[:, 4])

        flat_offset = (local_t * X * Y * Z) + (local_x * Y * Z) + (local_y * Z) + local_z

        coord_str = torch.cat([new_coords[:, 0:1], new_coords[:, 1:5]], dim=1)
        unique_coords, inverse_indices, counts = torch.unique(coord_str, dim=0, return_inverse=True, return_counts=True)

        batch_N = unique_coords.shape[0]

        if channel_average > 1:
            merged_feats_intermediate = torch.zeros((batch_N, new_C), dtype=self.dtype, device=self.device)
        else:
            merged_feats = torch.zeros((batch_N, new_C), dtype=self.dtype, device=self.device)

        merged_coords = unique_coords.to(dtype=self.coords.dtype)

        batch_idx = inverse_indices.unsqueeze(1)
        feature_indices = flat_offset.unsqueeze(1) * C + torch.arange(C, device=self.device).unsqueeze(0)

        input_feats_flat = self.feats.reshape(-1)
        batch_indices_flat = batch_idx.repeat(1, C).reshape(-1)
        feature_indices_flat = feature_indices.reshape(-1)

        if channel_average > 1:
            merged_feats_intermediate[batch_indices_flat, feature_indices_flat] = input_feats_flat
            merged_feats_reshaped = merged_feats_intermediate.reshape(batch_N, final_C, channel_average)
            merged_feats = torch.mean(merged_feats_reshaped, dim=2)
        else:
            merged_feats[batch_indices_flat, feature_indices_flat] = input_feats_flat

        batch_indices_np = merged_coords[:, 0].cpu().numpy()
        batch_size = int(torch.max(merged_coords[:, 0]).item()) + 1 if batch_N > 0 else 0

        layout_ends = np.zeros(batch_size + 1, dtype=np.int64)
        np.add.at(layout_ends[1:], batch_indices_np, 1)
        layout_ends = np.cumsum(layout_ends)

        new_layout = [slice(layout_ends[i], layout_ends[i + 1]) for i in range(batch_size)]

        new_shape = torch.Size([self.shape[0], final_C])

        result = SparseTensor(feats=merged_feats, coords=merged_coords, shape=new_shape, layout=new_layout)
        result._scale = self._scale
        result._spatial_cache = self._spatial_cache

        return result

    def slice_temporal_range(
        self,
        start_frame: int,
        end_frame: int,
        adjust_temporal: bool = True,
        offset: int = 0,
    ) -> "SparseTensor":
        """Slice the sparse tensor to keep only points within a temporal range.

        Args:
            start_frame: Starting frame index (inclusive).
            end_frame: Ending frame index (exclusive).
            adjust_temporal: If True, adjust temporal coordinates to start from 0.
            offset: Offset to apply when adjusting temporal coordinates.

        Returns:
            SparseTensor: A new sparse tensor containing only points in the temporal range.
        """
        if start_frame >= end_frame:
            empty_coords = torch.empty(
                (0, self.coords.shape[1]),
                dtype=self.coords.dtype,
                device=self.coords.device,
            )
            empty_feats = torch.empty(
                (0,) + self.feats.shape[1:],
                dtype=self.feats.dtype,
                device=self.feats.device,
            )
            result = SparseTensor(feats=empty_feats, coords=empty_coords)
            if adjust_temporal:
                result.register_spatial_cache("temporal_offset", start_frame - offset)
            return result

        if not self.has_special_tokens():
            start_value = torch.tensor(start_frame, dtype=self.coords.dtype, device=self.coords.device)
            end_value = torch.tensor(end_frame, dtype=self.coords.dtype, device=self.coords.device)
            batch_ranges: list[tuple[int, int]] = []

            for batch_slice in self.layout:
                if batch_slice.start >= batch_slice.stop:
                    batch_ranges.append((0, 0))
                    continue

                batch_times = self.coords[batch_slice, 1].contiguous()
                _assert_temporal_coords_sorted(batch_times)
                relative_start = int(torch.searchsorted(batch_times, start_value, right=False).item())
                relative_end = int(torch.searchsorted(batch_times, end_value, right=False).item())
                batch_ranges.append((relative_start, relative_end))

            temporal_offset = start_frame - offset if adjust_temporal else None
            result = self._select_contiguous_batch_ranges(
                batch_ranges,
                temporal_shift=temporal_offset,
                has_special_tokens=False,
            )
            if adjust_temporal:
                result.register_spatial_cache("temporal_offset", start_frame - offset)
            return result

        mask = (self.coords[:, 1] >= start_frame) & (self.coords[:, 1] < end_frame)
        temporal_offset = start_frame - offset if adjust_temporal else None

        if not mask.any():
            empty_coords = torch.empty(
                (0, self.coords.shape[1]),
                dtype=self.coords.dtype,
                device=self.coords.device,
            )
            empty_feats = torch.empty(
                (0,) + self.feats.shape[1:],
                dtype=self.feats.dtype,
                device=self.feats.device,
            )
            result = SparseTensor(feats=empty_feats, coords=empty_coords)
        else:
            new_coords = self.coords[mask].clone()
            new_feats = self.feats[mask]
            if temporal_offset is not None:
                new_coords[:, 1] -= temporal_offset
            result = SparseTensor(feats=new_feats, coords=new_coords)

        if adjust_temporal:
            result.register_spatial_cache("temporal_offset", start_frame - offset)

        return result

    def get_temporal_range(self) -> tuple[int | None, int | None]:
        """Get the temporal range of the sparse tensor.

        Returns:
            Tuple of (min_frame, max_frame) or (None, None) if empty.
        """
        if self.coords.shape[0] == 0:
            return None, None

        temporal_coords = self.coords[:, 1]
        return temporal_coords.min().item(), temporal_coords.max().item()

    def get_last_k_timesteps(self, k: int) -> "SparseTensor":
        """Return a sparse tensor containing only the last k consecutive time steps.

        Uses global max time step across all batches as reference.
        Preserves batch structure including empty batches for compatibility with concat operations.

        Args:
            k: Number of last time steps to keep.

        Returns:
            SparseTensor with only the last k time steps per batch.
            Batches without the required time steps will be empty but preserved.
        """
        if k <= 0 or self.coords.shape[0] == 0:
            # Return empty sparse tensor with same batch structure
            empty_coords = torch.empty(
                (0, self.coords.shape[1]),
                dtype=self.coords.dtype,
                device=self.coords.device,
            )
            empty_feats = torch.empty(
                (0,) + self.feats.shape[1:],
                dtype=self.feats.dtype,
                device=self.feats.device,
            )
            # Preserve the batch size in shape even if empty
            empty_shape = torch.Size([self.shape[0]] + list(self.feats.shape[1:]))
            # Create empty layout for all batches
            empty_layout = [slice(0, 0) for _ in range(self.shape[0])]
            return SparseTensor(
                feats=empty_feats,
                coords=empty_coords,
                shape=empty_shape,
                layout=empty_layout,
            )

        # Get global max time step
        global_max_time = self.coords[:, 1].max()

        # Calculate time threshold for last k steps
        time_threshold = global_max_time - k + 1

        if not self.has_special_tokens():
            batch_ranges: list[tuple[int, int]] = []
            for batch_slice in self.layout:
                if batch_slice.start >= batch_slice.stop:
                    batch_ranges.append((0, 0))
                    continue

                batch_times = self.coords[batch_slice, 1].contiguous()
                _assert_temporal_coords_sorted(batch_times)
                relative_start = int(torch.searchsorted(batch_times, time_threshold, right=False).item())
                batch_ranges.append((relative_start, batch_slice.stop - batch_slice.start))

            return self._select_contiguous_batch_ranges(
                batch_ranges,
                temporal_shift=time_threshold,
                has_special_tokens=False,
            )

        # Create mask for points in the last k time steps
        valid_mask = self.coords[:, 1] >= time_threshold

        # Extract selected coordinates and features and shift time dimension to start at 0.
        selected_coords = self.coords[valid_mask].clone()
        selected_coords[:, 1] -= time_threshold.to(dtype=selected_coords.dtype)
        selected_feats = self.feats[valid_mask]

        # Preserve the original batch size in the shape
        new_shape = torch.Size([self.shape[0]] + list(self.feats.shape[1:]))
        new_layout = self._build_layout_from_batch_indices(selected_coords[:, 0], self.shape[0])

        return SparseTensor(
            feats=selected_feats,
            coords=selected_coords,
            shape=new_shape,
            layout=new_layout,
            has_special_tokens=False if not self.has_special_tokens() else None,
        )

    def concat_temporal_at_batch_start(self, other: "SparseTensor") -> "SparseTensor":
        """Concatenate another sparse tensor at the beginning of each batch.

        This is used for KV caching - injecting cached data at the start of each batch.
        Handles all edge cases correctly including empty batches.

        Args:
            other: The sparse tensor to inject at the beginning of each batch.

        Returns:
            New sparse tensor with 'other' injected at the start of each batch.
        """
        # Handle empty tensor cases
        if other.coords.shape[0] == 0:
            return self
        if self.coords.shape[0] == 0:
            return other

        # Check feature dimensions compatibility
        assert self.feats.shape[1:] == other.feats.shape[1:], (
            f"Feature shape mismatch: {self.feats.shape[1:]} vs {other.feats.shape[1:]}"
        )

        current_batch_size = self.shape[0]

        if other.shape[0] == 0:
            return self

        if not self.has_special_tokens() and not other.has_special_tokens():
            combined_coords_parts = []
            combined_feats_parts = []
            new_layout: list[slice] = []
            offset = 0
            has_cached_tokens = False

            for batch_idx in range(current_batch_size):
                batch_len = 0

                if batch_idx < other.shape[0]:
                    other_slice = other.layout[batch_idx]
                    if other_slice.start < other_slice.stop:
                        has_cached_tokens = True
                        combined_coords_parts.append(other.coords[other_slice])
                        combined_feats_parts.append(other.feats[other_slice])
                        batch_len += other_slice.stop - other_slice.start

                self_slice = self.layout[batch_idx]
                if self_slice.start < self_slice.stop:
                    combined_coords_parts.append(self.coords[self_slice])
                    combined_feats_parts.append(self.feats[self_slice])
                    batch_len += self_slice.stop - self_slice.start

                new_layout.append(slice(offset, offset + batch_len))
                offset += batch_len

            if not has_cached_tokens:
                return self

            combined_coords = torch.cat(combined_coords_parts, dim=0)
            combined_feats = torch.cat(combined_feats_parts, dim=0)
            new_shape = torch.Size([current_batch_size] + list(self.feats.shape[1:]))
            result = SparseTensor(
                feats=combined_feats,
                coords=combined_coords,
                shape=new_shape,
                layout=new_layout,
                has_special_tokens=False,
            )
            result._scale = self._scale
            return result

        combined_coords_parts = []
        combined_feats_parts = []
        new_layout: list[slice] = []
        offset = 0
        has_cached_tokens = False

        # Preserve the SparseTensor invariant that special tokens, when present,
        # stay ahead of spatial tokens within each batch segment.
        for batch_idx in range(current_batch_size):
            special_coords_parts = []
            special_feats_parts = []
            normal_coords_parts = []
            normal_feats_parts = []

            if batch_idx < other.shape[0]:
                other_slice = other.layout[batch_idx]
                if other_slice.start < other_slice.stop:
                    has_cached_tokens = True
                    other_batch_coords = other.coords[other_slice]
                    other_batch_feats = other.feats[other_slice]
                    other_special_mask = (other_batch_coords[:, 1:] == -1).all(dim=1)
                    if other_special_mask.any():
                        special_coords_parts.append(other_batch_coords[other_special_mask])
                        special_feats_parts.append(other_batch_feats[other_special_mask])
                    if (~other_special_mask).any():
                        normal_coords_parts.append(other_batch_coords[~other_special_mask])
                        normal_feats_parts.append(other_batch_feats[~other_special_mask])

            self_slice = self.layout[batch_idx]
            if self_slice.start < self_slice.stop:
                self_batch_coords = self.coords[self_slice]
                self_batch_feats = self.feats[self_slice]
                self_special_mask = (self_batch_coords[:, 1:] == -1).all(dim=1)
                if self_special_mask.any():
                    special_coords_parts.append(self_batch_coords[self_special_mask])
                    special_feats_parts.append(self_batch_feats[self_special_mask])
                if (~self_special_mask).any():
                    normal_coords_parts.append(self_batch_coords[~self_special_mask])
                    normal_feats_parts.append(self_batch_feats[~self_special_mask])

            batch_coords_parts = [*special_coords_parts, *normal_coords_parts]
            batch_feats_parts = [*special_feats_parts, *normal_feats_parts]

            batch_len = 0
            for coords_part, feats_part in zip(batch_coords_parts, batch_feats_parts):
                batch_len += coords_part.shape[0]
                combined_coords_parts.append(coords_part)
                combined_feats_parts.append(feats_part)

            new_layout.append(slice(offset, offset + batch_len))
            offset += batch_len

        if not has_cached_tokens:
            return self

        combined_coords = torch.cat(combined_coords_parts, dim=0)
        combined_feats = torch.cat(combined_feats_parts, dim=0)
        new_shape = torch.Size([current_batch_size] + list(self.feats.shape[1:]))
        result = SparseTensor(
            feats=combined_feats,
            coords=combined_coords,
            shape=new_shape,
            layout=new_layout,
            has_special_tokens=True,
        )
        result._scale = self._scale
        return result

    def split_by_temporal_batches(
        self,
        frame_batch_size: int,
        frame_batch_strides: int | None = None,
        adjust_temporal: bool = True,
        offset: int = 0,
    ) -> list["SparseTensor"]:
        """Split the sparse tensor into temporal batches with sliding window support.

        Args:
            frame_batch_size: Number of frames per batch (window size).
            frame_batch_strides: Stride between consecutive windows (step size).
                If None, defaults to frame_batch_size (non-overlapping).
            adjust_temporal: Whether to adjust temporal coordinates to start from 0.
            offset: Offset to apply when adjusting temporal coordinates.

        Returns:
            List of temporal slices with sliding window overlap.

        Example:
            With frame_batch_size=16, frame_batch_strides=12:
            - Slice 0: frames [0, 16)   -> temporal range 0-15
            - Slice 1: frames [12, 28)  -> temporal range 12-27
            - Slice 2: frames [24, 40)  -> temporal range 24-39
        """
        min_frame, max_frame = self.get_temporal_range()

        # Handle empty tensor
        if min_frame is None or max_frame is None:
            return []

        # Validate parameters
        if frame_batch_size <= 0:
            raise ValueError("frame_batch_size must be positive")

        if frame_batch_strides is None:
            frame_batch_strides = frame_batch_size

        if frame_batch_strides <= 0:
            raise ValueError("frame_batch_strides must be positive")

        slices = []
        current_start = min_frame

        # Continue creating slices while there's still data to process
        while current_start <= max_frame:
            # Calculate end frame for current slice
            current_end = current_start + frame_batch_size

            # No offset for the first slice
            update_offset = 0 if current_start == min_frame else offset

            slice_tensor = self.slice_temporal_range(
                current_start,
                current_end,
                adjust_temporal=adjust_temporal,
                offset=update_offset,
            )

            slices.append(slice_tensor)

            # Move to next window start position
            current_start += frame_batch_strides

            # Stop if we've moved beyond the available data
            if current_start > max_frame:
                break

        return slices


def sparse_batch_broadcast(input: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    """Broadcast a tensor to a sparse tensor along the batch dimension.

    Args:
        input: SparseTensor to broadcast to.
        other: Tensor to broadcast.

    Returns:
        Broadcasted tensor matching the sparse tensor layout.
    """
    coords, feats = input.coords, input.feats
    broadcasted = torch.zeros_like(feats)
    for k in range(input.shape[0]):
        broadcasted[input.layout[k]] = other[k]
    return broadcasted


def sparse_batch_op(input: SparseTensor, other: torch.Tensor, op: callable = torch.add) -> SparseTensor:
    """Broadcast a tensor and perform an operation with a sparse tensor.

    Args:
        input: SparseTensor to operate on.
        other: Tensor to broadcast.
        op: Operation to perform (default: torch.add).

    Returns:
        SparseTensor with the operation applied.
    """
    return input.replace(op(input.feats, sparse_batch_broadcast(input, other)))


def sparse_cat(inputs: list[SparseTensor], dim: int = 0) -> SparseTensor:
    """Concatenate a list of sparse tensors.

    Args:
        inputs: List of sparse tensors to concatenate.
        dim: Dimension to concatenate along (0 for batch, >0 for features).

    Returns:
        Concatenated SparseTensor.
    """
    if dim == 0:
        start = 0
        coords = []
        for input in inputs:
            coords.append(input.coords.clone())
            coords[-1][:, 0] += start
            start += input.shape[0]
        coords = torch.cat(coords, dim=0)
        feats = torch.cat([input.feats for input in inputs], dim=0)
        output = SparseTensor(
            coords=coords,
            feats=feats,
        )
    else:
        feats = torch.cat([input.feats for input in inputs], dim=dim)
        output = inputs[0].replace(feats)

    return output


def sparse_unbind(input: SparseTensor, dim: int) -> list[SparseTensor]:
    """Unbind a sparse tensor along a dimension.

    Args:
        input: SparseTensor to unbind.
        dim: Dimension to unbind.

    Returns:
        List of SparseTensors.
    """
    if dim == 0:
        return [input[i] for i in range(input.shape[0])]
    else:
        feats = input.feats.unbind(dim)
        return [input.replace(f) for f in feats]


def reconstruct_from_temporal_slices(
    slices: list[SparseTensor],
    target_coords: torch.Tensor | None = None,
    use_cached_offsets: bool = True,
) -> SparseTensor:
    """Reconstruct a sparse tensor from temporal slices.

    Args:
        slices: List of temporal slices.
        target_coords: Target coordinates to match exactly.
        use_cached_offsets: Whether to use cached temporal offsets.

    Returns:
        Reconstructed SparseTensor with original temporal coordinates.
    """
    if len(slices) == 0:
        raise ValueError("Cannot reconstruct from empty slice list")

    non_empty_slices = [s for s in slices if s.coords.shape[0] > 0]

    if len(non_empty_slices) == 0:
        if target_coords is not None:
            empty_feats = torch.empty(
                (0,) + slices[0].feats.shape[1:],
                dtype=slices[0].feats.dtype,
                device=slices[0].feats.device,
            )
            return SparseTensor(feats=empty_feats, coords=target_coords[:0])
        else:
            first_slice = slices[0]
            empty_coords = torch.empty(
                (0, first_slice.coords.shape[1]),
                dtype=first_slice.coords.dtype,
                device=first_slice.coords.device,
            )
            empty_feats = torch.empty(
                (0,) + first_slice.feats.shape[1:],
                dtype=first_slice.feats.dtype,
                device=first_slice.feats.device,
            )
            return SparseTensor(feats=empty_feats, coords=empty_coords)

    restored_slices = []
    has_special_tokens = False
    for slice_tensor in non_empty_slices:
        if slice_tensor.has_special_tokens():
            has_special_tokens = True
        if use_cached_offsets:
            offset = slice_tensor.get_spatial_cache("temporal_offset")
            if offset is not None:
                new_coords = slice_tensor.coords.clone()
                new_coords[:, 1] += offset
                restored_slice = SparseTensor(
                    feats=slice_tensor.feats,
                    coords=new_coords,
                    shape=slice_tensor.shape,
                    layout=slice_tensor.layout,
                    scale=slice_tensor._scale,
                    has_special_tokens=slice_tensor._has_special_tokens,
                )
            else:
                restored_slice = slice_tensor
        else:
            restored_slice = slice_tensor

        restored_slices.append(restored_slice)

    all_coords = []
    all_feats = []

    for restored_slice in restored_slices:
        all_coords.append(restored_slice.coords)
        all_feats.append(restored_slice.feats)

    combined_coords = torch.cat(all_coords, dim=0)
    combined_feats = torch.cat(all_feats, dim=0)

    if target_coords is not None:
        fast_path = _reconstruct_in_target_batch_order(
            restored_slices,
            target_coords,
            has_special_tokens=has_special_tokens,
        )
        if fast_path is not None:
            return fast_path
        return _match_target_coordinates(
            combined_coords,
            combined_feats,
            target_coords,
            has_special_tokens=has_special_tokens,
        )
    else:
        sort_key = (combined_coords[:, 0].long() << 20) + combined_coords[:, 1].long()
        sorted_indices = torch.argsort(sort_key)

        final_coords = combined_coords[sorted_indices]
        final_feats = combined_feats[sorted_indices]

        return SparseTensor(feats=final_feats, coords=final_coords, has_special_tokens=has_special_tokens)


def _match_target_coordinates(
    available_coords: torch.Tensor,
    available_feats: torch.Tensor,
    target_coords: torch.Tensor,
    has_special_tokens: bool | None = None,
) -> SparseTensor:
    """Match available coordinates and features to target coordinates exactly."""
    if len(available_coords) == 0:
        device = target_coords.device
        feat_dim = available_feats.shape[1:] if len(available_feats) > 0 else (1,)
        zero_feats = torch.zeros(
            (len(target_coords),) + feat_dim,
            dtype=available_feats.dtype if len(available_feats) > 0 else torch.float32,
            device=device,
        )
        return SparseTensor(feats=zero_feats, coords=target_coords, has_special_tokens=has_special_tokens)

    available_hash = _compute_coord_hash(available_coords)  # [N_available]
    target_hash = _compute_coord_hash(target_coords)  # [N_target]

    # Vectorized hash lookup using sort + searchsorted (no CPU-GPU sync)
    # This replaces the Python dict + loop which caused O(n) .item() calls
    sorted_available_hash, sort_indices = torch.sort(available_hash)  # [N_available], [N_available]
    search_positions = torch.searchsorted(sorted_available_hash, target_hash)  # [N_target]
    valid_positions = search_positions < sorted_available_hash.numel()  # [N_target]
    safe_positions = search_positions.clamp(max=sorted_available_hash.numel() - 1)  # [N_target]
    reorder_indices = sort_indices[safe_positions]  # [N_target]
    matched_coords = available_coords[reorder_indices]  # [N_target,5]
    coord_matches = valid_positions & (matched_coords == target_coords).all(dim=1)  # [N_target]
    # This sync only runs on the target-order fallback path, not the fast
    # non-overlapping reconstruction path used by uniform temporal slices.
    if not bool(coord_matches.all().item()):
        mismatch_count = int((~coord_matches).sum().item())
        raise ValueError(
            "Coordinate hash lookup failed for "
            f"{mismatch_count}/{target_coords.shape[0]} target coords. "
            "This usually means target coords are missing or two coords collided in the 12-bit-per-axis hash."
        )

    reordered_feats = available_feats[reorder_indices]  # [N_target,*F]

    return SparseTensor(feats=reordered_feats, coords=target_coords, has_special_tokens=has_special_tokens)


def _reconstruct_in_target_batch_order(
    restored_slices: list[SparseTensor],
    target_coords: torch.Tensor,
    has_special_tokens: bool,
) -> SparseTensor | None:
    """Fast reconstruction path for non-overlapping temporal slices."""
    if has_special_tokens or len(restored_slices) == 0:
        return None

    batch_size = restored_slices[0].shape[0]
    scale = restored_slices[0]._scale
    if any(slice_tensor.shape[0] != batch_size or slice_tensor._scale != scale for slice_tensor in restored_slices):
        return None

    combined_coords_parts = []
    combined_feats_parts = []
    new_layout: list[slice] = []
    offset = 0

    for batch_idx in range(batch_size):
        batch_len = 0
        for slice_tensor in restored_slices:
            batch_slice = slice_tensor.layout[batch_idx]
            if batch_slice.start >= batch_slice.stop:
                continue
            combined_coords_parts.append(slice_tensor.coords[batch_slice])
            combined_feats_parts.append(slice_tensor.feats[batch_slice])
            batch_len += batch_slice.stop - batch_slice.start

        new_layout.append(slice(offset, offset + batch_len))
        offset += batch_len

    if not combined_coords_parts:
        return SparseTensor(
            feats=torch.empty(
                (0,) + restored_slices[0].feats.shape[1:],
                dtype=restored_slices[0].feats.dtype,
                device=restored_slices[0].feats.device,
            ),
            coords=target_coords[:0],
            shape=torch.Size([batch_size] + list(restored_slices[0].feats.shape[1:])),
            layout=new_layout,
            scale=scale,
            has_special_tokens=False,
        )

    combined_coords = torch.cat(combined_coords_parts, dim=0)
    if combined_coords.shape != target_coords.shape or not torch.equal(combined_coords, target_coords):
        return None

    return SparseTensor(
        feats=torch.cat(combined_feats_parts, dim=0),
        coords=target_coords,
        shape=torch.Size([batch_size] + list(restored_slices[0].feats.shape[1:])),
        layout=new_layout,
        scale=scale,
        has_special_tokens=False,
    )


def _compute_coord_hash(coords: torch.Tensor) -> torch.Tensor:
    """Compute compact coordinate keys for hash lookup.

    The layout uses five 12-bit coordinate fields in int64, so the lookup is
    collision-free only for non-negative coordinates below 4096 on every axis.
    `_match_target_coordinates` validates the exact coord selected by each key
    before returning, which turns out-of-domain collisions into loud failures.
    """
    if coords.ndim != 2 or coords.shape[1] != 5:
        raise ValueError(f"coords must have shape [N,5], got {tuple(coords.shape)}")

    hash_vals = coords[:, 0].long()  # [N]
    hash_vals = (hash_vals << 12) + coords[:, 1].long()  # [N]
    hash_vals = (hash_vals << 12) + coords[:, 2].long()  # [N]
    hash_vals = (hash_vals << 12) + coords[:, 3].long()  # [N]
    hash_vals = (hash_vals << 12) + coords[:, 4].long()  # [N]

    return hash_vals
