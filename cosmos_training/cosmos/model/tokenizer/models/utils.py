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

"""Generic utilities for Cosmos3 tokenizer.

This module provides general-purpose utilities:
    - Image/video processing: sparse_to_img_list, crop_tensors_to_match, resize_and_crop
    - Logging: SampleLogger
    - Tensor operations: average_with_scatter_add
    - Temporal utilities: split_temporal_dimension, restore_original_shape,
      reconstruct_from_temporal_slices

Note: Metrics like calculate_psnr have been moved to projects.cosmos3.tokenizer.evaluation.reconstruction_metrics
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision.utils import make_grid

from cosmos.model.tokenizer.models.modules import SparseTensor
from cosmos.model.tokenizer.models.modules.sparse_tensor import (
    _compute_coord_hash as _compute_coord_hash,
)
from cosmos.model.tokenizer.models.modules.sparse_tensor import (
    _match_target_coordinates as _match_target_coordinates,
)
from cosmos.model.tokenizer.models.modules.sparse_tensor import (
    _reconstruct_in_target_batch_order as _reconstruct_in_target_batch_order,
)
from cosmos.model.tokenizer.models.modules.sparse_tensor import (
    reconstruct_from_temporal_slices as reconstruct_from_temporal_slices,
)

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

EPS = 1e-6


# =============================================================================
# Tensor Operations
# =============================================================================


def average_with_scatter_add(features: torch.Tensor, batch_mapping: torch.Tensor) -> torch.Tensor:
    """Average features per batch using scatter_add.

    Args:
        features: Feature tensor of shape (N, feature_dim).
        batch_mapping: Tensor mapping each feature to its batch index.

    Returns:
        Averaged features of shape (num_batches, feature_dim).
    """
    device = features.device
    feature_dim = features.shape[1]
    num_original_batches = batch_mapping.max().item() + 1

    averaged_feats = torch.zeros((num_original_batches, feature_dim), dtype=features.dtype, device=device)
    batch_mapping = batch_mapping.long()

    counts = torch.zeros(num_original_batches, dtype=torch.float32, device=device)
    counts.scatter_add_(0, batch_mapping, torch.ones_like(batch_mapping, dtype=torch.float32))

    averaged_feats.scatter_add_(0, batch_mapping.unsqueeze(1).expand(-1, feature_dim), features)

    counts = counts.clamp(min=1)
    averaged_feats = averaged_feats / counts.unsqueeze(1)

    return averaged_feats


# =============================================================================
# Temporal Utilities
# =============================================================================


def split_temporal_dimension(x: torch.Tensor, t_fix: int) -> tuple[torch.Tensor, int, int, int, int, int, int, int]:
    """Split a 5D tensor along temporal dimension for batch processing.

    Args:
        x: Input tensor of shape (B, C, T, H, W).
        t_fix: Fixed temporal dimension to split into.

    Returns:
        Tuple of (x_flat, b, t_i, t_fix, h, w, c, padding).
    """
    b, c, t, h, w = x.shape
    padding = 0
    if t % t_fix != 0:
        padding = t_fix - t % t_fix
    if padding > 0:
        pad = (0, padding, 0, 0, 0, 0)
        x = rearrange(x, "b c t h w -> (b c) h w t")
        x = torch.nn.functional.pad(x, pad, mode="reflect")
        x = rearrange(x, "(b c) h w t -> b c t h w", b=b)
    t_padded = t + padding
    t_i = t_padded // t_fix
    x_flat = rearrange(
        x,
        "b c (t_i t_fix) h w -> (b t_i) c t_fix h w",
        t_i=t_i,
        t_fix=t_fix,
    )
    return x_flat, b, t_i, t_fix, h, w, c, padding


def restore_original_shape(
    x_processed: torch.Tensor,
    b: int,
    t_i: int,
    t_fix: int,
    h: int,
    w: int,
    c: int,
    padding: int,
) -> torch.Tensor:
    """Restore original shape after temporal split processing.

    Args:
        x_processed: Processed tensor.
        b: Original batch size.
        t_i: Number of temporal splits.
        t_fix: Fixed temporal dimension.
        h: Height.
        w: Width.
        c: Channels.
        padding: Amount of temporal padding applied.

    Returns:
        Tensor restored to original shape.
    """
    x_original_padded = rearrange(
        x_processed,
        "(b t_i) t_fix c h w -> b c (t_i t_fix) h w",
        t_i=t_i,
        t_fix=t_fix,
        b=b,
        h=h,
        w=w,
        c=c,
    )
    if padding > 0:
        x_original = x_original_padded[:, :, :-padding, :, :]
    else:
        x_original = x_original_padded
    return x_original


# =============================================================================
# Image/Video Processing
# =============================================================================


@lru_cache(maxsize=128)
def _get_cached_sparse_coords(
    batch: int,
    num_temporal_patches: int,
    num_height_patches: int,
    num_width_patches: int,
    feature_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Size, tuple[slice, ...]]:
    """Build and cache canonical sparse metadata for fixed input shapes.

    Coordinate templates are cached on CPU so the LRU does not pin GPU memory
    across many distinct shapes or devices.
    """
    coords = (
        torch.stack(
            torch.meshgrid(
                torch.arange(batch),
                torch.arange(num_temporal_patches),
                torch.arange(num_height_patches),
                torch.arange(num_width_patches),
                torch.arange(1),
                indexing="ij",
            ),
            dim=-1,
        )
        .reshape(-1, 5)
        .int()
    )
    tokens_per_batch = num_temporal_patches * num_height_patches * num_width_patches
    layout = tuple(
        slice(batch_idx * tokens_per_batch, (batch_idx + 1) * tokens_per_batch) for batch_idx in range(batch)
    )
    shape = torch.Size((batch, *feature_shape))
    return coords, shape, layout


def batch_tensor_to_sparse(batch_tensor: torch.Tensor, patch_size: tuple[int, int, int]) -> SparseTensor:
    """Convert a batch tensor to SparseTensor.

    Args:
        batch_tensor: Input tensor of shape [B, C, H, W] or [B, T, H, W, C].
        patch_size: Patch size (Pt, Ph, Pw).

    Returns:
        SparseTensor with patches as features.
    """
    if batch_tensor.dim() == 4:
        if batch_tensor.shape[1] == 3:
            batch_tensor = rearrange(batch_tensor, "b c h w -> b h w c")
        batch_tensor = batch_tensor.unsqueeze(1)
    batch, T, H, W, _ = batch_tensor.shape
    Pt, Ph, Pw = patch_size
    assert T % Pt == 0 and H % Ph == 0 and W % Pw == 0
    num_temporal_patches = T // Pt
    num_height_patches = H // Ph
    num_width_patches = W // Pw
    # reshape to (T//Pt, H//Ph, W//Pw, Pt*Ph*Pw*C)
    feats = rearrange(
        batch_tensor,
        "b (nt pt) (nh ph) (nw pw) c -> (b nt nh nw) (pt ph pw c)",
        nt=num_temporal_patches,
        nh=num_height_patches,
        nw=num_width_patches,
        pt=Pt,
        ph=Ph,
        pw=Pw,
    )
    coords, shape, layout = _get_cached_sparse_coords(
        batch,
        num_temporal_patches,
        num_height_patches,
        num_width_patches,
        tuple(feats.shape[1:]),
    )
    if coords.device != feats.device:
        coords = coords.to(device=feats.device)
    return SparseTensor(
        feats=feats,
        coords=coords,
        shape=shape,
        layout=list(layout),
        has_special_tokens=False,
    )


def _infer_uniform_batch_grid_shape(sparse_tensor: SparseTensor) -> tuple[int, int, int, int] | None:
    """Infer a uniform (T, H, W, Z) patch grid for all batches when possible."""
    if sparse_tensor.shape[0] == 0:
        return 0, 0, 0, 0

    if sparse_tensor.coords.shape[0] == 0 or sparse_tensor.coords.shape[1] != 5:
        return None

    if hasattr(sparse_tensor, "get_batch_seq_lens"):
        seq_lens = sparse_tensor.get_batch_seq_lens()
    else:
        layout = getattr(sparse_tensor, "layout", None)
        if layout is None:
            return None

        seq_lens = []
        for item in layout:
            if isinstance(item, slice):
                start = 0 if item.start is None else int(item.start)
                stop = start if item.stop is None else int(item.stop)
                seq_lens.append(max(0, stop - start))
            else:
                seq_lens.append(int(len(item)))

    if not seq_lens or any(seq_len == 0 for seq_len in seq_lens):
        return None
    if len(set(seq_lens)) != 1:
        return None

    special_mask = (sparse_tensor.coords[:, 1:] == -1).all(dim=1)
    if bool(special_mask.any()):
        return None

    batch_size = sparse_tensor.shape[0]
    batch_indices = sparse_tensor.coords[:, 0].long()
    spatial_coords = sparse_tensor.coords[:, 1:].long()
    device = getattr(sparse_tensor, "device", sparse_tensor.feats.device)

    max_per_axis = []
    for axis in range(spatial_coords.shape[1]):
        axis_max = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        axis_max = axis_max.scatter_reduce(
            0,
            batch_indices,
            spatial_coords[:, axis],
            reduce="amax",
            include_self=True,
        )
        max_per_axis.append(axis_max)

    temporal_min = torch.full((batch_size,), torch.iinfo(torch.long).max, dtype=torch.long, device=device)
    temporal_min = temporal_min.scatter_reduce(
        0,
        batch_indices,
        spatial_coords[:, 0],
        reduce="amin",
        include_self=True,
    )

    grid_shape = torch.stack(max_per_axis, dim=1) + 1
    image_like_batches = temporal_min == max_per_axis[0]
    if bool(image_like_batches.all()):
        grid_shape[:, 0] = 1

    if not torch.equal(grid_shape, grid_shape[0].unsqueeze(0).expand_as(grid_shape)):
        return None

    t, h, w, z = (int(dim.item()) for dim in grid_shape[0])
    if seq_lens[0] != t * h * w * z:
        return None

    return t, h, w, z


def sparse_to_batched_tensor(
    sparse_tensor: SparseTensor,
    patch_size: tuple[int, int, int],
    var_patch_axis: int = 0,
    task_type: str = "image",
    channels: int = 3,
) -> torch.Tensor | None:
    """Convert a uniform SparseTensor batch to a dense `[B, T, C, H, W]` tensor."""
    if re.search(r"3D", task_type):
        return None

    Pt, Ph, Pw = patch_size
    full_patch_size = [Pt, Ph, Pw, channels]
    variable_factor = np.prod(full_patch_size) // sparse_tensor.feats.shape[1]
    select_size = full_patch_size[var_patch_axis] // variable_factor
    if var_patch_axis == 0:
        Pt = select_size
    else:
        raise ValueError("variable_axis value wrong")

    grid_shape = _infer_uniform_batch_grid_shape(sparse_tensor)
    if grid_shape is None:
        return None

    batch_size = sparse_tensor.shape[0]
    if batch_size == 0:
        return sparse_tensor.feats.new_empty((0, 0, channels, 0, 0))

    T, H, W, Z = grid_shape
    if Z != 1:
        return None

    images = rearrange(
        sparse_tensor.feats,
        "(b t h w) (Pt Ph Pw c) -> b (t Pt) c (h Ph) (w Pw)",
        b=batch_size,
        t=T,
        h=H,
        w=W,
        Pt=Pt,
        Ph=Ph,
        Pw=Pw,
    )

    if "image_rec" in task_type and images.shape[1] > 1:
        images = images[:, 0:1]

    return images


def sparse_to_img_list(
    sparse_tensor: SparseTensor,
    patch_size: tuple[int, int, int],
    var_patch_axis: int = 0,
    task_type: str = "image",
    channels: int = 3,
) -> list[torch.Tensor]:
    """Convert SparseTensor to a list of image tensors.

    Args:
        sparse_tensor: Input SparseTensor to convert.
        patch_size: Tuple of (Pt, Ph, Pw) patch dimensions.
        var_patch_axis: Axis with variable patch size.
        task_type: Type of task (e.g., "image", "video", "3D").
        channels: Number of channels.

    Returns:
        List of image tensors.
    """
    # Check if this is a 3D batch (coords with different z index)
    if re.search(r"3D", task_type):
        return [sparse_tensor]

    Pt, Ph, Pw = patch_size

    # Select patch_size based on input
    full_patch_size = [Pt, Ph, Pw, channels]
    variable_factor = np.prod(full_patch_size) // sparse_tensor.feats.shape[1]
    select_size = full_patch_size[var_patch_axis] // variable_factor
    if var_patch_axis == 0:
        Pt = select_size
    else:
        raise ValueError("variable_axis value wrong")

    batched_tensor = sparse_to_batched_tensor(
        sparse_tensor,
        patch_size,
        var_patch_axis=var_patch_axis,
        task_type=task_type,
        channels=channels,
    )
    if batched_tensor is not None:
        return [image for image in batched_tensor]

    images = []
    for batch_idx in range(sparse_tensor.shape[0]):
        batch_sparse_tensor = sparse_tensor[batch_idx]
        T, H, W, Z = (int(dim) for dim in (batch_sparse_tensor.coords.max(dim=0)[0][1:] + 1).tolist())
        # Avoid Python's built-in all() on tensors; it scalarizes every element.
        if bool((batch_sparse_tensor.coords[:, 1] == (T - 1)).all().item()):
            T = 1
        img = rearrange(
            batch_sparse_tensor.feats,
            "(t h w) (Pt Ph Pw c) -> (t Pt) c (h Ph) (w Pw)",
            t=T,
            h=H,
            w=W,
            Pt=Pt,
            Ph=Ph,
            Pw=Pw,
        )

        if "image_rec" in task_type and img.shape[0] > 1:
            img = img[0:1]

        images.append(img)

    return images


def crop_tensors_to_match(rec_list: list[torch.Tensor], img_list: list[torch.Tensor]) -> list[torch.Tensor]:
    """Crop tensors in rec_list to match shapes in img_list.

    Args:
        rec_list: List of tensors to be cropped.
        img_list: List of reference tensors with target shapes.

    Returns:
        List of cropped tensors.
    """
    if len(rec_list) != len(img_list):
        raise ValueError(f"Lists must have same length. Got {len(rec_list)} and {len(img_list)}")

    cropped_list = []
    for rec_tensor, img_tensor in zip(rec_list, img_list):
        # Get the minimum shape for each dimension
        min_shapes = [min(r, i) for r, i in zip(rec_tensor.shape, img_tensor.shape)]

        # Create slicing for each dimension
        slices = tuple(slice(0, min_shape) for min_shape in min_shapes)

        # Crop the rec_tensor
        cropped_tensor = rec_tensor[slices]
        cropped_list.append(cropped_tensor.clamp(-1 + EPS, 1 - EPS))

    return cropped_list


def resize_and_crop(
    imgs: list[torch.Tensor],
    target_length: int | tuple[int, int] = 256,
    crop: Literal["center", "given", "random"] = "center",
    crop_pos: tuple[tuple[int, int], ...] = ((0, 0),),
) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[tuple[int, int]]]:
    """Resize images based on shortest edge and perform crop.

    Args:
        imgs: List of input tensors, each of shape [C, H, W] or [B, C, H, W].
        target_length: Target length for shortest edge.
        crop: Crop mode ("center", "random", or "given").
        crop_pos: Position to crop when crop="given".

    Returns:
        List of resized and cropped images, or tuple with crop positions for random.
    """
    processed_imgs = []
    all_crop_pos = []

    for idx, img in enumerate(imgs):
        # Add batch dimension if needed
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        # Get current dimensions
        _, _, h, w = img.shape

        # Handle tuple vs int target length
        if isinstance(target_length, tuple):
            target_h, target_w = target_length
        else:
            target_h = target_w = target_length

        # Calculate scale factor
        scale_h = target_h / h
        scale_w = target_w / w
        scale = max(scale_h, scale_w)

        # Calculate new dimensions after scaling
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize image
        img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # For safety, ensure new dimensions are at least as large as target
        if new_h < target_h or new_w < target_w:
            scale_fix = max(target_h / new_h, target_w / new_w)
            new_h = int(new_h * scale_fix)
            new_w = int(new_w * scale_fix)
            img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)

        if crop == "center":
            start_h = (new_h - target_h) // 2
            start_w = (new_w - target_w) // 2
        elif crop == "random":
            h_range = max(1, new_h - target_h + 1)
            w_range = max(1, new_w - target_w + 1)
            start_h = torch.randint(0, h_range, (1,)).item()
            start_w = torch.randint(0, w_range, (1,)).item()
        elif crop == "given":
            if idx < len(crop_pos):
                start_h, start_w = crop_pos[idx]
            else:
                start_h = (new_h - target_h) // 2
                start_w = (new_w - target_w) // 2
        else:
            raise ValueError(f"Invalid crop mode: {crop}")

        # Ensure start positions don't go out of bounds
        start_h = min(start_h, new_h - target_h)
        start_w = min(start_w, new_w - target_w)
        start_h = max(0, start_h)
        start_w = max(0, start_w)

        # Perform crop
        img = img[:, :, start_h : start_h + target_h, start_w : start_w + target_w]
        processed_imgs.append(img)
        all_crop_pos.append((start_h, start_w))

    if crop == "random":
        return processed_imgs, all_crop_pos

    return processed_imgs


# =============================================================================
# Logging
# =============================================================================
# Note: calculate_psnr has been moved to projects.cosmos3.tokenizer.evaluation.reconstruction_metrics
# =============================================================================


class SampleLogger:
    """Logger for visualizing samples during training."""

    def __init__(self, log_every_n: int = 10, max_samples: int = 1):
        """Initialize sample logger.

        Args:
            log_every_n: Log every n samples.
            max_samples: Maximum samples to log.
        """
        self.task_counts: dict[str, int] = {}
        self.log_every_n = log_every_n
        self.max_samples = max_samples

    def should_log(self, task_type: str) -> bool:
        """Check if we should log for this task type.

        Args:
            task_type: Type of task.

        Returns:
            True if should log.
        """
        if task_type not in self.task_counts:
            self.task_counts[task_type] = 0

        should_log = self.task_counts[task_type] % self.log_every_n == 0
        self.task_counts[task_type] += 1
        return should_log

    def log_visuals(
        self,
        original: torch.Tensor | list[torch.Tensor],
        reconstruction: torch.Tensor | list[torch.Tensor],
        trainer: Any,
        tag: str = "train",
    ) -> None:
        """Log original and reconstructed visuals.

        Args:
            original: Original tensor(s).
            reconstruction: Reconstructed tensor(s).
            trainer: Training context with accelerator and logger.
            tag: Tag for logging.
        """
        # Handle lists of tensors
        if isinstance(original, list) and isinstance(reconstruction, list):
            for i in range(min(self.max_samples, len(original))):
                self.log_visuals(original[i][None], reconstruction[i][None], trainer, tag)
            return

        B = min(self.max_samples, original.shape[0])

        if original.shape[1] == 1:
            self.log_image(original[:B], reconstruction[:B], trainer, tag)
        elif original.shape[1] > 1:
            self.log_video(original[:B], reconstruction[:B], trainer, tag)
        else:
            raise ValueError("Invalid shape for logging")

    def log_video(
        self,
        original: torch.Tensor,
        reconstruction: torch.Tensor | None,
        trainer: Any,
        tag: str = "train",
        fps: int = 24,
    ) -> None:
        """Log original and reconstructed videos side by side.

        Args:
            original: Original video tensor.
            reconstruction: Reconstructed video tensor.
            trainer: Training context.
            tag: Tag for logging.
            fps: Frames per second.
        """
        if not trainer.accelerator.is_main_process:
            return

        if reconstruction is None:
            combined = original
        else:
            combined = torch.cat([original, reconstruction], dim=-1)

        if combined.ndim == 4:
            self._tracker_log_video(f"{tag}_vis/video_0", combined, fps, trainer)
        else:
            for i in range(combined.shape[0]):
                self._tracker_log_video(f"{tag}_vis/video_{i}", combined[i][None], fps, trainer)

    @staticmethod
    def _tracker_log_video(name: str, video: torch.Tensor, fps: int, trainer: Any) -> None:
        """Log video to tracker.

        Args:
            name: Log name.
            video: Video tensor.
            fps: Frames per second.
            trainer: Training context.
        """
        for tracker in trainer.accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_video(name, video, trainer.global_step, fps=fps)
            elif tracker.name == "wandb" and HAS_WANDB:
                tracker.log(
                    {name: wandb.Video((video.cpu().numpy()[0] * 255).astype("uint8"), fps=fps)},
                    step=trainer.global_step,
                )
            else:
                trainer.logger.warning("video logging not implemented for %s", tracker.name)

    def log_image(
        self,
        original: torch.Tensor,
        reconstruction: torch.Tensor,
        trainer: Any,
        tag: str = "train",
    ) -> None:
        """Log original and reconstructed images.

        Args:
            original: Original image tensor.
            reconstruction: Reconstructed image tensor.
            trainer: Training context.
            tag: Tag for logging.
        """
        original = original.squeeze(1)
        reconstruction = reconstruction.squeeze(1)

        combined = torch.cat([original, reconstruction], dim=-1)

        if combined.ndim == 3:
            grid = combined
        else:
            grid = make_grid(combined, nrow=4, padding=2, normalize=False)

        for tracker in trainer.accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_images(
                    f"{tag}_vis/images",
                    grid[None],
                    trainer.global_step,
                    dataformats="NCHW",
                )
            elif tracker.name == "wandb" and HAS_WANDB:
                tracker.log_images(
                    {f"{tag}_vis/images": [wandb.Image(image) for image in grid[None].cpu()]},
                    step=trainer.global_step,
                )
            else:
                trainer.logger.warning("image logging not implemented for %s", tracker.name)
