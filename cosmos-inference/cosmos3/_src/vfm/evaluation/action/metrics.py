# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Evaluation metrics for Action models.

This module provides standard metrics used for evaluating video prediction
and action prediction quality.

All metric functions follow a unified structure:
1. Check if gt and pred share the same shape
2. Check if shape is expected
3. Detach tensor while keeping the device (if tensor)
4. Compute metric
5. Return float
"""

from __future__ import annotations

import warnings

import numpy as np
import torch


def _get_dtype_info(
    gt: np.ndarray | torch.Tensor,
    pred: np.ndarray | torch.Tensor,
) -> tuple[float, float, float]:
    """
    Get dtype info and derive data range.

    Args:
        gt: Ground truth array/tensor.
        pred: Predicted array/tensor.

    Returns:
        Tuple of (data_range, expected_min, expected_max).

    Raises:
        ValueError: If dtypes don't match or are unsupported.
    """
    if isinstance(gt, torch.Tensor):
        gt_is_float = gt.dtype.is_floating_point
        gt_is_uint8 = gt.dtype == torch.uint8
        gt_dtype_str = str(gt.dtype)
    else:
        gt_is_float = np.issubdtype(gt.dtype, np.floating)
        gt_is_uint8 = gt.dtype == np.uint8
        gt_dtype_str = str(gt.dtype)

    if isinstance(pred, torch.Tensor):
        pred_is_float = pred.dtype.is_floating_point
        pred_is_uint8 = pred.dtype == torch.uint8
        pred_dtype_str = str(pred.dtype)
    else:
        pred_is_float = np.issubdtype(pred.dtype, np.floating)
        pred_is_uint8 = pred.dtype == np.uint8
        pred_dtype_str = str(pred.dtype)

    if gt_is_float != pred_is_float or gt_is_uint8 != pred_is_uint8:
        raise ValueError(f"Dtype mismatch: gt {gt_dtype_str} vs pred {pred_dtype_str}")

    if gt_is_float:
        return 2.0, -1.0, 1.0
    elif gt_is_uint8:
        return 255.0, 0.0, 255.0
    else:
        raise ValueError(f"Unsupported dtype: {gt_dtype_str}. Expected float or uint8.")


def _compute_motion_mask(
    video: np.ndarray | torch.Tensor,
    threshold_percentile: float,
) -> np.ndarray | torch.Tensor:
    """
    Compute per-frame motion mask based on frame differences.

    For each frame t, computes |video[t] - video[t-1]| and thresholds based on
    the given percentile. Frame 0 uses the same mask as frame 1.

    Args:
        video: Video array/tensor of shape (C, T, H, W) or (B, C, T, H, W).
        threshold_percentile: Percentile threshold (0-100). Pixels with motion
            magnitude above this percentile are marked as dynamic.
            E.g., 80.0 means top 20% of motion is considered "dynamic".

    Returns:
        Boolean mask of shape (T, H, W) or (B, T, H, W) where True = dynamic pixel.
    """
    is_tensor = isinstance(video, torch.Tensor)

    # Handle both (C, T, H, W) and (B, C, T, H, W) shapes
    if video.ndim == 4:
        # (C, T, H, W) -> time is dim 1
        time_dim = 1
    else:
        # (B, C, T, H, W) -> time is dim 2
        time_dim = 2

    if is_tensor:
        video = video.detach().float()
        # Compute frame differences: |video[t] - video[t-1]|
        if time_dim == 1:
            diff = torch.abs(video[:, 1:] - video[:, :-1])  # [C,T-1,H,W]
            # Average over channels to get motion magnitude
            motion_magnitude = diff.mean(dim=0)  # [T-1,H,W]
        else:
            diff = torch.abs(video[:, :, 1:] - video[:, :, :-1])  # [B,C,T-1,H,W]
            motion_magnitude = diff.mean(dim=1)  # [B,T-1,H,W]

        # Compute threshold value from percentile
        threshold_value = torch.quantile(motion_magnitude.flatten().float(), threshold_percentile / 100.0)

        # Create mask for frames 1..T-1
        mask_after_first = motion_magnitude > threshold_value

        # For frame 0, use the same mask as frame 1
        if time_dim == 1:
            first_frame_mask = mask_after_first[0:1]  # [1,H,W]
            mask = torch.cat([first_frame_mask, mask_after_first], dim=0)  # [T,H,W]
        else:
            first_frame_mask = mask_after_first[:, 0:1]  # [B,1,H,W]
            mask = torch.cat([first_frame_mask, mask_after_first], dim=1)  # [B,T,H,W]
    else:
        video = np.asarray(video, dtype=np.float32)
        # Compute frame differences
        if time_dim == 1:
            diff = np.abs(video[:, 1:] - video[:, :-1])  # [C,T-1,H,W]
            motion_magnitude = diff.mean(axis=0)  # [T-1,H,W]
        else:
            diff = np.abs(video[:, :, 1:] - video[:, :, :-1])  # [B,C,T-1,H,W]
            motion_magnitude = diff.mean(axis=1)  # [B,T-1,H,W]

        # Compute threshold value from percentile
        threshold_value = np.percentile(motion_magnitude.flatten(), threshold_percentile)

        # Create mask for frames 1..T-1
        mask_after_first = motion_magnitude > threshold_value

        # For frame 0, use the same mask as frame 1
        if time_dim == 1:
            first_frame_mask = mask_after_first[0:1]  # [1,H,W]
            mask = np.concatenate([first_frame_mask, mask_after_first], axis=0)  # [T,H,W]
        else:
            first_frame_mask = mask_after_first[:, 0:1]  # [B,1,H,W]
            mask = np.concatenate([first_frame_mask, mask_after_first], axis=1)  # [B,T,H,W]

    return mask


def compute_psnr(
    gt: np.ndarray | torch.Tensor,
    pred: np.ndarray | torch.Tensor,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between ground truth and prediction.

    PSNR is defined as: 20 * log10(MAX / sqrt(MSE))

    The data range is automatically derived from dtype:
    - float dtype: expects values in [-1, 1], data_range = 2.0
    - uint8 dtype: expects values in [0, 255], data_range = 255.0

    Args:
        gt: Ground truth array/tensor of shape (C, T, H, W) or (B, C, T, H, W).
            - If float dtype: values must be in [-1, 1] range.
            - If uint8 dtype: values must be in [0, 255] range.
        pred: Predicted array/tensor of same shape and dtype as gt.

    Returns:
        PSNR value in decibels (dB). Higher is better.

    Raises:
        ValueError: If shapes don't match, dtypes don't match, dtype is unsupported,
            or values are out of expected range.

    Example:
        >>> gt_video = torch.randn(3, 16, 96, 96).clamp(-1, 1)  # (C, T, H, W), float in [-1, 1]
        >>> pred_video = torch.randn(3, 16, 96, 96).clamp(-1, 1)
        >>> psnr = compute_psnr(gt_video, pred_video)
    """
    # 1. Check if gt and pred share the same shape
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    # 2. Check if shape is expected: (C, T, H, W) or (B, C, T, H, W)
    if gt.ndim not in (4, 5):
        raise ValueError(
            f"Expected gt to have 4 dims (C, T, H, W) or 5 dims (B, C, T, H, W), got {gt.ndim} dims: {gt.shape}"
        )

    # 3. Check dtype and derive data range
    data_range, expected_min, expected_max = _get_dtype_info(gt, pred)

    # 4. Validate value range
    if isinstance(gt, torch.Tensor):
        gt_min, gt_max = gt.min().item(), gt.max().item()
    else:
        gt_min, gt_max = float(gt.min()), float(gt.max())

    if isinstance(pred, torch.Tensor):
        pred_min, pred_max = pred.min().item(), pred.max().item()
    else:
        pred_min, pred_max = float(pred.min()), float(pred.max())

    if gt_min < expected_min or gt_max > expected_max:
        raise ValueError(f"gt values out of range: got [{gt_min}, {gt_max}], expected [{expected_min}, {expected_max}]")
    if pred_min < expected_min or pred_max > expected_max:
        raise ValueError(
            f"pred values out of range: got [{pred_min}, {pred_max}], expected [{expected_min}, {expected_max}]"
        )

    # 5. Detach tensor while keeping the device (if tensor)
    # 6. Compute metric
    if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
        gt_t = gt.detach().float()
        pred_t = pred.detach().float()
        mse = torch.mean((gt_t - pred_t) ** 2).item()
    else:
        gt_np = np.asarray(gt, dtype=np.float32)
        pred_np = np.asarray(pred, dtype=np.float32)
        mse = float(np.mean((gt_np - pred_np) ** 2))

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20.0 * np.log10(data_range / np.sqrt(mse))

    # 7. Return float
    return float(psnr)


def compute_dynamic_psnr(
    gt: np.ndarray | torch.Tensor,
    pred: np.ndarray | torch.Tensor,
    motion_threshold_percentile: float = 80.0,
) -> tuple[float, np.ndarray | torch.Tensor]:
    """
    Compute PSNR only on dynamic (moving) pixels, ignoring static background.

    This metric is more meaningful for videos with large static backgrounds,
    as it focuses on the quality of dynamic content rather than being dominated
    by easy-to-predict static regions.

    Dynamic pixels are detected using frame differences: for each frame t,
    pixels where |gt[t] - gt[t-1]| exceeds a percentile threshold are considered
    "dynamic". PSNR is computed only on these pixels.

    Args:
        gt: Ground truth array/tensor of shape (C, T, H, W) or (B, C, T, H, W).
            - If float dtype: values must be in [-1, 1] range.
            - If uint8 dtype: values must be in [0, 255] range.
        pred: Predicted array/tensor of same shape and dtype as gt.
        motion_threshold_percentile: Percentile threshold for motion detection.
            Default 80.0 means top 20% of motion magnitude is considered "dynamic".
            Higher values = stricter threshold = fewer dynamic pixels.

    Returns:
        Tuple of (psnr, motion_mask):
            - psnr: PSNR value in decibels (dB) computed only on dynamic regions.
              Higher is better. Returns regular PSNR if no dynamic pixels are found.
            - motion_mask: Boolean mask of shape (T, H, W) or (B, T, H, W) where
              True indicates a dynamic pixel. Same type as input (tensor or ndarray).

    Raises:
        ValueError: If shapes don't match, dtypes don't match, dtype is unsupported,
            or values are out of expected range.

    Example:
        >>> gt_video = torch.randn(3, 16, 96, 96).clamp(-1, 1)  # (C, T, H, W)
        >>> pred_video = torch.randn(3, 16, 96, 96).clamp(-1, 1)
        >>> dynamic_psnr, motion_mask = compute_dynamic_psnr(gt_video, pred_video)
        >>> print(f"Dynamic PSNR: {dynamic_psnr:.2f}, Mask shape: {motion_mask.shape}")
    """

    # 1. Check if gt and pred share the same shape
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    # 2. Check if shape is expected: (C, T, H, W) or (B, C, T, H, W)
    if gt.ndim not in (4, 5):
        raise ValueError(
            f"Expected gt to have 4 dims (C, T, H, W) or 5 dims (B, C, T, H, W), got {gt.ndim} dims: {gt.shape}"
        )

    # 3. Check dtype and derive data range
    data_range, expected_min, expected_max = _get_dtype_info(gt, pred)

    # 4. Validate value range
    if isinstance(gt, torch.Tensor):
        gt_min, gt_max = gt.min().item(), gt.max().item()
    else:
        gt_min, gt_max = float(gt.min()), float(gt.max())

    if isinstance(pred, torch.Tensor):
        pred_min, pred_max = pred.min().item(), pred.max().item()
    else:
        pred_min, pred_max = float(pred.min()), float(pred.max())

    if gt_min < expected_min or gt_max > expected_max:
        raise ValueError(f"gt values out of range: got [{gt_min}, {gt_max}], expected [{expected_min}, {expected_max}]")
    if pred_min < expected_min or pred_max > expected_max:
        raise ValueError(
            f"pred values out of range: got [{pred_min}, {pred_max}], expected [{expected_min}, {expected_max}]"
        )

    # 5. Compute motion mask from GT
    motion_mask = _compute_motion_mask(gt, motion_threshold_percentile)

    # 6. Check if there are any dynamic pixels
    if isinstance(motion_mask, torch.Tensor):
        num_dynamic = motion_mask.sum().item()
    else:
        num_dynamic = int(motion_mask.sum())

    if num_dynamic == 0:
        warnings.warn(
            "No dynamic pixels found in video. Returning regular PSNR. Consider lowering motion_threshold_percentile.",
            stacklevel=2,
        )
        return compute_psnr(gt, pred), motion_mask

    # 7. Compute MSE only on dynamic pixels
    if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
        gt_t = gt.detach().float()
        pred_t = pred.detach().float()
        # motion_mask is a tensor when gt is a tensor
        mask_t = motion_mask if isinstance(motion_mask, torch.Tensor) else torch.from_numpy(motion_mask)

        # Expand mask to match video shape (add channel dimension)
        if gt.ndim == 4:
            # (C, T, H, W) - mask is (T, H, W)
            expanded_mask = mask_t.unsqueeze(0).expand_as(gt_t)  # [C,T,H,W]
        else:
            # (B, C, T, H, W) - mask is (B, T, H, W)
            expanded_mask = mask_t.unsqueeze(1).expand_as(gt_t)  # [B,C,T,H,W]

        # Compute squared error only on dynamic pixels
        squared_error = (gt_t - pred_t) ** 2
        masked_squared_error = squared_error[expanded_mask]
        mse = masked_squared_error.mean().item()
    else:
        gt_np = np.asarray(gt, dtype=np.float32)
        pred_np = np.asarray(pred, dtype=np.float32)
        # motion_mask is an ndarray when gt is an ndarray
        mask_np = motion_mask if isinstance(motion_mask, np.ndarray) else motion_mask.numpy()

        # Expand mask to match video shape
        if gt_np.ndim == 4:
            # (C, T, H, W) - mask is (T, H, W)
            expanded_mask = np.broadcast_to(mask_np[np.newaxis, :, :, :], gt_np.shape)
        else:
            # (B, C, T, H, W) - mask is (B, T, H, W)
            expanded_mask = np.broadcast_to(mask_np[:, np.newaxis, :, :, :], gt_np.shape)

        # Compute squared error only on dynamic pixels
        squared_error = (gt_np - pred_np) ** 2
        masked_squared_error = squared_error[expanded_mask]
        mse = float(masked_squared_error.mean())

    # 8. Convert MSE to PSNR
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20.0 * np.log10(data_range / np.sqrt(mse))

    return float(psnr), motion_mask


def compute_ssim(
    gt: np.ndarray | torch.Tensor,
    pred: np.ndarray | torch.Tensor,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between ground truth and prediction.

    SSIM measures the structural similarity between two images, considering
    luminance, contrast, and structure. It is computed per-frame and averaged.

    Uses skimage.metrics.structural_similarity for computation, following the
    implementation in projects/cosmos/tokenizer/evaluation/metric.py.

    The data range is automatically derived from dtype:
    - float dtype: expects values in [-1, 1], data_range = 2.0
    - uint8 dtype: expects values in [0, 255], data_range = 255.0

    Args:
        gt: Ground truth array/tensor of shape (C, T, H, W) or (B, C, T, H, W).
            If float dtype: values must be in [-1, 1] range.
            If uint8 dtype: values must be in [0, 255] range.
        pred: Predicted array/tensor of same shape and dtype as gt.

    Returns:
        SSIM value in range [-1, 1]. Higher is better (1.0 = identical).

    Raises:
        ValueError: If shapes don't match, dtypes don't match, dtype is unsupported,
            or values are out of expected range.

    Example:
        >>> gt_video = torch.randn(3, 16, 96, 96).clamp(-1, 1)  # (C, T, H, W), float in [-1, 1]
        >>> pred_video = torch.randn(3, 16, 96, 96).clamp(-1, 1)
        >>> ssim_val = compute_ssim(gt_video, pred_video)
    """
    from skimage.metrics import structural_similarity as ssim

    # 1. Check if gt and pred share the same shape
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    # 2. Check if shape is expected: (C, T, H, W) or (B, C, T, H, W)
    if gt.ndim not in (4, 5):
        raise ValueError(
            f"Expected gt to have 4 dims (C, T, H, W) or 5 dims (B, C, T, H, W), got {gt.ndim} dims: {gt.shape}"
        )

    # 3. Check dtype and derive data range
    data_range, expected_min, expected_max = _get_dtype_info(gt, pred)

    # 4. Validate value range
    if isinstance(gt, torch.Tensor):
        gt_min, gt_max = gt.min().item(), gt.max().item()
    else:
        gt_min, gt_max = float(gt.min()), float(gt.max())

    if isinstance(pred, torch.Tensor):
        pred_min, pred_max = pred.min().item(), pred.max().item()
    else:
        pred_min, pred_max = float(pred.min()), float(pred.max())

    if isinstance(gt, torch.Tensor) and gt.dtype == torch.bfloat16:
        gt = gt.float()
    if isinstance(pred, torch.Tensor) and pred.dtype == torch.bfloat16:
        pred = pred.float()

    if gt_min < expected_min or gt_max > expected_max:
        raise ValueError(f"gt values out of range: got [{gt_min}, {gt_max}], expected [{expected_min}, {expected_max}]")
    if pred_min < expected_min or pred_max > expected_max:
        raise ValueError(
            f"pred values out of range: got [{pred_min}, {pred_max}], expected [{expected_min}, {expected_max}]"
        )

    # 5. Convert to numpy arrays
    if isinstance(gt, torch.Tensor):
        gt_np = gt.detach().cpu().numpy()
    else:
        gt_np = np.asarray(gt)

    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = np.asarray(pred)

    # 6. Reshape to (N, C, H, W) where N = number of frames
    if gt_np.ndim == 4:
        # (C, T, H, W) -> (T, C, H, W)
        gt_frames = np.transpose(gt_np, (1, 0, 2, 3))
        pred_frames = np.transpose(pred_np, (1, 0, 2, 3))
    else:
        # (B, C, T, H, W) -> (B*T, C, H, W)
        b, c, t, h, w = gt_np.shape
        gt_frames = np.transpose(gt_np, (0, 2, 1, 3, 4)).reshape(b * t, c, h, w)
        pred_frames = np.transpose(pred_np, (0, 2, 1, 3, 4)).reshape(b * t, c, h, w)

    # 7. Compute SSIM per frame using skimage
    ssim_values = []
    for gt_frame, pred_frame in zip(gt_frames, pred_frames):
        # gt_frame and pred_frame are (C, H, W)
        frame_ssim = ssim(gt_frame, pred_frame, channel_axis=0, data_range=data_range)
        ssim_values.append(frame_ssim)

    # 8. Return average SSIM
    return float(np.mean(ssim_values))


def compute_action_mse(
    gt_action: np.ndarray | torch.Tensor,
    pred_action: np.ndarray | torch.Tensor,
) -> float:
    """
    Compute Mean Squared Error (MSE) between ground truth and predicted actions.

    Args:
        gt_action: Ground truth array/tensor of shape (T, D) or (B, T, D),
            where T is the number of timesteps and D is the action dimension.
        pred_action: Predicted array/tensor of same shape as gt_action.

    Returns:
        MSE value. Lower is better.

    Example:
        >>> gt = np.random.randn(16, 2)  # (T, D) - 16 timesteps, 2D actions
        >>> pred = np.random.randn(16, 2)
        >>> mse = compute_action_mse(gt, pred)
    """
    # 1. Check if gt and pred share the same shape
    if gt_action.shape != pred_action.shape:
        raise ValueError(f"Shape mismatch: gt_action {gt_action.shape} vs pred_action {pred_action.shape}")

    # 2. Check if shape is expected: (T, D) or (B, T, D)
    if gt_action.ndim not in (2, 3):
        raise ValueError(
            f"Expected gt_action to have 2 dims (T, D) or 3 dims (B, T, D), got {gt_action.ndim} dims: {gt_action.shape}"
        )

    # 3. Detach tensor while keeping the device (if tensor)
    # 4. Compute metric
    if isinstance(gt_action, torch.Tensor) and isinstance(pred_action, torch.Tensor):
        gt_t = gt_action.detach()
        pred_t = pred_action.detach()
        mse = torch.mean((gt_t - pred_t) ** 2).item()
    else:
        gt_np = np.asarray(gt_action)
        pred_np = np.asarray(pred_action)
        mse = float(np.mean((gt_np - pred_np) ** 2))

    # 5. Return float
    return float(mse)


def compute_grouped_action_mse(
    gt_action: np.ndarray | torch.Tensor,
    pred_action: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    """
    Compute grouped MSE for translation, rotation, and gripper action components.

    This metric is useful for robotics tasks where actions are structured as
    [translation(3), rotation(9), gripper(1)] using 9D rotation matrix representation.

    NOTE: All actions must be converted to 9D rotation matrix format (13D total)
    before calling this function. Use conversion utilities in the inference stage
    to convert from other rotation representations (axis-angle, 6D) to 9D.

    Args:
        gt_action: Ground truth array/tensor of shape (T, 13) or (B, T, 13),
            where T is the number of timesteps.
            Expected format: [translation(3), rotation_matrix(9), gripper(1)]
        pred_action: Predicted array/tensor of same shape as gt_action.

    Returns:
        Dictionary with MSE values for each component:
            - "translation": MSE for x, y, z (dims 0-2)
            - "rotation": MSE for flattened rotation matrix (dims 3-11)
            - "gripper": MSE for gripper (dim 12)
        Values are 0.0 if the corresponding dimensions are not present.
    """
    if gt_action.shape != pred_action.shape:
        raise ValueError(f"Shape mismatch: gt_action {gt_action.shape} vs pred_action {pred_action.shape}")

    if gt_action.ndim not in (2, 3):
        raise ValueError(
            f"Expected gt_action to have 2 dims (T, D) or 3 dims (B, T, D), got {gt_action.ndim} dims: {gt_action.shape}"
        )

    if isinstance(gt_action, torch.Tensor):
        gt_np = gt_action.detach().cpu().numpy()
    else:
        gt_np = np.asarray(gt_action)

    if isinstance(pred_action, torch.Tensor):
        pred_np = pred_action.detach().cpu().numpy()
    else:
        pred_np = np.asarray(pred_action)

    result: dict[str, float] = {"translation": 0.0, "rotation": 0.0, "gripper": 0.0}
    action_dim = gt_np.shape[-1]
    gripper_idx = 12

    if action_dim >= 3:
        result["translation"] = float(np.mean((gt_np[..., :3] - pred_np[..., :3]) ** 2))

    # Rotation: dimensions 3-11 (flattened 3x3 rotation matrix)
    if action_dim >= gripper_idx:
        result["rotation"] = float(np.mean((gt_np[..., 3:gripper_idx] - pred_np[..., 3:gripper_idx]) ** 2))

    # Gripper: dimension 12
    if action_dim > gripper_idx:
        result["gripper"] = float(np.mean((gt_np[..., gripper_idx] - pred_np[..., gripper_idx]) ** 2))

    return result


def compute_action_mae(
    gt_action: np.ndarray | torch.Tensor,
    pred_action: np.ndarray | torch.Tensor,
) -> float:
    """
    Compute Mean Absolute Error (MAE) between ground truth and predicted actions.

    Args:
        gt_action: Ground truth array/tensor of shape (T, D) or (B, T, D),
            where T is the number of timesteps and D is the action dimension.
        pred_action: Predicted array/tensor of same shape as gt_action.

    Returns:
        MAE value. Lower is better.

    Example:
        >>> gt = np.random.randn(16, 2)  # (T, D) - 16 timesteps, 2D actions
        >>> pred = np.random.randn(16, 2)
        >>> mae = compute_action_mae(gt, pred)
    """
    # 1. Check if gt and pred share the same shape
    if gt_action.shape != pred_action.shape:
        raise ValueError(f"Shape mismatch: gt_action {gt_action.shape} vs pred_action {pred_action.shape}")

    # 2. Check if shape is expected: (T, D) or (B, T, D)
    if gt_action.ndim not in (2, 3):
        raise ValueError(
            f"Expected gt_action to have 2 dims (T, D) or 3 dims (B, T, D), got {gt_action.ndim} dims: {gt_action.shape}"
        )

    # 3. Detach tensor while keeping the device (if tensor)
    # 4. Compute metric
    if isinstance(gt_action, torch.Tensor) and isinstance(pred_action, torch.Tensor):
        gt_t = gt_action.detach()
        pred_t = pred_action.detach()
        mae = torch.mean(torch.abs(gt_t - pred_t)).item()
    else:
        gt_np = np.asarray(gt_action)
        pred_np = np.asarray(pred_action)
        mae = float(np.mean(np.abs(gt_np - pred_np)))

    # 5. Return float
    return float(mae)


def compute_geodesic_rotation_error(
    gt_rot: np.ndarray,
    pred_rot: np.ndarray,
) -> np.ndarray:
    """Geodesic angular error between rotation matrices on SO(3).

    Computes ``arccos((tr(R_gt^T @ R_pred) - 1) / 2)`` for each pair,
    returning the angular distance in **degrees**.

    Args:
        gt_rot: Ground-truth rotation matrices of shape ``(N, 3, 3)``.
        pred_rot: Predicted rotation matrices of shape ``(N, 3, 3)``.

    Returns:
        Per-element angular errors in degrees, shape ``(N,)``.
    """
    if gt_rot.shape != pred_rot.shape or gt_rot.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (N,3,3) arrays, got gt={gt_rot.shape}, pred={pred_rot.shape}")

    R_err = np.matmul(np.transpose(gt_rot, (0, 2, 1)), pred_rot)  # [N,3,3]
    trace = np.trace(R_err, axis1=1, axis2=2)  # [N]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)  # [N]
    return np.degrees(np.arccos(cos_angle))  # [N]
