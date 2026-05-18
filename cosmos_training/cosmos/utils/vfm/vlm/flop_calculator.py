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

"""
FLOP calculator for Qwen3VL dynamic batching.

This module computes theoretical FLOPs for Qwen3VL samples to enable
FLOP-based batching (instead of token-based batching).

Key insight: Runtime scales linearly with FLOPs based on fitted curve from benchmarks.
"""

import torch

from cosmos.tools.flops.qwen3_vl import compute_qwen3vl_flops_from_config


class FlopCalculator:
    """Calculate theoretical FLOPs for Qwen3VL samples."""

    # The default fitted_slope / fitted_intercept were calibrated against
    # compute_qwen3vl_flops_from_config(is_causal=False) -- the bidirectional
    # upper-bound FLOP count. Causal-correct counting halves the text-decoder
    # S^2 attention terms and would invalidate the fit, causing the runtime
    # estimator to underestimate per-sample work and the dynamic batcher to
    # pack batches too large. Keep this False until the slope and intercept
    # are refit against is_causal=True benchmark data.

    # benchmark runs and flip _IS_CAUSAL_FOR_CALIBRATION to True so this
    # calculator inherits the algorithmically correct FLOP count by default.
    _IS_CAUSAL_FOR_CALIBRATION: bool = False

    def __init__(
        self,
        config,
        batch_multiplier: float = 3.0,
        fitted_slope: float = 5.078355e-12,  # ms/FLOP from fitted curve
        fitted_intercept: float = 133.88,  # ms from fitted curve
    ):
        """
        Initialize FLOP calculator.

        Args:
            config: Qwen3VLConfig object or dict with model parameters
            batch_multiplier: Multiplier for forward+backward pass (default: 3.0)
                             forward = 1x, backward = 2x, total = 3x
            fitted_slope: Slope from runtime_ms vs flops fitted curve (ms/FLOP)
            fitted_intercept: Intercept from fitted curve (ms)

        Fitted curve from benchmarks (R² = 0.9460):
            runtime_ms = 5.078355e-12 * flops + 133.88
        """
        self.config = config
        self.batch_multiplier = batch_multiplier
        self.fitted_slope = fitted_slope
        self.fitted_intercept = fitted_intercept

        # Extract config parameters
        if hasattr(config, "vision_config"):
            self.spatial_merge_size = config.vision_config.spatial_merge_size
        elif isinstance(config, dict):
            self.spatial_merge_size = config.get("spatial_merge_size", 2)
        else:
            self.spatial_merge_size = 2

    def get_num_visual_tokens(self, sample: dict) -> int:
        """
        Extract number of visual tokens from sample.

        Args:
            sample: Data sample containing visual information

        Returns:
            Number of visual tokens (after spatial merging)
        """
        if "pixel_values" in sample:
            # pixel_values: [num_patches, 1536] where 1536 = 3*2*16*16
            num_patches = sample["pixel_values"].shape[0]
            return num_patches // (self.spatial_merge_size**2)
        elif "image_grid_thw" in sample:
            # image_grid_thw: [num_images, 3] where each row is [t, h, w]
            # num_patches = sum(t * h * w for each image)
            thw = sample["image_grid_thw"]
            if isinstance(thw, torch.Tensor):
                num_patches = (thw[:, 0] * thw[:, 1] * thw[:, 2]).sum().item()
            else:
                # Handle numpy or list
                import numpy as np

                thw = np.array(thw)
                num_patches = (thw[:, 0] * thw[:, 1] * thw[:, 2]).sum()
            return int(num_patches // (self.spatial_merge_size**2))
        else:
            # Text-only sample
            return 0

    def get_num_patches(self, sample: dict) -> int:
        """
        Extract number of patches from sample.

        Args:
            sample: Data sample containing visual information

        Returns:
            Number of patches (before spatial merging)
        """
        if "pixel_values" in sample:
            return sample["pixel_values"].shape[0]
        elif "image_grid_thw" in sample:
            thw = sample["image_grid_thw"]
            if isinstance(thw, torch.Tensor):
                return int((thw[:, 0] * thw[:, 1] * thw[:, 2]).sum().item())
            else:
                import numpy as np

                thw = np.array(thw)
                return int((thw[:, 0] * thw[:, 1] * thw[:, 2]).sum())
        else:
            return 0

    def compute_single_sample_flops(self, sample: dict) -> float:
        """
        Compute FLOPs for single sample (batch_size=1).

        Args:
            sample: Data sample

        Returns:
            Total FLOPs for forward + backward pass
        """

        total_tokens = len(sample["input_ids"])
        num_visual_tokens = self.get_num_visual_tokens(sample)
        num_patches = self.get_num_patches(sample)

        result = compute_qwen3vl_flops_from_config(
            self.config,
            total_tokens=total_tokens,
            visual_tokens=num_visual_tokens,
            num_patches=num_patches,
            is_causal=self._IS_CAUSAL_FOR_CALIBRATION,
        )

        # Return total FLOPs including forward + backward
        return result["total_flops"] * self.batch_multiplier

    def compute_batch_flops(self, samples: list[dict]) -> float:
        """
        Compute FLOPs for a batch of samples.

        Key insight: In a batch, all samples are padded to max sequence length.
        Attention is O(n²) where n = max(sequence_lengths).

        Args:
            samples: list of samples in batch

        Returns:
            Total FLOPs for forward + backward pass on this batch
        """
        if not samples:
            return 0.0

        # Find max sequence length (determines padding)
        max_total_tokens = max(len(s["input_ids"]) for s in samples)

        # Sum visual tokens (vision encoder processes each separately)
        total_visual_tokens = sum(self.get_num_visual_tokens(s) for s in samples)
        total_num_patches = sum(self.get_num_patches(s) for s in samples)

        # Compute FLOPs as if all samples have max_total_tokens
        # This is the actual cost after padding
        batch_size = len(samples)

        # Average visual tokens per sample
        avg_visual_tokens = total_visual_tokens / batch_size if batch_size > 0 else 0
        avg_num_patches = total_num_patches / batch_size if batch_size > 0 else 0

        result = compute_qwen3vl_flops_from_config(
            self.config,
            total_tokens=max_total_tokens,
            visual_tokens=int(avg_visual_tokens),
            num_patches=int(avg_num_patches),
            is_causal=self._IS_CAUSAL_FOR_CALIBRATION,
        )

        # Scale by batch size and forward+backward multiplier
        return result["total_flops"] * batch_size * self.batch_multiplier

    def estimate_runtime_ms(self, flops: float) -> float:
        """
        Estimate runtime in milliseconds based on fitted curve.

        Args:
            flops: Theoretical FLOPs

        Returns:
            Estimated runtime in milliseconds

        Fitted curve (R² = 0.9460):
            runtime_ms = 5.078355e-12 * flops + 133.88
        """
        return self.fitted_slope * flops + self.fitted_intercept

    def compute_max_flops_for_runtime(self, target_runtime_seconds: float) -> float:
        """
        Compute maximum FLOPs for a target runtime.

        Args:
            target_runtime_seconds: Target runtime in seconds

        Returns:
            Maximum FLOPs to stay within runtime budget

        Solves: target_runtime_ms = fitted_slope * max_flops + fitted_intercept
                max_flops = (target_runtime_ms - fitted_intercept) / fitted_slope
        """
        target_runtime_ms = target_runtime_seconds * 1000
        max_flops = (target_runtime_ms - self.fitted_intercept) / self.fitted_slope
        return max_flops
