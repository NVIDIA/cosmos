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

"""Augmentor for creating sequence plans with random conditional frames.

Supports two sampling strategies:
- weighted dict (``conditioning_config``): explicit frame-count → probability pairs
- uniform (``uniform_conditioning=True``): k ~ Uniform{0, T_latent-1}, where T_latent
  is computed from the actual video length using the VAE temporal compression factor
"""

import random
from typing import Optional

import torch

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos3._src.vfm.datasets.sequence_packing import SequencePlan


class SequencePlanAugmentor(Augmentor):
    """Augmentor that creates SequencePlan with random conditional frames.

    Samples k conditioning frames and writes ``condition_frame_indexes_vision = list(range(k))``
    into the SequencePlan. Downstream packing code reads this field to set condition_mask.

    Args:
        input_keys: List of input keys (not used, but required by Augmentor interface).
        output_keys: List of output keys (not used, but required by Augmentor interface).
        args: Dictionary containing:
            - "conditioning_config" (dict[int, float], optional): Weighted distribution
              mapping latent-frame counts to unnormalized probabilities.
              Example: {0: 0.5, 4: 0.3, 8: 0.2}. Clamped to T_latent-1 at runtime.
            - "uniform_conditioning" (bool, default False): When True, samples
              k ~ Uniform{0, T_latent-1}. Takes precedence over conditioning_config when
              both are set. At least one of uniform_conditioning or conditioning_config
              must be provided.
            - "temporal_compression_factor" (int, default 4): VAE temporal compression
              factor used to convert pixel frame count N to T_latent = 1 + (N-1) // tcf.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

        if args is None:
            args = {}

        self.conditioning_config = args.get("conditioning_config")
        self.uniform_conditioning = args.get("uniform_conditioning", False)
        self.temporal_compression_factor = args.get("temporal_compression_factor", 4)

        if self.conditioning_config is None and not self.uniform_conditioning:
            raise ValueError("args must provide 'conditioning_config' or set 'uniform_conditioning=True'")

        # Validate and normalize probabilities
        if self.conditioning_config is not None:
            # Validate keys are non-negative integers
            for num_frames, prob in self.conditioning_config.items():
                if not isinstance(num_frames, int) or num_frames < 0:
                    raise ValueError(f"conditioning_config keys must be non-negative integers, got {num_frames}")
                if not isinstance(prob, (int, float)) or prob < 0:
                    raise ValueError(f"conditioning_config values must be non-negative numbers, got {prob}")

            # Normalize probabilities to sum to 1.0
            total_prob = sum(self.conditioning_config.values())
            if total_prob <= 0:
                raise ValueError("conditioning_config probabilities must sum to a positive number")

            self.normalized_config = {k: v / total_prob for k, v in self.conditioning_config.items()}
        else:
            self.normalized_config = {0: 1.0}

    def __call__(self, data_dict: dict) -> dict:
        """Create a SequencePlan with random conditional frames.

        Args:
            data_dict: Input data dictionary. Should contain "video" key to determine
                the number of frames available.

        Returns:
            data_dict: Output dictionary with "sequence_plan" key added.
        """
        # Get video to determine available frames
        video = data_dict.get("video")
        if video is None or (self.conditioning_config is None and not self.uniform_conditioning):
            # This is an image batch
            sequence_plan = SequencePlan(
                has_text=True,  # Has text prompt!
                has_vision=True,
                condition_frame_indexes_vision=[],  # No conditioning frames!
            )
            data_dict["sequence_plan"] = sequence_plan
            return data_dict

        # Determine number of frames
        # Video should be a tensor with shape (C, T, H, W) by this point in the pipeline
        if isinstance(video, torch.Tensor):
            assert video.ndim == 4, "video should be a tensor with shape (C, T, H, W)"
            num_frames = video.shape[1]
        else:
            # If video is not a tensor or dict, we can't determine the exact number
            # Use a conservative approach - will be limited by max available frames
            num_frames = None

        T_latent = 1 + (num_frames - 1) // self.temporal_compression_factor if num_frames is not None else 1

        # Sample number of conditional frames
        if self.uniform_conditioning:
            num_conditional_frames = random.randint(0, max(0, T_latent - 1))
        else:
            frames_options = list(self.normalized_config.keys())
            weights = list(self.normalized_config.values())
            num_conditional_frames = random.choices(frames_options, weights=weights, k=1)[0]
            num_conditional_frames = min(num_conditional_frames, T_latent - 1) if num_frames is not None else 0

        # Create condition_frame_indexes_vision list
        # Conditional frames are always the first N frames
        condition_frame_indexes_vision = list(range(num_conditional_frames))

        # Create SequencePlan
        sequence_plan = SequencePlan(
            has_text=True,
            has_vision=True,
            condition_frame_indexes_vision=condition_frame_indexes_vision,
        )

        # Add sequence plan to data dict
        data_dict["sequence_plan"] = sequence_plan

        return data_dict
