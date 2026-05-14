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

import random
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.lazy_config import LazyDict
from projects.cosmos3.vfm.conditioner import BooleanFlag, GeneralConditioner, ReMapkey, Text2WorldCondition
from cosmos.utils.vfm.context_parallel import broadcast_split_tensor


@dataclass(frozen=True)
class Video2WorldCondition(Text2WorldCondition):
    use_video_condition: bool = False
    # the following two attributes are used to set the video condition; during training, inference
    gt_frames: Optional[torch.Tensor] = None
    condition_video_input_mask: Optional[torch.Tensor] = None

    def set_video_condition(
        self,
        gt_frames: torch.Tensor,
        random_min_num_conditional_frames: int,
        random_max_num_conditional_frames: int,
        num_conditional_frames: Optional[int] = None,
        conditional_frames_probs: Optional[Dict[int, float]] = None,
    ) -> "Video2WorldCondition":
        """
        Sets the video conditioning frames for video-to-video generation.

        This method creates a conditioning mask for the input video frames that determines
        which frames will be used as context frames for generating new frames. The method
        handles both image batches (T=1) and video batches (T>1) differently.

        Args:
            gt_frames: A tensor of ground truth frames with shape [B, C, T, H, W], where:
                B = batch size
                C = number of channels
                T = number of frames
                H = height
                W = width

            random_min_num_conditional_frames: Minimum number of frames to use for conditioning
                when randomly selecting a number of conditioning frames.

            random_max_num_conditional_frames: Maximum number of frames to use for conditioning
                when randomly selecting a number of conditioning frames.

            num_conditional_frames: Optional; If provided, all examples in the batch will use
                exactly this many frames for conditioning. If None, a random number of frames
                between random_min_num_conditional_frames and random_max_num_conditional_frames
                will be selected for each example in the batch.

            conditional_frames_probs: Optional; Dictionary mapping number of frames to probabilities.
                If provided, overrides the random_min/max_num_conditional_frames with weighted sampling.
                Example: {0: 0.5, 1: 0.25, 2: 0.25} for 50% chance of 0 frames, 25% for 1, 25% for 2.

        Returns:
            A new Video2WorldCondition object with the gt_frames and conditioning mask set.
            The conditioning mask (condition_video_input_mask) is a binary tensor
            of shape [B, 1, T, H, W] where 1 indicates frames used for conditioning and 0
            indicates frames to be generated.

        Notes:
            - For image batches (T=1), no conditioning frames are used (num_conditional_frames_B = 0).
            - For video batches:
                - If num_conditional_frames is provided, all examples use that fixed number of frames.
                - Otherwise, each example randomly uses between random_min_num_conditional_frames and
                random_max_num_conditional_frames frames.
            - The mask marks the first N frames as conditioning frames (set to 1) for each example.
        """
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = gt_frames

        # condition_video_input_mask
        B, _, T, H, W = gt_frames.shape
        condition_video_input_mask = torch.zeros(B, 1, T, H, W, dtype=gt_frames.dtype, device=gt_frames.device)
        if T == 1:  # handle image batch
            num_conditional_frames_B = torch.zeros(B, dtype=torch.int32)
        else:  # handle video batch
            if num_conditional_frames is not None:
                num_conditional_frames_B = torch.ones(B, dtype=torch.int32) * num_conditional_frames
            elif conditional_frames_probs is not None:
                # Use weighted sampling based on provided probabilities
                frames_options = list(conditional_frames_probs.keys())
                weights = list(conditional_frames_probs.values())
                num_conditional_frames_B = torch.tensor(
                    random.choices(frames_options, weights=weights, k=B), dtype=torch.int32
                )
            else:
                num_conditional_frames_B = torch.randint(
                    random_min_num_conditional_frames, random_max_num_conditional_frames + 1, size=(B,)
                )
        for idx in range(B):
            condition_video_input_mask[idx, :, : num_conditional_frames_B[idx], :, :] += 1

        kwargs["condition_video_input_mask"] = condition_video_input_mask
        return type(self)(**kwargs)

    def edit_for_inference(
        self, is_cfg_conditional: bool = True, num_conditional_frames: int = 1
    ) -> "Video2WorldCondition":
        _condition = self.set_video_condition(
            gt_frames=self.gt_frames,
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
        )
        if not is_cfg_conditional:
            # Do not use classifier free guidance on conditional frames.
            # YB found that it leads to worse results.
            _condition.use_video_condition.fill_(True)
        return _condition

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> "Video2WorldCondition":
        if self.is_broadcasted:
            return self
        # extra efforts
        gt_frames = self.gt_frames
        condition_video_input_mask = self.condition_video_input_mask
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = None
        kwargs["condition_video_input_mask"] = None
        new_condition = Text2WorldCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )

        kwargs = new_condition.to_dict(skip_underscore=False)
        _, _, T, _, _ = gt_frames.shape
        if process_group is not None:
            if T > 1 and process_group.size() > 1:
                gt_frames = broadcast_split_tensor(gt_frames, seq_dim=2, process_group=process_group)
                condition_video_input_mask = broadcast_split_tensor(
                    condition_video_input_mask, seq_dim=2, process_group=process_group
                )
        kwargs["gt_frames"] = gt_frames
        kwargs["condition_video_input_mask"] = condition_video_input_mask
        return type(self)(**kwargs)


class Video2WorldConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Video2WorldCondition:
        output = super()._forward(batch, override_dropout_rate)
        return Video2WorldCondition(**output)


_SHARED_CONFIG = dict(
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    use_video_condition=L(BooleanFlag)(
        input_key="fps",
        output_key="use_video_condition",
        dropout_rate=0.2,
    ),
)

VideoPredictionConditioner: LazyDict = L(Video2WorldConditioner)(
    **_SHARED_CONFIG,
)


def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_conditioner",
        node=VideoPredictionConditioner,
    )
