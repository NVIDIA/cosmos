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
Augmentors for transfer (control-conditioned) image and video generation in the cosmos3 VFM pipeline.

Transfer training conditions the model on control signals (edge, blur, depth, or segmentation)
to generate images or videos, aligned with cosmos/transfer2. This module provides:

- **TransferToTrainingFormat**: Converts (control_input, target) into the joint dataloader format
  with SequencePlan (condition frame + generated frame), for both image and video outputs.

- **VideoTransferSampleFrame**: For video→image transfer: samples a single frame index consistently
  across control and video tensors, producing image-sized tensors from 4D video inputs.
- **AddControlFromVideoComb**: Uses AddControlInputComb (in transfer_control_input) to compute one of edge/blur/depth/seg
  from video or precomputed fields and writes the chosen control to data_dict["control_input"].
- **SampleResolution**: Samples a resolution from a list and sets data_dict["_res_size_map"] so downstream
  resize/padding use that resolution (used to combine multiple resolutions in one dataloader).
"""

from __future__ import annotations

import random
from typing import cast

import torch
import torchvision.transforms.functional as transforms_F

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.vfm.datasets.augmentors.transfer_control_input import AddControlInputComb
from cosmos3._src.vfm.datasets.sequence_packing import SequencePlan
from cosmos3._src.vfm.datasets.utils import VIDEO_RES_SIZE_INFO


class SampleResolution(Augmentor):
    """Sample one resolution from a list and set data_dict['_res_size_map'] for downstream resize/padding.

    When used before ResizeLargestSideAspectPreserving and ReflectionPadding, those augmentors will
    use obtain_augmentation_size(), which reads _res_size_map when present. This allows one dataloader
    to produce samples at different resolutions (e.g. 480 and 720) by sampling per sample.

    resolutions_weights: Optional sampling weights for each resolution (same length as resolutions).
    Weights are used by random.choices and need not sum to 1. If None, sampling is uniform.
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: list | None = None,
        args: dict | None = None,
        resolutions: list[str] | None = None,
        resolutions_weights: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.resolutions = list(resolutions) if resolutions else []
        assert len(self.resolutions) > 0, "SampleResolution requires at least one resolution."
        for r in self.resolutions:
            assert r in VIDEO_RES_SIZE_INFO, f"Unknown resolution {r}; known: {list(VIDEO_RES_SIZE_INFO.keys())}"
        self.resolutions_weights = resolutions_weights
        if self.resolutions_weights is not None:
            assert len(self.resolutions_weights) == len(self.resolutions), (
                "resolutions_weights must have same length as resolutions."
            )

    def __call__(self, data_dict: dict) -> dict:
        if self.resolutions_weights is not None:
            res = random.choices(self.resolutions, weights=self.resolutions_weights, k=1)[0]
        else:
            res = random.choice(self.resolutions)
        data_dict["_res_size_map"] = VIDEO_RES_SIZE_INFO[res]
        return data_dict


class TransferToTrainingFormat(Augmentor):
    """Convert (control_input, target) into joint-dataloader training format with SequencePlan.

    Reads data_dict["control_input"] and data_dict["video"] (target). Normalizes control to
    mean/std 0.5, then writes [control_tensor, target_tensor] into data_dict[output_media_key]
    ("images" for image transfer, "video" for video transfer). Also sets num_frames,
    dataset_name, fps, ai_caption, selected_caption_type, sequence_plan, and image_size.

    Supports both image (3D: C,H,W) and video (4D: C,T,H,W); for image output, 4D tensors
    are sliced to the first frame. Same output structure as ImageEditingToTrainingFormat.
    """

    def __init__(
        self,
        input_keys: list | None = None,
        mean: float = 0.5,
        std: float = 0.5,
        output_media_key: str = "images",
        conditioning_config: dict[int, float] | None = None,
        share_vision_temporal_positions: bool = True,
        args: dict | None = None,
    ) -> None:
        super().__init__(input_keys or [], None, args)
        self.mean = mean
        self.std = std
        self.output_media_key = output_media_key
        self.conditioning_config = conditioning_config
        self.share_vision_temporal_positions = share_vision_temporal_positions

        if self.conditioning_config is not None:
            for num_frames, prob in self.conditioning_config.items():
                if not isinstance(num_frames, int) or num_frames < 0:
                    raise ValueError(f"conditioning_config keys must be non-negative integers, got {num_frames}")
                if not isinstance(prob, (int, float)) or prob < 0:
                    raise ValueError(f"conditioning_config values must be non-negative numbers, got {prob}")
            total_prob = sum(self.conditioning_config.values())
            if total_prob <= 0:
                raise ValueError("conditioning_config probabilities must sum to a positive number")
            self.normalized_conditioning_config = {k: v / total_prob for k, v in self.conditioning_config.items()}
        else:
            self.normalized_conditioning_config = None

    def _normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize channel-wise to given mean/std. Accepts values in [0,1] or [0,255] (auto-detected)."""
        if x.dtype == torch.uint8 or x.max() > 1.0:
            x = x.float() / 255.0
        return transforms_F.normalize(x, mean=[self.mean] * 3, std=[self.std] * 3)

    def __call__(self, data_dict: dict) -> dict | None:
        control_norm = data_dict.get("control_input")
        target_norm = data_dict.get("video")

        if control_norm is None or target_norm is None:
            log.warning(
                f"TransferToTrainingFormat: missing control or target (video): {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None

        try:
            if control_norm.dim() == 2:
                control_norm = control_norm.unsqueeze(0).expand(3, -1, -1)  # [3,H,W]
            # is_video = control.dim() == 4 and isinstance(target_norm, torch.Tensor) and target_norm.dim() == 4
            if self.output_media_key == "video":
                # Video: (C, T, H, W) each; normalize per frame
                # control_norm = self._normalize_tensor(control.float())
                num_frames = control_norm.shape[1]
                data_dict["video"] = [control_norm, target_norm]
                data_dict["num_frames"] = num_frames
                data_dict["dataset_name"] = "video_transfer"
                data_dict["fps"] = data_dict.get("fps", 24.0)
            else:
                # Image: (C, H, W)
                if target_norm.dim() == 4:
                    target_norm = target_norm[:, 0]
                if control_norm.dim() == 4:
                    control_norm = control_norm[:, 0]
                # control_norm = self._normalize_tensor(control.float())
                data_dict["images"] = [control_norm, target_norm]
                data_dict["num_frames"] = 2
                data_dict["dataset_name"] = "image_transfer"
                data_dict["fps"] = 30.0
            data_dict.setdefault("ai_caption", "")
            data_dict.setdefault("selected_caption_type", "transfer_caption")

            num_condition_frames = 0
            if self.normalized_conditioning_config is not None:
                frames_options = list(self.normalized_conditioning_config.keys())
                weights = list(self.normalized_conditioning_config.values())
                num_condition_frames = random.choices(frames_options, weights=weights, k=1)[0]
                if self.output_media_key == "video" and target_norm.dim() == 4:
                    max_cond = target_norm.shape[1] - 1
                    num_condition_frames = min(num_condition_frames, max_cond)

            if num_condition_frames > 0 and target_norm.shape[1] > 1:
                condition_frames_indexes = list(range(num_condition_frames))
            else:
                condition_frames_indexes = []

            data_dict["sequence_plan"] = SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=condition_frames_indexes,
                # ControlNet-style transfer: control item and target item are
                # spatio-temporally aligned (same source video, frame-synced).
                # Forces shared temporal mRoPE grid across both items so the
                # model sees control_t=k and target_t=k as the same time index.
                share_vision_temporal_positions=self.share_vision_temporal_positions,
            )
        except Exception as e:
            log.warning(
                f"TransferToTrainingFormat error: {data_dict.get('__key__', 'unknown')}, {e}",
                rank0_only=False,
            )
            return None

        # duplicate image_size for each vision input/output
        data_dict["image_size"] = [data_dict["image_size"]] * len(data_dict[self.output_media_key])

        return data_dict


class VideoTransferSampleFrame(Augmentor):
    """Sample a single frame index from video tensors for image→image transfer.

    For each key in input_keys (default ["control_input", "video"]), resolves the
    tensor (e.g. unwraps data_dict["video"]["video"]). Picks one temporal index t
    (random if random_frame=True, else 0) and for every 4D tensor (C, T, H, W)
    replaces it in-place with the slice at t, yielding (C, 1, H, W). 3D tensors
    are left unchanged. All keys must be present; returns None if any is missing.
    """

    def __init__(
        self,
        input_keys: list | None = None,
        args: dict | None = None,
        random_frame: bool = True,
    ) -> None:
        self.input_keys = input_keys or ["control_input", "video"]
        super().__init__(self.input_keys, None, args)
        self.random_frame = random_frame

    def _get_tensor(self, data_dict: dict, key: str) -> torch.Tensor | None:
        """Return the tensor for key; if key is 'video' and value is a dict, return value['video']."""
        val = data_dict.get(key)
        if val is None:
            return None
        if isinstance(val, dict) and key == "video":
            return val.get("video")
        return val

    def __call__(self, data_dict: dict) -> dict | None:
        # Resolve tensors; find T from first 4D tensor. Require all keys present.
        tensors: list[tuple[str, torch.Tensor]] = []
        T: int | None = None
        for key in self.input_keys:
            raw = self._get_tensor(data_dict, key)
            if raw is None or not isinstance(raw, torch.Tensor):
                return None
            tensor = cast(torch.Tensor, raw)
            if tensor.dim() == 4:
                if T is None:
                    T = tensor.shape[1]
                    if T == 0:
                        return None
                tensors.append((key, tensor))
            else:
                tensors.append((key, tensor))

        if T is None:
            # No 4D tensor; nothing to sample
            return data_dict

        t_idx = random.randint(0, T - 1) if self.random_frame else 0

        for key, tensor in tensors:
            if tensor.dim() == 4:
                sampled = tensor[:, t_idx : t_idx + 1]
            else:
                sampled = tensor
            data_dict[key] = sampled

        return data_dict


class AddControlFromVideoComb(Augmentor):
    """Compute one control signal from video via AddControlInputComb and set control_input.

    Delegates to AddControlInputComb (edge/blur computed from video; depth/seg from data_dict
    when present). After the comb runs, selects the first non-zero control among
    control_input_edge, control_input_blur, control_input_depth, control_input_seg,
    writes it to data_dict["control_input"], and removes the temporary control keys.

    Args:
        control_input_type: e.g. "edge_blur", "edge_blur_depth_seg" (which controls to consider).
        num_control_inputs_prob: Probability distribution over number of combined controls;
            this wrapper uses only the single chosen control.
    """

    CONTROL_KEYS = ("control_input_edge", "control_input_blur", "control_input_depth", "control_input_seg")

    def __init__(
        self,
        input_keys: list,
        output_keys: list | None = None,
        args: dict | None = None,
        control_input_type: str = "edge_blur_depth_seg",
        use_random: bool = True,
        num_control_inputs_prob: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
        num_control_inputs: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys or ["control_input"], args)
        self._comb = AddControlInputComb(
            input_keys=input_keys,
            output_keys=None,
            use_random=use_random,
            control_input_type=control_input_type,
            num_control_inputs_prob=list(num_control_inputs_prob),
            num_control_inputs=num_control_inputs,
            **kwargs,
        )

    def __call__(self, data_dict: dict) -> dict | None:
        data_dict = self._comb(data_dict)
        if data_dict is None:
            return None
        # Pick first control key that exists and has non-zero data (comb sets unchosen to zeros).
        for key in self.CONTROL_KEYS:
            if key in data_dict:
                t = data_dict[key]
                if isinstance(t, torch.Tensor) and t.numel() > 0 and t.abs().sum() > 0:
                    data_dict["control_input"] = t
                    break
        else:
            # No break: no valid control found (e.g. all chosen controls failed or are zero).
            log.warning("AddControlFromVideoComb: no non-zero control found", rank0_only=False)
            return None
        for key in self.CONTROL_KEYS:
            data_dict.pop(key, None)
            data_dict.pop(key + "_mask", None)
        return data_dict
