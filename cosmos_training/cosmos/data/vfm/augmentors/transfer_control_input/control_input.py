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
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from pycocotools import mask as mask_utils

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.utils import log
from cosmos.data.vfm.augmentors.transfer_control_input.blur import Blur, BlurConfig
from cosmos.data.vfm.augmentors.transfer_control_input.seg import (
    decode_partial_rle_width1,
    segmentation_color_mask,
)

# Constants for segmentation color processing
# These parameters control the color-based mask extraction process in AddControlInputSeg

# Color quantization bin size for grouping similar colors together
# Range: 1-100 (smaller values = more granular color detection, larger values = more color grouping)
# Typical range: 10-50, where 25 provides good balance between precision and grouping
_BIN_SIZE = 25

# Maximum number of unique colors to examine for mask generation (to limit computation time)
# Range: 10-500 (smaller values = faster processing, larger values = more thorough color search)
# Typical range: 50-200, where 100 balances thoroughness with performance
_MAX_UNIQUE_COLORS = 100

# Color distance tolerance for considering pixels as the same color
# Range: 1-100 (smaller values = stricter color matching, larger values = more lenient matching)
# Typical range: 10-60, where 30 provides good tolerance for natural color variations
_COLOR_TOLERANCE = 30

# RGB value threshold below which a color is considered "black" and filtered out
# Range: 0-100 (smaller values = stricter black detection, larger values = more colors considered black)
# Typical range: 20-80, where 50 effectively filters out dark/black regions
_BLACK_THRESHOLD = 50


def _maybe_torch_to_numpy(frames: Union[torch.Tensor, list]) -> np.ndarray:
    try:
        return frames.numpy()
    except AttributeError:
        return np.array(frames)


class AddControlInputEdge(Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_edge"],
        args: Optional[dict] = None,
        use_random: Optional[bool] = True,
        preset_strength: Optional[str] = "medium",
        edge_t_lower: Optional[int] = None,
        edge_t_upper: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.use_random = use_random
        self.preset_strength = preset_strength
        self.t_lower = edge_t_lower
        self.t_upper = edge_t_upper

    def __call__(self, data_dict: dict) -> dict:
        if "control_input_edge" in data_dict:
            return data_dict
        key_img = self.input_keys[0]
        key_out = self.output_keys[0]
        frames = data_dict[key_img]
        # log.info(f"Adding control input edge. Input key: {key_img}, Output key: {key_out}. Use random: {self.use_random}, Preset strength: {self.preset_strength}")
        # Get lower and upper threshold for canny edge detection.
        if self.use_random:  # always on for training, always off for inference
            if self.t_lower is not None and self.t_upper is not None:
                # Use provided t_lower and t_upper values
                t_lower = self.t_lower
                t_upper = self.t_upper
            else:
                # Generate random values as before
                t_lower = np.random.randint(20, 100)  # Get a random lower thre
                t_diff = np.random.randint(50, 150)  # Get a random diff between lower and upper
                t_upper = t_lower + t_diff  # The upper thre is lower added by the diff
        else:
            if self.preset_strength == "none" or self.preset_strength == "very_low":
                t_lower, t_upper = 20, 50
            elif self.preset_strength == "low":
                t_lower, t_upper = 50, 100
            elif self.preset_strength == "medium":
                t_lower, t_upper = 100, 200
            elif self.preset_strength == "high":
                t_lower, t_upper = 200, 300
            elif self.preset_strength == "very_high":
                t_lower, t_upper = 300, 400
            else:
                raise ValueError(f"Preset {self.preset_strength} not recognized.")

        frames = _maybe_torch_to_numpy(frames).astype(np.uint8)
        is_image = len(frames.shape) < 4

        # Compute the canny edge map by the two thresholds.
        if is_image:
            edge_maps = cv2.Canny(frames, t_lower, t_upper)[None, None]
        else:
            edge_maps = [
                cv2.Canny(img, t_lower, t_upper) for img in frames.transpose((1, 2, 3, 0))
            ]  # (C, T, H, W) -> (T, H, W)
            edge_maps = np.stack(edge_maps)[None]
        edge_maps = torch.from_numpy(edge_maps).expand(3, -1, -1, -1)  # [3,T,H,W]
        if is_image:
            edge_maps = edge_maps[:, 0]
        data_dict[key_out] = edge_maps
        return data_dict


class AddControlInputBlur(Augmentor):
    """
    Main class for adding blurred input to the data dictionary.
    self.output_keys[0] indicates the types of blur added to the input.
    For example, control_input_gaussian_guided indicates that both Gaussian and Guided filters are applied
    """

    def __init__(
        self,
        input_keys: list,  # [key_load, key_img]
        output_keys: Optional[list] = ["control_input_blur"],
        args: Optional[dict] = None,  # not used
        use_random: bool = True,  # whether to use random parameters
        blur_config: BlurConfig | None = None,
        downup_preset: str | int = "medium",  # preset strength for downup factor
        min_downup_factor: int = 4,  # minimum downup factor
        max_downup_factor: int = 16,  # maximum downup factor
        downsize_before_blur: bool = True,  # whether to downsize before applying blur and then upsize or downup after blur
        blur_downsize_factor: list[int] = list(range(1, 5)),  # downscale factor for blur
        resize_cuda: bool = False,  # whether to do resizing on GPU, the result is still moved back to CPU for compatibility.
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.use_random = use_random
        downup_preset_values = {
            "none": 1,
            "very_low": min_downup_factor,
            "low": min_downup_factor,
            "medium": (min_downup_factor + max_downup_factor) // 2,
            "high": max_downup_factor,
            "very_high": max_downup_factor,
        }
        blur_downup_preset_values = {
            "none": 1,
            "very_low": 1,
            "low": 4,
            "medium": 2,
            "high": 1,
            "very_high": 4,
        }
        self.blur = Blur(config=blur_config, use_random=use_random)

        self.preset_strength = downup_preset
        self.downup_preset = downup_preset if isinstance(downup_preset, int) else downup_preset_values[downup_preset]
        self.downsize_before_blur = downsize_before_blur
        self.min_downup_factor = min_downup_factor
        self.max_downup_factor = max_downup_factor
        self.blur_downsize_factor = blur_downsize_factor
        self.blur_downup_preset = blur_downup_preset_values[downup_preset]
        self.resize_cuda = resize_cuda
        assert not (self.use_random and self.resize_cuda), "Cannot use resize on GPU during training."

    def _load_frame(self, data_dict: dict) -> tuple[np.ndarray, bool]:
        key_img = self.input_keys[0]
        frames = data_dict[key_img]
        frames = _maybe_torch_to_numpy(frames)
        is_image = False
        if len(frames.shape) < 4:
            frames = frames.transpose((2, 0, 1))[:, None]
            is_image = True
        return frames, is_image

    def __call__(self, data_dict: dict) -> dict:
        if "control_input_blur" in data_dict:
            # already processed
            data_dict[self.output_keys[0]] = data_dict["control_input_blur"]
            return data_dict

        key_out = self.output_keys[0]

        frames, is_image = self._load_frame(data_dict)
        if self.preset_strength == "none":
            if is_image:
                frames = frames[:, 0]
            data_dict[key_out] = torch.from_numpy(frames)
            return data_dict
        C, T, H, W = frames.shape

        # --- 1. Downscale Before Blur ---
        if self.use_random:
            downscale_factor = random.choice(self.blur_downsize_factor)
        else:
            downscale_factor = self.blur_downup_preset

        if self.downsize_before_blur:
            new_W, new_H = W // downscale_factor, H // downscale_factor
            downscaled = np.empty((C, T, new_H, new_W), dtype=frames.dtype)

            for t in range(T):
                frame = np.ascontiguousarray(frames[:, t].transpose(1, 2, 0))
                resized = cv2.resize(frame, (new_W, new_H), interpolation=cv2.INTER_AREA)
                if len(resized.shape) == 2:
                    resized = resized[..., None]
                downscaled[:, t] = resized.transpose(2, 0, 1)

            frames = downscaled
        # --------------------------

        # --- 2. Apply Blur ---
        frames = self.blur(frames)
        # --------------------------

        # --- 3. Upscale After Blur ---
        if self.downsize_before_blur:
            upscaled = np.empty((C, T, H, W), dtype=frames.dtype)
            for t in range(T):
                frame = np.ascontiguousarray(frames[:, t].transpose(1, 2, 0))
                resized = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
                if len(resized.shape) == 2:
                    resized = resized[..., None]
                upscaled[:, t] = resized.transpose(2, 0, 1)
            frames = upscaled

        # --- 4. Final Downup Augmentation ---
        if self.use_random:
            scale_factor = random.randint(self.min_downup_factor, self.max_downup_factor + 1)
        else:
            scale_factor = self.downup_preset

        final_W, final_H = int(W / scale_factor), int(H / scale_factor)

        final_frames = np.empty_like(frames)
        for t in range(T):
            frame = np.ascontiguousarray(frames[:, t].transpose(1, 2, 0))
            small = cv2.resize(frame, (final_W, final_H), interpolation=cv2.INTER_CUBIC)
            large = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
            if len(large.shape) == 2:
                large = large[..., None]
            final_frames[:, t] = large.transpose(2, 0, 1)

        if is_image:
            final_frames = final_frames[:, 0]
        data_dict[key_out] = torch.from_numpy(final_frames)

        return data_dict


class AddControlInputDepth(Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_depth"],
        args: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        if "control_input_depth" in data_dict:
            # already processed
            return data_dict

        key_out = self.output_keys[0]
        depth = data_dict["depth"]

        frames = data_dict["video"]
        _, T, H, W = frames.shape
        depth = transforms_F.resize(
            depth,
            size=(H, W),
            interpolation=transforms_F.InterpolationMode.BILINEAR,
        )
        data_dict[key_out] = depth
        return data_dict


class AddControlInputSeg(Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_seg"],
        thres_mb_python_decode: Optional[int] = 256,  # required: <= 512 for 7b
        use_fixed_color_list: bool = False,
        num_masks_max: int = 100,
        random_sample_num_masks: bool = True,
        min_mask_size: float = 0.2,
        args: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            thres_mb_python_decode: int, threshold of memory usage for python decode, in MB
            use_fixed_color_list: bool, if True, use predefined colors for segmentation masks. If False, generate random colors for segmentation masks.
            num_masks_max: int, maximum number of masks to sample
            random_sample_num_masks: bool, if True, sample number of masks randomly. If False, sample all masks in the data.
            min_mask_size: float, minimum size of the mask area, in fraction of the entire frame.
        """
        super().__init__(input_keys, output_keys, args)
        self.use_fixed_color_list = use_fixed_color_list
        self.num_masks_max = num_masks_max
        self.thres_mb_python_decode = thres_mb_python_decode
        self.random_sample_num_masks = random_sample_num_masks
        self.min_mask_size = min_mask_size

    def get_masks(self, data_dict: dict, num_masks: int = 1) -> tuple[torch.Tensor, bool]:
        """
        Get a single mask from the data dictionary.
        segmentation: list of dicts
            phrase: str
            segmentation_mask_rle: dict
                data: dict
                    size: [N, 1]
                    counts: bytes
                mask_shape: [T, H, W]
        """
        frames = data_dict["video"]
        _, T, H, W = frames.shape

        if not isinstance(data_dict["segmentation"], dict) and num_masks == 1:
            # this video is a color-coded segmentation mask, where each color corresponds to a different object
            # we need to extract the binary mask for a single object from the video
            seg_video = data_dict["segmentation"]
            seg_video = transforms_F.resize(
                seg_video,
                size=(H, W),
                interpolation=transforms_F.InterpolationMode.NEAREST,
            )
            # Get the first frame of the segmentation video
            first_frame = seg_video[:, 0, :, :]
            # Get a list of unique colors from the first frame and calculate mask size for each unique color
            unique_colors = (first_frame // _BIN_SIZE).view(3, -1).permute(1, 0).unique(
                dim=0
            ) * _BIN_SIZE  # [N_colors,3]
            # Randomly shuffle unique colors and take first N colors
            perm = torch.randperm(len(unique_colors))
            unique_colors = unique_colors[perm]
            unique_colors = unique_colors[:_MAX_UNIQUE_COLORS]  # check up to max colors to save time
            mask_sizes = []
            for color in unique_colors:
                color_diff = first_frame.to(torch.float32) - color[:, None, None]
                color_dists = torch.sqrt(torch.sum(color_diff**2, dim=0))
                mask = color_dists < _COLOR_TOLERANCE
                mask_size = mask.sum() / (H * W)  # Size as fraction of frame
                mask_sizes.append(mask_size)

            # Only keep colors that produce masks >= min_mask_size of frame and not black
            valid_color_indices = [
                i
                for i, size in enumerate(mask_sizes)
                if size >= self.min_mask_size and (unique_colors[i] > _BLACK_THRESHOLD).sum() > 0
            ]
            if len(valid_color_indices) == 0:
                # If no masks are large enough, return all ones
                log.critical("No masks are large enough, returning all ones")
                all_masks = np.ones((num_masks, T, H, W)).astype(bool)
                return torch.from_numpy(all_masks), False  # [num_masks,T,H,W]
            else:
                # Randomly select one of the valid large masks
                valid_color_idx = valid_color_indices[np.random.randint(len(valid_color_indices))]
                target_color = unique_colors[valid_color_idx]
                # Create binary mask where True means within tolerance of target color
                color_diff = seg_video.to(torch.float32) - target_color[:, None, None, None]
                color_dists = torch.sqrt(torch.sum(color_diff**2, dim=0, keepdim=True))  # [1,T,H,W]
                mask = (color_dists < _COLOR_TOLERANCE).to(torch.bool)  # [1,T,H,W]
                return mask, True
        frame_indices = data_dict["frame_indices"]
        frame_start, frame_end = frame_indices[0], frame_indices[-1] + 1
        is_continuous_frame_indices = (frame_end - frame_start) == T
        assert len(frame_indices) == T, (
            f"frame_indices length {len(frame_indices)} != T {T}, likely due to video decoder using different fps, i.e. sample with stride. Need to return frame indices from video decoder."
        )

        all_masks = np.ones((num_masks, T, H, W)).astype(bool)  # [num_masks,T,H,W]

        # sample number of masks
        mask_ids = np.arange(len(data_dict["segmentation"])).tolist()
        if len(data_dict["segmentation"]) == 0 or num_masks == 0:
            return torch.from_numpy(all_masks), False  # [num_masks,T,H,W]
        if num_masks == 1:  # Try up to 16 masks to find a large enough mask
            mask_ids_select = np.random.choice(mask_ids, min(len(mask_ids), 16), replace=False)
        else:
            mask_ids_select = np.random.choice(mask_ids, num_masks, replace=False)

        for idx, mid in enumerate(mask_ids_select):
            mask = data_dict["segmentation"][mid]
            if type(mask) != dict:  # data has sharding issue, skip this mask
                return torch.from_numpy(all_masks), False  # [num_masks,T,H,W]
            shape = mask["segmentation_mask_rle"]["mask_shape"]
            num_byte_per_mb = 1024 * 1024
            # total number of elements in uint8 (1 byte) / num_byte_per_mb
            if mask["segmentation_mask_rle"]["data"]["size"][0] / num_byte_per_mb > self.thres_mb_python_decode:
                # Switch to python decode if the mask is too large to avoid out of shared memory
                if is_continuous_frame_indices and (
                    T * shape[1] * shape[2] / num_byte_per_mb <= self.thres_mb_python_decode
                ):
                    log.critical(
                        f"Using python decode for mask of shape {shape}, Continuous frame indices, frame_start: {frame_start}, frame_end: {frame_end}"
                    )
                    rle = decode_partial_rle_width1(
                        mask["segmentation_mask_rle"]["data"],
                        frame_start * shape[1] * shape[2],
                        frame_end * shape[1] * shape[2],
                    )
                    partial_shape = (frame_end - frame_start, shape[1], shape[2])
                    rle = rle.reshape(partial_shape) * 255
                    rle = np.stack(
                        [cv2.resize(_image_np, (W, H), interpolation=cv2.INTER_NEAREST) for _image_np in rle]
                    )
                else:  # need to call decode_partial_rle_width1 multiple times
                    # It takes too much time to decode the mask, so we skip it and select another modality instead
                    log.critical(f"Skipping python decode for mask of shape {shape}")
                    return torch.from_numpy(all_masks), False  # [num_masks,T,H,W]
            else:
                rle = mask_utils.decode(mask["segmentation_mask_rle"]["data"])
                rle = rle.reshape(shape) * 255
                # Select the frames that are in the video
                if len(rle) < frame_end:  # Pad the mask if it is shorter than original video
                    rle = np.vstack([rle, [rle[-1]] * (frame_end - len(rle))])
                rle = np.stack([cv2.resize(rle[i], (W, H), interpolation=cv2.INTER_NEAREST) for i in frame_indices])
            if num_masks == 1:  # if we only need one mask and the current mask is large enough, return it
                if (rle > 0).sum() / rle.size >= self.min_mask_size:
                    # log.critical(f"Found a large enough mask with size {(rle > 0).sum() / rle.size}")
                    all_masks[0] = rle.astype(bool)
                    break
                elif idx == len(mask_ids_select) - 1:
                    log.critical("No large enough mask found, returning all ones")
            else:  # if we need multiple masks, return all masks
                all_masks[idx] = rle.astype(bool)
            del rle
        return torch.from_numpy(all_masks), True  # [num_masks,T,H,W]

    def __call__(self, data_dict: dict) -> dict:
        if "control_input_seg" in data_dict:
            # already processed
            return data_dict

        key_out = self.output_keys[0]
        if not isinstance(data_dict["segmentation"], dict):
            # already have a color-coded segmentation mask video, directly use it
            seg = data_dict["segmentation"]
            seg = transforms_F.resize(
                seg,
                size=data_dict["video"].shape[-2:],
                interpolation=transforms_F.InterpolationMode.NEAREST,
            )
            data_dict[key_out] = seg
            return data_dict

        # sample number of masks
        if self.random_sample_num_masks:
            num_masks = np.random.randint(0, min(self.num_masks_max + 1, len(data_dict["segmentation"]) + 1))
        else:
            num_masks = len(data_dict["segmentation"])

        all_masks, success = self.get_masks(data_dict, num_masks)

        if not success:
            data_dict["preprocess_failed"] = True
            del all_masks  # free memory
            return data_dict

        key_out = self.output_keys[0]
        # control_input_seg is the colored segmentation mask, value in [0,255], shape (3, T, H, W)
        data_dict[key_out] = torch.from_numpy(
            segmentation_color_mask(all_masks, self.use_fixed_color_list)
        )  # [3,T,H,W]
        if num_masks > 0:
            data_dict[key_out + "_mask"] = all_masks[random.randint(0, num_masks - 1)].clone()[None]  # [1,T,H,W]
        del all_masks  # free memory
        return data_dict


class AddControlInputIdentity(Augmentor):
    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_identity"],
        args: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        key_img = self.input_keys[0]
        key_out = self.output_keys[0]
        frames = _maybe_torch_to_numpy(data_dict[key_img])  # CTHW for video, HWC for image
        is_image = len(frames.shape) < 4
        if is_image:
            frames = frames.transpose((2, 0, 1))
        data_dict[key_out] = torch.from_numpy(frames).clone()  # [C,T,H,W] for video, [C,H,W] for image
        return data_dict


class AddControlInputHdmapBbox(Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_hdmap_bbox"],
        args: Optional[dict] = None,
        use_random: Optional[bool] = True,
        preset_strength: Optional[str] = "medium",
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.use_random = use_random
        self.preset_strength = preset_strength

    def __call__(self, data_dict: dict) -> dict:
        if "control_input_hdmap_bbox" in data_dict:
            return data_dict
        key_input = self.input_keys[0]
        key_out = self.output_keys[0]
        data_dict[key_out] = data_dict[key_input]
        return data_dict


CTRL_HINT_KEYS = {
    "control_input_edge": AddControlInputEdge,
    "control_input_blur": AddControlInputBlur,
    "control_input_depth": AddControlInputDepth,
    "control_input_seg": AddControlInputSeg,
    "control_input_inpaint": AddControlInputIdentity,
    "control_input_hdmap_bbox": AddControlInputHdmapBbox,
}


class AddControlInputComb(Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = None,
        args: Optional[dict] = None,
        use_random: bool = True,
        control_input_type: str = "edge_blur_depth_seg",
        use_control_mask_prob: float = 0.0,
        num_control_inputs_prob: list[float] = [1.0, 0.0, 0.0, 0.0],
        num_control_inputs: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.use_random = use_random
        self.control_hint_keys = ["control_input_" + key for key in control_input_type.split("_")]
        self.use_control_mask_prob = use_control_mask_prob
        self.num_control_inputs_prob = num_control_inputs_prob[: len(self.control_hint_keys)]
        self.num_control_inputs = num_control_inputs
        self.comb = {}
        for output_key, class_name in CTRL_HINT_KEYS.items():
            aug = class_name(
                input_keys=input_keys, output_keys=[output_key], args=args, use_random=use_random, **kwargs
            )
            self.comb[output_key] = aug

    def __call__(self, data_dict: dict) -> dict:
        success = False
        if self.use_random:
            # Randomly select a number of control inputs (or use fixed num_control_inputs when set)
            num_keys_prob = self.num_control_inputs_prob
            ctrl_hint_keys = self.control_hint_keys
            if self.num_control_inputs is not None:
                num_keys = max(1, min(self.num_control_inputs, len(ctrl_hint_keys)))
            else:
                num_keys = random.choices(range(len(ctrl_hint_keys)), weights=num_keys_prob, k=1)[0] + 1
            output_keys = np.random.choice(ctrl_hint_keys, size=num_keys, replace=False)
            # output_keys = np.random.choice(["control_input_edge", "control_input_blur", "control_input_depth"], size=num_keys, replace=False)
            zero_input = torch.zeros_like(data_dict[self.input_keys[0]])  # [C,T,H,W]
            zero_mask = torch.zeros(*data_dict[self.input_keys[0]][:1].shape, dtype=torch.bool)  # [1,T,H,W]
            ones_mask = torch.ones(*data_dict[self.input_keys[0]][:1].shape, dtype=torch.bool)  # [1,T,H,W]
            use_control_mask = random.random() < self.use_control_mask_prob
            for cur_key in ctrl_hint_keys:
                cur_mask_key = cur_key + "_mask"
                if cur_key in output_keys:
                    data_dict["preprocess_failed"] = False
                    data_dict = self.comb[cur_key](data_dict)
                    # log.critical(f"self.use_control_mask_prob: {self.use_control_mask_prob}")
                    if use_control_mask or cur_key == "control_input_inpaint":
                        # Get mask for the control input
                        if cur_mask_key not in data_dict:
                            data_dict[cur_mask_key], success = self.comb["control_input_seg"].get_masks(
                                data_dict, num_masks=1
                            )
                    else:
                        data_dict[cur_mask_key] = ones_mask

                    # If preprocess failed or cannot get inpaint mask, use control_input_edge instead
                    if data_dict["preprocess_failed"] or (cur_key == "control_input_inpaint" and not success):
                        data_dict[cur_key] = zero_input
                        data_dict[cur_mask_key] = zero_mask
                        if num_keys == 1 and "control_input_edge" in ctrl_hint_keys:
                            new_key = "control_input_edge"
                            log.critical(f"Preprocess failed for {cur_key}, using {new_key} instead")
                            if new_key in data_dict:
                                del data_dict[new_key]
                            data_dict = self.comb[new_key](data_dict)
                            data_dict[new_key + "_mask"] = ones_mask
                else:
                    data_dict[cur_key] = zero_input
                    data_dict[cur_mask_key] = zero_mask

            # Free memory: remove unused depth/segmentation to avoid OOM later
            if "control_input_depth" not in output_keys and "depth" in data_dict:
                del data_dict["depth"]
            if "control_input_seg" not in output_keys and "segmentation" in data_dict:
                del data_dict["segmentation"]
            if "segmentation" in data_dict and isinstance(data_dict["segmentation"], dict):
                del data_dict["segmentation"]

            if "control_input_inpaint" in output_keys and success:  # Post-process the inpaint mask
                inpaint_mask_key = "control_input_inpaint_mask"
                if random.random() < 0.5:  # randomly negate the mask
                    data_dict[inpaint_mask_key] = ~data_dict[inpaint_mask_key]
                # Make sure the inpaint mask does not overlap with other masks
                for cur_key in ctrl_hint_keys:
                    cur_mask_key = cur_key + "_mask"
                    if cur_mask_key == inpaint_mask_key:
                        continue
                    if torch.all(data_dict[cur_mask_key]) or torch.all(~data_dict[cur_mask_key]):  # dummy mask
                        continue
                    # Remove overlap by zeroing overlapping regions in mask1
                    overlap = data_dict[cur_mask_key] & data_dict[inpaint_mask_key]
                    if overlap.any():
                        data_dict[inpaint_mask_key] = data_dict[inpaint_mask_key] & ~overlap

        else:
            for k, v in self.comb.items():
                data_dict = v(data_dict)
        return data_dict
