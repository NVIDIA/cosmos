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

import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.io
import torchvision.transforms.functional as TF
from PIL import Image

from cosmos3._src.vfm.datasets.sequence_packing import SequencePlan
from cosmos3._src.vfm.datasets.utils import VIDEO_RES_SIZE_INFO


def resize_pil_image(image: Image.Image, max_size: int, padding_constant: int) -> Image.Image:
    """Resize a PIL image so the max side length is at most *max_size* and both
    dimensions are divisible by *padding_constant*.

    Args:
        image: Input PIL image.
        max_size: Maximum allowed side length (longest edge will be at most this).
        padding_constant: Both height and width are rounded down to the nearest
            multiple of this value.

    Returns:
        Resized PIL image.
    """
    orig_w, orig_h = image.size
    scale = max_size / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    new_w = (new_w // padding_constant) * padding_constant
    new_h = (new_h // padding_constant) * padding_constant
    new_w = max(new_w, padding_constant)
    new_h = max(new_h, padding_constant)
    return image.resize(
        (new_w, new_h),
        Image.LANCZOS,  # type: ignore
    )


def _resize_and_center_crop(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Aspect-ratio-preserving resize followed by center crop."""
    orig_h, orig_w = frames.shape[2], frames.shape[3]
    scaling_ratio = max(target_w / orig_w, target_h / orig_h)
    resize_h = int(math.ceil(scaling_ratio * orig_h))
    resize_w = int(math.ceil(scaling_ratio * orig_w))
    frames = TF.resize(frames, [resize_h, resize_w])  # [...,resize_h,resize_w]
    frames = TF.center_crop(frames, [target_h, target_w])  # [...,target_h,target_w]
    return frames


def load_conditioning_image(image_path: Path, target_h: int, target_w: int) -> torch.Tensor:
    """Load an image as conditioning frames from local or remote path; returns (3, 1, H, W) in [-1, 1]."""
    with image_path.open("rb") as f:
        img = Image.open(f).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0)  # [1,3,H,W]
    img_tensor = _resize_and_center_crop(img_tensor, target_h, target_w)  # [1,3,target_h,target_w]
    img_tensor = img_tensor.squeeze(0) / 127.5 - 1.0  # [3,target_h,target_w]
    return img_tensor.unsqueeze(1)  # [3,1,target_h,target_w]


def load_conditioning_video(
    video_path: Path,
    target_h: int,
    target_w: int,
    max_frames: int,
    *,
    keep: Literal["first", "last"] = "first",
) -> torch.Tensor:
    """Load video frames for conditioning; returns (3, T, H, W) in [-1, 1].

    ``keep`` selects which ``max_frames`` to take when the input is longer.
    """
    frames, _, _ = torchvision.io.read_video(str(video_path), pts_unit="sec")
    frames = frames[-max_frames:] if keep == "last" else frames[:max_frames]  # [T,H,W,3]
    frames_tchw = frames.permute(0, 3, 1, 2).float()  # [T,3,H,W]
    frames_resized = _resize_and_center_crop(frames_tchw, target_h, target_w)  # [T,3,target_h,target_w]
    frames_normalized = frames_resized / 127.5 - 1.0  # [T,3,target_h,target_w]
    return frames_normalized.permute(1, 0, 2, 3)  # [3,T,target_h,target_w]


def pil_to_conditioning_frames(pil_img: Image.Image) -> tuple[torch.Tensor, int, int]:
    """Convert a PIL image to a conditioning tensor in [-1, 1] and return (frames, h, w)."""
    w, h = pil_img.size
    img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float()  # [3,H,W]
    return (img_tensor / 127.5 - 1.0).unsqueeze(1), h, w  # [3,1,H,W]


def build_conditioned_video_batch(
    conditioning_frames: torch.Tensor,
    condition_frames_vision: list[int],
    w: int,
    h: int,
    num_frames: int,
    fps: float,
    batch_size: int = 1,
) -> dict:
    """Build a data batch with conditioning frames and sequence plans for generation."""
    t_cond = conditioning_frames.shape[1]
    video_data = torch.zeros(1, 3, num_frames, h, w, dtype=torch.bfloat16)  # [1,3,num_frames,h,w]
    t_fill = min(t_cond, num_frames)
    video_data[0, :, :t_fill, :, :] = conditioning_frames[:, :t_fill, :, :].to(dtype=torch.bfloat16)  # [3,t_fill,h,w]
    if t_fill < num_frames:
        video_data[0, :, t_fill:, :, :] = video_data[0, :, t_fill - 1 : t_fill, :, :].expand(
            -1, num_frames - t_fill, -1, -1
        )  # [3,num_frames-t_fill,h,w]
    video_list = [video_data.cuda() for _ in range(batch_size)]  # list of [1,3,num_frames,h,w]
    image_size = [torch.tensor([[h, w, h, w]], dtype=torch.float32).cuda() for _ in range(batch_size)]  # list of [1,4]
    sequence_plans = [
        SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=list(condition_frames_vision))
        for _ in range(batch_size)
    ]
    return {
        "dataset_name": "video_data",
        "video": video_list,
        "image_size": image_size,
        "t5_text_embeddings": torch.randn(batch_size, 512, 1024).cuda().to(dtype=torch.bfloat16),  # [B,512,1024]
        "fps": torch.full((batch_size,), float(fps)).cuda(),  # [B]
        "conditioning_fps": torch.full((batch_size,), float(fps)).cuda(),  # [B]
        "num_frames": torch.full((batch_size,), num_frames).cuda(),  # [B]
        "is_preprocessed": True,
        "sequence_plan": sequence_plans,
    }


def build_image_edit_batch(
    conditioning_frames: torch.Tensor,
    h: int,
    w: int,
    batch_size: int = 1,
) -> dict:
    """Build a data batch for image-to-image editing."""
    image = conditioning_frames.unsqueeze(0).cuda().to(dtype=torch.bfloat16)  # [1,3,1,h,w]
    sequence_plans = [
        SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[]) for _ in range(batch_size)
    ]
    image_size = torch.tensor([[h, w, h, w]], dtype=torch.float32).cuda()  # [1,4]
    return {
        "dataset_name": "image_data",
        "images": [image, image] * batch_size,
        "image_size": [image_size, image_size] * batch_size,
        "num_frames": [torch.tensor([2], dtype=torch.int64).cuda() for _ in range(batch_size)],  # list of [1]
        "num_vision_items_per_sample": [2 for _ in range(batch_size)],
        "is_preprocessed": True,
        "sequence_plan": sequence_plans,
    }


_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def detect_aspect_ratio(width: int, height: int) -> str:
    """Return the closest supported aspect-ratio key for a frame size."""
    aspect_ratios = np.array([16 / 9, 4 / 3, 1, 3 / 4, 9 / 16])
    aspect_ratio_keys = ["16,9", "4,3", "1,1", "3,4", "9,16"]
    current = width / height
    return aspect_ratio_keys[int(np.argmin((aspect_ratios - current) ** 2))]


def read_media_frames(path: Path, max_frames: int) -> tuple[torch.Tensor, float]:
    """Read an image or video into a uint8 tensor of shape (C, T, H, W)."""
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTENSIONS:
        with path.open("rb") as f:
            image = Image.open(f).convert("RGB")
        frames = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(1)
        return frames, 1.0
    if ext not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported media extension: {ext}")
    frames, _, info = torchvision.io.read_video(str(path), pts_unit="sec")
    frames = frames[:max_frames].permute(0, 3, 1, 2).permute(1, 0, 2, 3)
    fps = float(info.get("video_fps", 24.0))
    return frames, fps


def read_and_resize_media(
    path: Path,
    *,
    resolution: str,
    aspect_ratio: str | None,
    max_frames: int,
) -> tuple[torch.Tensor, float, str, tuple[int, int]]:
    """Read an image/video and resize it to the requested resolution bucket."""
    raw_frames, fps = read_media_frames(path, max_frames=max_frames)
    original_hw = (raw_frames.shape[2], raw_frames.shape[3])
    detected_aspect_ratio = detect_aspect_ratio(raw_frames.shape[3], raw_frames.shape[2])
    final_aspect_ratio = aspect_ratio or detected_aspect_ratio
    width, height = VIDEO_RES_SIZE_INFO[resolution][final_aspect_ratio]
    resized = _resize_and_center_crop(raw_frames.permute(1, 0, 2, 3), height, width)
    return resized.permute(1, 0, 2, 3), fps, final_aspect_ratio, original_hw


def uint8_to_normalized_float(tensor: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Convert uint8 [0, 255] frames into normalized [-1, 1] frames."""
    return tensor.to(dtype=dtype) / 127.5 - 1.0


def pad_temporal_frames(frames: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Pad a (C, T, H, W) tensor along time using reflection/repeat behavior."""
    num_frames = frames.shape[1]
    if num_frames >= target_frames:
        return frames
    if num_frames == 0:
        raise ValueError("Cannot pad an empty frame tensor.")
    padded = frames
    while padded.shape[1] < target_frames:
        pad_len = min(padded.shape[1] - 1, target_frames - padded.shape[1])
        if pad_len <= 0:
            pad_frame = padded[:, -1:].repeat(1, target_frames - padded.shape[1], 1, 1)
            padded = torch.cat([padded, pad_frame], dim=1)
            break
        padded = torch.cat([padded, padded.flip(dims=[1])[:, :pad_len]], dim=1)
    return padded
