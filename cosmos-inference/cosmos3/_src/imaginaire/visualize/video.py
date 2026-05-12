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

from typing import IO, Any, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image as PILImage
from torch import Tensor

from cosmos3._src.imaginaire.utils.easy_io import easy_io


def save_video(grid, video_name, fps=30):
    import cv2
    import ffmpegcv

    grid = (grid * 255).astype(np.uint8)
    grid = np.transpose(grid, (1, 2, 3, 0))
    with ffmpegcv.VideoWriter(video_name, "h264", fps) as writer:
        for frame in grid:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            writer.write(frame)


def save_img_or_video(
    sample: Tensor,  # [C,T,H,W] in [0,1] range
    save_fp_wo_ext: Union[str, IO[Any]],
    fps: int = 24,
    quality=None,
    ffmpeg_params=None,
    **kwargs,
) -> None:
    """
    Save a tensor as an image or video file based on shape

        Args:
        sample (Tensor): Input tensor with shape (C, T, H, W) in [0, 1] range.
        save_fp_wo_ext (Union[str, IO[Any]]): File path without extension or file-like object.
        fps (int): Frames per second for video. Default is 24.
    """
    assert sample.ndim == 4, "Only support 4D tensor"
    assert isinstance(save_fp_wo_ext, str) or hasattr(save_fp_wo_ext, "write"), (
        "save_fp_wo_ext must be a string or file-like object"
    )

    if torch.is_floating_point(sample):
        sample = sample.clamp(0, 1)
    else:
        assert sample.dtype == torch.uint8, "Only support uint8 tensor"
        sample = sample.float().div(255)

    if ffmpeg_params is not None:
        kwargs["ffmpeg_params"] = ffmpeg_params

    if sample.shape[1] == 1:
        save_obj = PILImage.fromarray(
            rearrange((sample.cpu().float().numpy() * 255), "c 1 h w -> h w c").astype(np.uint8),
            mode="RGB",
        )
        ext = ".jpg" if isinstance(save_fp_wo_ext, str) else ""
        easy_io.dump(
            save_obj,
            f"{save_fp_wo_ext}{ext}" if isinstance(save_fp_wo_ext, str) else save_fp_wo_ext,
            file_format="jpg",
            format="JPEG",
            quality=85 if quality is None else quality,
            **kwargs,
        )
    else:
        if quality is not None:
            kwargs["quality"] = quality
        save_obj = rearrange((sample.cpu().float().numpy() * 255), "c t h w -> t h w c").astype(np.uint8)
        ext = ".mp4" if isinstance(save_fp_wo_ext, str) else ""
        easy_io.dump(
            save_obj,
            f"{save_fp_wo_ext}{ext}" if isinstance(save_fp_wo_ext, str) else save_fp_wo_ext,
            file_format="mp4",
            format="mp4",
            fps=fps,
            **kwargs,
        )
