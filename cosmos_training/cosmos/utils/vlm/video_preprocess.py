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

import numpy as np
from PIL import Image


def tensor_to_pil_images(video_tensor):
    """
    Convert a video tensor of shape (C, T, H, W) or (T, C, H, W) to a list of PIL images.

    Args:
        video_tensor (torch.Tensor): Video tensor with shape (C, T, H, W) or (T, C, H, W)

    Returns:
        list[PIL.Image.Image]: List of PIL images
    """
    # Check tensor shape and convert if needed
    if video_tensor.shape[0] == 3 and video_tensor.shape[1] > 3:  # (C, T, H, W)
        # Convert to (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # [T,C,H,W]

    # Convert to numpy array with shape (T, H, W, C)
    video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()  # [T,H,W,C]

    # Ensure values are in the right range for PIL (0-255, uint8)
    if video_np.dtype == np.float32 or video_np.dtype == np.float64:
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)

    # Convert each frame to a PIL image
    pil_images = [Image.fromarray(frame) for frame in video_np]

    return pil_images
