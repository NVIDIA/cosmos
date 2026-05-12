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

import matplotlib.colors as mcolors
import numpy as np

# Array of 23 highly distinguishable colors in RGB format
PREDEFINED_COLORS_SEGMENTATION = np.array(
    [
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0],  # Yellow
        [0, 255, 255],  # Cyan
        [255, 0, 255],  # Magenta
        [255, 140, 0],  # Dark Orange
        [255, 105, 180],  # Hot Pink
        [0, 0, 139],  # Dark Blue
        [0, 128, 128],  # Teal
        [75, 0, 130],  # Indigo
        [128, 0, 128],  # Purple
        [255, 69, 0],  # Red-Orange
        [34, 139, 34],  # Forest Green
        [128, 128, 0],  # Olive
        [70, 130, 180],  # Steel Blue
        [255, 215, 0],  # Gold
        [255, 222, 173],  # Navajo White
        [144, 238, 144],  # Light Green
        [255, 99, 71],  # Tomato
        [221, 160, 221],  # Plum
        [0, 255, 127],  # Spring Green
        [255, 255, 255],  # White
    ]
)


def generate_distinct_colors() -> np.ndarray:
    """
    Generate `n` visually distinguishable and randomized colors.

    Returns:
        np.ndarray, (3)
    """
    # Randomize hue, saturation, and lightness within a range
    hue = random.uniform(0, 1)  # Full spectrum of hues
    saturation = random.uniform(0.1, 1)  # Vibrant colors
    lightness = random.uniform(0.2, 1.0)  # Avoid too dark

    r, g, b = mcolors.hsv_to_rgb((hue, saturation, lightness))
    return (np.array([r, g, b]) * 255).astype(np.uint8)


def segmentation_color_mask(segmentation_mask: np.ndarray, use_fixed_color_list: bool = False) -> np.ndarray:
    """
    Convert segmentation mask to color mask
    Args:
        segmentation_mask: np.ndarray, shape (num_masks, T, H, W)
    Returns:
        np.ndarray, shape (3, T, H, W), with each mask converted to a color mask, value [0,255]
    """

    num_masks, T, H, W = segmentation_mask.shape
    segmentation_mask_sorted = [segmentation_mask[i] for i in range(num_masks)]
    # Sort the segmentation mask by the number of non-zero pixels, from most to least
    segmentation_mask_sorted = sorted(segmentation_mask_sorted, key=lambda x: np.count_nonzero(x), reverse=True)

    output = np.zeros((3, T, H, W), dtype=np.uint8)
    if use_fixed_color_list:
        predefined_colors_permuted = PREDEFINED_COLORS_SEGMENTATION[
            np.random.permutation(len(PREDEFINED_COLORS_SEGMENTATION))
        ]
    else:
        predefined_colors_permuted = [generate_distinct_colors() for _ in range(num_masks)]
    # index the segmentation mask from last channel to first channel, i start from num_masks-1 to 0
    for i in range(num_masks):
        mask = segmentation_mask_sorted[i]
        color = predefined_colors_permuted[i % len(predefined_colors_permuted)]

        # Create boolean mask and use it for assignment
        bool_mask = mask > 0
        for c in range(3):
            output[c][bool_mask] = color[c]

    return output


def decode_partial_rle_width1(rle_obj: dict, start_row: int, end_row: int) -> np.ndarray:
    """
    Decode a partial RLE encoded mask with width = 1. In SAM2 output, the video mask (num_frame, height, width) are reshaped to (total_size, 1).
    Sometimes the video mask could be large, e.g. 1001x1080x1092 shape and it takes >1GB memory if using pycocotools, resulting in segmentation faults when training with multiple GPUs and data workers.
    This function is used to decode the mask for a subset of frames to reduce memory usage.

    Args:
        rle_obj (dict): RLE object containing:
            - 'size': A list [height, width=1] indicating the dimensions of the mask.
            - 'counts': A bytes or string object containing the RLE encoded data.
        start_row (int): The starting row (inclusive). It's computed from frame_start * height * width.
        end_row (int): The ending row (exclusive). It's computed from frame_end * height * width.

    Returns:
        numpy.ndarray: Decoded binary mask for the specified rows as a 1D numpy array.
    """
    height, width = rle_obj["size"]

    # Validate row range
    if width != 1:
        raise ValueError("This function is optimized for width=1.")
    if start_row < 0 or end_row > height or start_row >= end_row:
        raise ValueError("Invalid row range specified.")

    # Decode the RLE counts
    counts = rle_obj["counts"]
    if isinstance(counts, str):
        counts = np.frombuffer(counts.encode("ascii"), dtype=np.uint8)
    elif isinstance(counts, bytes):
        counts = np.frombuffer(counts, dtype=np.uint8)
    else:
        raise ValueError("Unsupported format for counts. Must be str or bytes.")

    # Interpret counts as a sequence of run lengths
    run_lengths = []
    current_val = 0
    i = 0
    while i < len(counts):
        x = int(0)
        k = 0
        more = True
        while more:
            c = int(counts[i]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = (c & 0x20) != 0
            i += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(run_lengths) > 2:
            x += run_lengths[-2]

        run_lengths.append(x)
        current_val += x
        if current_val > end_row:
            break
    # Initialize the partial mask
    idx_start = start_row
    idx_end = end_row
    partial_mask = np.zeros(idx_end - idx_start, dtype=np.uint8)
    partial_height = end_row - start_row
    idx = 0  # Current global index
    for i, run in enumerate(run_lengths):
        run_start = idx
        run_end = idx + run
        if run_end <= idx_start:
            # Skip runs entirely before the region
            idx = run_end
            continue
        if run_start >= idx_end:
            # Stop decoding once we pass the region
            break

        # Calculate overlap with the target region
        start = max(run_start, idx_start)
        end = min(run_end, idx_end)
        if start < end:
            partial_start = start - idx_start
            partial_end = end - idx_start
            partial_mask[partial_start:partial_end] = i % 2

        idx = run_end
    return partial_mask.reshape((partial_height, 1), order="F")
