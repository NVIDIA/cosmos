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

import re
from typing import List, Tuple

IMAGE_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    # Our desired 256 resolution is the one below (commented).

    # Desired: "256": {"1,1": (336, 336), "4,3": (384, 288), "3,4": (288, 384), "16,9": (448, 256), "9,16": (256, 448)},
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
    "480": {"1,1": (640, 640), "4,3": (736, 544), "3,4": (544, 736), "16,9": (832, 480), "9,16": (480, 832)},
    # 704 resolutions are nicely divisible by 32
    "704": {"1,1": (960, 960), "4,3": (1088, 832), "3,4": (832, 1088), "16,9": (1280, 704), "9,16": (704, 1280)},
    "720": {"1,1": (960, 960), "4,3": (1104, 832), "3,4": (832, 1104), "16,9": (1280, 720), "9,16": (720, 1280)},
    "1080": {"1,1": (1440, 1440), "4,3": (1664, 1248), "3,4": (1248, 1664), "16,9": (1920, 1080), "9,16": (1080, 1920)},
    "1280": {"1,1": (1712, 1712), "4,3": (1968, 1472), "3,4": (1472, 1968), "16,9": (2272, 1280), "9,16": (1280, 2272)},
    "2048": {
        "1,1": (2728, 2728),
        "4,3": (3160, 2368),
        "3,4": (2368, 3160),
        "16,9": (3640, 2048),
        "9,16": (2048, 3640),
    },
    "gt_2048": {
        "1,1": (5464, 5464),
        "4,3": (6304, 4728),
        "3,4": (4728, 6304),
        "16,9": (7280, 4096),
        "9,16": (4096, 7280),
    },
}

VIDEO_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    # Our desired 256 resolution is the one below (commented).

    # Desired: "256": {"1,1": (336, 336), "4,3": (384, 288), "3,4": (288, 384), "16,9": (448, 256), "9,16": (256, 448)},
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
    "480": {"1,1": (640, 640), "4,3": (736, 544), "3,4": (544, 736), "16,9": (832, 480), "9,16": (480, 832)},
    # 704 resolutions are nicely divisible by 32
    "704": {"1,1": (960, 960), "4,3": (1088, 832), "3,4": (832, 1088), "16,9": (1280, 704), "9,16": (704, 1280)},
    "720": {"1,1": (960, 960), "4,3": (1104, 832), "3,4": (832, 1104), "16,9": (1280, 720), "9,16": (720, 1280)},
    "1080": {"1,1": (1440, 1440), "4,3": (1664, 1248), "3,4": (1248, 1664), "16,9": (1920, 1080), "9,16": (1080, 1920)},
    "1280": {"1,1": (1712, 1712), "4,3": (1968, 1472), "3,4": (1472, 1968), "16,9": (2272, 1280), "9,16": (1280, 2272)},
    "2048": {
        "1,1": (2728, 2728),
        "4,3": (3160, 2368),
        "3,4": (2368, 3160),
        "16,9": (3640, 2048),
        "9,16": (2048, 3640),
    },
    "gt_2048": {
        "1,1": (5464, 5464),
        "4,3": (6304, 4728),
        "3,4": (4728, 6304),
        "16,9": (7280, 4096),
        "9,16": (4096, 7280),
    },
}


def get_aspect_ratios_from_wdinfos(wdinfos: list[str]) -> list[str]:
    aspect_ratios = []
    for wdinfo in wdinfos:
        aspect_ratio_match = re.search(r"aspect_ratio_(\d+_\d+)", wdinfo)
        aspect_ratios.append(aspect_ratio_match.group(1))

    return aspect_ratios


def get_wdinfos_w_aspect_ratio(wdinfos: list[str]) -> List[Tuple[str, str]]:
    aspect_ratios = get_aspect_ratios_from_wdinfos(wdinfos)

    # return a list of (wdinfo_path, aspect_ratio) pairs
    return [(wdinfo, aspect_ratio.replace("_", ",")) for wdinfo, aspect_ratio in zip(wdinfos, aspect_ratios)]


def parse_frame_range_from_wdinfo(wdinfo: str) -> tuple[int, int] | None:
    """
    Parse frame range from wdinfo path.

    Args:
        wdinfo: wdinfo path string containing frames_X_Y pattern

    Returns:
        Tuple of (min_frames, max_frames) if found, None otherwise

    Example:
        >>> parse_frame_range_from_wdinfo("wdinfo/v4/tv_drama/resolution_720/aspect_ratio_16_9/frames_300_400/wdinfo.json")
        (300, 400)
    """
    match = re.search(r"frames_(\d+)_(\d+)", wdinfo)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def filter_wdinfos_by_frame_range(
    wdinfos: list[str],
    min_frames: int | None = None,
    max_frames: int | None = None,
) -> list[str]:
    """
    Filter wdinfo files based on frame range.

    The frame range in wdinfo path (e.g., frames_300_400) represents videos
    with frames between those values. This function filters wdinfo files
    based on the wdinfo's upper bound (wdinfo_max):
    - min_frames is EXCLUSIVE: wdinfo_max must be > min_frames
    - max_frames is INCLUSIVE: wdinfo_max must be <= max_frames

    Args:
        wdinfos: List of wdinfo paths
        min_frames: Minimum number of frames (exclusive). If None, no lower bound.
        max_frames: Maximum number of frames (inclusive). If None, no upper bound.

    Returns:
        Filtered list of wdinfo paths

    Example:
        >>> wdinfos = [
        ...     "wdinfo/frames_400_500/wdinfo.json",
        ...     "wdinfo/frames_500_600/wdinfo.json",
        ...     "wdinfo/frames_600_700/wdinfo.json",
        ... ]
        >>> filter_wdinfos_by_frame_range(wdinfos, min_frames=500, max_frames=600)
        ['wdinfo/frames_500_600/wdinfo.json']
        # frames_400_500 excluded because wdinfo_max (500) <= min_frames (500)
        # frames_500_600 included because wdinfo_max (600) > min_frames (500) AND <= max_frames (600)
        # frames_600_700 excluded because wdinfo_max (700) > max_frames (600)
    """
    if min_frames is None and max_frames is None:
        return wdinfos

    filtered = []
    for wdinfo in wdinfos:
        frame_range = parse_frame_range_from_wdinfo(wdinfo)
        if frame_range is None:
            # If no frame range in path, include by default
            filtered.append(wdinfo)
            continue

        wdinfo_min, wdinfo_max = frame_range

        # Filter based on wdinfo's upper bound (wdinfo_max):
        # - min_frames is exclusive: wdinfo_max must be > min_frames
        # - max_frames is inclusive: wdinfo_max must be <= max_frames
        include = True
        if min_frames is not None and wdinfo_max <= min_frames:
            include = False
        if max_frames is not None and wdinfo_max > max_frames:
            include = False

        if include:
            filtered.append(wdinfo)

    return filtered
