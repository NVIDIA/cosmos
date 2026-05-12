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

import io
import re

import numpy as np
from PIL import Image

_IMG_EXTENSIONS = "jpg jpeg png ppm pgm pbm pnm".split()


def lin2srgb(lin):
    """Convert sRGB values to physically linear ones. The transformation is
    uniform in RGB, so *srgb* can be of any shape.

    *srgb* values should range between 0 and 1, inclusively.

    """
    gamma = 1.055 * lin ** (1.0 / 2.4) - 0.055
    scale = 12.92 * lin
    return np.where(lin > 0.0031308, gamma, scale)


def srgb2lin(srgb):
    """Convert sRGB values to physically linear ones. The transformation is
    uniform in RGB, so *srgb* can be of any shape.

    *srgb* values should range between 0 and 1, inclusively.

    """
    gamma = ((srgb + 0.055) / 1.055) ** 2.4
    scale = srgb / 12.92
    return np.where(srgb > 0.04045, gamma, scale)


def replace_bg_color_u8(fg: np.array, fg_mask: np.array, bg_color_old: list, bg_color_new: list):
    r"""Given an image with background, as well as the foreground mask and old background color,
        Replace the old background color with the new one.
        Assuming everything is in uint8
    Args:
        fg [..., 3] np.array
        fg_mask[..., 1] np.array: 0 -> full background; 255 -> full foreground.
        bg_color_old [3] RGB 0-255: Old background.
        bg_color_new [3] RGB 0-255: New background
    """
    assert fg.dtype == np.uint8 and fg_mask.dtype == np.uint8
    fg_mask = fg_mask.astype(np.float32) / 255.0
    fg = fg.astype(np.float32) / 255.0
    bg_color_old = np.array(bg_color_old, dtype=np.float32) / 255.0
    bg_color_new = np.array(bg_color_new, dtype=np.float32) / 255.0
    bg_mask = 1.0 - fg_mask
    result = srgb2lin(fg) + bg_mask * (srgb2lin(bg_color_new) - srgb2lin(bg_color_old))
    result = lin2srgb(result)
    result = np.clip((result * 255.0).round(), 0, 255).astype(np.uint8)
    return result


def replace_bg_color_pil(fg_pil: Image.Image, fg_mask_pil: Image.Image, bg_color_old: list, bg_color_new: list):
    fg = np.array(fg_pil)
    fg_mask = np.array(fg_mask_pil)
    if fg_mask.ndim == 2:
        fg_mask = fg_mask[..., None]
    else:
        fg_mask = fg_mask[..., :1]
    result = replace_bg_color_u8(fg, fg_mask, bg_color_old, bg_color_new)
    return Image.fromarray(result)


def pil_loader_with_mask(key, data, background_color_new=None, background_color_old=[255, 255, 255], mask=None):
    r"""
    Function to load an image.
    If the image is corrupt, it returns a black image.
    Args:
        key: Image key.
        data: Image data stream.
    """
    extension = re.sub(r".*[.]", "", key)
    if extension.lower() not in _IMG_EXTENSIONS:
        return None

    with io.BytesIO(data) as stream:
        img = Image.open(stream)
        img = img.convert("RGB")
    if background_color_new is not None:
        assert mask is not None
        with io.BytesIO(mask) as stream:
            mask = Image.open(stream)
            mask.load()
            mask = mask.convert("L")
        img = replace_bg_color_pil(img, mask, background_color_old, background_color_new)
    return img


def pil_loader(key, data, type="RGB"):
    r"""
    Function to load an image.
    If the image is corrupt, it returns a black image.
    Args:
        key: Image key.
        data: Image data stream.
    """
    extension = re.sub(r".*[.]", "", key)
    if extension.lower() not in _IMG_EXTENSIONS:
        return None

    with io.BytesIO(data) as stream:
        img = Image.open(stream)
        img.load()
        img = img.convert(type)

    return img
