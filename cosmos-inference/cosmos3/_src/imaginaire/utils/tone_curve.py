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
from typing import Literal

import numpy as np
from PIL import Image


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


def commerce_tonemap(color):
    startCompression = 0.8 - 0.04
    desaturation = 0.15

    x = np.min(color, axis=-1, keepdims=True)
    offset = np.where(x < 0.08, x - 6.25 * x * x, 0.04)
    color -= offset
    peak = np.max(color, axis=-1, keepdims=True)
    uncompressed = color

    d = 1.0 - startCompression
    newPeak = 1.0 - d * d / (peak + d - startCompression)
    with np.errstate(divide="ignore", invalid="ignore"):  # Avoid error print
        color = color * (newPeak / peak)

    g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0)

    compressed = color * (1 - g) + newPeak * g

    return np.where(peak < startCompression, uncompressed, compressed)


# https://github.com/RenderKit/oidn/blob/master/training/color.py


# Computes the luminance of an RGB color
def luminance(r, g, b):
    return 0.212671 * r + 0.715160 * g + 0.072169 * b


# Computes an autoexposure value for a NumPy image
def autoexposure(image, mask, key=0.18):
    maxBinSize = 16  # downsampling amount
    eps = 1e-8

    image = image * mask
    # Compute the luminance of each pixel
    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]
    L = luminance(r, g, b)

    # Center crop if the image size is not whole multiple of maxBinSize
    crop_H = L.shape[0] // maxBinSize * maxBinSize
    pad_top = round((L.shape[0] - crop_H) / 2)
    crop_W = L.shape[1] // maxBinSize * maxBinSize
    pad_left = round((L.shape[1] - crop_W) / 2)
    L = L[pad_top : pad_top + crop_H, pad_left : pad_left + crop_W]
    mask = mask[pad_top : pad_top + crop_H, pad_left : pad_left + crop_W]

    # Downsample the image to minimize sensitivity to noise
    H = L.shape[0]  # original height
    W = L.shape[1]  # original width
    L = L.reshape(H // maxBinSize, maxBinSize, W // maxBinSize, maxBinSize)
    L = np.mean(L, axis=(1, 3))
    mask = mask.reshape(H // maxBinSize, maxBinSize, W // maxBinSize, maxBinSize)
    mask = np.mean(mask, axis=(1, 3))
    with np.errstate(divide="ignore", invalid="ignore"):  # Avoid error print
        L /= mask
    L = L[mask > eps]

    # Keep only values greater than epsilon
    L = L[L > eps]
    if L.size == 0:
        return 1.0

    # Compute the exposure value
    return float(key / np.exp2(np.log2(L).mean()))


# Default values changed to identity transformation, aka do nothing.
def apply_tone_curve(
    imgs: list[Image.Image],
    input_mapping: Literal["log", "straight"] = "log",
    output_mapping: Literal["commerce", "straight", "log"] = "commerce",
    exposure_bias: float = 1.5,
    auto: bool = True,
    ae_pregain: float = 1.0,
    ae_key: float = 0.18,
    ae_strength_below: float = 1.0,
    ae_strength_above: float = 1.0,
) -> tuple[list[Image.Image], float]:
    r"""Adjust the exposure of a list of images together.
        For cam_v1 data, use input_mapping="log"
        For cam_v2 data, use input_mapping="straight"
        Some of the previous models are trained with output_mapping="commerce". This is a very forgiving curve.
        But to match the style of PixelSquid, use output_mapping="straight"
        See https://docs.google.com/document/d/1z08rWvWzqd_tNPlh7_D4aIkdaLAerSSagXK4pQlQxCk/edit for detail

    Args:
        imgs: list of PIL images

    Returns:
        ret: list of PIL images with exposure adjusted
    """
    num_imgs = len(imgs)
    img = np.concatenate([np.asarray(x) for x in imgs], axis=0).astype(np.float32) / 255.0
    mask = img[..., 3:4].astype(np.float32)  # H,W,1
    img = img[..., :3]  # Remove alpha

    img = srgb2lin(img)

    if input_mapping == "log":
        img = np.exp(img) - 1
    elif input_mapping == "straight":
        pass
    else:
        raise NotImplementedError(f"Unknown input_mapping: {input_mapping}")

    if auto:
        img *= ae_pregain
        exposure = autoexposure(img, mask, key=ae_key)
        log_exposure = math.log2(exposure)
        if log_exposure <= 0:
            log_exposure *= ae_strength_below
        else:
            log_exposure *= ae_strength_above
        exposure = 2.0**log_exposure
    else:
        exposure = 1.0
    exposure *= exposure_bias

    img = img * exposure

    if output_mapping == "commerce":
        img = commerce_tonemap(img)
    elif output_mapping == "log":
        img = np.log(img + 1)
    elif output_mapping == "straight":
        pass
    else:
        raise NotImplementedError(f"Unknown output_mapping: {output_mapping}")

    img = lin2srgb(img)
    img = np.concatenate([img, mask], axis=-1)
    img = np.clip((img * 255.0).round(), 0, 255).astype(np.uint8)
    return [Image.fromarray(x) for x in np.split(img, num_imgs, axis=0)], exposure


def apply_exposure(img: Image, exposure: float) -> Image:
    r"""Apply exposure adjustment to a PIL image.
    Args:
        img: a PIL image, RGB or RGBA
        exposure: exposure value
    Returns:
        img: PIL image with exposure adjusted
    """
    img = np.asarray(img).astype(np.float32) / 255.0
    img[..., :3] = lin2srgb(srgb2lin(img[..., :3]) * exposure)
    img = np.clip((img * 255.0).round(), 0, 255).astype(np.uint8)
    return Image.fromarray(img)
