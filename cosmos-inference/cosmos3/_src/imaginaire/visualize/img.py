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

"""image visualization utilities.

based on https://gitlab.com/qsh.zh/jam/-/blob/master/jamviz/img.py MIT License
"""

import os
from typing import Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision.utils import make_grid

__all__ = [
    "show_batch_img",
    "save_batch_img",
]


def _reshape_viz_batch_img(img_data: torch.Tensor | np.ndarray, shape: int | str = 7) -> tuple:
    """
    Reshapes a batch of images for visualization, organizing them into a grid format.

    Args:
        img_data (torch.Tensor | np.ndarray): The image data to be reshaped, can be either a PyTorch tensor or a NumPy array.
        shape (int | str, optional): Defines the layout of the grid. If an integer is provided, it specifies both the number of rows and columns. If a string is provided in the format 'nrowxncol', it parses to individual row and column numbers. Defaults to 7.

    Returns:
        tuple: A tuple containing:
            img (np.ndarray | torch.Tensor): The image data arranged in grid format.
            nrow (int): Number of rows in the grid.
            ncol (int): Number of columns in the grid.

    Raises:
        RuntimeError: If the shape parameter is neither an int nor a string, or if it's a string that doesn't contain 'x'.

    Example:
        >>> tensor_images = torch.rand(64, 3, 28, 28)  # Example tensor of 64 images
        >>> img_grid, rows, cols = _reshape_viz_batch_img(tensor_images, '8x8')
        >>> img_grid.shape
        (224, 224, 3)
    """
    if isinstance(shape, int):
        nrow, ncol = shape, shape
    elif isinstance(shape, str):
        if "x" not in shape:
            nrow, ncol = int(shape), int(shape)
        else:
            shape = shape.split("x")
            nrow, ncol = int(shape[0]), int(shape[1])
    else:
        raise RuntimeError(f"shape {shape} not support")
    if isinstance(img_data, torch.Tensor):
        assert img_data.shape[1] in [1, 3]
        grid_img = make_grid(img_data[: nrow * ncol].detach().cpu(), ncol)
        img = grid_img.permute(1, 2, 0)
    elif isinstance(img_data, np.ndarray):
        if img_data.shape[1] in [1, 3]:
            img = rearrange(img_data[: nrow * ncol], "(b t) c h w -> (b h) (t w) c", b=nrow)
        else:
            img = rearrange(img_data[: nrow * ncol], "(b t) h w c -> (b h) (t w) c", b=nrow)
    return img, nrow, ncol


def show_batch_img(
    img_data: torch.Tensor | np.ndarray,
    shape: int | str = 7,
    grid: int = 3,
    is_n1p1: bool = False,
    auto_n1p1: bool = True,
) -> None:
    """
    Displays a batch of images using matplotlib after arranging them into a specified grid layout.

    Args:
        img_data (torch.Tensor | np.ndarray): The image data to be displayed.
        shape (int | str, optional): The grid shape to organize the images. Defaults to 7.
        grid (int, optional): Scaling factor for each image in the grid, affecting the overall size of the displayed figure. Defaults to 3.
        is_n1p1 (bool, optional): Whether to normalize the images from [-1, 1] to [0, 1] for visualization. Defaults to False.
        auto_n1p1 (bool, optional): If true, automatically adjusts images from [-1, 1] to [0, 1] based on minimum pixel value detection. Defaults to True.

    Returns:
        None: This function does not return anything but displays the image grid using matplotlib.

    Example:
        >>> tensor_images = torch.rand(64, 3, 28, 28)  # Example tensor of 64 images
        >>> show_batch_img(tensor_images, '8x8')
    """
    import matplotlib.pyplot as plt

    if is_n1p1:
        img_data = (img_data + 1) / 2
    else:
        if auto_n1p1:
            if isinstance(img_data, torch.Tensor):
                if img_data.min().item() < -0.5:
                    img_data = (img_data + 1) / 2
            elif isinstance(img_data, np.ndarray):
                if np.min(img_data) < -0.5:
                    img_data = (img_data + 1) / 2
    img, nrow, ncol = _reshape_viz_batch_img(img_data, shape)
    plt.figure(figsize=(ncol * grid, nrow * grid))
    plt.axis("off")
    plt.imshow(img)


def save_batch_img(fpath: str, img_data: Union[torch.Tensor, np.ndarray], shape: Union[int, str] = 7) -> None:
    """
    Saves a batch of images to a file after arranging them into a grid format. Handles both PyTorch tensors and NumPy arrays as input.

    Args:
        fpath (str): File path where the image will be saved.
        img_data (Union[torch.Tensor, np.ndarray]): The image data to be saved. Can be a PyTorch tensor or a NumPy array.
        shape (Union[int, str], optional): The grid shape to organize the images. Can be an integer specifying equal number of rows and columns, or a string specifying 'nrowxncol'. Defaults to 7.

    Returns:
        None: This function does not return anything but saves the image to the specified file path.

    Raises:
        RuntimeError: If the input shape is neither an integer nor a string, or it does not include 'x' when provided as a string.

    Example:
        >>> tensor_images = torch.rand(64, 3, 28, 28)  # Example tensor of 64 images
        >>> save_batch_img('path/to/save/image.png', tensor_images, '8x8')
        # This saves the image grid to 'path/to/save/image.png'
    """
    img, _, _ = _reshape_viz_batch_img(img_data, shape)
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    ndarr = img.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    im.save(fpath)
