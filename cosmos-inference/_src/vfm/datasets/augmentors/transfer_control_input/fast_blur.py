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

import ctypes
from ctypes import POINTER, c_float, c_int, c_ubyte, sizeof

import torch


class BilateralGaussian:
    class NppiSize(ctypes.Structure):
        _fields_ = [("width", c_int), ("height", c_int)]

    class NppiPoint(ctypes.Structure):
        _fields_ = [("x", c_int), ("y", c_int)]

    def __init__(self):
        self.npp_i_lib = self._load_npp_library()
        self._setup_buffer_size_function()
        self._setup_bilateral_function()

    def _load_npp_library(self):
        return ctypes.CDLL("libnppif.so")

    def _setup_buffer_size_function(self):
        self.get_buffer_size_func = self.npp_i_lib.nppiFilterCannyBorderGetBufferSize
        self.get_buffer_size_func.restype = c_int
        self.get_buffer_size_func.argtypes = [BilateralGaussian.NppiSize, POINTER(c_int)]  # oSizeROI  # bufferSize

    def _setup_bilateral_function(self):
        self.bilateral_function = self.npp_i_lib.nppiFilterBilateralGaussBorder_8u_C3R
        self.bilateral_function.restype = c_int
        self.bilateral_function.argtypes = [
            POINTER(c_ubyte),  # pSrc
            c_int,  # nSrcStep
            BilateralGaussian.NppiSize,  # oSrcSize
            BilateralGaussian.NppiPoint,  # oSrcOffset
            POINTER(c_ubyte),  # pDst
            c_int,  # nDstStep
            BilateralGaussian.NppiSize,  # oSizeROI
            c_int,  # nRadius
            c_int,  # nStepBetweenSrcPixels
            c_float,  # nValSquareSigma
            c_float,  # nPosSquareSigma
            c_int,  # eBorderType
        ]

    def _prepare_input(self, image_tensor):
        if not image_tensor.is_cuda:
            image_tensor = image_tensor.cuda()
        if image_tensor.dtype != torch.uint8:
            image_tensor = (image_tensor * 255).byte()
        return image_tensor

    def _get_buffer_size(self, roi):
        buffer_size = c_int(0)
        status = self.get_buffer_size_func(roi, ctypes.byref(buffer_size))
        if status != 0:
            raise RuntimeError(f"Failed to get buffer size, status: {status}")
        return buffer_size.value

    def __call__(self, image_tensor, radius=30, color_sigma_square=150 * 150, sigma_space_square=100 * 100):
        # Prepare input
        image_tensor = self._prepare_input(image_tensor)

        height, width, channels = image_tensor.shape
        output = torch.empty_like(image_tensor)

        src_ptr = ctypes.cast(image_tensor.data_ptr(), POINTER(c_ubyte))
        dst_ptr = ctypes.cast(output.data_ptr(), POINTER(c_ubyte))

        roi = BilateralGaussian.NppiSize(width, height)

        status = self.bilateral_function(
            src_ptr,
            width * channels * sizeof(c_ubyte),
            BilateralGaussian.NppiSize(width, height),
            BilateralGaussian.NppiPoint(0, 0),
            dst_ptr,
            width * channels * sizeof(c_ubyte),
            roi,
            c_int(radius),
            1,  # step size
            c_float(color_sigma_square),
            c_float(sigma_space_square),
            2,  # border replicate
        )

        if status != 0:
            raise RuntimeError(f"NPP Canny edge detection failed with status {status}")

        return output
