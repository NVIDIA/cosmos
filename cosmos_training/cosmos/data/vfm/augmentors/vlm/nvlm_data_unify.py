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

"""Visual-Text Transformations or Augmentations."""

import io
from typing import Dict, Optional

from PIL import Image

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.utils import log
from cosmos.data.vfm.augmentors.vlm.nvlm_sample_loaders_and_part_filters import (
    get_data_class,
    get_part_filter,
    get_sample_loader,
)


class NVLMImageDataUnify(Augmentor):
    """
    This augmentor is used to unify the data format of the nvlm data.
    It will take the raw nvlm data tar and convert it to a dictionary with the following keys:
    {
        "__url__": str,
        "__key__": str,
        "data_class": str,
        "images": List[PIL.Image.Image],
        "text": str,
        "words_boxes": Optional[List[List[int]]],
        "words_text": Optional[List[str]],
        "similarity_matrix": Optional[List[List[float]]],
    }
    """

    def __init__(
        self,
        input_keys: list = ["raw_nvlm"],
        output_keys: Optional[list] = [],
        args: Optional[dict] = None,
        data_path_prefix: list[str] = [
            "cosmos/ar/v2/nvlm/",
        ],  # prefix of the data in s3
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.data_path_prefix = data_path_prefix

    def convert_image(self, img):
        try:
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img)).convert("RGB")
            elif isinstance(img, Image.Image):
                img = img.convert("RGB")
                pass  # Image is already in PIL format
            elif isinstance(img, list):
                for i in range(len(img)):
                    img[i], success = self.convert_image(img[i])
                    if not success:
                        return Image.new("RGB", (256, 256), (0, 0, 0)), False
                return img, True
            else:
                raise ValueError(f"Invalid image type: {type(img)}")

            success = True
        except Exception as e:
            log.warning(f"Error processing image: {e}. Creating an empty black image.", rank0_only=False)
            img = Image.new("RGB", (256, 256), (0, 0, 0))  # Creates a 256x256 black image
            success = False
        return img, success

    def __call__(self, data_dict: Dict) -> Dict:
        url = data_dict["__url__"]
        data_path = "/".join(url.path.split("/")[:-1])  # remove the last part of the path
        sample_loader = get_sample_loader(data_path)
        part_filter = get_part_filter(data_path)
        data_class = get_data_class(data_path)
        assert sample_loader is not None and part_filter is not None and data_class is not None, (
            f"sample_loader({sample_loader}) or part_filter({part_filter}) or data_class({data_class}) is not found for {data_path}"
        )

        raw = {"__url__": url, "__key__": data_dict["__key__"]}
        output = {"__url__": url, "__key__": data_dict["__key__"]}
        for k, v in data_dict.items():
            ext = k.split(".")[-1]
            if part_filter(ext):
                raw[ext] = v
        try:
            output_converted = sample_loader(raw)
            # Here output_converted will be a dictionary with the following keys:
            # {
            #   "__key__": str,
            #   "image": PIL.Image.Image,
            #   "images": List[PIL.Image.Image],
            #   "text": str,
            #   "words_boxes": Optional
            #   "words_text": Optional
            #   "similarity_matrix": Optional
            # }
        except Exception as e:
            log.warning(
                f"Error in sample_loader: {e}, sample_loader: {sample_loader}, data_path: {data_path}, raw: {raw.keys()}, original_data_dict: {data_dict.keys()}, __url__: {url}, __key__: {data_dict['__key__']}"
            )
            return None

        output.update(output_converted)
        if "image" not in output_converted and "images" not in output_converted:
            success = False
            log.warning(f"image not found in {output_converted.keys()}")
        if "image" in output_converted:  # Single image case
            img, success = self.convert_image(output["image"])
            output["images"] = [img]  # What should be the format for the iamges
        elif "images" in output_converted:
            output["images"] = output_converted["images"]
            output["images"], success = self.convert_image(output["images"])
        if not success:
            log.warning(f"image conversion failed for {data_dict['__key__']} url: {url} | Skip this data")
            return None
        output["data_class"] = data_class

        return output
