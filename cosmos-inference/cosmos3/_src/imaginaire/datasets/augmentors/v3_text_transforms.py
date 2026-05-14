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
from typing import Optional

import numpy as np
import torch

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor


def pad_and_resize(
    arr_np: np.ndarray, ntokens: int, is_mask_all_ones: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Function for padding and resizing a numpy array.
    Args:
        arr (np.ndarray): Input array
        ntokens (int): Number of output tokens after padding
        is_mask_all_ones (bool): if true, set mask to ones
    Returns:
        arr_padded (torch.Tensor): Padded output tensor
        mask (torch.Tensor): Padding mask
    """

    if isinstance(arr_np, np.ndarray):
        arr = torch.from_numpy(arr_np)
    elif isinstance(arr_np, torch.Tensor):
        arr = arr_np.clone().detach()
    else:
        raise TypeError("`arr_np` should be a numpy array or torch tensor.")
    embed_dim = arr.shape[1]

    arr_padded = torch.zeros(ntokens, embed_dim, device=arr.device, dtype=torch.float32)

    # If the input text is larger than num_text_tokens, clip it.
    if arr.shape[0] > ntokens:
        arr = arr[0:ntokens]

    mask = torch.LongTensor(ntokens).zero_()
    if len(arr.shape) > 1:
        mask[0 : arr.shape[0]] = 1

    if len(arr.shape) > 1:
        arr_padded[0 : arr.shape[0]] = arr

    if is_mask_all_ones:
        mask.fill_(1)

    return arr_padded, mask


def _obtain_embeddings(cfg: dict, embeddings_captions: dict[str, list], caption_idx: int) -> dict:
    r"""Function for obtaining text embeddings and text mask.
    Args:
        cfg (dict): Config dict
        embeddings_captions (np.ndarray): Caption embeddings
        caption_idx (int): Caption index
    Returns:
        Dictionary containing embeddings and mask
    """
    out_dict = dict()
    is_mask_all_ones = cfg["is_mask_all_ones"]
    if "byt5_tokens" in cfg:
        out_byt5_text, out_byt5_text_mask = pad_and_resize(
            embeddings_captions["byt5_fp8"][caption_idx],
            cfg["byt5_tokens"]["num"],
            is_mask_all_ones=is_mask_all_ones,
        )
        out_dict["byt5_text_embeddings"] = out_byt5_text
        out_dict["byt5_text_mask"] = out_byt5_text_mask

    if "t5_tokens" in cfg:
        out_t5, out_t5_mask = pad_and_resize(
            embeddings_captions["t5_xxl_fp8"][caption_idx],
            cfg["t5_tokens"]["num"],
            is_mask_all_ones=is_mask_all_ones,
        )
        out_dict["t5_text_embeddings"] = out_t5
        out_dict["t5_text_mask"] = out_t5_mask

    return out_dict


def obtain_data_dict_from_mixed_gt_and_ai_captions(data_dict: dict, input_keys: list, args: Optional[dict] = None):
    out_pkl_dict = dict()

    captions_gt = data_dict[input_keys[0]]
    decoded_captions_ai = data_dict[input_keys[1]]
    embeddings_captions_gt = data_dict[input_keys[2]]
    embeddings_captions_ai = data_dict[input_keys[3]]

    assert args is not None, "Please specify args in augmentation"
    probabilities = [args["caption_probs"]["ground_truth"], args["caption_probs"]["vfc_fidelity"]]
    valid_captions_indices = list(range(len(probabilities)))
    caption_idx = random.choices(valid_captions_indices, weights=probabilities, k=1)[0]

    # If VFC Fidelity caption is not valid, we will use the ground truth caption
    if caption_idx == 1 and decoded_captions_ai["had_parse_issue"]:
        caption_idx = 0

    # Merging GT and AI caption raw text
    captions = captions_gt["text"] + [decoded_captions_ai["captions"]["vfc_fidelity"]]

    # Merging GT and AI caption embeddings
    gt_embeddings = []
    for key in ["ground_truth_headline", "ground_truth"]:
        if key in embeddings_captions_gt:
            if embeddings_captions_gt[key] is not None:
                gt_embeddings.append(embeddings_captions_gt[key])

    # Randomly select one of the GT embeddings
    gt_embedding = random.choice(gt_embeddings)
    embeddings_captions = {}
    for key in embeddings_captions_ai["vfc_fidelity"]["embeddings"].keys():
        embeddings_captions[key] = [
            gt_embedding["embeddings"][key],
            embeddings_captions_ai["vfc_fidelity"]["embeddings"][key],
        ]

    # Sampling raw caption and embeddings
    raw_captions = captions[caption_idx]
    data_dict["raw_captions"] = raw_captions

    embeddings_dict = _obtain_embeddings(
        cfg=args,
        embeddings_captions=embeddings_captions,
        caption_idx=caption_idx,
    )
    out_pkl_dict.update(embeddings_dict)

    data_dict.update(out_pkl_dict)
    for key in input_keys:
        del data_dict[key]

    return data_dict


class TextTransform(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs camera transformation.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with camera attributes added
        """
        return obtain_data_dict_from_mixed_gt_and_ai_captions(data_dict, self.input_keys, self.args)


class TextTransformAIOnly(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs text transform for datasets where there are only AI captions (ex., NVCC).

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with camera attributes added
        """

        out_pkl_dict = dict()
        decoded_captions_ai = data_dict[self.input_keys[0]]
        embeddings_captions_ai = data_dict[self.input_keys[1]]

        assert self.args is not None, "Please specify args in augmentation"

        raw_captions = decoded_captions_ai["captions"]["vfc"]
        embeddings_captions = {}

        if decoded_captions_ai["had_parse_issue"]:
            raw_captions = decoded_captions_ai["captions"]["kosmos_2"]
            _embeddings_captions = embeddings_captions_ai["kosmos2"]
        else:
            raw_captions = decoded_captions_ai["captions"]["vfc"]
            _embeddings_captions = embeddings_captions_ai["vfc_fidelity"]

        for key in _embeddings_captions["embeddings"].keys():
            embeddings_captions[key] = [
                _embeddings_captions["embeddings"][key],
            ]

        # Sampling raw caption and embeddings
        data_dict["raw_captions"] = raw_captions
        embeddings_dict = _obtain_embeddings(
            cfg=self.args,
            embeddings_captions=embeddings_captions,
            caption_idx=0,
        )
        out_pkl_dict.update(embeddings_dict)

        data_dict.update(out_pkl_dict)
        for key in self.input_keys:
            del data_dict[key]

        return data_dict
