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

import torch

from cosmos3._src.imaginaire.functional.batch_ops import batch_mul
from cosmos3._src.imaginaire.utils import log


def random_dropout(embeddings, drop_rate):
    r"""
    Function to perform random dropout for embeddings.
    When we drop embeddings, we zero them out.
    Args:
        embeddings (tensor): Input embeddings
        drop_rate (float): Rate of dropping the embedding.
    """
    num_samples = embeddings.shape[0]
    # Create a shape (num_samples, 1, 1, 1, 1, ...) depending on embeddings dim.
    # This is done to ensure we can broadcast the zero_flag to the embeddings.
    # embeddings.ndim is 3 for images, and 4 for videos, and the corresponding
    # shapes are (num_samples, 1, 1) and (num_samples, 1, 1, 1) respectively.
    tensor_shape = (num_samples,) + tuple([1] * (embeddings.ndim - 1))
    zero_flag = torch.ones(tensor_shape).to(embeddings.dtype) * (1 - drop_rate)
    zero_flag = torch.bernoulli(zero_flag).to(embeddings.device)
    embeddings = embeddings * zero_flag
    return embeddings


def random_embed_replace(src_embed, tgt_embed, drop_rate, position=0):
    r"""
    Function to perform random embedding replacement.
    With probability given by drop rate, we replace src_embed by tgt_embed
    Args:
        src_embed (tensor): Src embeddings
        tgt_embed (tensor): Tgt embeddings
        drop_rate (float): Rate of replacing the embedding.
        position (int): Starting position to replace the sequence
    """
    for i in range(src_embed.shape[0]):
        coin_flip = torch.rand(1).item()
        if coin_flip < drop_rate:
            src_embed[i][position:] = tgt_embed

    return src_embed


def to255_round_uint8_append(vis_images, total_vis_images):
    r"""
    Map pixel values of vis_images to [0 255] and quantize them to 256 bins
    """
    if vis_images is not None:
        vis_images = ((vis_images + 1) / 2).clamp_(0, 1).mul_(255).round_().type(torch.uint8)
        total_vis_images.append(vis_images)
    return vis_images


def no_round_append(vis_images, total_vis_images):
    r"""
    Append the images as is without type casting
    """
    if vis_images is not None:
        total_vis_images.append(vis_images)
    return vis_images


def sample_sigma_and_xt(
    sde,
    target_data: torch.Tensor,
    data_batch: dict = None,
    use_same_noise_multiview: bool = False,
    use_low_noise_first_view: bool = False,
):
    """Sample pertubation noise levels and generate noisy observations."""
    # Sample pertubation noise levels
    tensor_kwargs = {"device": "cuda", "dtype": target_data.dtype}
    t = sde.sample_t(batch_size=target_data.size()[0]).to(**tensor_kwargs)  # check precision and memory_format later
    if data_batch is not None and data_batch.get("num_view", None) is not None:
        if use_same_noise_multiview and not use_low_noise_first_view:
            t_shape = t.shape
            t = t.view(-1, int(data_batch["num_view"].view(-1)[0].item()))
            t[:, 1:] = t[:, 0:1]
            t = t.view(t_shape)
        elif use_low_noise_first_view and not use_same_noise_multiview:
            t_shape = t.shape
            t = t.view(-1, int(data_batch["num_view"].view(-1)[0].item()))
            t[:, 0] = 0.02
            t = t.view(t_shape)
        elif use_low_noise_first_view and use_same_noise_multiview:
            t_shape = t.shape
            t = t.view(-1, int(data_batch["num_view"].view(-1)[0].item()))
            t[:, 0] = 0.02
            t[:, 2:] = t[:, 1:2]
            t = t.view(t_shape)
    # Generate an N(0,1) noise map.
    epsilon = torch.randn_like(target_data, **tensor_kwargs)
    # Get the mean and stand deviation of the marginal probability distribution.
    mean, std = sde.marginal_prob(target_data, t)
    # Generate noisy observations
    xt = mean + batch_mul(std, epsilon)  # corrupted data

    data_batch = {}
    data_batch["t"] = t  # between model.sde.eps to 1
    data_batch["epsilon"] = epsilon  # Standard normal noise map
    data_batch["mean"] = mean  # mean of the marginal distribution
    data_batch["std"] = std  # std deviation of the marginal distribution
    data_batch["xt"] = xt  # corrupted data
    data_batch["target"] = target_data
    return data_batch


def form_loss_mask(
    data_batch: dict,
    x_shape: tuple,
    dtype: torch.dtype,
    device: torch.device,
    loss_masking_cfg: dict = {"human_body_mask": 2, "human_face_mask": 4, "human_hand_mask": 4, "padding_mask": 0},
) -> torch.Tensor:
    r"""
    Function to form a combined mask given several loss masks.
    If there are overlapping region between multiple masks, we assign the max value to the
    overlapping region. For the unmasked regions, we assign a value of 1.
    However, if there is a mask specifying zero value, we zero it out.
    Zeroing is crucial for padded loss.
    Copied from i3, kaal.py form_loss_mask function.
    zero_mask: mask out some region by setting them as zero
    For example,
        mask1: [0, 1, 1, 1, 0, 0], weight: 2
        mask2: [1, 0, 1, 0, 0, 0], weight: 4
        mask3: [0, 1, 0, 0, 0, 0], weight: 0

        Final loss mask: [4, 0, 4, 2, 1, 1]
    """
    loss_mask = torch.ones(x_shape, dtype=dtype, device=device)
    zero_mask = torch.ones(x_shape, dtype=dtype, device=device)

    for key in loss_masking_cfg:
        if key not in data_batch:
            if loss_masking_cfg[key] > 0:
                log.warning(f"You set {key} to have larger loss, but there is no such mask data")
            continue
        # Repeat mask along channel's dimension. ndim=4 for images.
        repeat_dims = (1, 3) + tuple([1] * (data_batch[key].ndim - 2))
        mask_key = torch.tile(data_batch[key], dims=repeat_dims)
        weight_key = loss_masking_cfg[key]

        assert weight_key >= 0, "Current support only for weight >= 0"

        if key == "zero_mask":
            zero_mask = zero_mask * mask_key
        elif weight_key == 0:
            zero_mask = zero_mask * (1 - mask_key)
        else:
            no_mask_region = (mask_key == 0).float()
            loss_mask = mask_key * weight_key + no_mask_region * loss_mask
            # loss_mask = torch.max(loss_mask_new, loss_mask)

    loss_mask_final = loss_mask * zero_mask
    return loss_mask_final
