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

from functools import lru_cache
from typing import Any, Tuple

import torch
import torch.nn.functional as F

from cosmos3._src.imaginaire.utils.s3_utils import (
    download_from_s3_with_cache as download_from_s3_with_cache,
)
from cosmos3._src.imaginaire.utils.s3_utils import (
    load_from_s3_with_cache as load_from_s3_with_cache,
)


def disabled_train(self: Any, mode: bool = True) -> Any:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def expand_dims_like(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)  # broadcast-compatible shape matching y.dim()
    return x


@lru_cache(maxsize=10)
def load_negative_prompt_t5(t5_lenght: int = 512) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """
    Loads and returns the raw caption and corresponding T5 embeddings for a negative prompt,
    along with a mask tensor from an S3 bucket cache.

    The T5 embeddings are truncated or padded to match the specified T5 length. The function
    utilizes LRU (Least Recently Used) cache to store up to 10 recent calls for efficiency.

    Parameters:
    t5_length (int): The desired length for the T5 embeddings. Defaults to 512.

    Returns:
    Tuple[str, torch.Tensor, torch.Tensor]: A tuple containing the raw caption (str),
    the T5 embeddings (torch.Tensor) truncated or padded to the specified length, and
    a mask tensor (torch.Tensor) indicating the valid positions in the T5 embeddings.

    The embeddings tensor is of shape [L, D] where L is the sequence length up to `t5_length`
    and D is the embeddings dimension. The mask is a LongTensor of shape [t5_length] with 1s
    at indices corresponding to actual token positions and 0s elsewhere.
    """
    negative_prompt_t5 = load_from_s3_with_cache("s3://bucket/edify_image_v4/test_data_batch/negative_prompt.pt")
    raw_caption: str = negative_prompt_t5["raw_captions"]
    t5_emb_LD: torch.Tensor = negative_prompt_t5["t5_text_embeddings"]  # [L,D]
    length = t5_emb_LD.shape[0]
    mask = torch.LongTensor(t5_lenght).zero_()  # [t5_length]
    mask[0:length] = 1
    if length < t5_lenght:
        t5_emb_LD = F.pad(t5_emb_LD, (0, 0, 0, t5_lenght - length), value=0)  # [t5_length,D]
    t5_emb_LD = t5_emb_LD[:t5_lenght]  # [t5_length,D]
    return raw_caption, t5_emb_LD, mask
