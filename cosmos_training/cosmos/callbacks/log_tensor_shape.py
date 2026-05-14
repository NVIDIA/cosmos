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

from __future__ import annotations

import torch

from cosmos.utils import log
from cosmos.utils.callback import Callback


class LogTensorShapeCallback(Callback):
    """Log the shape and dtype of every tensor in ``data_batch`` for the first
    ``num_log`` training iterations, on every rank. Used to verify dataloader
    geometry at the start of a run.
    """

    def __init__(self, num_log: int = 10):
        self.num_log = num_log

    def on_training_step_start(self, model_parts, data_batch, iteration):
        if iteration > self.num_log:
            return
        summary_str = f"[Tensor Shape] Iteration {iteration}"
        for key in data_batch.keys():
            if isinstance(data_batch[key], torch.Tensor):
                summary_str += f" | {key} shape: {data_batch[key].shape}, dtype: {data_batch[key].dtype} "
        summary_str += f"data_batch: {data_batch.keys()}"
        if iteration < 1000:
            # Only log the first 1000 iterations
            for key in ["__url__", "__key__", "image_grid_thw", "video_grid_thw"]:
                if key in data_batch:
                    summary_str += f" | {key}: {data_batch[key]}"
        log.info(summary_str, rank0_only=False)
