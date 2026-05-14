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

import html
import json
import tempfile
from typing import Literal

import torch
import wandb
from einops import rearrange

from cosmos.utils.config import JobConfig
from cosmos.utils import callback, distributed
from cosmos.utils.easy_io import easy_io


class VisualizationLoggingCallback(callback.WandBCallback):
    def __init__(
        self,
        every_n: int = 1,
        input_normalization: str | None = None,
        job: JobConfig | None = None,
        config: callback.Config | None = None,
        trainer: callback.ImaginaireTrainer | None = None,
    ):
        super().__init__(config, trainer)
        self.every_n = every_n
        self.input_normalization = input_normalization
        self.job = job

    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if iteration % self.every_n == 0 and distributed.is_rank0() and wandb.run is not None:
            # log images to wandb for rank0
            self.log_videos(log_type="train", data=data_batch, output=output_batch, iteration=iteration)

    def on_validation_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Collect the validation batch and aggregate the overall loss.
        super().on_validation_step_end(model, data_batch, output_batch, loss, iteration)
        if iteration % self.every_n == 0 and distributed.is_rank0() and wandb.run is not None:
            # log images to wandb for rank0
            self.log_videos(log_type="val", data=data_batch, output=output_batch, iteration=iteration)

    @torch.no_grad()
    def log_videos(
        self,
        log_type: Literal["train", "val"],
        data: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        iteration: int = 0,
    ):
        if "raw_image" in data:
            video = data["raw_image"][0].cpu()  # [3,T,H,W], range [0, 255], uint8
        elif "raw_video" in data:
            video = data["raw_video"][0].cpu()  # [3,T,H,W], range [0, 255], uint8
        video = video.permute(1, 0, 2, 3)  # [T,3,H,W]
        video = video.numpy()
        wandb_video = rearrange(video, "t c h w -> t h w c")  # [T,H,W,3]
        # create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file_path = temp_file.name
        easy_io.dump(wandb_video, temp_file_path, file_format="mp4", format="mp4", fps=10, quality=5)
        wandb.log({f"{log_type}/video": wandb.Video(temp_file_path)}, step=iteration)

        # Convert dialog string to JSON and format it for HTML display
        try:
            dialog_json = json.loads(data["dialog_str"][0], indent=4)
            formatted_html = f"""
            <pre style='font-size: 0.5em;'>{dialog_json}</pre>
            """
            wandb.log({f"{log_type}/prompt": wandb.Html(formatted_html)}, step=iteration)
        except Exception as e:
            # Fallback to original format if JSON conversion fails
            # Escape HTML tags in the dialog string to display them properly
            escaped_dialog = html.escape(data["dialog_str"][0])
            wandb.log(
                {f"{log_type}/prompt": wandb.Html(f"<pre style='font-size: 0.5em;'>{escaped_dialog}</pre>")},
                step=iteration,
            )
