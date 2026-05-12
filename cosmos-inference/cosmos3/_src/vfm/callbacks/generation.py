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

import glob
import os
from pathlib import Path

import einops
import numpy as np
import torch
import torchvision
import wandb
from PIL import Image

from cosmos3._src.imaginaire.callbacks.every_n import EveryN
from cosmos3._src.imaginaire.trainer import ImaginaireTrainer
from cosmos3._src.imaginaire.utils import distributed, log


class Generation(EveryN):
    def __init__(
        self,
        every_n: int = 500,
        num_vis: int = 10,
    ):
        r"""
        This callback enables us to perform full generation from class indices.
        The generated images are saved to s3.

        Args:
            every_n (int): Call this callback every_n steps
            num_vis (int): Number of visualizations to save
        """
        super().__init__(every_n)
        self.num_vis = num_vis

    def on_train_start(self, model: torch.nn.Module, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/generation"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")

    @torch.inference_mode()
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: torch.nn.Module,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        if not hasattr(model, "run_pipe_for_data_batch"):
            log.warning("Model does not have run_pipe_for_data_batch method, skipping generation")
            return
        if model.config.train_mllm_only:
            log.warning("Skipping generation in MLLM only mode")
            return
        assert (
            len(data_batch["diffusion_media_input"].shape) == 5 and data_batch["diffusion_media_input"].shape[1] == 3
        ), (
            f"`diffusion_media_input` must have the shape of (bs, 3, T, H, W), current shape is {data_batch['diffusion_media_input'].shape}"
        )

        log.info(f"Generating video for iteration {iteration}, data_batch keys: {data_batch.keys()}")
        video = model.run_pipe_for_data_batch(data_batch)

        input_video = data_batch["diffusion_media_input"]  # [B,3,T,H,W]

        log.info(f"Video list length: {len(video)}")
        rank = distributed.get_rank()
        output_path = os.path.join(self.local_dir, f"iter_{iteration}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        B, _, T, height, width = input_video.shape
        gt_image = einops.rearrange(input_video, "B C T H W -> (B T) C H W")  # [BT,3,H,W]
        gt_image = ((gt_image + 1) / 2).clamp(0, 1).float()  # [BT,3,H,W], range: 0-1
        gt_grid = torchvision.utils.make_grid(gt_image, nrow=B * T, padding=0).cpu()  # [3,H,W*BT]

        video = torch.stack(
            [torch.from_numpy(np.array(image.resize((width, height))) / 255.0) for image in video], dim=0
        )  # [BT,H,W,3]
        video = einops.rearrange(video, "BT H W C -> BT C H W")  # [BT,3,H,W]
        video_grid = torchvision.utils.make_grid(video, nrow=B * T, padding=0).cpu()  # [3,H,W*BT]
        if video_grid.shape[2] < gt_grid.shape[2]:
            # the output from sampling function is less than the ground truth, so we need to pad the video_grid on the left
            pad_width = gt_grid.shape[2] - video_grid.shape[2]
            video_grid = torch.nn.functional.pad(video_grid, (pad_width, 0))  # [3,H,W*BT]
            video_grid[:, :, :pad_width] = gt_grid[
                :, :, :pad_width
            ]  # Pad the generated grid with the ground truth images

        log.info(f"video_grid: {video_grid.shape}, gt_grid: {gt_grid.shape}")
        display_image = torch.stack([video_grid, gt_grid], dim=0)  # [2,3,H,W*BT]
        display_image = torchvision.utils.make_grid(
            display_image, nrow=1, padding=2, pad_value=1.0
        )  # [3,H_total,W_total]

        log.info(
            f"Generated image: {video[0].shape} -> {video_grid.shape}, gt_image: {gt_image[0].shape} -> {gt_grid.shape} | display_image: {display_image.shape}"
        )
        if rank <= self.num_vis:
            display_image = einops.rearrange(display_image.numpy(), "C H W -> H W C") * 255.0  # [H,W,3]
            display_image = display_image.astype(np.uint8)
            display_image = Image.fromarray(display_image)
            current_width, current_height = display_image.size
            # reduce the image size to half
            display_image = display_image.resize((current_width // 2, current_height // 2))
            display_image.save(output_path + f"_rank_{rank}.jpg")
            caption_list = data_batch["raw_captions"][0]
            with open(output_path + f"_rank_{rank}.txt", "w") as f:
                f.write("top: generation, bottom: ground truth. Left to right: condition, generation\n")
                f.write(caption_list)

        # barrier
        distributed.barrier()
        if rank == 0 and wandb.run is not None:
            file_list = (
                sorted(glob.glob(output_path + "*.jpg"))[: self.num_vis]
                + sorted(glob.glob(output_path + "*.mp4"))[: self.num_vis]
            )
            caption_file_list = [file.replace(".jpg", ".txt").replace(".mp4", ".txt") for file in file_list]
            caption_list = [Path(caption_file).read_text() for caption_file in caption_file_list]
            wandb.log(
                {
                    "vis/generation": [
                        wandb.Image(file, caption=caption) for file, caption in zip(file_list, caption_list)
                    ]
                },
                step=iteration,
            )
