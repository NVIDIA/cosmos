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
import os
from contextlib import nullcontext
from functools import partial
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as torchvision_F
import wandb
from einops import rearrange

from cosmos.callbacks.every_n import EveryN
from cosmos.model._base import ImaginaireModel
from cosmos.utils import distributed, log, misc
from cosmos.utils.easy_io import easy_io
from cosmos.tools.visualize.video import save_img_or_video
from cosmos.utils.vfm.data_utils import slice_data_batch


def resize_image(image: torch.Tensor, size: int = 1024) -> torch.Tensor:
    """
    Resize the image to the given size. This is done so that wandb can display the image correctly.
    """
    _, h, w = image.shape
    ratio = size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    return torchvision_F.resize(image, (new_h, new_w))


def is_primitive(value):
    return isinstance(value, (int, float, str, bool, type(None)))


def convert_to_primitive(value):
    if isinstance(value, (list, tuple)):
        return [convert_to_primitive(v) for v in value if is_primitive(v) or isinstance(v, (list, dict))]
    elif isinstance(value, dict):
        return {k: convert_to_primitive(v) for k, v in value.items() if is_primitive(v) or isinstance(v, (list, dict))}
    elif is_primitive(value):
        return value
    else:
        return "non-primitive"  # Skip non-primitive types


def pad_images_and_cat(images: List[torch.Tensor], max_w: int, max_h: int, t_crop: int = 1) -> torch.Tensor:
    """
    Pad images to a common size and concatenate them along the batch dimension.

    This function is needed because different samples in a batch can have different resolutions.
    To create a unified visualization grid, all images must be padded to the same dimensions.
    Images are center-padded to preserve their visual content in the middle.

    Args:
        images: List of image/video tensors with shape [B, C, T, H, W].
        max_w: Target width to pad all images to.
        max_h: Target height to pad all images to.
        t_crop: Number of temporal frames to keep for videos. If > 1 and the image
            has more than 1 frame, only the first t_crop frames are retained.

    Returns:
        Concatenated tensor of padded images with shape [total_B, C, T, max_h, max_w].
    """
    padded_images = []
    for image in images:
        # Pad the image to the center
        padding_h = (max_h - image.shape[-2]) // 2
        padding_w = (max_w - image.shape[-1]) // 2
        padded_image = torch.nn.functional.pad(
            image, (padding_w, max_w - image.shape[-1] - padding_w, padding_h, max_h - image.shape[-2] - padding_h)
        )  # [B,C,T,max_h,max_w]
        # Handle video case
        if image.shape[2] > 1 and t_crop > 1:
            padded_image = padded_image[:, :, 0:t_crop, :, :]

        padded_images.append(padded_image)
    return torch.cat(padded_images, dim=0)  # [total_B,C,T,max_h,max_w]  (total_B = sum of batch dims)


class EveryNDrawSample(EveryN):
    """
    This callback sample condition inputs from training data, run inference and save the results to wandb and s3.

    Args:
        every_n (int): The frequency at which the callback is invoked.
        step_size (int, optional): The step size for the callback. Defaults to 1.
        n_viz_sample (int, optional): for each batch, min(n_viz_sample, batch_size) samples will be saved to wandb. Defaults to 3.
        n_sample_to_save (int, optional): number of samples to save. The actual number of samples to save is min(n_sample_to_save, data parallel instances). Defaults to 128.
        num_sampling_step (int, optional): number of sampling steps. Defaults to 35.
        guidance (List[float], optional): guidance scale. Defaults to [0.0, 3.0, 7.0].
        do_x0_prediction (bool, optional): whether to do x0 prediction. Defaults to True.
        n_sigmas_for_x0_prediction (int, optional): number of sigmas to use for x0 prediction. Defaults to 4.
        save_s3 (bool, optional): whether to save to s3. Defaults to False.
        is_ema (bool, optional): whether the callback is run for ema model. Defaults to False.
        use_negative_prompt (bool, optional): whether to use negative prompt. Defaults to False.
        fps (int, optional): frames per second when saving the video. Defaults to 16.
    """

    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        n_viz_sample: int = 2,
        n_sample_to_save: int = 128,
        num_sampling_step: int = 35,
        guidance: List[float] = [0.0, 3.0, 7.0],
        do_x0_prediction: bool = True,
        n_sigmas_for_x0_prediction: int = 4,
        save_s3: bool = False,
        save_local: bool = False,
        is_ema: bool = False,
        use_negative_prompt: bool = False,
        prompt_type: str = "t5_xxl",
        fps: int = 16,
        run_at_start: bool = False,
    ):
        # s3: # files: min(n_sample_to_save, data instance)  # per file: min(batch_size, n_viz_sample)
        # wandb: 1 file, # per file: min(batch_size, n_viz_sample)
        super().__init__(every_n, step_size, run_at_start=run_at_start)

        self.n_viz_sample = n_viz_sample
        self.n_sample_to_save = n_sample_to_save
        self.save_s3 = save_s3
        self.save_local = save_local
        self.do_x0_prediction = do_x0_prediction
        self.n_sigmas_for_x0_prediction = n_sigmas_for_x0_prediction
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.use_negative_prompt = use_negative_prompt
        self.prompt_type = prompt_type
        self.guidance = guidance
        self.num_sampling_step = num_sampling_step
        self.rank = distributed.get_rank()
        self.fps = fps

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")

        self.data_parallel_id = self.rank

    @misc.timer("EveryNDrawSample: x0")
    @torch.no_grad()
    def x0_pred(self, trainer, model, data_batch, output_batch, loss, iteration):
        tag = "ema" if self.is_ema else "reg"

        log.debug("starting data and condition model", rank0_only=False)


        data_clean = model.get_data_and_condition(data_batch)
        raw_data = data_clean.raw_state_vision
        x0 = data_clean.x0_tokens_vision

        # Handle model parallelism if available (legacy models)
        if hasattr(model, "broadcast_split_for_model_parallelsim"):
            _, condition, x0, _ = model.broadcast_split_for_model_parallelsim(None, None, x0, None)

        log.debug("done data and condition model", rank0_only=False)
        batch_size = len(x0)
        sigmas = np.exp(
            np.linspace(
                math.log(model.sde.sigma_min), math.log(model.sde.sigma_max), self.n_sigmas_for_x0_prediction + 1
            )[1:]
        )

        to_show = []
        generator = torch.Generator(device="cuda")
        generator.manual_seed(0)
        random_noise = torch.randn(*x0.shape, generator=generator, **model.tensor_kwargs)  # same shape as x0
        _ones = torch.ones(batch_size, **model.tensor_kwargs)  # [B]
        mse_loss_list = []
        for _, sigma in enumerate(sigmas):
            x_sigma = sigma * random_noise + x0
            log.debug(f"starting denoising {sigma}", rank0_only=False)
            sample = model.denoise(x_sigma, None).x0
            log.debug(f"done denoising {sigma}", rank0_only=False)
            mse_loss = distributed.dist_reduce_tensor(F.mse_loss(sample, x0))
            mse_loss_list.append(mse_loss)

            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())
        to_show.append(
            raw_data.float().cpu(),
        )

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_x0_Iter{iteration:09d}"

        local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
        return local_path, torch.tensor(mse_loss_list).cuda(), sigmas  # [N_sigmas]

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "every_n_sampling")
        else:
            context = nullcontext

        tag = "ema" if self.is_ema else "reg"
        sample_counter = getattr(trainer, "sample_counter", iteration)
        batch_info = {
            "data": {
                k: convert_to_primitive(v)
                for k, v in data_batch.items()
                if is_primitive(v) or isinstance(v, (list, dict))
            },
            "sample_counter": sample_counter,
            "iteration": iteration,
        }
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            easy_io.dump(
                batch_info,
                f"s3://rundir/{self.name}/Iter{iteration:09d}/BatchInfo_ReplicateID{self.data_parallel_id:04d}_Iter{iteration:09d}.json",
            )

        log.debug("entering, every_n_impl", rank0_only=False)
        with context():
            if self.do_x0_prediction:
                log.debug("entering, x0_pred", rank0_only=False)
                x0_img_fp, mse_loss, sigmas = self.x0_pred(
                    trainer,
                    model,
                    data_batch,
                    output_batch,
                    loss,
                    iteration,
                )
                log.debug("done, x0_pred", rank0_only=False)
                if self.save_s3 and self.rank == 0:
                    easy_io.dump(
                        {
                            "mse_loss": mse_loss.tolist(),
                            "sigmas": sigmas.tolist(),
                            "iteration": iteration,
                        },
                        f"s3://rundir/{self.name}/{tag}_MSE_Iter{iteration:09d}.json",
                    )

            log.debug("entering, sample", rank0_only=False)
            sample_img_fp = self.sample(
                trainer,
                model,
                data_batch,
                output_batch,
                loss,
                iteration,
            )
            log.debug("done, sample", rank0_only=False)

            log.debug("waiting for all ranks to finish", rank0_only=False)
            dist.barrier()
        if wandb.run:
            sample_counter = getattr(trainer, "sample_counter", iteration)
            data_type = "image" if model.is_image_batch(data_batch) else "video"
            tag += f"_{data_type}"
            info = {
                "trainer/global_step": iteration,
                "sample_counter": sample_counter,
            }
            if self.do_x0_prediction:
                info[f"{self.name}/{tag}_x0"] = wandb.Image(x0_img_fp, caption=f"{sample_counter}")
                # convert mse_loss to a dict
                mse_loss = mse_loss.tolist()
                info.update({f"x0_pred_mse_{tag}/Sigma{sigmas[i]:0.5f}": mse_loss[i] for i in range(len(mse_loss))})

            info[f"{self.name}/{tag}_sample"] = wandb.Image(sample_img_fp, caption=f"{sample_counter}")
            wandb.log(
                info,
                step=iteration,
            )
        torch.cuda.empty_cache()

    @misc.timer("EveryNDrawSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        data_batch = slice_data_batch(data_batch, start=0, limit=self.n_viz_sample)

        tag = "ema" if self.is_ema else "reg"

        # Obtain text embeddings online
        text_encoder_config = getattr(model.config, "text_encoder_config", None)
        if text_encoder_config is not None and text_encoder_config.compute_online:
            text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )  # [B,N_tokens]  (all tokens valid)

        data_clean = model.get_data_and_condition(data_batch)
        raw_data = data_clean.raw_state_vision
        x0 = data_clean.x0_tokens_vision

        # determine the number of visualized samples
        n_viz_sample = min(self.n_viz_sample, data_clean.batch_size)

        # Check if this is a multi-item vision batch (image editing)
        num_items = data_clean.num_vision_items_per_sample
        is_multi_item = num_items is not None

        if is_multi_item:
            # Image editing: raw_data is flat [src1, tgt1, src2, tgt2, ...].
            # Split into per-sample condition (source) and GT target images.
            condition_images: list[torch.Tensor] = []
            gt_target_images: list[torch.Tensor] = []
            vis_offset = 0
            for sample_idx in range(data_clean.batch_size):
                n_vis = num_items[sample_idx]
                # First item(s) are condition, last item is generation target

                # but we need to support multiple conditions per sample in the future. Current code
                # can handle this without throwing an error.
                condition_images.append(raw_data[vis_offset])  # source image (1, C, 1, H, W)
                gt_target_images.append(raw_data[vis_offset + n_vis - 1])  # target image (1, C, 1, H, W)
                vis_offset += n_vis

            # Use target images for max_w/max_h/t_crop (generated samples match target size)
            max_w = max(img.shape[-1] for img in gt_target_images)
            max_h = max(img.shape[-2] for img in gt_target_images)
            t_crop = min(img.shape[-3] for img in gt_target_images)
        else:
            max_w = max(image.shape[-1] for image in raw_data)
            max_h = max(image.shape[-2] for image in raw_data)
            t_crop = min(image.shape[-3] for image in raw_data)

        to_show = []

        # Row 0 (image editing only): condition (source) images
        if is_multi_item:
            to_show.append(pad_images_and_cat(condition_images[:n_viz_sample], max_w, max_h, t_crop).float().cpu())

        for guidance in self.guidance:
            sample = model.generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                n_sample=n_viz_sample,
                num_steps=self.num_sampling_step,
                has_negative_prompt=True if self.use_negative_prompt else False,
                seed=list(range(iteration, iteration + n_viz_sample)),
            )
            sample_vision = sample["vision"]
            assert hasattr(model, "decode")
            sample_vision_decoded = [model.decode(sample_vision_i) for sample_vision_i in sample_vision]
            assert len(sample_vision_decoded) == n_viz_sample
            to_show.append(pad_images_and_cat(sample_vision_decoded, max_w, max_h, t_crop).float().cpu())

        # Last row: ground truth
        if is_multi_item:
            # Image editing: show GT target images (not the flat raw_data which mixes src + tgt)
            assert len(gt_target_images) == n_viz_sample
            to_show.append(pad_images_and_cat(gt_target_images, max_w, max_h, t_crop).float().cpu())
        else:
            assert len(raw_data) == n_viz_sample
            to_show.append(pad_images_and_cat(raw_data, max_w, max_h, t_crop).float().cpu())

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"
        base_fp_wo_ext = f"Iter{iteration:09d}/{base_fp_wo_ext}"

        batch_size = data_clean.batch_size
        local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
        return local_path

    def run_save(self, to_show, batch_size, base_fp_wo_ext) -> Optional[str]:
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [N_rows,B,C,T,H,W]  range [0,1]
        is_single_frame = to_show.shape[3] == 1
        n_viz_sample = min(self.n_viz_sample, batch_size)
        to_show = to_show[:, :n_viz_sample]

        # ! we only save first n_sample_to_save video!
        video_grid = rearrange(to_show, "n b c t h w -> c t (n h) (b w)")  # [C,T,N_rows*H,B*W]
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            save_img_or_video(
                video_grid,
                f"s3://rundir/{self.name}/{base_fp_wo_ext}",
                fps=self.fps,
            )
        if self.save_local and self.data_parallel_id < self.n_sample_to_save:
            local_video_path = f"{self.local_dir}/{base_fp_wo_ext}"
            os.makedirs(os.path.dirname(local_video_path), exist_ok=True)
            save_img_or_video(video_grid, local_video_path, fps=self.fps)

        file_base_fp = f"{base_fp_wo_ext}_resize.jpg"
        local_path = f"{self.local_dir}/{file_base_fp}"

        if self.rank == 0 and wandb.run:
            if is_single_frame:  # image case
                to_show = rearrange(
                    to_show[:, :n_viz_sample],
                    "n b c t h w -> t c (n h) (b w)",
                )  # [1,C,N_rows*H,B*W]  (t=1 for images)
                image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
                # resize so that wandb can handle it
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                torchvision.utils.save_image(resize_image(image_grid, 1024), local_path, nrow=1, scale_each=True)
            else:
                to_show = to_show[:, :n_viz_sample]  # [N_rows,B,C,T,H,W]

                # resize 3 frames frames so that we can display them on wandb
                _T = to_show.shape[3]
                three_frames_list = [0, _T // 2, _T - 1]
                to_show = to_show[:, :, :, three_frames_list]  # [N_rows,B,C,3,H,W]  (3 sampled frames)
                log_image_size = 1024
                to_show = rearrange(
                    to_show,
                    "n b c t h w -> 1 c (n h) (b t w)",
                )  # [1,C,N_rows*H,B*3*W]  (t=3 sampled frames)

                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                # resize so that wandb can handle it
                image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                torchvision.utils.save_image(
                    resize_image(image_grid, log_image_size), local_path, nrow=1, scale_each=True
                )

            return local_path
        return None
