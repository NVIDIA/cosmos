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

import re
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
import torchvision
import wandb
from einops import rearrange, repeat

from cosmos3._src.imaginaire.utils import log, misc
from cosmos3._src.imaginaire.utils.easy_io import easy_io
from cosmos3._src.imaginaire.visualize.video import save_img_or_video
from cosmos3._src.vfm.callbacks.every_n_draw_sample import (
    EveryNDrawSample,
    convert_to_primitive,
    is_primitive,
    pad_images_and_cat,
    resize_image,
)
from cosmos3._src.vfm.utils.data_utils import slice_data_batch


class EveryNDurationFPSDrawSample(EveryNDrawSample):
    """
    Callback to visualize samples with specific Duration/FPS metadata control.
    It performs two types of generation:
    1. Standard generation (using the batch as-is).
    2. "Consistent" generation: Rewrites the duration/FPS metadata in the caption
       to match the actual video FPS and generated frame count, then generates again.
    The "Consistent" results are logged as individual windows per sample to show
    the exact caption used.
    """

    @misc.timer("EveryNDurationFPSDrawSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        data_batch = slice_data_batch(data_batch, start=0, limit=self.n_viz_sample)

        tag = "ema" if self.is_ema else "reg"
        results = {}

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

        # Setup negative prompts if needed
        if self.use_negative_prompt:
            batch_size = len(x0)
            if self.negative_prompt_data["t5_text_embeddings"].shape != data_batch["t5_text_embeddings"].shape:
                data_batch["neg_t5_text_embeddings"] = misc.to(
                    repeat(
                        self.negative_prompt_data["t5_text_embeddings"],
                        "... -> b ...",
                        b=batch_size,
                    ),  # [B,N_tokens,D]
                    **model.tensor_kwargs,
                )
            else:
                data_batch["neg_t5_text_embeddings"] = misc.to(
                    self.negative_prompt_data["t5_text_embeddings"],
                    **model.tensor_kwargs,
                )
            data_batch["neg_t5_text_mask"] = data_batch["t5_text_mask"]

        # Compute max dimensions for padding (supports variable shapes)
        max_w = max(image.shape[-1] for image in raw_data)
        max_h = max(image.shape[-2] for image in raw_data)
        t_crop = min(image.shape[-3] for image in raw_data)

        # Helper to run generation for a specific data batch configuration
        def _generate_and_save(batch_to_use, suffix="", split_batch=False, save_fps=None):
            to_show = []
            for guidance in self.guidance:
                sample = model.generate_samples_from_batch(
                    batch_to_use,
                    guidance=guidance,
                    n_sample=self.n_viz_sample,
                    num_steps=self.num_sampling_step,
                    has_negative_prompt=True if self.use_negative_prompt else False,
                    seed=list(range(iteration, iteration + self.n_viz_sample)),
                )
                sample_vision = sample["vision"]
                if hasattr(model, "decode"):
                    sample_vision_decoded = [model.decode(sample_vision[i]) for i in range(len(sample_vision))]
                else:
                    sample_vision_decoded = sample_vision
                to_show.append(pad_images_and_cat(sample_vision_decoded, max_w, max_h, t_crop).float().cpu())

            to_show.append(
                pad_images_and_cat(raw_data[: len(sample_vision_decoded)], max_w, max_h, t_crop).float().cpu()
            )

            base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}{suffix}"
            batch_size = len(x0)

            if split_batch:
                # When splitting, run_save_split returns keys like "_0", "_1"
                # We need to prepend the suffix (e.g. "_consistent") to these keys
                # so they become "_consistent_0", "_consistent_1"
                split_results = self.run_save_split(to_show, batch_size, base_fp_wo_ext, save_fps=save_fps)
                return {f"{suffix}{k}": v for k, v in split_results.items()}
            else:
                return self.run_save(to_show, batch_size, base_fp_wo_ext)

        # 1. Standard generation
        results[""] = _generate_and_save(data_batch)

        # 2. "Consistent Duration/FPS" Variation
        is_video = not model.is_image_batch(data_batch)
        input_caption_key = getattr(model, "input_caption_key", "ai_caption")

        if is_video and input_caption_key in data_batch and "conditioning_fps" in data_batch:
            original_captions = data_batch[input_caption_key]

            batch_copy = data_batch.copy()
            fps_values = data_batch["conditioning_fps"]

            new_captions = []
            for i, cap in enumerate(original_captions):
                new_captions.append(cap)

            batch_copy[input_caption_key] = new_captions

            # Re-compute embeddings
            if text_encoder_config is not None and text_encoder_config.compute_online:
                text_embeddings = model.text_encoder.compute_text_embeddings_online(batch_copy, input_caption_key)
                batch_copy["t5_text_embeddings"] = text_embeddings
                batch_copy["t5_text_mask"] = torch.ones(
                    text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
                )  # [B,N_tokens]  (all tokens valid)
                if "neg_t5_text_embeddings" in batch_copy:
                    pass

            # Pass captions back so we can log them
            batch_result = _generate_and_save(batch_copy, suffix="_consistent", split_batch=True, save_fps=fps_values)
            # Attach captions to the result dictionary for logging
            batch_result["__captions__"] = new_captions
            results.update(batch_result)

        return results

    def run_save_split(self, to_show, batch_size, base_fp_wo_ext, save_fps=None) -> dict:
        """
        Similar to run_save but splits the batch into individual images.
        """
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [N_rows,B,C,T,H,W]  range [0,1]
        n_viz_sample = min(self.n_viz_sample, batch_size)

        # We assume video here since we checked is_video
        to_show_full = to_show[:, :n_viz_sample]  # [N_rows,B,C,T,H,W] - Keep full video for S3 saving
        _T = to_show_full.shape[3]

        # Save individual FULL videos to S3 if enabled (before 3-frame reduction)
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            for i in range(n_viz_sample):
                # Extract individual FULL video from batch
                individual_video = to_show_full[:, i : i + 1]  # [n, 1, c, T, h, w] - FULL video

                # Get FPS for this specific batch item
                item_fps = self.fps  # fallback
                if save_fps is not None:
                    if isinstance(save_fps, torch.Tensor):
                        item_fps = save_fps[i].item() if save_fps.ndim > 0 else save_fps.item()
                    elif isinstance(save_fps, (list, tuple)):
                        item_fps = save_fps[i]
                    else:
                        item_fps = float(save_fps)

                # Save individual FULL video to S3
                save_img_or_video(
                    rearrange(individual_video, "n b c t h w -> c t (n h) (b w)"),  # [C,T,N_rows*H,W]
                    f"s3://rundir/{self.name}/{base_fp_wo_ext}_{i}",
                    fps=item_fps,
                )

        # NOW reduce to 3 frames for WandB visualization only
        three_frames_list = [0, _T // 2, _T - 1]
        to_show_3frames = to_show_full[:, :, :, three_frames_list]  # [N_rows,B,C,3,H,W]
        log_image_size = 1024

        # Save individual FULL videos to S3 if enabled (before 3-frame reduction)
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            for i in range(n_viz_sample):
                # Extract individual FULL video from batch
                individual_video = to_show_full[:, i : i + 1]  # [N_rows,1,C,T,H,W]

                # Get FPS for this specific batch item
                item_fps = self.fps  # fallback
                if save_fps is not None:
                    if isinstance(save_fps, torch.Tensor):
                        item_fps = save_fps[i].item() if save_fps.ndim > 0 else save_fps.item()
                    elif isinstance(save_fps, (list, tuple)):
                        item_fps = save_fps[i]
                    else:
                        item_fps = float(save_fps)

                # Save individual FULL video to S3
                save_img_or_video(
                    rearrange(individual_video, "n b c t h w -> c t (n h) (b w)"),  # [C,T,N_rows*H,W]
                    f"s3://rundir/{self.name}/{base_fp_wo_ext}_{i}",
                    fps=item_fps,
                )

        # NOW reduce to 3 frames for WandB visualization only
        three_frames_list = [0, _T // 2, _T - 1]
        to_show_3frames = to_show_full[:, :, :, three_frames_list]  # [N_rows,B,C,3,H,W]
        log_image_size = 1024

        paths = {}
        for i in range(n_viz_sample):
            sample_data = to_show_3frames[:, i : i + 1]  # [N_rows,1,C,3,H,W]
            sample_grid_data = rearrange(sample_data, "n b c t h w -> 1 c (n h) (b t w)")  # [1,C,N_rows*H,3*W]  (t=3)

            sample_path = f"{self.local_dir}/{base_fp_wo_ext}_{i}_resize.jpg"
            if self.rank == 0:
                image_grid = torchvision.utils.make_grid(sample_grid_data, nrow=1, padding=0, normalize=False)
                torchvision.utils.save_image(
                    resize_image(image_grid, log_image_size), sample_path, nrow=1, scale_each=True
                )
            paths[f"_{i}"] = sample_path
        return paths

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
        # Log batch info logic from base class...
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
                f"s3://rundir/{self.name}/BatchInfo_ReplicateID{self.data_parallel_id:04d}_Iter{iteration:09d}.json",
            )

        log.debug("entering, every_n_impl", rank0_only=False)
        with context():
            # Skipping x0_pred for brevity in this specialized callback
            log.debug("entering, sample", rank0_only=False)
            sample_img_paths = self.sample(
                trainer,
                model,
                data_batch,
                output_batch,
                loss,
                iteration,
            )
            log.debug("done, sample", rank0_only=False)
            dist.barrier()

        if wandb.run:
            data_type = "image" if model.is_image_batch(data_batch) else "video"
            tag += f"_{data_type}"
            info = {
                "trainer/global_step": iteration,
                "sample_counter": sample_counter,
            }
            # Handle dictionary of paths
            if isinstance(sample_img_paths, dict):
                # Retrieve captions if available
                consistent_captions = sample_img_paths.get("__captions__", [])

                # Log standard (key "")
                if "" in sample_img_paths:
                    info[f"{self.name}/{tag}_sample"] = wandb.Image(sample_img_paths[""], caption=f"{sample_counter}")

                # Log consistent variations (keys "_consistent_0", etc)
                for suffix, path in sample_img_paths.items():
                    if suffix == "" or suffix == "__captions__":
                        continue

                    caption_text = f"{sample_counter}{suffix}"

                    if "_consistent_" in suffix:
                        try:
                            idx = int(suffix.split("_")[-1])
                            if idx < len(consistent_captions):
                                full_caption = consistent_captions[idx]
                                # Extract duration and FPS values and prepend for WandB display
                                duration_match = re.search(r"(\d+\.?\d*)\s+seconds?", full_caption)
                                fps_match = re.search(r"(\d+\.?\d*)\s+FPS", full_caption, re.IGNORECASE)

                                if duration_match and fps_match:
                                    duration = duration_match.group(1)
                                    fps = fps_match.group(1)
                                    caption_text = f"(Dur: {duration}s, FPS: {fps}fps) {full_caption}"
                                else:
                                    caption_text = full_caption  # No metadata found, use as-is
                        except Exception as e:
                            log.warning(f"Failed to parse suffix '{suffix}' for caption lookup: {e}")

                    info[f"{self.name}/{tag}_sample{suffix}"] = wandb.Image(path, caption=caption_text)

            wandb.log(info, step=iteration)
        torch.cuda.empty_cache()
