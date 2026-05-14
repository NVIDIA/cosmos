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

"""Callback for sampling and visualizing joint audio-video generation.

Extends the video sampling callback to also handle sound:
- Generates video and audio samples via model.generate_samples_from_batch()
- Logs video frames as image grids to WandB (same as EveryNDrawSample)
- Logs audio as WandB Audio objects
- Optionally creates combined video+audio MP4 files via ffmpeg
"""

import os
import subprocess
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
import torchvision
import wandb
from einops import rearrange

from cosmos3._src.imaginaire.callbacks.every_n import EveryN
from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.utils import distributed, log, misc
from cosmos3._src.vfm.callbacks.every_n_draw_sample import pad_images_and_cat, resize_image
from cosmos3._src.vfm.utils.data_utils import slice_data_batch


class EveryNDrawAudioVideoSample(EveryN):
    """Callback for sampling and visualizing joint audio-video generation.

    Samples from the model and logs both video frames and audio to WandB.

    Args:
        every_n: Frequency at which the callback is invoked
        step_size: Step size for the callback (default: 1)
        n_viz_sample: Number of samples to visualize in WandB (default: 3)
        num_sampling_step: Number of ODE integration steps (default: 35)
        guidance: List of guidance scales to try (default: [1.0, 3.0, 7.0])
        save_s3: Whether to save to S3 (default: False)
        is_ema: Whether to use EMA model (default: False)
        video_fps: FPS for video visualization (default: 24)
    """

    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        n_viz_sample: int = 3,
        num_sampling_step: int = 35,
        guidance: list[float] | None = None,
        save_s3: bool = False,
        is_ema: bool = False,
        video_fps: int = 24,
        generation_mode: str = "t2vs",
        run_at_start: bool = False,
    ):
        super().__init__(every_n, step_size, run_at_start=run_at_start)
        self.n_viz_sample = n_viz_sample
        self.save_s3 = save_s3
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.guidance = guidance if guidance is not None else [1.0, 3.0, 7.0]
        self.num_sampling_step = num_sampling_step
        self.rank = distributed.get_rank()
        self.video_fps = video_fps
        self.generation_mode = generation_mode

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")

        self.data_parallel_id = self.rank

        # Check if model has sound tokenizer
        self.has_sound = hasattr(model, "tokenizer_sound_gen") and model.tokenizer_sound_gen is not None
        if self.has_sound:
            self.audio_sample_rate = model.tokenizer_sound_gen.sample_rate
            log.info(
                f"[{self.name}] Audio-video callback initialized: "
                f"audio_sample_rate={self.audio_sample_rate}, is_ema={self.is_ema}"
            )
        else:
            self.audio_sample_rate = 48000
            log.warning(f"[{self.name}] Model does not have tokenizer_sound_gen, audio sampling disabled.")

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "every_n_av_sampling")
        else:
            context = nullcontext

        tag = "ema" if self.is_ema else "reg"

        with context():
            sample_results = self.sample(trainer, model, data_batch, output_batch, loss, iteration)
            dist.barrier()

        if wandb.run and self.rank == 0:
            info = {"trainer/global_step": iteration}

            # Log video grid
            if sample_results.get("video_grid_path"):
                info[f"{self.name}/{tag}_video"] = wandb.Image(
                    sample_results["video_grid_path"],
                    caption=f"iter={iteration}, guidance={self.guidance}",
                )

            # Log video+audio MP4s
            for i, (mp4_path, caption) in enumerate(sample_results.get("video_audio_samples", [])):
                if os.path.exists(mp4_path):
                    info[f"{self.name}/{tag}_video_audio_{i}"] = wandb.Video(
                        mp4_path, fps=self.video_fps, caption=caption
                    )

            # Log standalone audio
            for i, (audio_path, caption) in enumerate(sample_results.get("audio_samples", [])):
                if os.path.exists(audio_path):
                    info[f"{self.name}/{tag}_audio_{i}"] = wandb.Audio(
                        audio_path, sample_rate=self.audio_sample_rate, caption=caption
                    )

            # Log captions
            if sample_results.get("captions"):
                captions_text = "\n".join([f"{i}: {c}" for i, c in enumerate(sample_results["captions"])])
                info[f"{self.name}/{tag}_captions"] = wandb.Html(f"<pre>{captions_text}</pre>")

            wandb.log(info, step=iteration)

        torch.cuda.empty_cache()

    @misc.timer("EveryNDrawAudioVideoSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        """Generate audio-video samples and save results.

        Mode-aware behavior:
        - t2vs/ti2sv: Decode both video and sound from model output
        - tv2s: Use raw conditioning video for visualization, only decode sound
        - ts2v: Only decode video, skip sound visualization (conditioned)
        """
        data_batch = slice_data_batch(data_batch, start=0, limit=self.n_viz_sample)

        tag = "ema" if self.is_ema else "reg"
        mode = self.generation_mode
        results = {}

        # Get conditioning data
        gen_data_clean = model.get_data_and_condition(data_batch)
        raw_data = gen_data_clean.raw_state_vision
        x0 = gen_data_clean.x0_tokens_vision
        batch_size = len(x0)

        # Get captions for logging
        captions = data_batch.get(model.input_caption_key, [""] * batch_size)
        if isinstance(captions, torch.Tensor):
            captions = ["[tensor]"] * batch_size
        results["captions"] = captions[: self.n_viz_sample]

        # Determine what to decode based on mode
        # tv2s: Video is conditioning input (use raw_data), only sound is generated
        # ts2v: Sound is conditioning input, only video is generated
        # t2vs/ti2sv: Both are generated
        decode_video = mode not in ("tv2s",)  # Skip video decode when video is conditioning
        decode_sound = mode not in ("ts2v",)  # Skip sound decode when sound is conditioning

        video_samples_all = []
        audio_samples_all = []

        max_w = max(image.shape[-1] for image in raw_data)
        max_h = max(image.shape[-2] for image in raw_data)
        t_crop = min(image.shape[-3] for image in raw_data)

        for guidance in self.guidance:
            sample_output = model.generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                n_sample=self.n_viz_sample,
                num_steps=self.num_sampling_step,
                seed=list(range(iteration, iteration + self.n_viz_sample)),
            )

            # Video handling based on mode — decode one at a time and move to CPU immediately
            if decode_video:
                sample_vision = sample_output.get("vision", [])
                if sample_vision and hasattr(model, "decode"):
                    decoded_cpu = []
                    for i in range(len(sample_vision)):
                        dec = model.decode(sample_vision[i]).float().cpu()
                        decoded_cpu.append(dec)
                    video_samples_all.append(pad_images_and_cat(decoded_cpu, max_w, max_h, t_crop))
                    del decoded_cpu

            # Sound handling based on mode — decode one at a time and move to CPU immediately
            if decode_sound:
                sound_latents = sample_output.get("sound", [])
                if sound_latents and self.has_sound:
                    audio_cpu = []
                    for s in sound_latents:
                        dec = model.decode_sound(s).float().cpu()
                        audio_cpu.append(dec)
                    audio_samples_all.append(audio_cpu)
                    del audio_cpu

            # Free all GPU memory from this guidance iteration before next one
            del sample_output
            torch.cuda.empty_cache()

        # For tv2s: Use raw conditioning video instead of decoded (avoids wasteful VAE round-trip)
        if mode == "tv2s":
            conditioning_video = (
                pad_images_and_cat(raw_data[: min(self.n_viz_sample, batch_size)], max_w, max_h, t_crop).float().cpu()
            )
            # Use conditioning video for all guidance scales (same video, different audio)
            video_for_mp4 = conditioning_video
        else:
            video_for_mp4 = None  # Will use per-guidance decoded video below

        # Add ground truth video for comparison (skip for tv2s where video isn't generated)
        if decode_video:
            video_samples_all.append(
                pad_images_and_cat(raw_data[: min(self.n_viz_sample, batch_size)], max_w, max_h, t_crop).float().cpu()
            )

        # Save video grid (skip for tv2s — video evaluation should be done separately)
        if video_samples_all and decode_video:
            video_grid_path = self._save_video_grid(video_samples_all, batch_size, tag, iteration)
            if video_grid_path:
                results["video_grid_path"] = video_grid_path

        # Save audio samples and video+audio MP4s
        if audio_samples_all and self.rank == 0:
            audio_paths = []
            video_audio_paths = []

            # Get conditioning FPS for video playback
            conditioning_fps = data_batch.get("conditioning_fps", None)
            if conditioning_fps is not None and isinstance(conditioning_fps, (torch.Tensor, list)):
                video_write_fps = float(
                    conditioning_fps[0].item() if isinstance(conditioning_fps, torch.Tensor) else conditioning_fps[0]
                )
            else:
                video_write_fps = self.video_fps

            for g_idx, audio_batch in enumerate(audio_samples_all):
                # Determine which video to pair with audio for MP4
                if video_for_mp4 is not None:
                    # tv2s: Use conditioning video for all guidance scales
                    video_batch = video_for_mp4
                elif g_idx < len(video_samples_all) - 1:
                    # t2vs/ti2sv: Use decoded video from this guidance scale (exclude GT at end)
                    video_batch = video_samples_all[g_idx]
                else:
                    video_batch = None

                for sample_idx in range(min(self.n_viz_sample, len(audio_batch))):
                    audio_waveform = audio_batch[sample_idx]  # [C,N_samples]

                    # Save standalone audio
                    audio_path = self._save_audio(audio_waveform, tag, iteration, g_idx, sample_idx)
                    if audio_path:
                        caption = f"mode={mode}, guidance={self.guidance[g_idx]}, sample={sample_idx}"
                        if sample_idx < len(captions):
                            caption += f", caption: {captions[sample_idx][:100]}"
                        audio_paths.append((audio_path, caption))

                    # Create video+audio MP4
                    if video_batch is not None and sample_idx < video_batch.shape[0]:
                        video_tensor = video_batch[sample_idx]  # [C,T,H,W]
                        mp4_path = self._save_video_with_audio(
                            video_tensor,
                            audio_waveform,
                            tag,
                            iteration,
                            g_idx,
                            sample_idx,
                            fps=video_write_fps,
                        )
                        if mp4_path:
                            video_audio_paths.append((mp4_path, caption))

            results["audio_samples"] = audio_paths
            results["video_audio_samples"] = video_audio_paths

        return results

    def _save_video_grid(
        self, video_samples: list[torch.Tensor], batch_size: int, tag: str, iteration: int
    ) -> str | None:
        """Save video samples as image grid for WandB."""
        if self.rank != 0 or not wandb.run:
            return None

        to_show = (1.0 + torch.stack(video_samples, dim=0).clamp(-1, 1)) / 2.0  # [N_rows,B,C,T,H,W]  range [0,1]
        n_viz_sample = min(self.n_viz_sample, batch_size)
        is_single_frame = to_show.shape[3] == 1

        file_base_fp = f"{tag}_AV_Video_Iter{iteration:09d}.jpg"
        local_path = f"{self.local_dir}/{file_base_fp}"

        if is_single_frame:
            to_show = rearrange(
                to_show[:, :n_viz_sample], "n b c t h w -> t c (n h) (b w)"
            )  # [1,C,N_rows*H,B*W]  (t=1)
            image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
            torchvision.utils.save_image(resize_image(image_grid, 1024), local_path)
        else:
            to_show = to_show[:, :n_viz_sample]  # [N_rows,B,C,T,H,W]
            _T = to_show.shape[3]
            three_frames_list = [0, _T // 2, _T - 1]
            to_show = to_show[:, :, :, three_frames_list]  # [N_rows,B,C,3,H,W]
            to_show = rearrange(to_show, "n b c t h w -> 1 c (n h) (b t w)")  # [1,C,N_rows*H,B*3*W]  (t=3)
            image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
            torchvision.utils.save_image(resize_image(image_grid, 1024), local_path)

        return local_path

    def _save_audio(
        self, audio_waveform: torch.Tensor, tag: str, iteration: int, guidance_idx: int, sample_idx: int
    ) -> str | None:
        """Save audio waveform as WAV file."""
        if self.rank != 0:
            return None
        try:
            import soundfile as sf

            file_name = f"{tag}_Audio_Iter{iteration:09d}_g{guidance_idx}_s{sample_idx}.wav"
            local_path = f"{self.local_dir}/{file_name}"

            audio_np = audio_waveform.clamp(-1, 1).numpy()
            if audio_np.ndim == 2:
                audio_np = audio_np.T  # [C, N] → [N, C] for soundfile

            sf.write(local_path, audio_np, self.audio_sample_rate)
            return local_path
        except Exception as e:
            log.warning(f"Failed to save audio: {e}", rank0_only=False)
            return None

    def _save_video_with_audio(
        self,
        video_tensor: torch.Tensor,
        audio_tensor: torch.Tensor,
        tag: str,
        iteration: int,
        guidance_idx: int,
        sample_idx: int,
        fps: float | None = None,
    ) -> str | None:
        """Create MP4 video with audio using ffmpeg."""
        video_fps = fps if fps is not None else self.video_fps
        if self.rank != 0:
            return None
        try:
            import soundfile as sf

            file_base = f"{tag}_VideoAudio_Iter{iteration:09d}_g{guidance_idx}_s{sample_idx}"
            mp4_path = f"{self.local_dir}/{file_base}.mp4"
            temp_video_path = f"{self.local_dir}/{file_base}_temp.mp4"
            temp_audio_path = f"{self.local_dir}/{file_base}_temp.wav"

            # Save video frames as temp MP4
            video_frames = video_tensor.permute(1, 0, 2, 3)  # [T,C,H,W]
            video_frames = (video_frames.clamp(-1, 1) + 1) / 2  # [T,C,H,W]  range [0,1]
            video_frames = (video_frames * 255).to(torch.uint8)  # [T,C,H,W]

            torchvision.io.write_video(
                temp_video_path,
                video_frames.permute(0, 2, 3, 1).cpu(),  # [T,H,W,C]
                fps=video_fps,
                video_codec="libx264",
            )

            # Save audio as temp WAV
            audio_np = audio_tensor.clamp(-1, 1).numpy()
            if audio_np.ndim == 2:
                audio_np = audio_np.T
            sf.write(temp_audio_path, audio_np, self.audio_sample_rate)

            # Combine with ffmpeg
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                temp_video_path,
                "-i",
                temp_audio_path,
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-shortest",
                "-loglevel",
                "error",
                mp4_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log.warning(f"ffmpeg failed: {result.stderr}")
                return None

            # Cleanup
            for f in [temp_video_path, temp_audio_path]:
                if os.path.exists(f):
                    os.remove(f)

            return mp4_path
        except Exception as e:
            log.warning(f"Failed to create video with audio: {e}", rank0_only=False)
            return None
