# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""ModelOpt FP8 calibration machinery for the Cosmos3 DiT.

Everything the generation (t2v / t2i / i2v) calibration needs: the quant config,
the never-quantize skip filter, the calibration prompt/image loaders, and
:func:`make_forward_loop` — a forward loop that replays real denoising so the
ModelOpt ``max`` calibrator sees production-representative activations.

Generation-only by design: this cookbook does not calibrate the Qwen3-VL reasoning
tower (no reasoning overlay), so the LM-calibration phase of the NIM pipeline is
intentionally omitted.
"""

from __future__ import annotations

import copy
import itertools
import json
import re
from pathlib import Path

import modelopt.torch.quantization as mtq
import torch
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.model.generator.diffusion.samplers.fixed_step import (
    FixedStepSampler,
)
from cosmos_framework.model.generator.utils.data_and_condition import (
    GenerationDataClean,
)
from datasets import load_dataset
from modelopt.torch.utils.vlm_dataset_utils import get_vlm_dataset_dataloader
from PIL import Image

QUANT_CONFIGS = {"fp8": mtq.FP8_DEFAULT_CFG}

# Modules that should never be quantized (norms, embeddings, RoPE, patch
# projections). ``_NO_LM`` keeps the UND transformer linears eligible (we quantize
# the generation + understanding towers); the plain variant additionally excludes
# the whole ``language_model`` subtree.
_BASE_SKIP_TOKENS = (
    "language_model|time_embedder|vae2llm|llm2vae|embed_tokens|"
    "norm_q|norm_k|input_layernorm|post_attention_layernorm|"
    "norm_moe_gen|rotary_emb|action2llm|llm2action|sound2llm|llm2sound"
)
_BASE_SKIP_TOKENS_NO_LM = (
    "time_embedder|vae2llm|llm2vae|embed_tokens|"
    "norm_q|norm_k|input_layernorm|post_attention_layernorm|"
    "norm_moe_gen|rotary_emb|action2llm|llm2action|sound2llm|llm2sound"
)

# Video latent channel count (Cosmos3/Wan VAE).
NUM_CHANNELS = 48


def build_quant_config(algo: str = "max") -> dict:
    """FP8 quant config for the requested calibration algorithm.

    Only per-tensor ``max`` (the production recipe) is wired up here; the NIM
    pipeline's smoothquant/awq/percentile variants are out of scope for the cookbook.
    """
    cfg = copy.deepcopy(QUANT_CONFIGS["fp8"])
    if algo != "max":
        raise ValueError(f"cookbook supports --quant-algo max only, got {algo!r}")
    return cfg


def count_gen_layers(mdl) -> int:
    """Detect how many ``gen_layers.<i>.*`` blocks exist in the model."""
    max_idx = -1
    for name, _ in mdl.named_modules():
        m = re.match(r"gen_layers\.(\d+)\.", name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def build_filter(
    num_gen_layers: int,
    quantize_language_model: bool = True,
    skip_generation_quant: bool = False,
):
    """Return ``module_name -> True`` if its quantizers should be disabled."""
    tokens = [_BASE_SKIP_TOKENS_NO_LM if quantize_language_model else _BASE_SKIP_TOKENS]
    if skip_generation_quant:
        tokens.append(r"gen_layers\.")
    pattern = re.compile(r".*(" + "|".join(tokens) + r").*")

    def filter_func(name: str) -> bool:
        return pattern.match(name) is not None

    return filter_func


@torch.no_grad()
def prepare_calibration_prompts(num_samples: int):
    """Load calibration captions for framework-owned tokenization.

    Streams ``WenhaoWang/VideoUFO`` and materializes only the first ``num_samples``
    prompts (the large, video-backed dataset is never downloaded in full). Only the
    ``Detailed_Caption`` text column is read — never the image column.
    """
    dataset = load_dataset(
        "WenhaoWang/VideoUFO", split="Full", streaming=True
    ).select_columns("Detailed_Caption")
    raw_prompts = [row["Detailed_Caption"] for row in itertools.islice(dataset, num_samples)]

    return raw_prompts


def load_i2v_images_from_dir(image_dir, num_samples):
    """Load up to ``num_samples`` conditioning images (PIL) from a local directory."""
    from PIL import Image

    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No conditioning images found in {image_dir} (expected {exts}).")
    paths = paths[:num_samples]
    print(f"[calib/i2v] loading {len(paths)} conditioning images from {image_dir}")
    return [Image.open(p) for p in paths]


def load_i2v_images_from_dataset(dataset_name, num_samples, subsets=None):
    """Load raw conditioning images (PIL) from a ModelOpt VLM calibration dataset.

    Uses the RAW images (VAE-encoded here as clean first frames), not the
    vision-processor pixel_values a reasoner dataloader would yield.
    """

    def keep_raw_images(*, images, **_kwargs):
        return {"images": images}

    print(f"[calib/i2v] loading conditioning images from VLM dataset {dataset_name} (subsets={subsets or 'default'})")
    dataloader = get_vlm_dataset_dataloader(
        dataset_name=dataset_name,
        processor=keep_raw_images,
        batch_size=1,
        num_samples=num_samples * 4,
        require_image=True,
        subsets=subsets,
    )
    images = []
    for batch in dataloader:
        img = batch["images"][0]
        if isinstance(img, Image.Image):
            images.append(img)
            if len(images) >= num_samples:
                break
    if not images:
        raise ValueError(f"No conditioning images loaded from {dataset_name}.")
    print(f"[calib/i2v] loaded {len(images)} conditioning images")
    return images


def vae_encode_cond_images(vae, images, height, width, num_frames):
    """VAE-encode PIL conditioning images into clean first-frame latents for i2v.

    Each image is resized to (height, width), normalized to [-1, 1], repeated across
    ``num_frames``, VAE-encoded, and normalized by the VAE's latents_mean/std (or
    scaling_factor). Returns a list of ``[1, C, T_lat, H, W]`` bf16 latents (CPU).
    """
    import numpy as np

    has_mean_std = hasattr(vae.config, "latents_mean") and hasattr(vae.config, "latents_std")
    latents = []
    for img in images:
        img = img.convert("RGB").resize((width, height))
        px = torch.from_numpy(np.asarray(img)).float().div(255.0)  # [H,W,3] in [0,1]
        px = px.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0          # [1,3,H,W] in [-1,1]
        video = px.unsqueeze(2).expand(-1, -1, num_frames, -1, -1).contiguous()
        video = video.to(device="cuda", dtype=vae.dtype)
        with torch.no_grad():
            lat = vae.encode(video).latent_dist.mode()
        if has_mean_std:
            m = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(lat.device, lat.dtype)
            s = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(lat.device, lat.dtype)
            lat = (lat - m) / s
        else:
            lat = lat * getattr(vae.config, "scaling_factor", 1.0)
        latents.append(lat.to(torch.bfloat16).cpu())
    return latents


def resolve_distilled_sigmas(input_dir: Path) -> list[float] | None:
    """Fixed distilled step sigmas from the checkpoint, or None for normal models.

    Prefers ``modular_model_index.json['distilled_sigmas']`` and falls back to
    ``scheduler/scheduler_config.json`` -> ``fixed_step_sampler_config.t_list``.
    Both carry identical values today; reading both means a checkpoint that ships
    only one still calibrates on the true few-step schedule.
    """
    def _load(name: str) -> dict:
        try:
            with open(Path(input_dir) / name) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    sigmas = (
        _load("modular_model_index.json").get("distilled_sigmas")
        or (_load("scheduler/scheduler_config.json").get("fixed_step_sampler_config") or {}).get("t_list")
    )
    return [float(s) for s in sigmas] if sigmas else None


def _pack_framework_denoiser_input(
    mdl,
    *,
    hidden: torch.Tensor,
    timestep: torch.Tensor,
    text_indexes: list[int],
    frame_rate: float,
    i2v: bool,
):
    """Build the framework-owned packed input for one cookbook denoiser call."""
    if hidden.shape[0] != 1:
        raise ValueError(f"Calibration only supports batch size 1, got {hidden.shape[0]}.")

    # ``GenerationDataClean.x0_tokens_vision`` stores one batched latent per
    # vision item: [B, C, T, H, W].  Do not strip the batch dimension here.
    # The sequence packer uses it to distinguish samples from latent channels.
    latent = hidden
    clean_data = GenerationDataClean(
        batch_size=1,
        is_image_batch=latent.shape[2] == 1,
        x0_tokens_vision=[latent],
        fps_vision=torch.tensor([frame_rate], dtype=torch.float32),
    )
    packed_sequence = mdl._pack_input_sequence(
        [SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0] if i2v else [])],
        [text_indexes],
        clean_data,
        timestep.detach().reshape(1).float().cpu(),
        include_end_of_generation_token=mdl._derive_include_end_of_generation_token(),
    )
    if hidden.is_cuda:
        packed_sequence.to_cuda()
    else:
        packed_sequence.prepare_sequence_pack_metadata()
    return packed_sequence


def _run_framework_denoiser(
    mdl,
    *,
    hidden: torch.Tensor,
    timestep: torch.Tensor,
    text_indexes: list[int],
    frame_rate: float,
    i2v: bool,
) -> torch.Tensor:
    """Run a framework model through its canonical packing and denoise APIs."""
    packed_sequence = _pack_framework_denoiser_input(
        mdl,
        hidden=hidden,
        timestep=timestep,
        text_indexes=text_indexes,
        frame_rate=frame_rate,
        i2v=i2v,
    )
    prediction = mdl.denoise(data_batch_packed=packed_sequence)["preds_vision"]
    return torch.stack(prediction)


def make_forward_loop(
    *,
    vae,
    scheduler,
    prompts,
    num_inference_steps: int,
    height: int,
    width: int,
    num_frames: int,
    guidance_scale: float,
    frame_rate: float,
    flow_shift: float,
    negative_prompt: str = "",
    seed: int = 0,
    sigma_max: float | None = None,
    use_karras_sigmas: bool | None = None,
    use_flow_sigmas: bool | None = None,
    i2v: bool = False,
    cond_latents: list | None = None,
    explicit_sigmas: list[float] | None = None,
):
    """Build a ``forward_loop(mdl)`` that simulates real denoising for calibration.

    The framework sampler selects the sampling regime: its fixed-step sampler
    drives distilled students, while its UniPC sampler drives the base models.
    The cookbook still owns prompt selection, initial noise, CFG, and I2V clean
    frame conditioning.

    ``i2v=True`` calibrates the image-to-video regime: frame 0 of the latent is a
    clean VAE-encoded conditioning latent (from ``cond_latents``, round-robin), the
    rest is noise, and the framework sequence plan marks frame 0 as clean. The
    closure restores that frame before every denoiser invocation.
    """
    vae_scale_factor_temporal = getattr(
        vae, "temporal_compression_factor", getattr(getattr(vae, "config", None), "scale_factor_temporal", 4)
    )
    vae_scale_factor_spatial = getattr(
        vae, "spatial_compression_factor", getattr(getattr(vae, "config", None), "scale_factor_spatial", 16)
    )

    T = (num_frames - 1) // vae_scale_factor_temporal + 1
    H = height // vae_scale_factor_spatial
    W = width // vae_scale_factor_spatial
    _is_fixed_step = hasattr(scheduler, "t_list")
    if _is_fixed_step:
        if explicit_sigmas is not None and list(scheduler.t_list[:-1]) != list(explicit_sigmas):
            scheduler = FixedStepSampler(
                t_list=list(explicit_sigmas),
                sample_type=scheduler.sample_type,
                num_train_timesteps=scheduler.num_train_timesteps,
            )
        print(f"[calib] scheduler: framework fixed-step sigmas={scheduler.t_list}")
    else:
        print(
            f"[calib] scheduler: framework UniPC shift={flow_shift} "
            f"(sigma_max={sigma_max}, karras={use_karras_sigmas}, flow_sigmas={use_flow_sigmas})"
        )

    # Classifier-free guidance mirrors the pipeline: on iff guidance_scale != 1.0.
    # Distilled (DMD2) runs guidance 1.0 -> cond-only; calibrating the uncond branch
    # would sample activations the served model never produces.
    do_cfg = guidance_scale != 1.0
    if not do_cfg:
        print("[calib] guidance_scale == 1.0 -> CFG disabled (cond-only calibration)")

    condition_mask = None
    if i2v:
        if not cond_latents:
            raise ValueError("i2v=True requires cond_latents (VAE-encoded conditioning frames).")
        condition_mask = torch.zeros(1, 1, T, 1, 1, dtype=torch.bfloat16, device="cuda")
        condition_mask[:, :, 0, :, :] = 1.0

    def _run(mdl, hidden, ts, text_indexes):
        return _run_framework_denoiser(
            mdl,
            hidden=hidden,
            timestep=ts,
            text_indexes=text_indexes,
            frame_rate=frame_rate,
            i2v=i2v,
        )

    def forward_loop(mdl):
        n_prompts = len(prompts)
        _mode = "i2v (clean frame-0 conditioning)" if i2v else "t2v/t2i"
        print(f"[calib] {_mode} diffusion calibration — {n_prompts} prompts x {num_inference_steps} steps")
        data_batch = {mdl.input_caption_key: prompts}
        has_negative_prompt = bool(negative_prompt)
        if has_negative_prompt:
            data_batch["neg_" + mdl.input_caption_key] = [negative_prompt] * n_prompts
        cond_text_tokens, uncond_text_tokens = mdl._get_inference_text_tokens(data_batch, has_negative_prompt)
        torch.manual_seed(seed)
        for prompt_idx, (cond_text_indexes, uncond_text_indexes) in enumerate(
            zip(cond_text_tokens, uncond_text_tokens, strict=True)
        ):
            print(f"[calib] prompt {prompt_idx + 1}/{n_prompts}")
            noise = torch.randn(
                1, NUM_CHANNELS, T, H, W, dtype=torch.bfloat16, device="cuda",
            )
            image_latent = None
            if i2v:
                cond_latent = cond_latents[prompt_idx % len(cond_latents)].to(
                    dtype=torch.bfloat16, device="cuda"
                )
                image_latent = cond_latent[:, :, 0:1]  # clean frame 0 for re-injection
                latents = condition_mask * cond_latent + (1.0 - condition_mask) * noise
            else:
                latents = noise

            def velocity_fn(
                latent,
                timestep,
                *,
                image_latent=image_latent,
                cond_text_indexes=cond_text_indexes,
                uncond_text_indexes=uncond_text_indexes,
            ):
                if image_latent is not None:
                    latent[:, 0:1] = image_latent[0]
                hidden = latent.unsqueeze(0)
                noise_pred_cond = _run(mdl, hidden, timestep, cond_text_indexes)
                if do_cfg:
                    noise_pred_uncond = _run(mdl, hidden, timestep, uncond_text_indexes)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond
                return noise_pred[0]

            if _is_fixed_step:
                scheduler(
                    velocity_fn,
                    latents[0],
                    seed=seed,
                    condition_reference=image_latent[0] if image_latent is not None else None,
                    condition_mask=condition_mask[0] if condition_mask is not None else None,
                )
            else:
                scheduler(
                    velocity_fn,
                    latents[0],
                    num_steps=num_inference_steps,
                    shift=flow_shift,
                    seed=seed,
                )

    return forward_loop
