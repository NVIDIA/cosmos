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

import torch
from datasets import load_dataset
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
import modelopt.torch.quantization as mtq


QUANT_CONFIGS = {"fp8": mtq.FP8_DEFAULT_CFG}

# Modules that should never be quantized (norms, embeddings, RoPE, patch
# projections). ``_NO_LM`` keeps the UND transformer linears eligible (we quantize
# the generation + understanding towers); the plain variant additionally excludes
# the whole ``language_model`` subtree.
_BASE_SKIP_TOKENS = (
    "language_model|time_embedder|vae2llm|llm2vae|embed_tokens|"
    "norm_q|norm_k|input_layernorm|post_attention_layernorm|"
    "norm_moe_gen|rotary_emb"
)
_BASE_SKIP_TOKENS_NO_LM = (
    "time_embedder|vae2llm|llm2vae|embed_tokens|"
    "norm_q|norm_k|input_layernorm|post_attention_layernorm|"
    "norm_moe_gen|rotary_emb"
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
def prepare_calibration_prompts(tokenizer, num_samples: int, max_seq_len: int = 512):
    """Tokenize prompts with the same chat template as the production pipeline.

    Streams ``WenhaoWang/VideoUFO`` and materializes only the first ``num_samples``
    prompts (the large, video-backed dataset is never downloaded in full). Only the
    ``Detailed_Caption`` text column is read — never the image column.
    """
    dataset = load_dataset(
        "WenhaoWang/VideoUFO", split="Full", streaming=True
    ).select_columns("Detailed_Caption")
    raw_prompts = [row["Detailed_Caption"] for row in itertools.islice(dataset, num_samples)]

    batches = []
    for prompt in raw_prompts:
        conversations = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True,
        )
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids = token_ids[:max_seq_len]
        token_ids.append(tokenizer.eos_token_id)
        token_ids.append(tokenizer.convert_tokens_to_ids("<|vision_start|>"))

        seq_len = len(token_ids)
        pad_len = max_seq_len - seq_len
        attention_mask = [1] * seq_len + [0] * pad_len
        token_ids = token_ids + [tokenizer.pad_token_id or 0] * pad_len

        cond_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
        cond_mask = torch.tensor([attention_mask], dtype=torch.float, device="cuda")
        batches.append((cond_ids, cond_mask))
    return batches


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
    from PIL import Image
    from modelopt.torch.utils.vlm_dataset_utils import (
        SUPPORTED_VLM_DATASET_CONFIG,
        _get_vlm_dataset,
        _extract_image_ref_from_example,
        _maybe_load_image,
    )

    repo_id = SUPPORTED_VLM_DATASET_CONFIG.get(dataset_name, {}).get("config", {}).get("path")
    print(f"[calib/i2v] loading conditioning images from VLM dataset {dataset_name} (subsets={subsets or 'default'})")
    ds = _get_vlm_dataset(dataset_name, num_samples=num_samples * 4, require_image=True, subsets=subsets)
    images = []
    for ex in ds:
        img = _maybe_load_image(_extract_image_ref_from_example(ex), repo_id=repo_id, image_root=None)
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


def make_forward_loop(
    *,
    tokenizer,
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

    The scheduler *class* selects the sampling regime:

    * ``FlowMatchEulerDiscreteScheduler`` (distilled students) -> fixed explicit
      sigmas + a stochastic (SDE) step; guidance is cond-only (guidance_scale=1.0).
    * ``UniPCMultistepScheduler`` (everyone else) -> flow_shift / sigma_max /
      karras / flow_sigmas knobs, with classifier-free guidance.

    ``i2v=True`` calibrates the image-to-video regime: frame 0 of the latent is a
    clean VAE-encoded conditioning latent (from ``cond_latents``, round-robin), the
    rest is noise, a ``noisy_frame_mask`` (frame 0 = 0) is fed to the transformer so
    the clean frame gets no timestep embedding, and the clean frame is re-injected
    after every scheduler step — mirroring ``_prepare_latents_i2v`` in serving.
    """
    vae_scale_factor_temporal = getattr(getattr(vae, "config", None), "scale_factor_temporal", 4)
    vae_scale_factor_spatial = getattr(getattr(vae, "config", None), "scale_factor_spatial", 16)

    T = (num_frames - 1) // vae_scale_factor_temporal + 1
    H = height // vae_scale_factor_spatial
    W = width // vae_scale_factor_spatial
    video_shape = (T, H, W)

    neg_conversations = [{"role": "user", "content": negative_prompt}]
    neg_text = tokenizer.apply_chat_template(
        neg_conversations, tokenize=False, add_generation_prompt=True,
    )
    neg_ids_raw = tokenizer.encode(neg_text, add_special_tokens=False)
    neg_ids_raw.append(tokenizer.eos_token_id)
    neg_ids_raw.append(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
    max_seq_len = prompts[0][0].shape[1]
    pad_len = max_seq_len - len(neg_ids_raw)
    neg_mask = [1] * len(neg_ids_raw) + [0] * pad_len
    neg_ids_raw = neg_ids_raw + [tokenizer.pad_token_id or 0] * pad_len
    uncond_ids = torch.tensor([neg_ids_raw], dtype=torch.long, device="cuda")
    uncond_mask = torch.tensor([neg_mask], dtype=torch.float, device="cuda")

    _is_flow_match = isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    if _is_flow_match:
        _scheduler = FlowMatchEulerDiscreteScheduler.from_config(dict(scheduler.config))
        _fixed_sigmas = explicit_sigmas or (
            (dict(scheduler.config).get("fixed_step_sampler_config") or {}).get("t_list")
        )
        _stochastic = bool(dict(scheduler.config).get("stochastic_sampling"))
        print(f"[calib] scheduler: FlowMatchEulerDiscreteScheduler sigmas={_fixed_sigmas} stochastic={_stochastic}")
    else:
        _scheduler_cfg = dict(scheduler.config)
        _scheduler_cfg["flow_shift"] = flow_shift
        if sigma_max is not None:
            _scheduler_cfg["sigma_max"] = sigma_max
        if use_karras_sigmas is not None:
            _scheduler_cfg["use_karras_sigmas"] = use_karras_sigmas
        if use_flow_sigmas is not None:
            _scheduler_cfg["use_flow_sigmas"] = use_flow_sigmas
        _scheduler = UniPCMultistepScheduler.from_config(_scheduler_cfg)
        _fixed_sigmas = None
        print(
            f"[calib] scheduler: flow_shift={_scheduler_cfg.get('flow_shift')} "
            f"sigma_max={_scheduler_cfg.get('sigma_max')} "
            f"use_karras_sigmas={_scheduler_cfg.get('use_karras_sigmas')} "
            f"use_flow_sigmas={_scheduler_cfg.get('use_flow_sigmas')}"
        )

    # Classifier-free guidance mirrors the pipeline: on iff guidance_scale != 1.0.
    # Distilled (DMD2) runs guidance 1.0 -> cond-only; calibrating the uncond branch
    # would sample activations the served model never produces.
    do_cfg = guidance_scale != 1.0
    _step_gen = torch.Generator(device="cuda").manual_seed(seed)
    if not do_cfg:
        print("[calib] guidance_scale == 1.0 -> CFG disabled (cond-only calibration)")
    _init_noise_sigma = float(getattr(_scheduler, "init_noise_sigma", 1.0))

    noisy_frame_mask = None
    condition_mask = None
    if i2v:
        if not cond_latents:
            raise ValueError("i2v=True requires cond_latents (VAE-encoded conditioning frames).")
        condition_mask = torch.zeros(1, 1, T, 1, 1, dtype=torch.bfloat16, device="cuda")
        condition_mask[:, :, 0, :, :] = 1.0
        noisy_frame_mask = 1.0 - condition_mask  # [1,1,T,1,1], frame 0 = 0 (clean)

    def _run(mdl, hidden, ts, txt_ids, txt_mask):
        return mdl(
            hidden_states=hidden,
            timestep=ts,
            text_ids=txt_ids,
            text_mask=txt_mask,
            video_shape=video_shape,
            fps=frame_rate,
            noisy_frame_mask=noisy_frame_mask,
        )

    def forward_loop(mdl):
        n_prompts = len(prompts)
        _mode = "i2v (clean frame-0 conditioning)" if i2v else "t2v/t2i"
        print(f"[calib] {_mode} diffusion calibration — {n_prompts} prompts x {num_inference_steps} steps")
        torch.manual_seed(seed)
        for prompt_idx, (cond_ids, cond_mask) in enumerate(prompts):
            print(f"[calib] prompt {prompt_idx + 1}/{n_prompts}")
            if _is_flow_match and _fixed_sigmas:
                _scheduler.set_timesteps(sigmas=list(_fixed_sigmas), device="cuda")
            else:
                _scheduler.set_timesteps(num_inference_steps, device="cuda")
            timesteps_list = list(_scheduler.timesteps)
            N = len(timesteps_list)

            noise = torch.randn(
                1, NUM_CHANNELS, T, H, W, dtype=torch.bfloat16, device="cuda",
            ) * _init_noise_sigma
            image_latent = None
            if i2v:
                cond_latent = cond_latents[prompt_idx % len(cond_latents)].to(
                    dtype=torch.bfloat16, device="cuda"
                )
                image_latent = cond_latent[:, :, 0:1]  # clean frame 0 for re-injection
                latents = condition_mask * cond_latent + (1.0 - condition_mask) * noise
            else:
                latents = noise

            for step_idx, t in enumerate(timesteps_list):
                if step_idx % 10 == 0:
                    print(f"[calib] prompt {prompt_idx + 1}/{n_prompts} step {step_idx + 1}/{N}")
                timestep = t.unsqueeze(0)
                noise_pred_cond = _run(mdl, latents, timestep, cond_ids, cond_mask)
                if do_cfg:
                    noise_pred_uncond = _run(mdl, latents, timestep, uncond_ids, uncond_mask)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond
                if _is_flow_match:
                    latents = _scheduler.step(noise_pred, t, latents, generator=_step_gen).prev_sample
                else:
                    latents = _scheduler.step(noise_pred, t, latents).prev_sample
                if i2v:
                    latents[:, :, 0:1] = image_latent  # keep the conditioning frame clean

    return forward_loop
