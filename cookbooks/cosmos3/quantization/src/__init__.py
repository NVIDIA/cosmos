# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Cosmos3 FP8 quantization cookbook — self-contained ``src/`` package.

The model (:class:`Cosmos3VFMTransformer`) plus the ModelOpt static-scale FP8
recipe that produced the shipped Cosmos3 FP8 checkpoints: calibration
(:mod:`.calibration`), export (:mod:`.export`) and a one-call orchestrator
(:func:`quantize_fp8_checkpoint`).

The design centers on the :class:`Sampler`: a base model and its distilled student
share the same model, shape and conditioning and **differ only by their sampler**
(scheduler class + steps + guidance), so quantizing the pair is the same call twice
with a different :class:`Sampler`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import modelopt.torch.quantization as mtq

from .cosmos3_vfm import Cosmos3VFMTransformer, compute_mrope_position_ids_text
from . import calibration as _calib
from . import export as _export
from .checkpoint_io import (
    load_transformer,
    load_tokenizer,
    load_scheduler,
    load_vae,
    load_sharded_safetensors,
    save_sharded_safetensors,
)

FP8_E4M3_MAX = 448.0


@dataclass
class Shape:
    """Calibration latent shape (pixel-space; the VAE scale factors are applied internally)."""
    height: int
    width: int
    num_frames: int


@dataclass
class Sampler:
    """The sampler-specific calibration knobs — the only axis that differs between a
    base model and its distilled student.

    ``explicit_sigmas`` / the FlowMatchEuler regime are auto-detected from the loaded
    checkpoint (the distilled students ship a ``FlowMatchEulerDiscreteScheduler`` and
    a fixed sigma ``t_list``); the UniPC knobs below are ignored for those.
    """
    num_inference_steps: int
    guidance_scale: float
    flow_shift: float = 10.0
    sigma_max: float | None = 80.0
    use_karras_sigmas: bool = False
    use_flow_sigmas: bool = True
    explicit_sigmas: list[float] | None = None
    label: str = ""


# --- Production shapes (models.mk) -------------------------------------------
SHAPE_VIDEO = Shape(height=720, width=1280, num_frames=189)   # t2v / i2v
SHAPE_IMAGE = Shape(height=720, width=1280, num_frames=1)     # t2i
# Small shapes for the demo runs (a single prompt, seconds not minutes).
SHAPE_VIDEO_DEMO = Shape(height=480, width=720, num_frames=29)
SHAPE_IMAGE_DEMO = Shape(height=512, width=512, num_frames=1)

# --- Production samplers (models.mk) -----------------------------------------
# t2v / i2v base: UniPC "runtime" preset (flow_shift=10, sigma_max=80, flow sigmas).
SAMPLER_VIDEO_BASE = Sampler(
    num_inference_steps=50, guidance_scale=6.0,
    flow_shift=10.0, sigma_max=80.0, use_karras_sigmas=False, use_flow_sigmas=True,
    label="video base (UniPC runtime)",
)
# t2i base: UniPC with flow_shift=3, sigma_max=200, karras sigmas.
SAMPLER_IMAGE_BASE = Sampler(
    num_inference_steps=50, guidance_scale=6.0,
    flow_shift=3.0, sigma_max=200.0, use_karras_sigmas=True, use_flow_sigmas=True,
    label="t2i base (UniPC karras)",
)
# Distilled students (t2i / i2v): FlowMatchEuler, 4 fixed steps, CFG-free.
SAMPLER_DISTILLED = Sampler(
    num_inference_steps=4, guidance_scale=1.0, explicit_sigmas=None,
    label="distilled (FlowMatchEuler 4-step, CFG-free)",
)


class _StubVAE:
    """Minimal stand-in for the VAE when only the scale factors are needed (t2v/t2i).

    ``make_forward_loop`` reads ``vae.config.scale_factor_{temporal,spatial}`` to size
    the latent; the full VAE is only needed to encode i2v conditioning frames.
    """
    def __init__(self, temporal: int = 4, spatial: int = 16):
        import types
        self.config = types.SimpleNamespace(scale_factor_temporal=temporal, scale_factor_spatial=spatial)


def quantize_fp8_checkpoint(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    sampler: Sampler,
    shape: Shape,
    profile: str = "t2v",
    num_samples: int = 8,
    quant_algo: str = "max",
    seed: int = 0,
    negative_prompt: str = "",
    fps: float = 24.0,
    quantize_language_model: bool = True,
    i2v_cond_dir: str | Path | None = None,
    i2v_cond_dataset: str | None = None,
    tokenizer_id: str | Path | None = None,
    variant: str | None = None,
    keep_model: bool = False,
):
    """Quantize a diffusers-layout Cosmos3 checkpoint to FP8 and write a drop-in dir.

    Mirrors the NIM ``quantize_cosmos3.py`` gen-only ``max`` path end to end:
    load -> calibrate (real denoising) -> ModelOpt FP8 -> disable skip filter ->
    export vllm-omni FP8 diffusers checkpoint -> assemble drop-in dir.

    ``profile`` is one of ``"t2v"``, ``"t2i"``, ``"i2v"``. ``i2v`` requires either
    ``i2v_cond_dir`` (a local image dir) or ``i2v_cond_dataset`` (a ModelOpt VLM
    dataset name). Returns the assembled ``output_dir`` (and, if ``keep_model``, the
    calibrated model for inspection).
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if profile not in ("t2v", "t2i", "i2v"):
        raise ValueError(f"profile must be t2v/t2i/i2v, got {profile!r}")
    i2v = profile == "i2v"

    model, transformer_dir = load_transformer(input_dir, variant=variant)
    tokenizer = load_tokenizer(input_dir, tokenizer_id=tokenizer_id)
    scheduler = load_scheduler(input_dir)
    vae = load_vae(input_dir) if i2v else _StubVAE()

    num_gen_layers = _calib.count_gen_layers(model)
    print(f"[init] gen_layers={num_gen_layers} profile={profile} sampler={sampler.label!r}")

    prompts = _calib.prepare_calibration_prompts(tokenizer, num_samples=num_samples)

    cond_latents = None
    if i2v:
        if i2v_cond_dataset:
            images = _calib.load_i2v_images_from_dataset(i2v_cond_dataset, num_samples)
        elif i2v_cond_dir:
            images = _calib.load_i2v_images_from_dir(i2v_cond_dir, num_samples)
        else:
            raise ValueError("profile='i2v' requires i2v_cond_dir or i2v_cond_dataset.")
        cond_latents = _calib.vae_encode_cond_images(
            vae, images, shape.height, shape.width, shape.num_frames,
        )

    forward_loop = _calib.make_forward_loop(
        tokenizer=tokenizer,
        vae=vae,
        scheduler=scheduler,
        prompts=prompts,
        num_inference_steps=sampler.num_inference_steps,
        height=shape.height,
        width=shape.width,
        num_frames=shape.num_frames,
        guidance_scale=sampler.guidance_scale,
        frame_rate=fps,
        flow_shift=sampler.flow_shift,
        negative_prompt=negative_prompt,
        seed=seed,
        sigma_max=sampler.sigma_max,
        use_karras_sigmas=sampler.use_karras_sigmas,
        use_flow_sigmas=sampler.use_flow_sigmas,
        i2v=i2v,
        cond_latents=cond_latents,
        explicit_sigmas=sampler.explicit_sigmas or _calib.resolve_distilled_sigmas(input_dir),
    )

    quant_cfg = _calib.build_quant_config(quant_algo)
    skip_filter = _calib.build_filter(
        num_gen_layers=num_gen_layers,
        quantize_language_model=quantize_language_model,
    )

    print(f"[quant] format=fp8 algo={quant_algo}")
    mtq.quantize(model, quant_cfg, forward_loop)
    mtq.disable_quantizer(model, skip_filter)
    mtq.print_quant_summary(model)

    staging_dir = output_dir.parent / f".quantized_transformer_{profile}.tmp"
    _export.export_quantized_transformer(model, staging_dir, transformer_dir)
    _export.assemble_output_dir(input_dir, output_dir, staging_dir)

    print(f"[done] {output_dir}")
    return (output_dir, model) if keep_model else output_dir


def diffusers_input_scales(model) -> dict[str, torch.Tensor]:
    """Extract the calibrated per-module ``input_scale``s, keyed like the diffusers
    checkpoint (``scale = amax / 448``). Handy for bit-identity checks vs. a reference.
    """
    net_amax = {}
    for name, m in model.named_modules():
        iq = getattr(m, "input_quantizer", None)
        if iq is not None and getattr(iq, "is_enabled", True) and getattr(iq, "amax", None) is not None:
            net_amax[f"{name}.input_scale"] = (iq.amax.detach().float().reshape(-1) / FP8_E4M3_MAX).cpu()
    return _export.reverse_remap_state_dict(net_amax)


__all__ = [
    "Cosmos3VFMTransformer",
    "compute_mrope_position_ids_text",
    "Sampler",
    "Shape",
    "SHAPE_VIDEO",
    "SHAPE_IMAGE",
    "SHAPE_VIDEO_DEMO",
    "SHAPE_IMAGE_DEMO",
    "SAMPLER_VIDEO_BASE",
    "SAMPLER_IMAGE_BASE",
    "SAMPLER_DISTILLED",
    "quantize_fp8_checkpoint",
    "diffusers_input_scales",
    "load_transformer",
    "load_tokenizer",
    "load_scheduler",
    "load_vae",
    "load_sharded_safetensors",
    "save_sharded_safetensors",
]
