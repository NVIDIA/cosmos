# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Checkpoint I/O for the Cosmos3 FP8 quantization cookbook.

Loads the pieces the calibration + export stages need from a diffusers-layout
Cosmos3 checkpoint (``transformer/``, ``vae/``, ``scheduler/`` and — for the
unified nano/super checkpoints — a bundled Qwen3-VL tokenizer at the root), and
writes the quantized transformer back out as sharded safetensors.

The transformer is the cookbook's own :class:`~src.cosmos3_vfm.Cosmos3VFMTransformer`;
nothing here depends on any external pipeline package.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .cosmos3_vfm import Cosmos3VFMTransformer


def load_sharded_safetensors(checkpoint_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load every ``*.safetensors`` shard in ``checkpoint_dir`` into one state dict."""
    weights: dict[str, torch.Tensor] = {}
    shards = sorted(Path(checkpoint_dir).glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(
            f"No .safetensors shards found in {checkpoint_dir}. The checkpoint download "
            "is incomplete; do not quantize from config/index files alone."
        )
    for shard in shards:
        weights.update(load_file(str(shard)))
    return weights


def save_sharded_safetensors(
    state_dict: dict[str, torch.Tensor],
    out_dir: Path,
    max_shard_bytes: int = 5_000_000_000,
) -> int:
    """Write ``state_dict`` as sharded safetensors matching the bf16 checkpoint layout.

    Produces ``diffusion_pytorch_model-{i:05d}-of-{n:05d}.safetensors`` shards plus a
    single ``diffusion_pytorch_model.safetensors.index.json`` (the diffusers format
    the vllm-omni loader recognizes), instead of one large ``model.safetensors``.
    Tensors are packed greedily so each shard stays under ``max_shard_bytes``; a lone
    tensor larger than the cap gets its own shard. Returns the shard count.

    NB: emit exactly ONE weight index and no consolidated ``model.safetensors`` — the
    loader errors if two index files are present and drops files absent from the index.
    """
    def _nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for key, tensor in state_dict.items():
        size = _nbytes(tensor)
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current, current_bytes = {}, 0
        current[key] = tensor
        current_bytes += size
    if current:
        shards.append(current)

    n = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard in enumerate(shards, start=1):
        fname = f"diffusion_pytorch_model-{i:05d}-of-{n:05d}.safetensors"
        save_file(shard, str(out_dir / fname), metadata={"format": "pt"})
        for key, tensor in shard.items():
            weight_map[key] = fname
            total_size += _nbytes(tensor)

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(out_dir / "diffusion_pytorch_model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    return n


def detect_variant(input_dir: Path, transformer_dir: Path) -> str:
    """Infer the model size (``8b``/``32b``) from the dir name or the transformer config."""
    lower = str(input_dir).lower()
    if "super" in lower:
        return "32b"
    if "nano" in lower:
        return "8b"
    cfg_path = transformer_dir / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except FileNotFoundError as e:
        raise ValueError(
            f"Cannot determine model variant from input_dir={input_dir!r} "
            f"(no 'nano'/'super' substring) and {cfg_path} is missing."
        ) from e
    n_layers = int(cfg.get("num_hidden_layers", 0))
    if n_layers == 36:
        return "8b"
    if n_layers >= 60:
        return "32b"
    raise ValueError(
        f"Cannot determine model variant: {cfg_path} has num_hidden_layers={n_layers} "
        "(expected 36 for 8B or >=60 for 32B)."
    )


def load_transformer(input_dir: Path, variant: str | None = None):
    """Load the Cosmos3 DiT (bf16, on CUDA, eval) from ``input_dir/transformer``."""
    transformer_dir = input_dir / "transformer"
    if not transformer_dir.is_dir():
        raise FileNotFoundError(f"Expected {transformer_dir} (a diffusers-layout transformer/).")
    variant = variant or detect_variant(input_dir, transformer_dir)
    print(f"[load] transformer from {transformer_dir} (variant={variant})")
    model = Cosmos3VFMTransformer(variant=variant)
    model.load_weights(load_sharded_safetensors(transformer_dir))
    model.to(torch.bfloat16)
    model.post_load_weights()
    model.to("cuda").eval()
    return model, transformer_dir


def load_tokenizer(input_dir: Path, tokenizer_id: str | Path | None = None):
    """Load the Qwen3-VL tokenizer.

    Defaults to the tokenizer bundled at the checkpoint root (``local_files_only``)
    so the cookbook never contacts the HF hub — the shared hub cache lock files are
    often unwritable on root-squash NFS. Pass ``tokenizer_id`` to override.
    """
    from transformers import AutoTokenizer

    src = str(tokenizer_id) if tokenizer_id is not None else str(input_dir)
    local = tokenizer_id is None or Path(src).exists()
    print(f"[load] tokenizer: {src} (local_files_only={local})")
    return AutoTokenizer.from_pretrained(src, local_files_only=local)


def load_scheduler(input_dir: Path):
    """Load the checkpoint's scheduler, picking the class from ``scheduler_config.json``.

    Base video/image models ship a ``UniPCMultistepScheduler``; the distilled few-step
    students ship a ``FlowMatchEulerDiscreteScheduler`` (fixed sigmas + stochastic SDE
    step). The class is what makes a base model and its distilled student "differ only
    by scheduler" — the calibration loop branches on it.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

    scheduler_dir = input_dir / "scheduler"
    if not scheduler_dir.is_dir():
        raise FileNotFoundError(f"Expected {scheduler_dir} (a diffusers scheduler/).")
    cfg = json.load(open(scheduler_dir / "scheduler_config.json"))
    cls = {
        "UniPCMultistepScheduler": UniPCMultistepScheduler,
        "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
    }.get(cfg.get("_class_name"), UniPCMultistepScheduler)
    scheduler = cls.from_pretrained(str(scheduler_dir))
    print(f"[load] scheduler class: {type(scheduler).__name__}")
    return scheduler


def load_vae(input_dir: Path):
    """Load the checkpoint's VAE (bf16, CUDA) — only needed for i2v conditioning."""
    from diffusers import AutoencoderKLWan

    vae_dir = input_dir / "vae"
    if not vae_dir.is_dir():
        raise FileNotFoundError(f"Expected {vae_dir} (a diffusers vae/).")
    print(f"[load] vae from {vae_dir}")
    return AutoencoderKLWan.from_pretrained(str(vae_dir), torch_dtype=torch.bfloat16).to("cuda")
