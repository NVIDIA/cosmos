# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Checkpoint I/O for the Cosmos3 FP8 quantization cookbook.

Loads a diffusers-layout Cosmos3 checkpoint through the supported
:class:`~cosmos_framework.inference.inference.OmniInference` path and provides
the sharded-safetensors helpers that export uses to overlay FP8 weights onto
the original bf16 transformer layout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import gettempdir

import torch
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.inference import OmniInference
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


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


def resolve_checkpoint_path(model_name_or_path: str) -> Path:
    if os.path.exists(model_name_or_path):
        return Path(model_name_or_path)
    else:
        return Path(snapshot_download(model_name_or_path))


def load_transformer(model_name_or_path: str):
    """Load the Cosmos3 model through the supported ``OmniInference`` path."""
    input_dir = resolve_checkpoint_path(model_name_or_path)
    transformer_dir = input_dir / "transformer"
    if not transformer_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {transformer_dir} (a diffusers-layout transformer/)."
        )

    setup_args = OmniSetupOverrides(
        checkpoint_path=str(input_dir),
        output_dir=Path(gettempdir()) / "cosmos3-quantization-load",
        dp_replicate_size=1,
        dp_shard_size=1,
        cp_size=1,
        cfgp_size=1,
        use_cuda_graphs=False,
        use_torch_compile=False,
        guardrails=False,
    ).build_setup(world_size=1, local_world_size=1)
    inference = OmniInference.create(setup_args)
    return inference.model, transformer_dir


def load_legacy_scheduler(input_dir: str | Path):
    """Load the checkpoint's Diffusers scheduler for equivalence experiments.

    The normal cookbook path uses the scheduler supplied by ``OmniInference``.
    This opt-in helper exists solely to compare its sampling trajectory with the
    historical cookbook, whose scheduler was a Diffusers component in the
    checkpoint layout.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

    scheduler_dir = Path(input_dir) / "scheduler"
    with open(scheduler_dir / "scheduler_config.json") as f:
        config = json.load(f)
    scheduler_cls = {
        "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
        "UniPCMultistepScheduler": UniPCMultistepScheduler,
    }.get(config.get("_class_name"), UniPCMultistepScheduler)
    scheduler = scheduler_cls.from_pretrained(str(scheduler_dir))
    print(f"[calib] legacy-equivalence scheduler: {type(scheduler).__name__}")
    return scheduler
