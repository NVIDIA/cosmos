# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Export a calibrated Cosmos3 DiT as a vllm-omni-loadable FP8 diffusers checkpoint.

The calibrated model (this cookbook's :class:`~src.cosmos3_vfm.Cosmos3VFMTransformer`)
is read directly: its FP8 weights + per-tensor scales are materialized, reverse-
remapped to the diffusers key layout, and overlaid onto the full bf16 diffusers base
so the projections / audio-action towers / lm_head / LM final norm are preserved.
Then a drop-in checkpoint dir is assembled (transformer/ new, everything else
symlinked back) with a ModelOpt FP8 ``quantization_config`` + a fresh root index.

FP8 only — the NIM pipeline's NVFP4 exporter and mixed-precision recipes are out of
scope for the cookbook.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import torch
import modelopt
from modelopt.torch.export.diffusers_utils import hide_quantizers_from_state_dict
from modelopt.torch.export.unified_export_hf import (
    _fuse_qkv_linears_diffusion,
    _process_quantized_modules,
)

from .checkpoint_io import load_sharded_safetensors, save_sharded_safetensors
from .safetensors_index import build_root_index


def collapse_input_amax_to_scalar(mdl) -> int:
    """Collapse per-channel ``input_quantizer.amax`` to scalar (max over channels).

    ModelOpt's ``_fuse_qkv_linears_diffusion`` asserts input quantizer amax is scalar.
    Per-tensor ``max`` calibration already yields scalars, so this is normally a no-op;
    the conservative max-collapse keeps the widest dynamic range if a per-channel
    quantizer ever slips through.
    """
    try:
        from modelopt.torch.quantization.nn import TensorQuantizer
    except Exception:
        from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer  # type: ignore

    cnt = 0
    for name, module in mdl.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        if "input_quantizer" not in name:
            continue
        if getattr(module, "amax", None) is None:
            continue
        if module.amax.numel() <= 1:
            continue
        with torch.no_grad():
            module.amax = module.amax.abs().max().reshape(()).clone()
        cnt += 1
    return cnt


def reverse_remap_state_dict(state_dict: dict) -> dict:
    """Remap cosmos3_vfm internal keys -> diffusers/HF Cosmos3 checkpoint format.

    Produces a top-level (no ``model.`` prefix) layout structurally identical to the
    BF16 NGC ``transformer/`` shards, so the loader takes the same diffusers-style
    code path for both BF16 and quantized checkpoints. ``time_embedder`` is rewritten
    from the ``nn.Sequential`` indices ``time_embedder.mlp.{0,2}.*`` back to diffusers
    ``linear_{1,2}.*``.
    """
    remapped: dict = {}

    for key, value in state_dict.items():
        if key.startswith("time_embedder.mlp.0."):
            remapped["time_embedder.linear_1." + key[len("time_embedder.mlp.0."):]] = value
            continue
        if key.startswith("time_embedder.mlp.2."):
            remapped["time_embedder.linear_2." + key[len("time_embedder.mlp.2."):]] = value
            continue

        if key.startswith(("vae2llm.", "llm2vae.", "time_embedder.")):
            remapped[key] = value
            continue

        if key.startswith("norm_moe_gen."):
            remapped[key] = value
            continue

        if key.startswith("language_model.embed_tokens.") or key.startswith("language_model.norm."):
            remapped[key[len("language_model."):]] = value
            continue

        m = re.match(r"^language_model\.layers\.(\d+)\.(.+)$", key)
        if m:
            i, rest = m.group(1), m.group(2)
            base = f"layers.{i}."
            attn_map = {
                "self_attn.to_q.":     "self_attn.to_q.",
                "self_attn.to_k.":     "self_attn.to_k.",
                "self_attn.to_v.":     "self_attn.to_v.",
                "self_attn.to_out.0.": "self_attn.to_out.",
                "self_attn.norm_q.":   "self_attn.norm_q.",
                "self_attn.norm_k.":   "self_attn.norm_k.",
            }
            matched = False
            for src, dst in attn_map.items():
                if rest.startswith(src):
                    remapped[base + dst + rest[len(src):]] = value
                    matched = True
                    break
            if not matched:
                remapped[base + rest] = value
            continue

        m = re.match(r"^gen_layers\.(\d+)\.(.+)$", key)
        if m:
            i, rest = m.group(1), m.group(2)
            base = f"layers.{i}."
            attn_map = {
                "cross_attention.to_q.":     "self_attn.add_q_proj.",
                "cross_attention.to_k.":     "self_attn.add_k_proj.",
                "cross_attention.to_v.":     "self_attn.add_v_proj.",
                "cross_attention.to_out.0.": "self_attn.to_add_out.",
                "cross_attention.norm_q.":   "self_attn.norm_added_q.",
                "cross_attention.norm_k.":   "self_attn.norm_added_k.",
            }
            matched = False
            for src, dst in attn_map.items():
                if rest.startswith(src):
                    remapped[base + dst + rest[len(src):]] = value
                    matched = True
                    break
            if not matched:
                if rest.startswith("input_layernorm."):
                    remapped[base + "input_layernorm_moe_gen." + rest[len("input_layernorm."):]] = value
                elif rest.startswith("post_attention_layernorm."):
                    remapped[base + "post_attention_layernorm_moe_gen." + rest[len("post_attention_layernorm."):]] = value
                elif rest.startswith("mlp."):
                    remapped[base + "mlp_moe_gen." + rest[len("mlp."):]] = value
            continue

    return remapped


# Top-level projections produced by reverse_remap use the cosmos3_vfm names; the
# bf16 diffusers base provides the equivalently-named proj_in/proj_out, so we drop
# ours and keep the base copies.
_DROP_FROM_QUANTIZED = ("vae2llm.", "llm2vae.")
_FP8_DTYPES = {torch.float8_e4m3fn, getattr(torch, "float8_e5m2", torch.float8_e4m3fn)}
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale", ".input_scale_2", ".alpha", ".pre_quant_scale")

# Linears the diffusers checkpoint keeps in bf16 (never quantized), emitted in the
# ModelOpt ``ignore`` list so a reasoning-path loader (vllm Cosmos3ForConditionalGeneration)
# does not FP8-quantize the vision tower / projections it can't supply scales for.
_DIFFUSERS_IGNORE_MODULES = [
    "proj_in", "proj_out", "time_embedder*",
    "audio_proj_in", "audio_proj_out", "action_proj_in", "action_proj_out",
    "lm_head",
    "model.visual*", "visual*",
]


class Fp8DiffusersExporter:
    """Wrap a modelopt-FP8-quantized cosmos3 DiT as a vllm-omni diffusers checkpoint.

    Materializes the quantized tensors (fuse QKV + process + reverse-remap), overlays
    them onto the full bf16 diffusers transformer (which supplies proj_in/proj_out,
    the audio/action towers, lm_head and the LM final norm), and stamps a ModelOpt
    FP8 ``quantization_config``. FP8 weights keep the bf16 shape (dtype-only change).
    """

    quant_format = "fp8"

    def build_quant_config_json(self) -> dict:
        return {
            "config_groups": {
                "group_0": {
                    "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                    "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                    "targets": ["Linear"],
                }
            },
            "ignore": list(_DIFFUSERS_IGNORE_MODULES),
            "quant_algo": "FP8",
            "producer": {"name": "modelopt", "version": modelopt.__version__},
            "quant_method": "modelopt",
        }

    def overlay(self, bf16: dict, quantized: dict) -> dict:
        merged = dict(bf16)
        n_weight = n_scale = n_dropped = 0
        quantized_modules: set[str] = set()
        for k, v in quantized.items():
            if k.startswith(_DROP_FROM_QUANTIZED):
                n_dropped += 1
                continue
            if k.endswith(_SCALE_SUFFIXES):
                merged[k] = v
                n_scale += 1
                continue
            if k.endswith(".weight") and v.dtype in _FP8_DTYPES:
                if k not in bf16:
                    raise KeyError(
                        f"quantized weight {k!r} has no bf16 counterpart — layer-key "
                        "schema mismatch; the overlay assumption is broken."
                    )
                if bf16[k].shape != v.shape:
                    raise ValueError(
                        f"shape mismatch for {k}: bf16 {tuple(bf16[k].shape)} vs fp8 {tuple(v.shape)}"
                    )
                merged[k] = v
                quantized_modules.add(k[: -len(".weight")])
                n_weight += 1
                continue
            # Remaining bf16 tensors (norms, embed_tokens, renamed projections) are
            # identical to / superseded by the base — keep base.

        missing = [f"{m}.weight_scale" for m in quantized_modules if f"{m}.weight_scale" not in merged]
        if missing:
            raise ValueError(f"{len(missing)} quantized modules missing weight_scale, e.g. {missing[:3]}")
        print(
            f"[export] overlaid {n_weight} fp8 weights + {n_scale} scales onto bf16 base "
            f"(dropped {n_dropped} vae2llm/llm2vae); {len(merged)} tensors total"
        )
        return merged

    def export(self, mdl, transformer_export_dir: Path, src_transformer_dir: Path, dtype=torch.bfloat16) -> None:
        transformer_export_dir.mkdir(parents=True, exist_ok=True)

        n_collapsed = collapse_input_amax_to_scalar(mdl)
        if n_collapsed > 0:
            print(f"[export] collapsed per-channel input amax to scalar on {n_collapsed} quantizers")

        def dummy_forward():
            B, C, T, H, W = 1, 48, 5, 8, 8
            mdl(
                hidden_states=torch.randn(B, C, T, H, W, dtype=dtype, device="cuda"),
                timestep=torch.tensor([0.5], device="cuda"),
                text_ids=torch.ones(B, 16, dtype=torch.long, device="cuda"),
                text_mask=torch.ones(B, 16, dtype=torch.float, device="cuda"),
                video_shape=(T, H, W),
            )

        _fuse_qkv_linears_diffusion(mdl, dummy_forward)
        _process_quantized_modules(mdl, dtype)

        with hide_quantizers_from_state_dict(mdl):
            internal_state_dict = {k: v.detach().contiguous().cpu() for k, v in mdl.state_dict().items()}
        quantized = reverse_remap_state_dict(internal_state_dict)

        print(f"[export] loading bf16 base from {src_transformer_dir}")
        bf16 = load_sharded_safetensors(src_transformer_dir)
        merged = self.overlay(bf16, quantized)
        n_shards = save_sharded_safetensors(merged, transformer_export_dir)
        print(f"[export] wrote {len(merged)} tensors across {n_shards} shard(s)")

        src_config = src_transformer_dir / "config.json"
        if not src_config.is_file():
            raise FileNotFoundError(f"Missing source config.json at {src_config}; cannot stamp quantization_config.")
        with open(src_config, encoding="utf-8") as f:
            config = json.load(f)
        config["quantization_config"] = self.build_quant_config_json()
        with open(transformer_export_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        print(f"[export] wrote {len(merged)} tensors (fp8) to {transformer_export_dir}")


def export_quantized_transformer(mdl, transformer_export_dir: Path, src_transformer_dir: Path, dtype=torch.bfloat16):
    """Export the FP8-quantized DiT as a vllm-omni-loadable diffusers checkpoint dir."""
    Fp8DiffusersExporter().export(mdl, Path(transformer_export_dir), Path(src_transformer_dir), dtype)


def _write_root_hf_quant_config(root_dir: Path, transformer_dir: Path) -> None:
    """Mirror the transformer's ``quantization_config`` to a root ``hf_quant_config.json``.

    vllm-omni's diffusers pipeline reads the quant config from ``transformer/config.json``.
    Upstream vLLM's ``Cosmos3ForConditionalGeneration`` looks only at the model root, so
    we write a standalone copy there (the transformer config stays the source of truth).
    """
    with open(transformer_dir / "config.json", encoding="utf-8") as f:
        quant_config = json.load(f).get("quantization_config")
    if quant_config is None:
        raise ValueError(f"No 'quantization_config' in {transformer_dir}/config.json.")
    with open(root_dir / "hf_quant_config.json", "w", encoding="utf-8") as f:
        json.dump(quant_config, f, indent=4)
    print(f"[assemble] wrote {root_dir / 'hf_quant_config.json'} for upstream-vLLM discovery")


def _write_root_weight_index(root_dir: Path) -> None:
    """(Re)generate the root ``model.safetensors.index.json`` for whole-model loading."""
    index = build_root_index(root_dir)
    dst = root_dir / "model.safetensors.index.json"
    index.write(dst)
    print(f"[assemble] wrote {dst.name} ({len(index.weight_map)} tensors, {index.metadata['total_size']} bytes)")


def assemble_output_dir(input_dir: Path, output_dir: Path, quantized_transformer_dir: Path):
    """Materialize a drop-in checkpoint dir mirroring ``input_dir``.

    ``output_dir`` mirrors ``input_dir`` (vae, scheduler, model_index.json, tokenizers,
    vision_encoder symlinked back); only ``transformer/`` is physically new (the
    quantized export). A fresh root weight index + a root ``hf_quant_config.json`` are
    written so both the diffusers and transformers loaders resolve the new weights.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_transformer = output_dir / "transformer"
    if target_transformer.exists():
        shutil.rmtree(target_transformer)
    print(f"[assemble] moving quantized -> {target_transformer}")
    shutil.move(str(quantized_transformer_dir), str(target_transformer))

    # Wire everything except transformer/ + the stale root index as relative symlinks
    # back to input_dir (relative links survive bind-mount path differences).
    _SKIP_LINK = {"transformer", "model.safetensors.index.json"}
    input_dir_abs = input_dir.absolute()
    for entry in input_dir.iterdir():
        if entry.name in _SKIP_LINK:
            continue
        dst = output_dir / entry.name
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        src = input_dir_abs / entry.name
        rel_src = os.path.relpath(src, start=output_dir.resolve())
        print(f"[assemble] linking {entry.name} -> {rel_src}")
        dst.symlink_to(rel_src)
    _write_root_weight_index(output_dir)
    _write_root_hf_quant_config(output_dir, target_transformer)
    print(f"[assemble] drop-in dir ready at {output_dir}")
