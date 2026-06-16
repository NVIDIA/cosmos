#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for llava_ov (VLM SFT on
# lmms-lab/LLaVA-OneVision-Data via CosmosDataLoader). Drives
# cosmos_framework.scripts.train against toml/sft_config/llava_ov.toml.
#
# [job].task = "vlm" — picks cosmos_framework/configs/base/vlm/config.py as the base config.
#
# Requires an activated cosmos-framework venv (see the finetune README
# Prerequisites). Run from cookbooks/cosmos3/finetune/.
#
# The dataset streams from the HuggingFace Hub, so DATASET_PATH /
# WAN_VAE_PATH / BASE_CHECKPOINT_PATH are NOT required.
#
# Optional env:
#   HF_TOKEN               for gated Qwen3-VL-8B-Instruct downloads.
#   VLM_SAFETENSORS_PATH   local directory of pre-converted Qwen3-VL safetensors
#                          (e.g. a Cosmos3-Nano LM merged with Qwen3-VL visual via
#                          `cosmos_framework.scripts.convert_model_to_vlm_safetensors`).
#                          When set, plumbed to backbone.safetensors_path via a
#                          tail override. When unset, the framework falls back
#                          to the public Qwen/Qwen3-VL-8B-Instruct HF snapshot.
#
# Usage (8-GPU allocation, framework venv active, from cookbooks/cosmos3/finetune/):
#   bash launch_sft_llava_ov.sh

TOML_FILE="toml/sft_config/llava_ov.toml"

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

# When VLM_SAFETENSORS_PATH is set, plumb it to backbone.safetensors_path so the
# framework loads weights from the local snapshot (e.g. a Cosmos3-Nano LM merged
# with Qwen3-VL visual via `cosmos_framework.scripts.convert_model_to_vlm_safetensors`)
# while keeping the public HF model_name for tokenizer/architecture discovery.
if [[ -n "${VLM_SAFETENSORS_PATH:-}" ]]; then
    TAIL_OVERRIDES+=("model.config.policy.backbone.safetensors_path=$VLM_SAFETENSORS_PATH")
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
