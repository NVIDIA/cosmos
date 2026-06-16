#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for vision_sft_super (T2V / I2V / V2V LoRA SFT on
# Qwen3-VL-32B-Instruct, 8-GPU FSDP with CP=2 / DP=4). Drives
# cosmos_framework.scripts.train against toml/sft_config/vision_sft_super.toml.
#
# Requires an activated cosmos-framework venv (see the finetune README
# Prerequisites). Run from cookbooks/cosmos3/finetune/.
#
# Optional env vars (defaults below point under this cookbook dir; override to
# put data or checkpoints on a different filesystem):
#   DATASET_PATH          default: data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge
#                         (must contain train/video_dataset_file.jsonl)
#   BASE_CHECKPOINT_PATH  default: checkpoints/Cosmos3-Super
#   WAN_VAE_PATH          default: checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              if any tokenizer download requires gated HF access
#   OUTPUT_ROOT           default: outputs/train
#
# Usage (8-GPU allocation, framework venv active, from cookbooks/cosmos3/finetune/):
#   bash launch_sft_vision_super.sh

TOML_FILE="toml/sft_config/vision_sft_super.toml"
: "${DATASET_PATH:=data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=checkpoints/Cosmos3-Super}"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }'

# Super-variant env tweaks: clear LD_LIBRARY_PATH to avoid host CUDA/NCCL libs
# bleeding into the venv, switch the allocator to expandable_segments so the
# 32B backbone fits without OOM during compile/decode.
export LD_LIBRARY_PATH=""
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
