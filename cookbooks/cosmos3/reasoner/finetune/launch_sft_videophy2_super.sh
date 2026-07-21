#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Complete recipe: Reasoner physical-plausibility SFT on VideoPhy-2, Cosmos3-Super
# tier (Qwen3-VL-32B full fine-tune) (8x H100).
# Run from this folder with the cosmos-framework venv active (see README):
#   bash launch_sft_videophy2_super.sh
# It materializes the dataset, builds the Cosmos3-Super Reasoner checkpoint, and
# trains — in order. Paths are fixed under this folder.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Super-variant allocator tweak: expandable_segments so the 32B backbone fits
# without OOM. (We do NOT clear LD_LIBRARY_PATH — VideoPhy-2 clips are decoded with
# torchcodec, which dlopen()s the CUDA NPP + FFmpeg libs off LD_LIBRARY_PATH.)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

VIDEOPHYSICS_ROOT="$PWD/data/videophysics"
REASONER_CHECKPOINT="$PWD/checkpoints/Cosmos3-Super-Reasoner"

# 1. Materialize the VideoPhy-2 dataset (skipped if present).
if [[ ! -d "$VIDEOPHYSICS_ROOT/videophy2_train" ]]; then
    python -m cosmos_framework.scripts.reasoner.prepare_videophy2_from_hf --out_root "$VIDEOPHYSICS_ROOT" --split both
fi

# 2. Build the Cosmos3-Super Reasoner checkpoint: merge the Cosmos3-Super LM onto
#    the Qwen3-VL-32B visual tower (skipped if present).
if [[ ! -d "$REASONER_CHECKPOINT" ]]; then
    python -m cosmos_framework.scripts.convert_model_to_vlm_safetensors --checkpoint-path Cosmos3-Super --vlm-model-name Qwen/Qwen3-VL-32B-Instruct -o "$REASONER_CHECKPOINT"
fi

# 3. Train (8-GPU FSDP). VIDEOPHYSICS_ROOT is read from the environment; the
#    checkpoint is supplied as a config override after `--`.
export VIDEOPHYSICS_ROOT
# On a 4-GPU node (e.g. GB200x4), set --nproc_per_node=4 instead.
IMAGINAIRE_OUTPUT_ROOT="$PWD/outputs/train" torchrun --nproc_per_node=8 \
    -m cosmos_framework.scripts.train --sft-toml="toml/sft_config/videophy2_sft_super.toml" \
    -- model.config.policy.backbone.safetensors_path="$REASONER_CHECKPOINT"
