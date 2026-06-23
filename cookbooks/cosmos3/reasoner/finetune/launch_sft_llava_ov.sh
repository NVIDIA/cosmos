#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Complete recipe: Reasoner alignment SFT on LLaVA-OneVision (8x H100).
# Run from this folder with the cosmos-framework venv active (see README):
#   bash launch_sft_llava_ov.sh
# The dataset streams from HuggingFace, so there's nothing to download first. It
# builds the merged Cosmos3-Nano VLM checkpoint, then trains — in order. Paths
# are fixed under this (git-ignored) folder.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VLM_CHECKPOINT="$PWD/checkpoints/Cosmos3-Nano-VLM"

# 1. Build the merged Cosmos3-Nano VLM checkpoint (skipped if present).
if [[ ! -d "$VLM_CHECKPOINT" ]]; then
    python -m cosmos_framework.scripts.convert_model_to_vlm_safetensors --checkpoint-path Cosmos3-Nano -o "$VLM_CHECKPOINT"
fi

# 2. Train (8-GPU FSDP). The merged checkpoint is supplied as a config override
#    after `--`.
IMAGINAIRE_OUTPUT_ROOT="$PWD/outputs/train" torchrun --nproc_per_node=8 \
    -m cosmos_framework.scripts.train --sft-toml="toml/sft_config/llava_ov.toml" \
    -- model.config.policy.backbone.safetensors_path="$VLM_CHECKPOINT"
