#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Complete recipe: Vision SFT on Cosmos3-Edge (T2V / I2V / V2V, 8x H100 or 4x GB200).
# Full fine-tune of the compact 2B dense Nemotron backbone — same dataset and
# dataflow as the Cosmos3-Nano recipe.
# Run from this folder with the cosmos-framework venv active (see README):
#   bash launch_sft_vision_edge.sh
# It downloads the data, prepares the base checkpoint, and trains — in order.
# Paths are fixed under this folder; edit them below to relocate.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATASET_DIR="$PWD/data/BridgeData2-Subset-Synthetic-Captions"
CHECKPOINT_DIR="$PWD/checkpoints/Cosmos3-Edge"
VAE_PATH="$PWD/checkpoints/wan22_vae/Wan2.2_VAE.pth"

# 1. Download the SFT dataset (skipped if present; license-gated — accept terms + 'uvx hf@latest auth login').
if [[ ! -f "$DATASET_DIR/sft_dataset_bridge/train/video_dataset_file.jsonl" ]]; then
    uvx hf@latest download --repo-type dataset nvidia/BridgeData2-Subset-Synthetic-Captions \
        --revision 40d018ac1c1a2a4b9734f17fdb21f3d933c49a01 --local-dir "$DATASET_DIR"
fi

# 2. Download the Wan2.2 VAE (skipped if present).
if [[ ! -f "$VAE_PATH" ]]; then
    uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$(dirname "$VAE_PATH")"
fi

# 3. Convert the base checkpoint to DCP (skipped if present). Cosmos3-Edge is a
#    registered checkpoint name — the converter fetches it from HF, same as Nano.
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    python -m cosmos_framework.scripts.convert_model_to_dcp -o "$CHECKPOINT_DIR" --checkpoint-path Cosmos3-Edge
fi

# 4. Train (8-GPU FSDP). The TOML reads these three paths from the environment.
export DATASET_PATH="$DATASET_DIR/sft_dataset_bridge"
export BASE_CHECKPOINT_PATH="$CHECKPOINT_DIR"
export WAN_VAE_PATH="$VAE_PATH"
# The model configs reference their packaged JSONs relative to the framework
# root, so run torchrun from there (recipe paths stay pinned to this folder).
TOML_PATH="$PWD/toml/sft_config/vision_sft_edge.toml"
OUTPUT_ROOT="$PWD/outputs/train"
cd "$(python -c 'import pathlib, cosmos_framework; print(pathlib.Path(cosmos_framework.__file__).resolve().parents[1])')"
# Edge is only 2B — on a 4-GPU node (e.g. GB200x4), set --nproc_per_node=4 instead.
IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" torchrun --nproc_per_node=8 \
    -m cosmos_framework.scripts.train --sft-toml="$TOML_PATH"
