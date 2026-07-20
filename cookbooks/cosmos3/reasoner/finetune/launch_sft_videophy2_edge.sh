#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Complete recipe: Reasoner physical-plausibility SFT on VideoPhy-2, Cosmos3-Edge
# tier (Nemotron-2B-Dense-VL LM + SigLIP2 vision tower) (8x H100 or 4x GB200).
# Run from this folder with the cosmos-framework venv active (see README):
#   bash launch_sft_videophy2_edge.sh
# It materializes the dataset, pre-fetches the model snapshot, and trains — in
# order. Paths are fixed under this folder. Reasoner weights load directly from
# the public nvidia/Cosmos3-Edge snapshot via the TOML's model_name — no converter step.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VIDEOPHYSICS_ROOT="$PWD/data/videophysics"

# 1. Materialize the VideoPhy-2 dataset (skipped if present).
if [[ ! -d "$VIDEOPHYSICS_ROOT/videophy2_train" ]]; then
    python -m cosmos_framework.scripts.reasoner.prepare_videophy2_from_hf --out_root "$VIDEOPHYSICS_ROOT" --split both
fi

# 2. Pre-fetch the nvidia/Cosmos3-Edge snapshot into the HF cache (idempotent) so
#    the multi-GB first download stays out of the multi-rank torchrun job.
uvx hf@latest download nvidia/Cosmos3-Edge

# 3. Train (FSDP full shard). VIDEOPHYSICS_ROOT is read from the environment.
export VIDEOPHYSICS_ROOT
# Edge is only 2B — on a 4-GPU node (e.g. GB200x4), set --nproc_per_node=4 instead.
# The TOML defaults to flash_attention_2; if flash-attn is not installed, append
# `-- model.config.policy.attn_implementation=sdpa` to the torchrun command.
IMAGINAIRE_OUTPUT_ROOT="$PWD/outputs/train" torchrun --nproc_per_node=8 \
    -m cosmos_framework.scripts.train --sft-toml="toml/sft_config/videophy2_sft_edge.toml"
