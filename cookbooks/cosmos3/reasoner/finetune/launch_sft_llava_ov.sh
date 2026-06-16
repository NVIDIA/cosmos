#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Complete recipe: Reasoner alignment SFT on LLaVA-OneVision (8x H100).
# Run from this folder with the cosmos-framework venv active (see README):
#   bash launch_sft_llava_ov.sh
# The dataset streams from HuggingFace and the Qwen3-VL-8B-Instruct backbone is
# fetched at startup, so there's nothing to download first — this just trains.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Train (8-GPU FSDP).
IMAGINAIRE_OUTPUT_ROOT="$PWD/outputs/train" torchrun --nproc_per_node=8 \
    -m cosmos_framework.scripts.train --sft-toml="toml/sft_config/llava_ov.toml"
