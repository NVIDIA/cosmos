#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Complete recipe: DROID action-policy SFT on Cosmos3-Nano.
# The TOML pins the GB200 reference shape (HSDP 32x8 = 256 ranks, global batch
# 8192); on fewer GPUs override data_parallel_replicate_degree / grad_accum_iter
# (see README). Set NNODES / NODE_RANK / MASTER_ADDR for multi-node.
# Run from this folder with the cosmos-framework venv active (see README):
#   bash launch_sft_action_policy_droid_nano.sh
# It prepares the small dependencies, checks for the staged DROID dataset, and trains.
# Paths are fixed under this (git-ignored) folder, matching the reasoner finetune
# wrappers, while the TOML and tail-overrides match the cosmos-framework example.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TOML_FILE="$PWD/toml/sft_config/action_policy_droid_nano.toml"
: "${DATASET_PATH:=$PWD/data/Cosmos3-DROID/success}"
: "${BASE_CHECKPOINT_PATH:=$PWD/checkpoints/Cosmos3-Nano}"
: "${WAN_VAE_PATH:=$PWD/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${FILTER_PATH:=$PWD/data/droid_filters/keep_ranges_1_0_1.json}"

# 1. The full DROID dataset is large, so require users to stage it explicitly.
if [[ ! -f "$DATASET_PATH/meta/info.json" ]]; then
    cat >&2 <<EOF
ERROR: missing DROID dataset at:
  $DATASET_PATH

Expected a LeRobotDataset success split containing meta/info.json.
Stage nvidia/Cosmos3-DROID first, or export DATASET_PATH=/path/to/Cosmos3-DROID/success.
For example:
  uvx hf@latest download --repo-type dataset nvidia/Cosmos3-DROID --local-dir data/Cosmos3-DROID
EOF
    exit 1
fi

# 2. Download the Wan2.2 VAE (skipped if present).
if [[ ! -f "$WAN_VAE_PATH" ]]; then
    uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$(dirname "$WAN_VAE_PATH")"
fi

# 3. Convert the base checkpoint to DCP (skipped if present).
if [[ ! -d "$BASE_CHECKPOINT_PATH" ]]; then
    python -m cosmos_framework.scripts.convert_model_to_dcp -o "$BASE_CHECKPOINT_PATH" --checkpoint-path Cosmos3-Nano
fi

# 4. Download the keep-ranges filter used by the released reproduction recipe (skipped if present).
if [[ ! -f "$FILTER_PATH" ]]; then
    mkdir -p "$(dirname "$FILTER_PATH")"
    uvx hf@latest download KarlP/droid keep_ranges_1_0_1.json --local-dir "$(dirname "$FILTER_PATH")"
fi

# 5. Train. torchrun uses NPROC_PER_NODE GPUs (8 by default); the TOML's HSDP 32x8
#    shape needs 256 ranks, so scale nodes or override parallelism (see README).
#    The TOML reads these paths from the environment.
export DROID_ROOT="${DROID_ROOT:-$DATASET_PATH}"
export BASE_CHECKPOINT_PATH
export WAN_VAE_PATH

TAIL_OVERRIDES=(
    "dataloader_train.dataloader.datasets.droid.dataset.use_filter_dict=True"
    "dataloader_train.dataloader.datasets.droid.dataset.filter_dict_path=$FILTER_PATH"
)
if [[ -n "${EXTRA_TAIL_OVERRIDES:-}" ]]; then
    # EXTRA_TAIL_OVERRIDES is intentionally word-split to match the framework launcher UX.
    # shellcheck disable=SC2206
    EXTRA_OVERRIDES_ARRAY=(${EXTRA_TAIL_OVERRIDES})
    TAIL_OVERRIDES+=("${EXTRA_OVERRIDES_ARRAY[@]}")
fi

TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE:-8}")
TORCHRUN_ARGS+=(--master_port="${MASTER_PORT:-50012}")
[[ -n "${NNODES:-}" ]] && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]] && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")

OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/train}"
# The model configs reference their packaged JSONs relative to the framework
# root, so run torchrun from there (all recipe paths above are absolute).
cd "$(python -c 'import pathlib, cosmos_framework; print(pathlib.Path(cosmos_framework.__file__).resolve().parents[1])')"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}" torchrun "${TORCHRUN_ARGS[@]}" \
    -m cosmos_framework.scripts.train --sft-toml="$TOML_FILE" \
    -- "${TAIL_OVERRIDES[@]}"
