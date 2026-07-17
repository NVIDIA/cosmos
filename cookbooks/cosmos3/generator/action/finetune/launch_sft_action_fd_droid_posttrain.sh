#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for action_fd_droid_posttrain.
#
# This trains forward dynamics on the Cosmos3-DROID success + failure splits. See
# docs/action_fd_droid_posttrain.md.
#
# Env vars (override for your filesystem):
#   DATASET_PATH                 Cosmos3-DROID parent dir (success/ + failure/)
#   BASE_CHECKPOINT_PATH         Base DCP checkpoint
#   WAN_VAE_PATH                 Wan2.2 VAE .pth
#   WANDB_API_KEY                for online logging (TOML wandb_mode="online")
#   NPROC_PER_NODE               torchrun --nproc_per_node (default 8)
#   EXTRA_TAIL_OVERRIDES         space-separated Hydra overrides
#
# Single-node smoke:
#   export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10"
#   bash examples/launch_sft_action_fd_droid_posttrain.sh
#
# Multi-node: launch on every worker. For HSDP set
# model.parallelism.data_parallel_replicate_degree = <num_nodes> (shard stays 8).
# ============================================================================

TOML_FILE="examples/toml/sft_config/action_fd_droid_posttrain.toml"
: "${DATASET_PATH:=examples/data/Cosmos3-DROID}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

WAN_VAE_PATH="${WAN_VAE_PATH:-examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"

EXTRA_DATASET_CHECK='[[ "$WAN_VAE_PATH" = /* ]] || WAN_VAE_PATH="$WORKDIR/$WAN_VAE_PATH"; export WAN_VAE_PATH; [[ -f "$DATASET_PATH/success/meta/info.json" || -n "$(compgen -G "$DATASET_PATH/success/*/meta/info.json")" ]] || { echo "ERROR: missing Cosmos3-DROID success split under $DATASET_PATH (expected success/meta/info.json or success/*/meta/info.json; see docs/action_fd_droid_posttrain.md)" >&2; exit 1; }; [[ -f "$DATASET_PATH/failure/meta/info.json" || -n "$(compgen -G "$DATASET_PATH/failure/*/meta/info.json")" ]] || { echo "ERROR: missing Cosmos3-DROID failure split under $DATASET_PATH (expected failure/meta/info.json or failure/*/meta/info.json; see docs/action_fd_droid_posttrain.md)" >&2; exit 1; }; [[ -f "$WAN_VAE_PATH" ]] || { echo "ERROR: WAN_VAE_PATH not found: $WAN_VAE_PATH" >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
