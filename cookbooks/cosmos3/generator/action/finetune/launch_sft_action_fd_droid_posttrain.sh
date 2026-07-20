#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for action_fd_droid_posttrain.
#
# This script intentionally does not source _sft_launcher_common.sh. It contains
# the recipe-specific defaults, input checks, path anchoring, tail overrides, and
# torchrun invocation in one file.
#
# Run with the cosmos-framework venv active (see action_fd_droid_posttrain.md).
# Training starts from the installed cosmos-framework checkout root so relative
# model config paths (e.g. Qwen3-VL JSON) resolve correctly.
#
# Env vars (override for your filesystem):
#   DATASET_PATH            Cosmos3-DROID parent dir (success/ + failure/)
#   BASE_CHECKPOINT_PATH    Base DCP checkpoint
#   WAN_VAE_PATH            Wan2.2 VAE .pth
#   COSMOS_FRAMEWORK_ROOT   Optional override for the framework checkout root
#   NPROC_PER_NODE          torchrun --nproc_per_node (default 8)
#   EXTRA_TAIL_OVERRIDES    space-separated Hydra overrides
#   MASTER_PORT             torchrun --master_port (default 50012)
#   NNODES                  torchrun --nnodes for multi-node
#   NODE_RANK               torchrun --node_rank for multi-node
#   MASTER_ADDR             torchrun --master_addr for multi-node
#   OUTPUT_ROOT             default output root (default outputs/train)
#   LOG_FILENAME            override log filename
# ============================================================================

set -euo pipefail

# Cookbook recipe directory (TOML, logs, default relative paths).
COOKBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOML_FILE="$COOKBOOK_DIR/toml/sft_config/action_fd_droid_posttrain.toml"

: "${DATASET_PATH:=data/Cosmos3-DROID}"
: "${BASE_CHECKPOINT_PATH:=checkpoints/Cosmos3-Nano}"
: "${WAN_VAE_PATH:=checkpoints/wan22_vae/Wan2.2_VAE.pth}"

[[ "$DATASET_PATH" = /* ]] || DATASET_PATH="$COOKBOOK_DIR/$DATASET_PATH"
[[ "$BASE_CHECKPOINT_PATH" = /* ]] || BASE_CHECKPOINT_PATH="$COOKBOOK_DIR/$BASE_CHECKPOINT_PATH"
[[ "$WAN_VAE_PATH" = /* ]] || WAN_VAE_PATH="$COOKBOOK_DIR/$WAN_VAE_PATH"
export DATASET_PATH BASE_CHECKPOINT_PATH WAN_VAE_PATH

OUTPUT_ROOT="${OUTPUT_ROOT:-$COOKBOOK_DIR/outputs/train}"
LOG_DIR="$OUTPUT_ROOT/logs"
TOML_STEM="$(basename "$TOML_FILE" .toml)"
LOG_FILE="$LOG_DIR/${LOG_FILENAME:-${TOML_STEM}_sft.log}"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}"
mkdir -p "$LOG_DIR"

echo ">>> $(date '+%H:%M:%S') Checking inputs..."
[[ -f "$TOML_FILE" ]] || { echo "ERROR: TOML not found: $TOML_FILE" >&2; exit 1; }
[[ -d "$DATASET_PATH" ]] || { echo "ERROR: DATASET_PATH not found: $DATASET_PATH" >&2; exit 1; }
[[ -d "$BASE_CHECKPOINT_PATH" ]] || { echo "ERROR: BASE_CHECKPOINT_PATH not found: $BASE_CHECKPOINT_PATH" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]] || { echo "ERROR: WAN_VAE_PATH not found: $WAN_VAE_PATH" >&2; exit 1; }

[[ -f "$DATASET_PATH/success/meta/info.json" || -n "$(compgen -G "$DATASET_PATH/success/*/meta/info.json")" ]] || {
    echo "ERROR: missing Cosmos3-DROID success split under $DATASET_PATH (expected success/meta/info.json or success/*/meta/info.json)" >&2
    exit 1
}
[[ -f "$DATASET_PATH/failure/meta/info.json" || -n "$(compgen -G "$DATASET_PATH/failure/*/meta/info.json")" ]] || {
    echo "ERROR: missing Cosmos3-DROID failure split under $DATASET_PATH (expected failure/meta/info.json or failure/*/meta/info.json)" >&2
    exit 1
}

if [[ -n "${COSMOS_FRAMEWORK_ROOT:-}" ]]; then
    FRAMEWORK_ROOT="$(cd "$COSMOS_FRAMEWORK_ROOT" && pwd)"
else
    FRAMEWORK_ROOT="$(
        python -c 'import pathlib, cosmos_framework; print(pathlib.Path(cosmos_framework.__file__).resolve().parents[1])'
    )"
fi

QWEN3_VL_CONFIG="$FRAMEWORK_ROOT/cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
[[ -f "$QWEN3_VL_CONFIG" ]] || {
    cat >&2 <<EOF
ERROR: cannot resolve cosmos-framework model configs under:
  $FRAMEWORK_ROOT

Expected Qwen3-VL config at:
  $QWEN3_VL_CONFIG

Activate the cosmos-framework venv, or export COSMOS_FRAMEWORK_ROOT=/path/to/cosmos-framework.
EOF
    exit 1
}

TAIL_OVERRIDES=(
    "trainer.straggler_detection.enabled=False"
)
if [[ -n "${EXTRA_TAIL_OVERRIDES:-}" ]]; then
    # EXTRA_TAIL_OVERRIDES is intentionally word-split to match the framework launcher UX.
    # shellcheck disable=SC2206
    EXTRA_OVERRIDES_ARRAY=(${EXTRA_TAIL_OVERRIDES})
    TAIL_OVERRIDES+=("${EXTRA_OVERRIDES_ARRAY[@]}")
fi

TRAILING_ARGS=(-- "${TAIL_OVERRIDES[@]}")

TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE:-8}" --master_port="${MASTER_PORT:-50012}")
[[ -n "${NNODES:-}" ]] && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]] && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")

cd "$FRAMEWORK_ROOT"
echo ">>> $(date '+%H:%M:%S') cookbook:   $COOKBOOK_DIR"
echo ">>> $(date '+%H:%M:%S') framework:  $FRAMEWORK_ROOT"
echo ">>> $(date '+%H:%M:%S') TOML:       $TOML_FILE"
echo ">>> $(date '+%H:%M:%S') dataset:    $DATASET_PATH"
echo ">>> $(date '+%H:%M:%S') checkpoint: $BASE_CHECKPOINT_PATH"
echo ">>> $(date '+%H:%M:%S') log:        $LOG_FILE"

IMAGINAIRE_OUTPUT_ROOT="$IMAGINAIRE_OUTPUT_ROOT" \
PYTHONPATH="${FRAMEWORK_ROOT}${PYTHONPATH:+:$PYTHONPATH}" \
    torchrun "${TORCHRUN_ARGS[@]}" -m cosmos_framework.scripts.train \
    --sft-toml="$TOML_FILE" \
    "${TRAILING_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit $EXIT_CODE
