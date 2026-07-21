#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Complete recipe: LIBERO-all (4-suite) action-policy SFT on Cosmos3-Nano (HSDP 2x8).
# Run from this folder with the cosmos-framework venv active (see README):
#   bash launch_sft_action_policy_libero_all_nano.sh
# Trains on all 4 LIBERO suites (equal mix); it prepares the small dependencies,
# checks for the staged suites, and trains. The 4-suite mix needs longer training
# than libero_10-only (max_iter 5000 vs 2000). Paths are fixed under this
# (git-ignored) folder, matching the reasoner finetune wrappers.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TOML_FILE="toml/sft_config/action_policy_libero_all_nano.toml"
: "${LIBERO_ROOT:=$PWD/data/LIBERO_LeRobot_v3}"   # PARENT dir of the 4 suites
: "${BASE_CHECKPOINT_PATH:=$PWD/checkpoints/Cosmos3-Nano}"
: "${WAN_VAE_PATH:=$PWD/checkpoints/wan22_vae/Wan2.2_VAE.pth}"

# 1. Stage all 4 LIBERO suites (libero-all trains on the full mix).
_missing=""
for _s in libero_spatial libero_object libero_goal libero_10; do
    [[ -f "$LIBERO_ROOT/$_s/meta/info.json" ]] || _missing="$_missing $_s"
done
if [[ -n "$_missing" ]]; then
    echo "Downloading nvidia/LIBERO_LeRobot_v3 (all suites:$_missing) ..."
    uvx hf@latest download --repo-type dataset nvidia/LIBERO_LeRobot_v3 --local-dir "$LIBERO_ROOT"
fi
for _s in libero_spatial libero_object libero_goal libero_10; do
    if [[ ! -f "$LIBERO_ROOT/$_s/meta/info.json" ]]; then
        cat >&2 <<EOF
ERROR: missing LIBERO suite '$_s' under:
  $LIBERO_ROOT

Expected LIBERO_ROOT to be the LIBERO_LeRobot_v3 parent dir containing all 4 suites.
Stage them with:
  uvx hf@latest download --repo-type dataset nvidia/LIBERO_LeRobot_v3 \\
      --local-dir data/LIBERO_LeRobot_v3
or export LIBERO_ROOT=/path/to/LIBERO_LeRobot_v3.
EOF
        exit 1
    fi
done

# 2. Download the Wan2.2 VAE (skipped if present).
if [[ ! -f "$WAN_VAE_PATH" ]]; then
    uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$(dirname "$WAN_VAE_PATH")"
fi

# 3. Convert the base checkpoint to DCP (skipped if present).
if [[ ! -d "$BASE_CHECKPOINT_PATH" ]]; then
    python -m cosmos_framework.scripts.convert_model_to_dcp -o "$BASE_CHECKPOINT_PATH" --checkpoint-path Cosmos3-Nano
fi

# 4. Train (HSDP 2x8 per the TOML; set NNODES/NODE_RANK/MASTER_ADDR per node).
#    The TOML reads these paths from the environment.
export LIBERO_ROOT
export BASE_CHECKPOINT_PATH
export WAN_VAE_PATH

TAIL_OVERRIDES=()
if [[ -n "${EXTRA_TAIL_OVERRIDES:-}" ]]; then
    # EXTRA_TAIL_OVERRIDES is intentionally word-split to match the framework launcher UX.
    # shellcheck disable=SC2206
    TAIL_OVERRIDES=(${EXTRA_TAIL_OVERRIDES})
fi

TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE:-8}")
TORCHRUN_ARGS+=(--master_port="${MASTER_PORT:-50012}")
[[ -n "${NNODES:-}" ]] && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]] && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")

OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/train}"
if (( ${#TAIL_OVERRIDES[@]} )); then
    IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}" torchrun "${TORCHRUN_ARGS[@]}" \
        -m cosmos_framework.scripts.train --sft-toml="$TOML_FILE" \
        -- "${TAIL_OVERRIDES[@]}"
else
    IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}" torchrun "${TORCHRUN_ARGS[@]}" \
        -m cosmos_framework.scripts.train --sft-toml="$TOML_FILE"
fi
