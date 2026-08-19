#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Complete recipe: RoboCasa mobile-manipulation action-policy SFT on Cosmos3-Nano (HSDP 8x2).
# Run from this folder with the cosmos-framework venv active (see README):
#   bash launch_sft_action_policy_robocasa_nano.sh
#
# Trains a MOBILE-BASE manipulation policy on RoboCasa with the native base-motion contract
# (`base_encoding="raw"`, 15-D) and the full-resolution agentview_left | eye_in_hand composite.
# See the TOML header for the full contract.
#
# The recipe is not tied to one task set. TASK_SET selects which RoboCasa split to train on:
#
#   target_atomic     18 tasks   target/atomic          (default; the worked example)
#   pretrain_atomic   65 tasks   pretrain/atomic
#   pretrain_all     300 tasks   pretrain atomic + composite
#
# Anything beyond the default also needs ROBOCASA_ROOT pointed at the matching converted export
# and a longer schedule -- pass e.g.
#   EXTRA_TAIL_OVERRIDES="trainer.max_iter=40000 scheduler.cycle_lengths=[40000]"
#
# It prepares the small dependencies, checks for the staged dataset, and trains.
# Paths are fixed under this (git-ignored) folder, matching the DROID/LIBERO wrappers, while
# the TOML and tail-overrides match the cosmos-framework example.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TOML_FILE="${TOML_FILE:-toml/sft_config/action_policy_robocasa_nano.toml}"
# RoboCasa's released export is LeRobot v2.1; the installed lerobot is v3.0-only and rejects it.
# ROBOCASA_ROOT must point at a CONVERTED copy: <root>/<task>/<date>/lerobot/.
: "${ROBOCASA_ROOT:=$PWD/data/robocasa_v30}"
: "${BASE_CHECKPOINT_PATH:=$PWD/checkpoints/Cosmos3-Nano}"
: "${WAN_VAE_PATH:=$PWD/checkpoints/wan22_vae/Wan2.2_VAE.pth}"

# Resolve TASK_SET to a task list. The tuples live in the framework's
# robocasa_lerobot_dataset module, which is the single source of truth; reading them here keeps
# the names from drifting between the recipe and the dataset.
TASK_SET="${TASK_SET:-target_atomic}"
case "$TASK_SET" in
    target_atomic)   TASK_CONST=DEFAULT_ALL_ATOMIC_TASKS ;;
    pretrain_atomic) TASK_CONST=DEFAULT_PRETRAIN_ATOMIC_TASKS ;;
    pretrain_all)    TASK_CONST=DEFAULT_PRETRAIN_ALL_TASKS ;;
    *) echo "TASK_SET must be one of: target_atomic pretrain_atomic pretrain_all" >&2; exit 1 ;;
esac
_mod=cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset
_tasks="$(python -c "from ${_mod} import ${TASK_CONST} as T; print(' '.join(T))")" || {
    echo "ERROR: could not read ${TASK_CONST} from ${_mod}; is the cosmos-framework venv active?" >&2
    exit 1
}
read -ra TASKS <<< "$_tasks"
# The TOML pins the default set; anything else is selected with a hydra list override.
if [[ "$TASK_SET" != "target_atomic" ]]; then
    _list="$(IFS=,; echo "${TASKS[*]}")"
    EXTRA_TAIL_OVERRIDES="${EXTRA_TAIL_OVERRIDES:-} dataloader_train.dataloader.datasets.robocasa.dataset.task_names=[${_list}]"
fi
echo "task set   : ${TASK_SET} (${#TASKS[@]} tasks)"

# 1. Check the v3.0-converted dataset is staged. tasks.parquet is a v3.0-only marker, so its
#    presence distinguishes a converted copy from the raw v2.1 export.
missing=()
for t in "${TASKS[@]}"; do
    ls "$ROBOCASA_ROOT/$t"/*/lerobot/meta/tasks.parquet >/dev/null 2>&1 || missing+=("$t")
done
if (( ${#missing[@]} )); then
    cat >&2 <<EOF
ERROR: ${#missing[@]} of ${#TASKS[@]} RoboCasa tasks (${TASK_SET}) are missing under:
  $ROBOCASA_ROOT

Missing: ${missing[*]}

Expected per-task LeRobot v3.0 dirs: <ROBOCASA_ROOT>/<task>/<date>/lerobot/meta/tasks.parquet
The installed lerobot is v3.0-only, so the released v2.1 export must be converted first:
  SRC_ROOT=/path/to/robocasa/datasets/v1.0/target/atomic bash convert_robocasa_to_lerobot_v30.sh
or point at an existing converted root:
  export ROBOCASA_ROOT=/path/to/converted/v30/root

The loader skips absent tasks with a warning rather than failing, so this check runs up front
to avoid training on a partial task set by accident.
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

# 4. Train (HSDP 8x2 = 16 ranks per the TOML; set NNODES/NODE_RANK/MASTER_ADDR per node).
#    The TOML reads these paths from the environment.
export ROBOCASA_ROOT
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

# The training entrypoint does NOT chdir, but the model config loads release-root-relative
# resource files, so torchrun must run from the package root. Derive it from the installed
# package and make the TOML path absolute so it still resolves from that cwd. Every other
# path the run needs (ROBOCASA_ROOT / BASE_CHECKPOINT_PATH / WAN_VAE_PATH / OUTPUT_ROOT) is
# already absolute.
COSMOS_PKG_ROOT="$(python -c 'import cosmos_framework, pathlib; print(pathlib.Path(cosmos_framework.__file__).resolve().parent.parent)')"
TOML_ABS="$PWD/$TOML_FILE"
echo "torchrun cwd=$COSMOS_PKG_ROOT  toml=$TOML_ABS"

# torchrun workers inherit the launcher's cwd, so run it from the package root (subshell
# keeps the cd local). All paths passed in are absolute, so nothing else is affected.
if (( ${#TAIL_OVERRIDES[@]} )); then
    ( cd "$COSMOS_PKG_ROOT" && IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}" \
        torchrun "${TORCHRUN_ARGS[@]}" \
        -m cosmos_framework.scripts.train --sft-toml="$TOML_ABS" \
        -- "${TAIL_OVERRIDES[@]}" )
else
    ( cd "$COSMOS_PKG_ROOT" && IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}" \
        torchrun "${TORCHRUN_ARGS[@]}" \
        -m cosmos_framework.scripts.train --sft-toml="$TOML_ABS" )
fi
