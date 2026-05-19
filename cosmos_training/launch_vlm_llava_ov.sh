#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Launch the pre_exp012_llava_ov_datapacker smoke inside the released tree.
#
# Source experiment file (carried over via cosmos.toml mapping):
#   projects/cosmos3/vfm/configs/base/vlm/experiment/llava_ov_datapacker_experiment.py
# Released as:
#   configs/base/vlm/experiment/llava_ov_datapacker_experiment.py
# Registered Hydra name:
#   pre_exp012_llava_ov_datapacker
#
# Dataset: lmms-lab/LLaVA-OneVision-Data (streamed from HuggingFace Hub).
# Model:   Siglip2-Qwen3-1.7B-BF16-Alignment (HF safetensors).
#
# Usage (from a Slurm allocation with --container-name yangyangt_dev):
#   srun --overlap --jobid <JOB_ID> --container-name yangyangt_dev \
#        bash -s < /lustre/.../cosmos_training/launch_vlm_llava_ov.sh
set -uo pipefail

WORKDIR="/lustre/fsw/portfolios/cosmos/users/yangyangt/Cosmos-prerelease/cosmos_training"
MODEL_PATH="/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/maoshengl/models/Siglip2-Qwen3-1.7B-BF16-Alignment"

OUTPUT_ROOT="/lustre/fsw/portfolios/cosmos/users/yangyangt/Cosmos-prerelease/training_output"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_FILE="$LOG_DIR/pre_exp012_llava_ov_datapacker.log"

mkdir -p "$LOG_DIR"

echo ">>> $(date '+%H:%M:%S') Checking inputs..."
[[ -d "$WORKDIR" ]]    || { echo "ERROR: WORKDIR not found: $WORKDIR" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "ERROR: model dir not found: $MODEL_PATH" >&2; exit 1; }

cd "$WORKDIR"
echo ">>> $(date '+%H:%M:%S') WORKDIR:   $WORKDIR"
echo ">>> $(date '+%H:%M:%S') model:     $MODEL_PATH"
echo ">>> $(date '+%H:%M:%S') log:       $LOG_FILE"

# HuggingFace env: streamed dataset (no local download) + tokenizer/processor fetch.
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HUB_DISABLE_XET=1

# Determinism: PYTHONHASHSEED must be set before the interpreter starts.
export PYTHONHASHSEED=42

IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" PYTHONPATH=. \
    torchrun --nproc_per_node=4 --master_port=50012 -m scripts.train \
    --config=configs/base/vlm/config.py \
    --deterministic \
    -- \
    experiment=pre_exp012_llava_ov_datapacker \
    "model.config.policy.backbone.model_name=$MODEL_PATH" \
    trainer.max_iter=10 \
    trainer.logging_iter=1 \
    job.wandb_mode=disabled \
    checkpoint.load_from_object_store.enabled=false \
    checkpoint.save_to_object_store.enabled=false \
    upload_reproducible_setup=false \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit $EXIT_CODE
