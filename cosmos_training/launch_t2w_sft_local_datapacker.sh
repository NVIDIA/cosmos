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

# Launch the t2w_sft_8b_local_datapacker smoke inside the released tree.
#
# Source experiment file (carried over via cosmos.toml mapping):
#   projects/cosmos3/vfm/configs/base/experiment/posttrain_video/t2w_sft_8b_local_datapacker.py
# Released as:
#   configs/base/experiment/posttrain_video/t2w_sft_8b_local_datapacker.py
# Registered Hydra name: t2w_sft_8b_local_datapacker
#
# Dataset: same local sft_dataset_bridge JSONL used by mixed_modality_sft_8b
# (renamed key `s3_path` -> `vision_path` lives in the .bak'd JSONL already).
# Model: warm-start from the cosmos3-nano midtrain DCP (8B weights).
# Wan VAE: same local copy as the other smokes.
#
# Usage (from a Slurm allocation with --container-name yangyangt_dev):
#   srun --overlap --jobid <JOB_ID> --container-name yangyangt_dev \
#        bash -s < /lustre/.../cosmos_training/launch_t2w_sft_local_datapacker.sh

set -uo pipefail

WORKDIR="/lustre/fsw/portfolios/cosmos/users/yangyangt/Cosmos-prerelease/cosmos_training"
DATASET_JSONL="/lustre/fsw/portfolios/cosmos/users/yangyangt/cosmos_opensource/sft_dataset_bridge/train/video_dataset_file.jsonl"
DCP_LOAD_PATH="/lustre/fsw/portfolios/cosmos/users/yangyangt/midtrain"
WAN_VAE_PATH="/lustre/fsw/portfolios/cosmos/users/yangyangt/cosmos_opensource/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"

OUTPUT_ROOT="/lustre/fsw/portfolios/cosmos/users/yangyangt/Cosmos-prerelease/training_output"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_FILE="$LOG_DIR/t2w_sft_8b_local_datapacker.log"

mkdir -p "$LOG_DIR"

echo ">>> $(date '+%H:%M:%S') Checking inputs..."
[[ -d "$WORKDIR" ]]        || { echo "ERROR: WORKDIR not found: $WORKDIR" >&2; exit 1; }
[[ -f "$DATASET_JSONL" ]]  || { echo "ERROR: dataset jsonl not found: $DATASET_JSONL" >&2; exit 1; }
[[ -d "$DCP_LOAD_PATH" ]]  || { echo "ERROR: DCP checkpoint dir not found: $DCP_LOAD_PATH" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]]   || { echo "ERROR: Wan VAE not found: $WAN_VAE_PATH" >&2; exit 1; }

cd "$WORKDIR"
echo ">>> $(date '+%H:%M:%S') WORKDIR:    $WORKDIR"
echo ">>> $(date '+%H:%M:%S') dataset:    $DATASET_JSONL"
echo ">>> $(date '+%H:%M:%S') checkpoint: $DCP_LOAD_PATH"
echo ">>> $(date '+%H:%M:%S') WAN VAE:    $WAN_VAE_PATH"
echo ">>> $(date '+%H:%M:%S') log:        $LOG_FILE"

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HUB_DISABLE_XET=1

# Determinism: PYTHONHASHSEED must be set before the interpreter starts.
export PYTHONHASHSEED=42

# Overrides:
#   - bucket_name/credentials cleared so the Wan2pt2VAEInterface doesn't try to
#     open credentials/gcp_*.secret (we already have a local Wan VAE).
#   - vlm_config.tokenizer.config_variant=hf routes Qwen3-VL tokenizer fetch to
#     HF Hub instead of S3/GCP.
#   - vlm_config.pretrained_weights.enabled=false skips the S3 weights overlay.
#   - joint_attn_implementation=flex for non-GB200 nodes (Blackwell here).
#   - data_parallel_shard_degree=-1 auto-detects from torchrun world size.
IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" PYTHONPATH=. \
    torchrun --nproc_per_node=4 --master_port=50014 -m scripts.train \
    --config=configs/base/config.py \
    --deterministic \
    -- \
    experiment=t2w_sft_8b_local_datapacker \
    trainer.max_iter=10 \
    trainer.logging_iter=1 \
    trainer.run_validation=false \
    job.group=debug \
    job.wandb_mode=disabled \
    upload_reproducible_setup=false \
    "checkpoint.load_path=$DCP_LOAD_PATH" \
    checkpoint=local \
    ckpt_type=dummy \
    checkpoint.load_from_object_store.enabled=false \
    checkpoint.save_to_object_store.enabled=false \
    model.config.parallelism.data_parallel_shard_degree=-1 \
    model.config.parallelism.use_torch_compile=false \
    model.config.joint_attn_implementation=flex \
    model.config.vlm_config.tokenizer.config_variant=hf \
    model.config.vlm_config.pretrained_weights.enabled=false \
    "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
    model.config.tokenizer.bucket_name="" \
    model.config.tokenizer.object_store_credential_path_pretrained="" \
    dataloader_train.data_source.num_video_frames=61 \
    "dataloader_train.data_source.jsonl_paths=[$DATASET_JSONL]" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit $EXIT_CODE
