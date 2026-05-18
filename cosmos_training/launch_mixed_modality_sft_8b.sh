#!/usr/bin/env bash
# Launch mixed_modality_sft_8b training inside the cosmos_training released tree.
#
# Inputs (already provisioned by the user):
#   - dataset (jsonl + clips): /lustre/.../cosmos_opensource/sft_dataset_bridge/
#   - Wan2.2 VAE (~2.8 GB):    cosmos_training/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth
#   - Cosmos3-Nano DCP base:   /lustre/.../midtrain/
#
# Usage (from a 4-GPU Slurm allocation, inside the yangyangt_dev container):
#   bash launch_mixed_modality_sft_8b.sh
#   # or stream it:
#   srun --overlap --jobid <JOB_ID> --container-name yangyangt_dev \
#        bash -s < /lustre/.../cosmos_training/launch_mixed_modality_sft_8b.sh
set -uo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKDIR="/lustre/fsw/portfolios/cosmos/users/yangyangt/Cosmos-prerelease/cosmos_training"
DATASET_JSONL="/lustre/fsw/portfolios/cosmos/users/yangyangt/cosmos_opensource/sft_dataset_bridge/train/video_dataset_file.jsonl"
DCP_LOAD_PATH="/lustre/fsw/portfolios/cosmos/users/yangyangt/midtrain"
WAN_VAE_PATH="/lustre/fsw/portfolios/cosmos/users/yangyangt/cosmos_opensource/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"

OUTPUT_ROOT="/lustre/fsw/portfolios/cosmos/users/yangyangt/Cosmos-prerelease/training_output"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_FILE="$LOG_DIR/mixed_modality_sft_8b.log"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Sanity-check inputs
# ---------------------------------------------------------------------------
echo ">>> $(date '+%H:%M:%S') Checking inputs..."
[[ -d "$WORKDIR" ]] || { echo "ERROR: WORKDIR not found: $WORKDIR" >&2; exit 1; }
[[ -f "$DATASET_JSONL" ]] || { echo "ERROR: dataset jsonl not found: $DATASET_JSONL" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]] || { echo "ERROR: Wan VAE not found: $WAN_VAE_PATH" >&2; exit 1; }
[[ -d "$DCP_LOAD_PATH" ]] || { echo "ERROR: DCP checkpoint dir not found: $DCP_LOAD_PATH" >&2; exit 1; }

cd "$WORKDIR"
echo ">>> $(date '+%H:%M:%S') WORKDIR:   $WORKDIR"
echo ">>> $(date '+%H:%M:%S') dataset:   $DATASET_JSONL"
echo ">>> $(date '+%H:%M:%S') WAN VAE:   $WAN_VAE_PATH"
echo ">>> $(date '+%H:%M:%S') DCP load:  $DCP_LOAD_PATH"
echo ">>> $(date '+%H:%M:%S') log:       $LOG_FILE"

# ---------------------------------------------------------------------------
# HuggingFace environment (needed for tokenizer downloads via config_variant=hf).
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HUB_DISABLE_XET=1

# Determinism: PYTHONHASHSEED must be set before the interpreter starts.
# --deterministic on scripts/train.py applies the rest (CUBLAS_WORKSPACE_CONFIG,
# torch.use_deterministic_algorithms, cudnn.deterministic, etc.).
export PYTHONHASHSEED=42

# ---------------------------------------------------------------------------
# torchrun launch
# ---------------------------------------------------------------------------
# Override notes:
#   - experiment=mixed_modality_sft_8b → registered by experiments/sft/mixed_modality_sft_8b.py
#   - jsonl_paths drops the Hydra "???" MISSING marker.
#   - model.config.tokenizer.vae_path anchors the VAE to the local file.
#   - data_parallel_shard_degree=4 matches the 4-GPU node.
#   - checkpoint.load_path is the manually-prepared DCP base.
#   - upload_reproducible_setup=false avoids the startup S3 PUT.
IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" PYTHONPATH=. \
    torchrun --nproc_per_node=4 --master_port=50011 -m scripts.train \
    --config=configs/base/config.py \
    --deterministic \
    -- \
    experiment=mixed_modality_sft_8b \
    "dataloader_train.dataloader.datasets.video.dataset.jsonl_paths=[\"$DATASET_JSONL\"]" \
    "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
    "checkpoint.load_path=$DCP_LOAD_PATH" \
    model.config.parallelism.data_parallel_shard_degree=4 \
    job.wandb_mode=disabled \
    upload_reproducible_setup=false \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit $EXIT_CODE
