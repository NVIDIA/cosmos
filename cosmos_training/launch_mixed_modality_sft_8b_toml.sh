#!/usr/bin/env bash
# TOML-mode equivalent of launch_mixed_modality_sft_8b.sh. Loads the structured
# overrides (experiment, wandb_mode, ckpt.load_path, dp_shard, etc.) from
# toml/launch_mixed_modality_sft_8b.toml via scripts/train.py --toml=...
# and passes only the per-user data paths as the CLI tail (no interface-schema
# mapping in scripts/interface_toml.py for jsonl_paths / vae_path).
#
# Usage (4-GPU Slurm allocation, inside the yangyangt_dev container):
#   bash launch_mixed_modality_sft_8b_toml.sh
set -uo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKDIR="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/cosmos_training"
DATASET_JSONL="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/workdir/cosmos_opensource/sft_dataset_bridge/train/video_dataset_file.jsonl"
WAN_VAE_PATH="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/workdir/cosmos_opensource/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"
TOML_FILE="toml/launch_mixed_modality_sft_8b.toml"

OUTPUT_ROOT="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/training_output"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_FILE="$LOG_DIR/mixed_modality_sft_8b_toml.log"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Sanity-check inputs
# ---------------------------------------------------------------------------
echo ">>> $(date '+%H:%M:%S') Checking inputs..."
[[ -d "$WORKDIR" ]] || { echo "ERROR: WORKDIR not found: $WORKDIR" >&2; exit 1; }
[[ -f "$WORKDIR/$TOML_FILE" ]] || { echo "ERROR: TOML not found: $WORKDIR/$TOML_FILE" >&2; exit 1; }
[[ -f "$DATASET_JSONL" ]] || { echo "ERROR: dataset jsonl not found: $DATASET_JSONL" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]] || { echo "ERROR: Wan VAE not found: $WAN_VAE_PATH" >&2; exit 1; }

cd "$WORKDIR"
echo ">>> $(date '+%H:%M:%S') WORKDIR:   $WORKDIR"
echo ">>> $(date '+%H:%M:%S') TOML:      $TOML_FILE"
echo ">>> $(date '+%H:%M:%S') dataset:   $DATASET_JSONL"
echo ">>> $(date '+%H:%M:%S') WAN VAE:   $WAN_VAE_PATH"
echo ">>> $(date '+%H:%M:%S') log:       $LOG_FILE"

# ---------------------------------------------------------------------------
# HuggingFace environment (needed for tokenizer downloads via config_variant=hf).
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HUB_DISABLE_XET=1

# Determinism: PYTHONHASHSEED must be set before the interpreter starts.
export PYTHONHASHSEED=42

# ---------------------------------------------------------------------------
# torchrun launch
# ---------------------------------------------------------------------------
IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" PYTHONPATH=. \
    torchrun --nproc_per_node=4 --master_port=50011 -m scripts.train \
    --config=configs/base/config.py \
    --toml="$TOML_FILE" \
    --deterministic \
    -- \
    "dataloader_train.dataloader.datasets.video.dataset.jsonl_paths=[\"$DATASET_JSONL\"]" \
    "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit $EXIT_CODE
