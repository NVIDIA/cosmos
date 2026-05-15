#!/usr/bin/env bash
# TOML-mode equivalent of launch_t2w_sft_local_datapacker.sh. Loads structured
# overrides from toml/launch_t2w_sft_local_datapacker.toml; per-user paths +
# keys with no interface-schema mapping flow through the CLI tail.
set -uo pipefail

WORKDIR="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/cosmos_training"
DATASET_JSONL="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/workdir/cosmos_opensource/sft_dataset_bridge/train/video_dataset_file.jsonl"
WAN_VAE_PATH="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/workdir/cosmos_opensource/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"
TOML_FILE="toml/launch_t2w_sft_local_datapacker.toml"

OUTPUT_ROOT="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/training_output"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_FILE="$LOG_DIR/t2w_sft_8b_local_datapacker_toml.log"

mkdir -p "$LOG_DIR"

echo ">>> $(date '+%H:%M:%S') Checking inputs..."
[[ -d "$WORKDIR" ]]            || { echo "ERROR: WORKDIR not found: $WORKDIR" >&2; exit 1; }
[[ -f "$WORKDIR/$TOML_FILE" ]] || { echo "ERROR: TOML not found: $WORKDIR/$TOML_FILE" >&2; exit 1; }
[[ -f "$DATASET_JSONL" ]]      || { echo "ERROR: dataset jsonl not found: $DATASET_JSONL" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]]       || { echo "ERROR: Wan VAE not found: $WAN_VAE_PATH" >&2; exit 1; }

cd "$WORKDIR"
echo ">>> $(date '+%H:%M:%S') WORKDIR:    $WORKDIR"
echo ">>> $(date '+%H:%M:%S') TOML:       $TOML_FILE"
echo ">>> $(date '+%H:%M:%S') dataset:    $DATASET_JSONL"
echo ">>> $(date '+%H:%M:%S') WAN VAE:    $WAN_VAE_PATH"
echo ">>> $(date '+%H:%M:%S') log:        $LOG_FILE"

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HUB_DISABLE_XET=1

export PYTHONHASHSEED=42

IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" PYTHONPATH=. \
    torchrun --nproc_per_node=4 --master_port=50014 -m scripts.train \
    --config=configs/base/config.py \
    --toml="$TOML_FILE" \
    --deterministic \
    -- \
    checkpoint.load_from_object_store.enabled=false \
    checkpoint.save_to_object_store.enabled=false \
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
