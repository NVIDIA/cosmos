#!/usr/bin/env bash
# TOML-mode equivalent of launch_vlm_llava_ov.sh. Loads structured overrides
# from toml/launch_vlm_llava_ov.toml (variant=vfm-vlm); per-user paths +
# keys with no interface-schema mapping flow through the CLI tail.
set -uo pipefail

WORKDIR="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/cosmos_training"
MODEL_PATH="/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/maoshengl/models/Siglip2-Qwen3-1.7B-BF16-Alignment"
TOML_FILE="toml/launch_vlm_llava_ov.toml"

OUTPUT_ROOT="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/training_output"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_FILE="$LOG_DIR/pre_exp012_llava_ov_datapacker_toml.log"

mkdir -p "$LOG_DIR"

echo ">>> $(date '+%H:%M:%S') Checking inputs..."
[[ -d "$WORKDIR" ]]            || { echo "ERROR: WORKDIR not found: $WORKDIR" >&2; exit 1; }
[[ -f "$WORKDIR/$TOML_FILE" ]] || { echo "ERROR: TOML not found: $WORKDIR/$TOML_FILE" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]]         || { echo "ERROR: model dir not found: $MODEL_PATH" >&2; exit 1; }

cd "$WORKDIR"
echo ">>> $(date '+%H:%M:%S') WORKDIR:   $WORKDIR"
echo ">>> $(date '+%H:%M:%S') TOML:      $TOML_FILE"
echo ">>> $(date '+%H:%M:%S') model:     $MODEL_PATH"
echo ">>> $(date '+%H:%M:%S') log:       $LOG_FILE"

# HuggingFace env: streamed dataset (no local download) + tokenizer/processor fetch.
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HUB_DISABLE_XET=1

export PYTHONHASHSEED=42

IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" PYTHONPATH=. \
    torchrun --nproc_per_node=4 --master_port=50012 -m scripts.train \
    --config=configs/base/vlm/config.py \
    --toml="$TOML_FILE" \
    --deterministic \
    -- \
    "model.config.policy.backbone.model_name=$MODEL_PATH" \
    checkpoint.load_from_object_store.enabled=false \
    checkpoint.save_to_object_store.enabled=false \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit $EXIT_CODE
