#!/usr/bin/env bash
# Launch action_policy_sft_8b_datapacker smoke (8B LIBERO policy, 4-GPU FSDP, 10 iters)
# inside the released cosmos_training tree.
#
# Mirrors /lustre/fsw/portfolios/cosmos/users/yangyangt/imaginaire4/action_run.sh
# (the i4 reference launch), but:
#   - WORKDIR is the released tree
#   - --config is the released vfm config root (configs/base/config.py)
#   - LIBERO root points at the freshly downloaded nvidia/LIBERO_LeRobot_v3 snapshot
#
# Usage (from a Slurm allocation with --container-name yangyangt_dev):
#   srun --overlap --jobid <JOB_ID> --container-name yangyangt_dev \
#        bash -s < /lustre/.../cosmos_training/launch_action_libero.sh

set -uo pipefail

WORKDIR="/lustre/fsw/portfolios/cosmos/users/yangyangt/Cosmos-prerelease/cosmos_training"
LIBERO_BASE="/lustre/fsw/portfolios/cosmos/users/yangyangt/cosmos_opensource/LIBERO_LeRobot_v3"
WAN_VAE_PATH="/lustre/fsw/portfolios/cosmos/users/yangyangt/cosmos_opensource/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"

OUTPUT_ROOT="/lustre/fsw/portfolios/cosmos/users/yangyangt/Cosmos-prerelease/training_output"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_FILE="$LOG_DIR/action_policy_sft_8b_datapacker.log"

# The LIBERO loader's _resolve_libero_roots() expects versioned subdir names
# (libero_10_no_noops_1.0.0_lerobot_aligned_20260124, etc.). The HF snapshot
# uses short names. Create symlinks so the resolver finds them.
declare -A LIBERO_SYMLINKS=(
    ["libero_10_no_noops_1.0.0_lerobot_aligned_20260124"]="libero_10"
    ["libero_90_no_noops_lerobot_shuffled_20260124"]="libero_90"
    ["libero_object_no_noops_1.0.0_lerobot_aligned_20260124"]="libero_object"
    ["libero_spatial_no_noops_1.0.0_lerobot_20260124"]="libero_spatial"
    ["libero_goal_no_noops_1.0.0_lerobot_20260124"]="libero_goal"
)
for versioned in "${!LIBERO_SYMLINKS[@]}"; do
    short="${LIBERO_SYMLINKS[$versioned]}"
    target="$LIBERO_BASE/$versioned"
    if [ ! -e "$target" ]; then
        ln -s "$LIBERO_BASE/$short" "$target"
        echo ">>> Symlinked $versioned -> $short"
    fi
done

export LIBERO_LOCAL_DATA_ROOT="$LIBERO_BASE"

# Pretrained checkpoint to warm-start from. Override via:
#   CHECKPOINT_PATH=/your/path bash launch_action_libero.sh
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/maoshengl/inter_ckpts/internal}"

mkdir -p "$LOG_DIR"

echo ">>> $(date '+%H:%M:%S') Checking inputs..."
[[ -d "$WORKDIR" ]]      || { echo "ERROR: WORKDIR not found: $WORKDIR" >&2; exit 1; }
[[ -d "$LIBERO_BASE" ]]  || { echo "ERROR: LIBERO root not found: $LIBERO_BASE" >&2; exit 1; }
[[ -d "$CHECKPOINT_PATH" ]] || { echo "ERROR: checkpoint not found: $CHECKPOINT_PATH" >&2; exit 1; }

cd "$WORKDIR"
echo ">>> $(date '+%H:%M:%S') WORKDIR:    $WORKDIR"
echo ">>> $(date '+%H:%M:%S') LIBERO:     $LIBERO_BASE"
echo ">>> $(date '+%H:%M:%S') checkpoint: $CHECKPOINT_PATH"
echo ">>> $(date '+%H:%M:%S') log:        $LOG_FILE"

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HUB_DISABLE_XET=1

# Determinism: PYTHONHASHSEED must be set before the interpreter starts.
export PYTHONHASHSEED=42

LOGURU_LEVEL=DEBUG IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" PYTHONPATH=. \
    torchrun --nproc_per_node=4 --master_port=50013 -m scripts.train \
    --config=configs/base/config.py \
    --deterministic \
    -- \
    experiment=action_policy_sft_8b_datapacker \
    "checkpoint.load_path=$CHECKPOINT_PATH" \
    "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
    model.config.tokenizer.bucket_name="" \
    model.config.tokenizer.object_store_credential_path_pretrained="" \
    trainer.max_iter=10 \
    trainer.logging_iter=1 \
    trainer.run_validation=false \
    job.group=debug \
    job.wandb_mode=disabled \
    upload_reproducible_setup=false \
    checkpoint=local \
    ckpt_type=dummy \
    checkpoint.load_from_object_store.enabled=false \
    checkpoint.save_to_object_store.enabled=false \
    model.config.parallelism.data_parallel_shard_degree=-1 \
    model.config.parallelism.use_torch_compile=false \
    model.config.vlm_config.tokenizer.config_variant=hf \
    model.config.vlm_config.pretrained_weights.enabled=false \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit $EXIT_CODE
