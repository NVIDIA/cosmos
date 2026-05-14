#!/usr/bin/env bash
# Runtime import-trace for the mixed_modality_sft_8b experiment in the
# cosmos_training released tree.
#
# Adapted from /lustre/fsw/portfolios/cosmos/users/yangyangt/run_full_smoke_trace.sh
# (which traced the i4 source tree). Key differences here:
#   - Workdir is cosmos_training/ (released cosmos.* / configs.* / experiments.* / scripts namespaces)
#   - Uses launch_mixed_modality_sft_8b.sh's full set of overrides (jsonl, DCP, VAE, etc.)
#   - "In-repo" prefixes are cosmos / configs / experiments / scripts
#
# Usage (inside container on a 4-GPU node):
#   srun --overlap --jobid <JOB_ID> --container-name yangyangt_dev \
#        bash /lustre/.../cosmos_training/run_mixed_modality_sft_trace.sh
#
# Env vars:
#   MAX_ITER (default: 2)    trainer.max_iter — keep small for trace runs
#   PORT     (default: 12399) torchrun master port
#   OUT_LOG  (default: /lustre/.../cosmos_opensource/training_output/imports_mixed_modality_sft_8b.log)
set -uo pipefail

WORKDIR="/lustre/fsw/portfolios/cosmos/users/yangyangt/cosmos_opensource/cosmos_training"
DATASET_JSONL="/lustre/fsw/portfolios/cosmos/users/yangyangt/cosmos_opensource/sft_dataset_bridge/train/video_dataset_file.jsonl"
DCP_LOAD_PATH="/lustre/fsw/portfolios/cosmos/users/yangyangt/midtrain"
WAN_VAE_PATH="${WORKDIR}/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"

PORT="${PORT:-12399}"
MAX_ITER="${MAX_ITER:-2}"
OUT_LOG="${OUT_LOG:-/lustre/fsw/portfolios/cosmos/users/yangyangt/cosmos_opensource/training_output/imports_mixed_modality_sft_8b.log}"

TRACER_DIR="/tmp/cosmos_import_tracer"
LOG_DIR="/tmp/cosmos_import_logs"

echo "=== mixed_modality_sft_8b import trace ==="
echo "  WORKDIR:  $WORKDIR"
echo "  PORT:     $PORT"
echo "  MAX_ITER: $MAX_ITER"
echo "  OUT_LOG:  $OUT_LOG"
echo "=========================================="

# 1. Build the tracer payload (sitecustomize.py).
rm -rf "$TRACER_DIR" "$LOG_DIR"
mkdir -p "$TRACER_DIR" "$LOG_DIR"

cat > "$TRACER_DIR/sitecustomize.py" <<'PYEOF'
"""Auto-loaded by every Python process when COSMOS_IMPORT_TRACE=1.
Logs the name of every imported module to /tmp/cosmos_import_logs/import.<pid>.log.
"""
import os
import sys

if os.environ.get("COSMOS_IMPORT_TRACE") == "1":
    log_dir = os.environ.get("COSMOS_IMPORT_LOG_DIR", "/tmp/cosmos_import_logs")
    log_path = os.path.join(log_dir, f"import.{os.getpid()}.log")
    try:
        _f = open(log_path, "a", buffering=1)
    except OSError:
        _f = None

    def _hook(event, args):
        if _f is None:
            return
        if event == "import":
            try:
                _f.write(args[0] + "\n")
            except (BrokenPipeError, OSError):
                pass

    sys.addaudithook(_hook)
PYEOF

# 2. Sanity-check inputs (same as launch_mixed_modality_sft_8b.sh).
[[ -d "$WORKDIR" ]]      || { echo "ERROR: WORKDIR not found: $WORKDIR" >&2; exit 1; }
[[ -f "$DATASET_JSONL" ]] || { echo "ERROR: dataset jsonl not found: $DATASET_JSONL" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]]  || { echo "ERROR: Wan VAE not found: $WAN_VAE_PATH" >&2; exit 1; }
[[ -d "$DCP_LOAD_PATH" ]] || { echo "ERROR: DCP checkpoint dir not found: $DCP_LOAD_PATH" >&2; exit 1; }

cd "$WORKDIR"

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HUB_DISABLE_XET=1

echo ">>> $(date '+%H:%M:%S') Launching torchrun (port $PORT, max_iter $MAX_ITER) ..."

# 3. Run the smoke with the tracer on PYTHONPATH.
# PYTHONPATH order: tracer dir FIRST so its sitecustomize wins.
COSMOS_IMPORT_TRACE=1 \
COSMOS_IMPORT_LOG_DIR="$LOG_DIR" \
PYTHONPATH="$TRACER_DIR:$WORKDIR" \
IMAGINAIRE_OUTPUT_ROOT=/tmp/cosmos-trace-output \
    torchrun --nproc_per_node=4 --master_port=$PORT -m scripts.train \
    --config=configs/base/config.py \
    -- \
    experiment=mixed_modality_sft_8b \
    "dataloader_train.dataloader.datasets.video.dataset.jsonl_paths=[\"$DATASET_JSONL\"]" \
    "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
    "checkpoint.load_path=$DCP_LOAD_PATH" \
    model.config.parallelism.data_parallel_shard_degree=4 \
    trainer.max_iter=$MAX_ITER \
    trainer.logging_iter=1 \
    trainer.run_validation=false \
    job.wandb_mode=disabled \
    upload_reproducible_setup=false \
    2>&1 | tail -40

SMOKE_EXIT=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') smoke exit: $SMOKE_EXIT"

# 4. Summarize and merge.
NUM_PIDS=$(ls "$LOG_DIR"/import.*.log 2>/dev/null | wc -l)
echo
echo "=== per-PID log files: $NUM_PIDS"
ls -la "$LOG_DIR" 2>/dev/null | tail -10

if [[ $NUM_PIDS -eq 0 ]]; then
    echo "ERROR: no import logs were captured" >&2
    exit 2
fi

TOTAL=$(cat "$LOG_DIR"/*.log | wc -l)
UNIQUE=$(cat "$LOG_DIR"/*.log | sort -u | wc -l)
# In the released tree, in-repo namespaces are cosmos/configs/experiments/scripts.
INREPO=$(cat "$LOG_DIR"/*.log | sort -u \
    | grep -E '^(cosmos|configs|experiments|scripts)(\.|$)' | wc -l)

echo
echo "=== totals:"
echo "  total import events:    $TOTAL"
echo "  unique modules:         $UNIQUE"
echo "  in-repo (closure):      $INREPO"

# 5. Persist the merged dedup'd log to lustre.
mkdir -p "$(dirname "$OUT_LOG")"
cat "$LOG_DIR"/*.log | sort -u > "$OUT_LOG"
echo
echo "=== wrote merged log:"
wc -l "$OUT_LOG"

# 6. Also emit a separate in-repo-only file alongside.
INREPO_LOG="${OUT_LOG%.log}.inrepo.log"
grep -E '^(cosmos|configs|experiments|scripts)(\.|$)' "$OUT_LOG" > "$INREPO_LOG" || true
echo "=== wrote in-repo-only log:"
wc -l "$INREPO_LOG"
echo
echo ">>> $(date '+%H:%M:%S') Done."
