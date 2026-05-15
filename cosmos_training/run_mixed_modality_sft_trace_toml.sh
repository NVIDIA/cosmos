#!/usr/bin/env bash
# TOML-launch sibling of run_mixed_modality_sft_trace.sh.
#
# Same job (mixed_modality_sft_8b smoke + import tracer), but the common
# Hydra overrides live in toml/mixed_modality_sft_8b.toml instead
# of being inlined on the CLI. Only the project-specific paths
# (jsonl_paths, vae_path) and the env-controlled trainer.max_iter remain
# as CLI tail overrides.
#
# Usage (inside container on a 4-GPU node):
#   srun --overlap --jobid <JOB_ID> --container-name yangyangt_dev \
#        bash /lustre/.../cosmos_training/run_mixed_modality_sft_trace_toml.sh
#
# Env vars (same as the direct-CLI script):
#   MAX_ITER (default: 2)    trainer.max_iter — keep small for trace runs
#   PORT     (default: 12399) torchrun master port
#   OUT_LOG  (default: /lustre/.../cosmos_opensource/training_output/imports_mixed_modality_sft_8b.log)
set -uo pipefail

# Self-locate: WORKDIR is the directory this script sits in (cosmos_training/).
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_JSONL="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/workdir/cosmos_opensource/sft_dataset_bridge/train/video_dataset_file.jsonl"
WAN_VAE_PATH="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/workdir/cosmos_opensource/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"
# Note: DCP_LOAD_PATH is in the TOML ([train.ckpt].load_path), not here.

PORT="${PORT:-12399}"
MAX_ITER="${MAX_ITER:-2}"
OUT_LOG="${OUT_LOG:-/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/workdir/cosmos_opensource/training_output/imports_mixed_modality_sft_8b.log}"

TRACER_DIR="/tmp/cosmos_import_tracer"
LOG_DIR="/tmp/cosmos_import_logs"

echo "=== mixed_modality_sft_8b import trace (TOML launch) ==="
echo "  WORKDIR:  $WORKDIR"
echo "  PORT:     $PORT"
echo "  MAX_ITER: $MAX_ITER"
echo "  OUT_LOG:  $OUT_LOG"
echo "========================================================"

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

# 2. Sanity-check inputs.
[[ -d "$WORKDIR" ]]      || { echo "ERROR: WORKDIR not found: $WORKDIR" >&2; exit 1; }
[[ -f "$DATASET_JSONL" ]] || { echo "ERROR: dataset jsonl not found: $DATASET_JSONL" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]]  || { echo "ERROR: Wan VAE not found: $WAN_VAE_PATH" >&2; exit 1; }
[[ -f "$WORKDIR/toml/mixed_modality_sft_8b.toml" ]] \
    || { echo "ERROR: TOML not found: $WORKDIR/toml/mixed_modality_sft_8b.toml" >&2; exit 1; }

cd "$WORKDIR"

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HUB_DISABLE_XET=1

echo ">>> $(date '+%H:%M:%S') Launching torchrun (port $PORT, max_iter $MAX_ITER) ..."

# 3. Run the smoke with the tracer on PYTHONPATH.
# PYTHONPATH order: tracer dir FIRST so its sitecustomize wins.
#
# What goes where:
#   --toml  →  toml/mixed_modality_sft_8b.toml carries
#              job.wandb_mode, trainer.{max_iter, logging_iter, run_validation},
#              upload_reproducible_setup, experiment=mixed_modality_sft_8b,
#              checkpoint.load_path, model.config.parallelism.data_parallel_shard_degree
#   CLI tail →  the two project-specific deep paths that the interface schema
#               doesn't cover, plus the env-controlled trainer.max_iter
#               (CLI overrides win over TOML, so $MAX_ITER replaces the
#               TOML default of 2).
COSMOS_IMPORT_TRACE=1 \
COSMOS_IMPORT_LOG_DIR="$LOG_DIR" \
PYTHONPATH="$TRACER_DIR:$WORKDIR" \
IMAGINAIRE_OUTPUT_ROOT=/tmp/cosmos-trace-output \
    torchrun --nproc_per_node=4 --master_port=$PORT -m scripts.train \
    --config=configs/base/config.py \
    --toml=toml/mixed_modality_sft_8b.toml \
    -- \
    "dataloader_train.dataloader.datasets.video.dataset.jsonl_paths=[\"$DATASET_JSONL\"]" \
    "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
    trainer.max_iter=$MAX_ITER \
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
