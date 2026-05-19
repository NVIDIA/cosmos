#!/usr/bin/env bash
# TOML-launch sibling of run_mixed_modality_sft_trace.sh.
#
# Same job (mixed_modality_sft_8b smoke + import tracer), but all Hydra
# overrides — including trainer.max_iter — live in
# toml/mixed_modality_sft_8b.toml. No CLI tail overrides.
#
# Usage (inside container on a 4-GPU node):
#   srun --overlap --jobid <JOB_ID> --container-name yangyangt_dev \
#        bash /lustre/.../cosmos_training/run_mixed_modality_sft_trace_toml.sh
#
# Env vars:
#   PORT     (default: 12399) torchrun master port
#   OUT_LOG  (default: /lustre/.../cosmos_opensource/training_output/imports_mixed_modality_sft_8b.log)
set -uo pipefail

# Self-locate: WORKDIR is the directory this script sits in (cosmos_training/).
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Note: DCP_LOAD_PATH is in the TOML ([train.ckpt].load_path), not here.

PORT="${PORT:-12399}"
OUT_LOG="${OUT_LOG:-/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/workdir/cosmos_opensource/training_output/imports_mixed_modality_sft_8b.log}"

TRACER_DIR="/tmp/cosmos_import_tracer"
LOG_DIR="/tmp/cosmos_import_logs"

echo "=== mixed_modality_sft_8b import trace (TOML launch) ==="
echo "  WORKDIR:  $WORKDIR"
echo "  PORT:     $PORT"
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
[[ -f "$WORKDIR/toml/mixed_modality_sft_8b.toml" ]] \
    || { echo "ERROR: TOML not found: $WORKDIR/toml/mixed_modality_sft_8b.toml" >&2; exit 1; }

cd "$WORKDIR"

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HUB_DISABLE_XET=1

echo ">>> $(date '+%H:%M:%S') Launching torchrun (port $PORT) ..."

# 3. Run the smoke with the tracer on PYTHONPATH.
# PYTHONPATH order: tracer dir FIRST so its sitecustomize wins.
#
# All Hydra overrides — experiment, ckpt, dp_shard, trainer.{max_iter,
# logging_iter, run_validation}, dataset paths — live in
# toml/mixed_modality_sft_8b.toml. No CLI tail.
COSMOS_IMPORT_TRACE=1 \
COSMOS_IMPORT_LOG_DIR="$LOG_DIR" \
PYTHONPATH="$TRACER_DIR:$WORKDIR" \
IMAGINAIRE_OUTPUT_ROOT=/tmp/cosmos-trace-output \
    torchrun --nproc_per_node=4 --master_port=$PORT -m scripts.train \
    --toml=toml/mixed_modality_sft_8b.toml \
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
