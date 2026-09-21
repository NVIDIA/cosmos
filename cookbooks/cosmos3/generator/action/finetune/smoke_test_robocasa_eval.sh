#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Smoke test for the RoboCasa closed-loop evaluation handshake.
#
# Runs the exact two-process pair the README documents -- policy server in the
# cosmos-framework venv, evaluator in the robosuite/robocasa venv -- on a single task for a
# couple of short rollouts, and checks that the width and state contracts the two sides agree
# on are the ones this recipe trains with. The evaluator sends a 15-D state token; a server
# that resolved a different width rejects every request, so the mismatch has to be caught
# here rather than an hour into a real evaluation.
#
#   RUN_DIR=outputs/train/<project>/<group>/<name> \
#   FRAMEWORK_ROOT=/path/to/cosmos-framework \
#   SIM_PYTHON=/path/to/robocasa-venv/bin/python \
#   DATASET_DIR=/path/to/robocasa/datasets/v1.0/target/atomic/<task>/<date>/lerobot \
#       bash smoke_test_robocasa_eval.sh
#
# Run it from this folder with the cosmos-framework venv active.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

: "${RUN_DIR:?set RUN_DIR to a training run directory (it must contain checkpoints/ and config.yaml)}"
: "${DATASET_DIR:?set DATASET_DIR to an ORIGINAL RoboCasa export: .../<task>/<date>/lerobot}"
: "${SIM_PYTHON:?set SIM_PYTHON to the python of the robosuite/robocasa venv}"
# cosmos-framework is not installed in the simulator venv; the evaluator imports only
# pure-python modules from it, so a checkout on PYTHONPATH is enough.
: "${FRAMEWORK_ROOT:?set FRAMEWORK_ROOT to a cosmos-framework checkout}"

# The DCP checkpoint is served directly; evaluation needs no safetensors export.
if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
    LATEST="$RUN_DIR/checkpoints/$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")"
    CHECKPOINT_PATH="$LATEST/model"
fi
CONFIG_FILE="${CONFIG_FILE:-$RUN_DIR/config.yaml}"
PORT="${PORT:-8912}"
NUM_EPISODES="${NUM_EPISODES:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/smoke_eval}"
# The width this recipe trains with (use_base_action=True, base_encoding="raw").
EXPECTED_ACTION_DIM="${EXPECTED_ACTION_DIM:-15}"

for f in "$CHECKPOINT_PATH" "$CONFIG_FILE" "$DATASET_DIR" "$SIM_PYTHON" "$FRAMEWORK_ROOT"; do
    [[ -e "$f" ]] || { echo "ERROR: does not exist: $f" >&2; exit 1; }
done

mkdir -p "$OUTPUT_DIR"
SERVER_LOG="$OUTPUT_DIR/server.log"

echo "=== 1/3 starting policy server on port $PORT (log: $SERVER_LOG)"
# No --raw-action-dim: the point of the test is that the server resolves the width itself.
python -m cosmos_framework.scripts.action_policy_server_robocasa \
    --checkpoint-path "$CHECKPOINT_PATH" --config-file "$CONFIG_FILE" \
    --port "$PORT" --num-steps 10 --guidance 3.0 --fps 20 \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "=== 2/3 waiting for /info and checking the contract"
INFO=""
for _ in $(seq 1 120); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: the policy server exited during startup; tail of $SERVER_LOG:" >&2
        tail -30 "$SERVER_LOG" >&2
        exit 1
    fi
    if INFO="$(curl -sf -m 5 "http://127.0.0.1:$PORT/info")"; then
        break
    fi
    INFO=""
    sleep 5
done
[[ -n "$INFO" ]] || { echo "ERROR: /info never came up; tail of $SERVER_LOG:" >&2; tail -30 "$SERVER_LOG" >&2; exit 1; }

INFO="$INFO" EXPECTED_ACTION_DIM="$EXPECTED_ACTION_DIM" python - <<'PY'
import json, os, sys

info = json.loads(os.environ["INFO"])
expected = int(os.environ["EXPECTED_ACTION_DIM"])
print("[smoke] /info:", json.dumps(info, indent=2))
dim = info.get("raw_action_dim")
source = info.get("raw_action_dim_source")
if dim != expected:
    sys.exit(
        f"FAIL: server resolved raw_action_dim={dim} (from {source}), but this recipe is "
        f"{expected}-D. The evaluator would send a {expected}-D state token and every /predict "
        f"would be rejected."
    )
if not info.get("requires_state"):
    sys.exit(
        "FAIL: the server does not require a state token, so this checkpoint was not trained "
        "with use_state=True. Serving it under --use-state silently drops a conditioning signal."
    )
print(f"[smoke] OK: raw_action_dim={dim} (from {source}), requires_state=True")
PY

echo "=== 3/3 running $NUM_EPISODES rollouts through the evaluator"
# Same flags as the README; only the episode count is cut down. The rollout horizon comes
# from robocasa's task registry, so it is not shortened here.
MUJOCO_GL="${MUJOCO_GL:-egl}" PYTHONPATH="$FRAMEWORK_ROOT" \
"$SIM_PYTHON" -m cosmos_framework.simulation.robocasa.closed_loop_eval \
    --server-url "http://127.0.0.1:$PORT" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR" --num-test-episodes "$NUM_EPISODES" \
    --action-horizon 32 --camera-set left_wrist \
    --use-state --use-base-action --base-encoding raw \
    --seed 0 --image-size 256 --cam-size 256

# The rollouts are far too short to succeed; completing them without a contract error is the
# result being checked.
echo "=== PASS: the server/evaluator handshake works end to end; results in $OUTPUT_DIR"
