#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# run_paibench_c.sh - self-contained PAI-Bench-C generation + evaluation script.
#
# Mirrors every step of run_with_cosmos_framework.ipynb as a plain bash script.
# No assumptions about the calling environment beyond bash, git, and CUDA drivers.
# uv, git-lfs, and the physical-ai-bench evaluation library are all set up
# automatically on first run.
#
# --- Quick start --------------------------------------------------------------
#   # Smoke-test: 1 task, edge, Cosmos3-Nano
#   bash run_paibench_c.sh
#
#   # Demo with 4 tasks (default)
#   bash run_paibench_c.sh
#
#   # Full 600-task run, all modalities, Cosmos3-Super
#   PAIBENCH_C_NUM_SAMPLES=600 \
#   PAIBENCH_C_MODALITIES="edge blur depth seg" \
#   PAIBENCH_C_CHECKPOINT=Cosmos3-Super \
#   bash run_paibench_c.sh
#
#   # Skip dependency install on subsequent runs
#   SKIP_INSTALL=1 bash run_paibench_c.sh
#
#   # Generation only - skip evaluation
#   SKIP_EVAL=1 bash run_paibench_c.sh
#
# --- Required environment -----------------------------------------------------
#   HF_TOKEN            HuggingFace token for gated model/dataset access.
#                       Alternative: run `uvx hf auth login` once after setup.
#
# --- All overridable variables ------------------------------------------------
#   PAIBENCH_C_CHECKPOINT       Cosmos checkpoint        (default: Cosmos3-Nano)
#   PAIBENCH_C_MODALITIES       Space-separated list     (default: edge)
#   PAIBENCH_C_NUM_SAMPLES      Number of tasks to run   (default: 4)
#   COSMOS3_NUM_GPUS            GPUs for generation      (default: 4)
#   CUDA_VISIBLE_DEVICES        GPU IDs                  (default: 0,1,2,3)
#   COSMOS3_REPO                Path to cosmos-framework checkout
#   COSMOS3_GIT_URL             Git URL for cosmos-framework
#   COSMOS3_CACHE_ROOT          Root for all cache dirs  (default: .cache/ next to script)
#   UV_CACHE_DIR                uv package cache         (default: COSMOS3_CACHE_ROOT/uv)
#   HF_HOME                     HuggingFace cache        (default: COSMOS3_CACHE_ROOT/huggingface)
#   TRITON_CACHE_DIR            Triton JIT cache         (default: COSMOS3_CACHE_ROOT/triton)
#   COSMOS3_UV_ENV              cosmos-framework venv    (default: TMPDIR/paibench-<uid>/cosmos3-venv)
#   PAIBENCH_UV_ENV             physical-ai-bench venv   (default: TMPDIR/paibench-<uid>/eval-venv)
#   COSMOS3_VENV_ROOT           parent dir for cosmos3 venv  (overrides default TMPDIR path)
#   PAIBENCH_VENV_ROOT          parent dir for eval venv     (overrides default TMPDIR path)
#   TMPDIR                      scratch dir for venvs    (default: /tmp; set to local disk path
#                               if /tmp is RAM-backed tmpfs and too small for large venvs)
#   PAIBENCH_C_DATASET_ROOT     HF dataset clone path    (default: next to script)
#   PAIBENCH_C_OUTPUT_ROOT      Generation + metrics     (default: outputs/ next to script)
#   PAIBENCH_C_HF_URL           HF dataset URL
#   PAIBENCH_EVAL_ROOT          physical-ai-bench/conditional_generation path
#                               (default: physical-ai-bench/ next to script; auto-cloned)
#   PAIBENCH_EVAL_GIT_URL       Eval library git URL
#                               (default: https://github.com/trungtpham/pai-bench-c-cosmos3)
#   PAIBENCH_EVAL_GIT_BRANCH    branch to clone (default: main)
#   SKIP_INSTALL                Set to 1 to force-skip uv sync (use existing venvs as-is)
#   FORCE_INSTALL               Set to 1 to wipe + rebuild both venvs from scratch
#   UV_HTTP_TIMEOUT             uv download timeout in seconds (default: 300)
#                               Increase to 600 on slow networks (torch wheel is ~2 GB)
#   SKIP_GEN                    Set to 1 to skip generation (evaluate existing videos)
#   SKIP_EVAL                   Set to 1 to skip evaluation step
# =============================================================================
set -euo pipefail

# -- helpers -------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# Integer min - pure bash
int_min() { (( $1 < $2 )) && echo "$1" || echo "$2"; }

# -- locate the cosmos repo root (contains README.md + cookbooks/) -------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COSMOS_ROOT="$SCRIPT_DIR"
while [[ "$COSMOS_ROOT" != "/" ]]; do
    [[ -f "$COSMOS_ROOT/README.md" && -d "$COSMOS_ROOT/cookbooks" ]] && break
    COSMOS_ROOT="$(dirname "$COSMOS_ROOT")"
done
[[ -f "$COSMOS_ROOT/README.md" && -d "$COSMOS_ROOT/cookbooks" ]] \
    || die "Could not locate the cosmos repo root from $SCRIPT_DIR"
PAIBENCH_C_ROOT="$SCRIPT_DIR"
log "COSMOS_ROOT=$COSMOS_ROOT"

# -- defaults - all relative to the script directory so the script is portable -
: "${PAIBENCH_C_CHECKPOINT:=Cosmos3-Nano}"
: "${PAIBENCH_C_MODALITIES:=edge}"
: "${SKIP_GEN:=0}"
: "${SKIP_EVAL:=0}"
: "${SKIP_INSTALL:=0}"
# Default 4 tasks for a quick smoke-test; set to 600 for the full benchmark.
# SKIP_GEN=1 defaults to 600 since the videos are already generated.
if [[ "${SKIP_GEN}" == "1" ]]; then
    : "${PAIBENCH_C_NUM_SAMPLES:=600}"
else
    : "${PAIBENCH_C_NUM_SAMPLES:=4}"
fi
: "${COSMOS3_NUM_GPUS:=4}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${COSMOS3_GIT_URL:=https://github.com/NVIDIA/cosmos-framework.git}"

# Locate cosmos-framework inside the repo (packages/cosmos3 or packages/cosmos-framework)
if [[ -z "${COSMOS3_REPO:-}" ]]; then
    for _cand in "$COSMOS_ROOT/packages/cosmos3" "$COSMOS_ROOT/packages/cosmos-framework"; do
        if [[ -f "$_cand/pyproject.toml" && -d "$_cand/cosmos_framework" ]]; then
            COSMOS3_REPO="$_cand"; break
        fi
    done
    : "${COSMOS3_REPO:=$COSMOS_ROOT/packages/cosmos-framework}"
fi

# Cache dirs - all under PAIBENCH_C_ROOT/.cache to avoid home-dir quota issues.
_cache_root="${COSMOS3_CACHE_ROOT:-$PAIBENCH_C_ROOT/.cache}"
: "${UV_CACHE_DIR:=$_cache_root/uv}"
: "${HF_HOME:=$_cache_root/huggingface}"
: "${TRITON_CACHE_DIR:=$_cache_root/triton}"
export UV_CACHE_DIR HF_HOME TRITON_CACHE_DIR
export TORCH_HOME="${TRITON_CACHE_DIR%/triton}"

: "${PAIBENCH_C_HF_URL:=https://huggingface.co/datasets/shi-labs/physical-ai-bench-conditional-generation}"
: "${PAIBENCH_C_DATASET_ROOT:=$PAIBENCH_C_ROOT/physical-ai-bench-conditional-generation}"
: "${PAIBENCH_C_OUTPUT_ROOT:=$PAIBENCH_C_ROOT/outputs}"
: "${PAIBENCH_EVAL_ROOT:=$PAIBENCH_C_ROOT/physical-ai-bench/conditional_generation}"

# physical-ai-bench with depth SI-RMSE + DOVER fixes from
# https://github.com/SHI-Labs/physical-ai-bench/pull/8
# hosted publicly at https://github.com/trungtpham/pai-bench-c-cosmos3
# until PR #8 merges into SHI-Labs:main.
: "${PAIBENCH_EVAL_GIT_URL:=https://github.com/trungtpham/pai-bench-c-cosmos3.git}"
: "${PAIBENCH_EVAL_GIT_BRANCH:=main}"

# Venvs default to a node-local temp directory so that rm -rf during reinstall never
# hits Lustre "Stale file handle" errors (files written on a different node).
# Resolution order:
#   1. UV_PROJECT_ENVIRONMENT / COSMOS3_UV_ENV / PAIBENCH_UV_ENV  (explicit overrides)
#   2. COSMOS3_VENV_ROOT / PAIBENCH_VENV_ROOT                     (parent dir override)
#   3. ${TMPDIR}/paibench-<uid>-{cosmos3,eval}-venv               (auto: local scratch)
#      TMPDIR is the standard POSIX scratch variable; set it to a large local disk
#      path (e.g. /var/tmp or /scratch) if /tmp is RAM-backed and too small.
_uid="$(id -u)"
_default_venv_root="${TMPDIR:-/tmp}/paibench-${_uid}"
: "${COSMOS3_VENV_ROOT:=$_default_venv_root}"
: "${PAIBENCH_VENV_ROOT:=$_default_venv_root}"
: "${COSMOS3_UV_ENV:=$COSMOS3_VENV_ROOT/cosmos3-venv}"
: "${PAIBENCH_UV_ENV:=$PAIBENCH_VENV_ROOT/eval-venv}"
# Standard uv env var takes precedence over everything else.
[[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]] && COSMOS3_UV_ENV="$UV_PROJECT_ENVIRONMENT"
COSMOS3_UV_GROUP="${COSMOS3_UV_GROUP:-cu130-train}"

mkdir -p "$PAIBENCH_C_OUTPUT_ROOT" "$_cache_root"

log "=== Configuration ==="
log "  CHECKPOINT          = $PAIBENCH_C_CHECKPOINT"
log "  MODALITIES          = $PAIBENCH_C_MODALITIES"
log "  NUM_SAMPLES         = $PAIBENCH_C_NUM_SAMPLES"
log "  COSMOS3_REPO        = $COSMOS3_REPO"
log "  COSMOS3_UV_ENV      = $COSMOS3_UV_ENV"
log "  PAIBENCH_UV_ENV     = $PAIBENCH_UV_ENV"
log "  DATASET_ROOT        = $PAIBENCH_C_DATASET_ROOT"
log "  OUTPUT_ROOT         = $PAIBENCH_C_OUTPUT_ROOT"
log "  PAIBENCH_EVAL_ROOT  = $PAIBENCH_EVAL_ROOT"
log "  UV_CACHE_DIR        = $UV_CACHE_DIR"
log "  HF_HOME             = $HF_HOME"
log "  CUDA_VISIBLE_DEVICES= $CUDA_VISIBLE_DEVICES  (NGPU=$COSMOS3_NUM_GPUS)"

# -- HF token ------------------------------------------------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
    log "WARNING: HF_TOKEN is not set. Downloads will fail for gated repos."
    log "  Option 1 (recommended): export HF_TOKEN=hf_... before running."
    log "  Option 2: run 'uvx hf auth login' once (token cached in HF_HOME)."
else
    export HF_TOKEN
    _tok_len="${#HF_TOKEN}"
    _stars="$(printf '%*s' "$(( _tok_len - 4 ))" | tr ' ' '*')"
    log "HF_TOKEN: ${_stars}${HF_TOKEN: -4}"
fi

# -----------------------------------------------------------------------------
# STEP 1 - Ensure uv is available (auto-install if missing)
# -----------------------------------------------------------------------------
log "=== Step 1: uv ==="
if ! command -v uv >/dev/null 2>&1; then
    log "uv not found - installing to $_cache_root/uv-bin ..."
    mkdir -p "$_cache_root/uv-bin"
    UV_INSTALL_DIR="$_cache_root/uv-bin" \
        curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$_cache_root/uv-bin/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv install failed"
fi
log "uv: $(uv --version)"

# -----------------------------------------------------------------------------
# STEP 2 - Clone or reuse Cosmos Framework + install Python deps
# -----------------------------------------------------------------------------
log "=== Step 2: Cosmos Framework ==="
mkdir -p "$(dirname "$COSMOS3_REPO")"
if [[ -f "$COSMOS3_REPO/pyproject.toml" && -d "$COSMOS3_REPO/cosmos_framework" ]]; then
    log "Using existing framework checkout: $COSMOS3_REPO"
elif [[ -e "$COSMOS3_REPO" ]]; then
    die "COSMOS3_REPO exists but is not a Cosmos Framework checkout: $COSMOS3_REPO"
else
    log "Cloning $COSMOS3_GIT_URL into $COSMOS3_REPO ..."
    git clone "$COSMOS3_GIT_URL" "$COSMOS3_REPO"
fi

log "=== Step 3: Install Cosmos dependencies ==="
# Healthy = python runs AND all packages needed for inference are importable.
# Checking at install time prevents silent failures that only surface inside torchrun.
_cosmos3_venv_healthy() {
    "$COSMOS3_UV_ENV/bin/python" -c \
        'import pydantic, torch; from transformers import set_seed' 2>/dev/null
}
_cosmos3_venv_install() {
    # Remove stale venv first.  On Lustre, rm -rf can fail with "Stale file handle"
    # for files written on a different node — ignore those errors since the venv
    # defaults to local disk (TMPDIR) and won't have this issue normally.
    if [[ -d "$COSMOS3_UV_ENV" ]]; then
        log "  Removing existing venv at $COSMOS3_UV_ENV ..."
        rm -rf "$COSMOS3_UV_ENV" 2>/dev/null || true
    fi
    log "  Running uv sync (group=$COSMOS3_UV_GROUP, timeout=${UV_HTTP_TIMEOUT:-300}s) ..."
    # UV_EXTRA_INDEX_URL adds PyPI as a fallback for dependencies (like cuda-bindings)
    # that may be missing from the pytorch whl index on some platforms.
    GIT_LFS_SKIP_SMUDGE=1 \
    UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}" \
    UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-https://pypi.org/simple}" \
    UV_PROJECT_ENVIRONMENT="$COSMOS3_UV_ENV" \
    uv sync --all-extras --group="$COSMOS3_UV_GROUP" --directory "$COSMOS3_REPO"

    # pynvml is excluded from pyproject.toml (sys_platform=='never') but required at runtime.
    if ! "$COSMOS3_UV_ENV/bin/python" -c 'import pynvml' 2>/dev/null; then
        log "  Installing pynvml ..."
        UV_PROJECT_ENVIRONMENT="$COSMOS3_UV_ENV" \
        uv pip install --python "$COSMOS3_UV_ENV/bin/python" pynvml \
            || die "pynvml install failed"
    fi

    # transformers can end up with a broken editable install after a partial uv sync.
    # Reinstall it if set_seed is not importable.
    if ! "$COSMOS3_UV_ENV/bin/python" -c 'from transformers import set_seed' 2>/dev/null; then
        log "  Reinstalling transformers (set_seed not importable) ..."
        UV_PROJECT_ENVIRONMENT="$COSMOS3_UV_ENV" \
        uv pip install --python "$COSMOS3_UV_ENV/bin/python" --reinstall transformers \
            || die "transformers reinstall failed"
    fi

    # opencv-contrib-python (pulled in by cosmos3) links against X11 (libxcb.so.1) which is
    # absent on headless GPU nodes.  Replace it with the headless build if cv2 fails to import.
    if ! "$COSMOS3_UV_ENV/bin/python" -c 'import cv2' 2>/dev/null; then
        log "  cv2 import failed (likely missing libxcb) — replacing with opencv-*-headless ..."
        UV_PROJECT_ENVIRONMENT="$COSMOS3_UV_ENV" \
        uv pip install --python "$COSMOS3_UV_ENV/bin/python" \
            opencv-python-headless opencv-contrib-python-headless \
            || die "opencv-headless install failed"
    fi

    # Final validation — die early with a clear diagnostic rather than failing inside torchrun.
    if ! "$COSMOS3_UV_ENV/bin/python" -c \
            'import pydantic, torch, cv2; from transformers import set_seed' 2>/dev/null; then
        _torch_ver="$("$COSMOS3_UV_ENV/bin/python" -c 'import torch; print(torch.__version__)' 2>&1 || echo 'NOT INSTALLED')"
        _tf_ver="$("$COSMOS3_UV_ENV/bin/python" -c 'import transformers; print(transformers.__version__)' 2>&1 || echo 'NOT INSTALLED')"
        die "cosmos3 venv is incomplete after uv sync.
  torch/pydantic/transformers not importable from $COSMOS3_UV_ENV
  torch:        $_torch_ver
  transformers: $_tf_ver
  Possible fixes:
    Slow network: UV_HTTP_TIMEOUT=600 FORCE_INSTALL=1 bash run_paibench_c.sh
    aarch64/no wheel: check that torch+cu130 has an aarch64 whl at download.pytorch.org"
    fi
    _torch_ver="$("$COSMOS3_UV_ENV/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
    log "  cosmos3 venv OK: torch=$_torch_ver"
}

if [[ "$SKIP_INSTALL" == "1" ]]; then
    log "SKIP_INSTALL=1 - skipping uv sync"
elif _cosmos3_venv_healthy && [[ "${FORCE_INSTALL:-0}" != "1" ]]; then
    log "Existing cosmos3 venv OK - skipping install (FORCE_INSTALL=1 to rebuild)"
else
    _cosmos3_venv_install
fi

log "cosmos3 python: $COSMOS3_UV_ENV/bin/python"

# -----------------------------------------------------------------------------
# STEP 4 - Clone + install physical-ai-bench evaluation library
#           (done before generation so setup errors surface early)
# -----------------------------------------------------------------------------
if [[ "$SKIP_EVAL" != "1" ]]; then
    log "=== Step 4: Evaluation library (physical-ai-bench) ==="
    _paibench_repo="$(dirname "$PAIBENCH_EVAL_ROOT")"
    if [[ ! -d "$PAIBENCH_EVAL_ROOT" ]]; then
        log "Cloning physical-ai-bench → $_paibench_repo ..."
        mkdir -p "$(dirname "$_paibench_repo")"
        # GIT_TERMINAL_PROMPT=0 + credential.helper="" prevents IDE credential
        # helpers (e.g. VS Code socket) from being invoked in SLURM jobs.
        GIT_TERMINAL_PROMPT=0 \
        git -c credential.helper="" clone -b "$PAIBENCH_EVAL_GIT_BRANCH" \
            "$PAIBENCH_EVAL_GIT_URL" "$_paibench_repo" \
            || die "Could not clone physical-ai-bench from $PAIBENCH_EVAL_GIT_URL
Set PAIBENCH_EVAL_ROOT to an existing checkout or override PAIBENCH_EVAL_GIT_URL."
        pushd "$_paibench_repo" >/dev/null
        GIT_TERMINAL_PROMPT=0 git -c credential.helper="" submodule update --init --recursive
        popd >/dev/null
    else
        log "Using existing physical-ai-bench checkout: $PAIBENCH_EVAL_ROOT"
    fi

    # Healthy = torchrun exists AND the key runtime packages are importable.
    # Note: the `groundingdino` pip package is NOT checked here because
    # grounded_sam_v2.py uses HuggingFace transformers for object detection —
    # the groundingdino CUDA extension is never imported at runtime.
    _paibench_venv_healthy() {
        [[ -x "$PAIBENCH_UV_ENV/bin/torchrun" ]] \
        && "$PAIBENCH_UV_ENV/bin/python3" -c \
            'import torch, sam2, transformers' 2>/dev/null
    }
    _paibench_venv_install() {
        log "  Setting up physical-ai-bench venv at $PAIBENCH_UV_ENV ..."
        if [[ -d "$PAIBENCH_UV_ENV" ]]; then
            log "  Removing stale venv ..."
            rm -rf "$PAIBENCH_UV_ENV" 2>/dev/null || true
        fi
        # pyproject.toml already pins sam2==1.1.0 and all other runtime deps.
        # No CUDA extension compilation needed: grounded_sam_v2.py uses
        # HuggingFace transformers (AutoModelForZeroShotObjectDetection) for
        # object detection, so the groundingdino CUDA package is not required.
        UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}" \
        UV_PROJECT_ENVIRONMENT="$PAIBENCH_UV_ENV" \
        uv sync --directory "$PAIBENCH_EVAL_ROOT"

        # Validate — print the exact error if any package is missing.
        local _ok=1
        for _pkg in torch sam2 transformers torchvision; do
            if ! "$PAIBENCH_UV_ENV/bin/python3" -c "import $_pkg" 2>/dev/null; then
                log "  FAIL: import $_pkg"
                "$PAIBENCH_UV_ENV/bin/python3" -c "import $_pkg" 2>&1 | head -5 || true
                _ok=0
            fi
        done
        [[ "$_ok" == "1" ]] \
            || die "physical-ai-bench venv incomplete after install (see FAIL lines above).
  Try: FORCE_INSTALL=1 UV_HTTP_TIMEOUT=600 bash run_paibench_c.sh"
        log "  physical-ai-bench venv OK"
    }

    if [[ "${FORCE_INSTALL:-0}" == "1" ]]; then
        log "FORCE_INSTALL=1 - rebuilding physical-ai-bench venv ..."
        _paibench_venv_install
    elif _paibench_venv_healthy; then
        log "Existing physical-ai-bench venv OK - skipping install"
    else
        log "physical-ai-bench venv missing or incomplete - rebuilding ..."
        _paibench_venv_install
    fi

    # Always verify sam2==1.1.0 is installed — the version must exactly match
    # the internal evaluation to get reproducible mask propagation results.
    _sam2_ver="$("$PAIBENCH_UV_ENV/bin/python3" -c \
        "import importlib.metadata; print(importlib.metadata.version('sam2'))" \
        2>/dev/null || echo "missing")"
    if [[ "$_sam2_ver" != "1.1.0" ]]; then
        log "Pinning sam2==1.1.0 (current: $_sam2_ver) ..."
        UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}" \
        UV_PROJECT_ENVIRONMENT="$PAIBENCH_UV_ENV" \
        uv pip install "sam2==1.1.0" \
            || die "sam2==1.1.0 install failed"
    else
        log "sam2==1.1.0 already installed."
    fi

    # Download evaluation model checkpoints if any are missing.
    # get_checkpoint.sh downloads: sam2_hiera_large.pt, DOVER.pth,
    # video_depth_anything_vits.pth, and IDEA-Research/grounding-dino-tiny (HF).
    _ckpt_dir="$PAIBENCH_EVAL_ROOT/checkpoint"
    if [[ ! -f "$_ckpt_dir/sam2/sam2_hiera_large.pt" ]] || \
       [[ ! -f "$_ckpt_dir/DOVER/DOVER.pth" ]]; then
        log "Downloading evaluation checkpoints ..."
        (
            cd "$PAIBENCH_EVAL_ROOT"
            PATH="$PAIBENCH_UV_ENV/bin:$PATH" bash get_checkpoint.sh
        ) || die "get_checkpoint.sh failed"
    else
        log "Evaluation checkpoints already present."
    fi
else
    log "=== Step 4: Evaluation library - skipped (SKIP_EVAL=1) ==="
fi

# -----------------------------------------------------------------------------
# STEP 5 - Verify GPU environment
# -----------------------------------------------------------------------------
log "=== Step 5: GPU check ==="
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$COSMOS3_UV_ENV/bin/python" - <<'PY'
import torch
print(f"torch={torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  device {i}: {torch.cuda.get_device_name(i)}")
PY

# -----------------------------------------------------------------------------
# STEP 6 - Download PAI-Bench-C dataset (Git LFS)
# -----------------------------------------------------------------------------
log "=== Step 6: Dataset download (samples=$PAIBENCH_C_NUM_SAMPLES) ==="
if ! command -v git-lfs >/dev/null 2>&1; then
    log "git-lfs not found - downloading static binary ..."
    mkdir -p "$_cache_root/bin"
    _arch="$(uname -m)"
    [[ "$_arch" == "x86_64" ]] && _lfs_arch="amd64" || _lfs_arch="arm64"
    _lfs_ver="3.6.1"
    _lfs_url="https://github.com/git-lfs/git-lfs/releases/download/v${_lfs_ver}/git-lfs-linux-${_lfs_arch}-v${_lfs_ver}.tar.gz"
    curl -LsSf "$_lfs_url" \
        | tar -xz -C "$_cache_root/bin" --strip-components=1 "git-lfs-${_lfs_ver}/git-lfs" 2>/dev/null \
        || curl -LsSf "$_lfs_url" \
        | tar -xz -C "$_cache_root/bin" --wildcards --strip-components=1 "*/git-lfs"
    chmod +x "$_cache_root/bin/git-lfs"
    export PATH="$_cache_root/bin:$PATH"
    command -v git-lfs >/dev/null 2>&1 || die "git-lfs install failed. Install manually: https://git-lfs.com/"
    log "git-lfs: $(git-lfs version)"
fi

if [[ -d "$PAIBENCH_C_DATASET_ROOT/.git" ]]; then
    log "Dataset already cloned at $PAIBENCH_C_DATASET_ROOT"
else
    log "Cloning dataset skeleton (no LFS blobs) ..."
    mkdir -p "$(dirname "$PAIBENCH_C_DATASET_ROOT")"
    GIT_LFS_SKIP_SMUDGE=1 git clone "$PAIBENCH_C_HF_URL" "$PAIBENCH_C_DATASET_ROOT"
fi
# Always install LFS hooks in the repo - needed when git-lfs was just auto-downloaded
# and the repo was cloned in a previous run before git-lfs was on PATH.
# --force overwrites an existing pre-push hook (which git-lfs install exits 2 for,
# killing the script under set -e without it).
git -C "$PAIBENCH_C_DATASET_ROOT" lfs install --force

pushd "$PAIBENCH_C_DATASET_ROOT" >/dev/null
if [[ "$PAIBENCH_C_NUM_SAMPLES" -lt 600 ]]; then
    log "Pulling $PAIBENCH_C_NUM_SAMPLES task(s) starting at task_0000 ..."
    for i in $(seq 0 $((PAIBENCH_C_NUM_SAMPLES - 1))); do
        _t=$(printf "task_%04d" "$i")
        git lfs pull --include="videos/${_t}.mp4"
        git lfs pull --include="canny/${_t}.mp4"
        git lfs pull --include="blur/${_t}.mp4"
        git lfs pull --include="depth_vids/${_t}.mp4"
        git lfs pull --include="sam2_vids/${_t}.mp4"
        git lfs pull --include="depth_npzs/${_t}.npz"
        git lfs pull --include="sam2_pkls/${_t}.pkl"
    done
    log "Download complete ($PAIBENCH_C_NUM_SAMPLES task(s))."
else
    log "Pulling all 600 tasks (may take several minutes) ..."
    git lfs pull
    log "Download complete (600 tasks)."
fi
popd >/dev/null

# -----------------------------------------------------------------------------
# STEP 7 - Build input JSONLs and run generation
# -----------------------------------------------------------------------------
log "=== Step 7: Generation ==="

build_jsonl_and_run() {
    local MODALITY="$1"

    # Include checkpoint name in the output path so Nano and Super runs don't
    # overwrite each other when both are run in the same PAIBENCH_C_OUTPUT_ROOT.
    local _ckpt_slug
    _ckpt_slug="${PAIBENCH_C_CHECKPOINT//\//-}"   # e.g. Cosmos3-Nano, Cosmos3-Super

    local OUTPUT_DIR="$PAIBENCH_C_OUTPUT_ROOT/${_ckpt_slug}/$MODALITY"
    local INPUT_JSONL="$OUTPUT_DIR/input.jsonl"
    local RAW_DIR="$OUTPUT_DIR/raw"
    mkdir -p "$RAW_DIR"

    log "  Building JSONL: modality=$MODALITY  samples=$PAIBENCH_C_NUM_SAMPLES ..."
    PAIBENCH_C_ROOT="$PAIBENCH_C_ROOT" \
    PAIBENCH_C_DATASET_ROOT="$PAIBENCH_C_DATASET_ROOT" \
    PAIBENCH_C_OUTPUT_DIR="$OUTPUT_DIR" \
    PAIBENCH_C_MODALITY="$MODALITY" \
    PAIBENCH_C_NUM_SAMPLES="$PAIBENCH_C_NUM_SAMPLES" \
    PAIBENCH_C_SPEC="$PAIBENCH_C_ROOT/specs/$MODALITY.json" \
    "$COSMOS3_UV_ENV/bin/python" - <<'PY'
import json, os, pathlib

modality    = os.environ["PAIBENCH_C_MODALITY"]
dataset     = pathlib.Path(os.environ["PAIBENCH_C_DATASET_ROOT"])
output_dir  = pathlib.Path(os.environ["PAIBENCH_C_OUTPUT_DIR"])
num_samples = int(os.environ["PAIBENCH_C_NUM_SAMPLES"])

# Load eval spec as base template for all inference parameters.
# Per-task fields (aspect_ratio, fps, seed, prompt, negative_prompt, control_path)
# are overlaid from assets/tasks.json.
base_spec = json.loads(pathlib.Path(os.environ["PAIBENCH_C_SPEC"]).read_text())
# Remove file-reference fields — eval uses inline values instead
base_spec.pop("prompt_path", None)
base_spec.pop("negative_prompt_file", None)
# Remove the modality block — we'll replace control_path per task
modality_defaults = base_spec.pop(modality, {})

# Modalities that support on-the-fly control computation from vision_path.
# depth and seg require pre-computed control files (control_path).
VISION_PATH_MODALITIES = {"edge", "blur"}
CONTROL_KEY = {"edge": "canny_path", "blur": "blur_path", "depth": "depth_path", "seg": "seg_path"}

all_tasks = json.loads(
    (pathlib.Path(os.environ["PAIBENCH_C_ROOT"]) / "assets" / "tasks.json").read_text()
)
by_task = {t["task_id"]: t for t in all_tasks}

task_ids = sorted(by_task.keys())[:num_samples]

rows = []
for tid in task_ids:
    t = by_task[tid]

    # Start from spec template, then overlay per-task values
    row = dict(base_spec)
    row.update({
        "name":            tid,
        "aspect_ratio":    t["aspect_ratio"],  # per-video from assets/tasks.json
        "fps":             t["fps"],
        "seed":            t["seed"],
        "num_outputs":     1,
        "prompt":          t["prompt"],
        "negative_prompt": t["negative_prompt"],
    })
    # All modalities pass vision_path (source video) to match the internal run.
    row["vision_path"] = str(dataset / t["video_path"])
    if modality in VISION_PATH_MODALITIES:
        # edge/blur: control is computed on-the-fly; no control_path needed.
        row[modality] = dict(modality_defaults)
    else:
        # depth/seg: pre-computed control file required in addition to vision_path.
        control_block = dict(modality_defaults)
        control_block["control_path"] = str(dataset / t[CONTROL_KEY[modality]])
        row[modality] = control_block
    rows.append(row)

dst = output_dir / "input.jsonl"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(json.dumps(r) for r in rows))
print(f"Wrote {len(rows)} rows → {dst}")
PY

    # -------------------------------------------------------------------------
    # Run inference using N parallel single-GPU workers — one per GPU.
    # This matches the internal run_transferbench_oss_inference.sh exactly:
    #   - Each worker owns one GPU (CUDA_VISIBLE_DEVICES=<id>)
    #   - Each worker processes its interleaved task shard independently
    #   - Tasks with valid existing outputs are skipped (resume-safe)
    # Why this avoids GPU OOM: each worker handles only N/NGPU tasks (~150
    # for 600 tasks total), so accumulated tensors never exhaust VRAM.
    # With one shared multi-GPU torchrun (--parallelism-preset=latency), all
    # 600 tasks run in a single process; unreleased diffusion intermediates
    # accumulate after ~200 tasks and production of empty 7 KB stubs begins.
    # -------------------------------------------------------------------------
    local _min_valid_bytes=50000   # < 50 KB → empty/failed video stub

    # Unset vars that can contaminate cosmos3 Python workers when called from
    # a Jupyter/notebook environment.
    unset PYTHONPATH PYTHONSTARTUP PYTHONHOME MPLBACKEND 2>/dev/null || true
    export TORCH_HOME="${TRITON_CACHE_DIR%/triton}"

    # Parse GPU list from CUDA_VISIBLE_DEVICES (e.g. "0,1,2,3" → array).
    IFS=',' read -ra _gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
    local _num_gpus="${#_gpu_ids[@]}"
    local _py="${COSMOS3_UV_ENV}/bin/python"
    [[ -x "$_py" ]] || _py="$(command -v python3 || command -v python)"

    log "  Running inference: modality=$MODALITY  tasks=$PAIBENCH_C_NUM_SAMPLES  gpus=$_num_gpus (parallel single-GPU workers) ..."

    # Build per-GPU shard JSONLs — interleaved distribution matches the
    # internal script: GPU 0 → tasks 0,4,8,…; GPU 1 → tasks 1,5,9,… etc.
    local -a _shard_jsonls=()
    local gi
    for (( gi=0; gi<_num_gpus; gi++ )); do
        local _shard_jsonl="$OUTPUT_DIR/shard_${gi}.jsonl"
        _shard_jsonls+=("$_shard_jsonl")
        PAIBENCH_C_RAW_DIR="$RAW_DIR" \
        PAIBENCH_C_INPUT_JSONL="$INPUT_JSONL" \
        PAIBENCH_C_SHARD_JSONL="$_shard_jsonl" \
        PAIBENCH_C_SHARD_IDX="$gi" \
        PAIBENCH_C_NUM_SHARDS="$_num_gpus" \
        PAIBENCH_C_MIN_VALID_BYTES="$_min_valid_bytes" \
        "$_py" - <<'PY'
import json, os, pathlib

raw_dir     = pathlib.Path(os.environ["PAIBENCH_C_RAW_DIR"])
input_jsonl = pathlib.Path(os.environ["PAIBENCH_C_INPUT_JSONL"])
shard_jsonl = pathlib.Path(os.environ["PAIBENCH_C_SHARD_JSONL"])
shard_idx   = int(os.environ["PAIBENCH_C_SHARD_IDX"])
num_shards  = int(os.environ["PAIBENCH_C_NUM_SHARDS"])
min_bytes   = int(os.environ["PAIBENCH_C_MIN_VALID_BYTES"])

all_rows = [json.loads(l) for l in input_jsonl.read_text().strip().splitlines()]
# Interleaved: GPU 0 gets rows 0,4,8,…; GPU 1 gets rows 1,5,9,… etc.
shard_rows = [r for i, r in enumerate(all_rows) if i % num_shards == shard_idx]

pending = []
for row in shard_rows:
    name = row["name"]
    vision_mp4 = raw_dir / name / "vision.mp4"
    if vision_mp4.exists() and vision_mp4.stat().st_size >= min_bytes:
        continue  # already generated — skip
    # Remove stale empty stub so the framework recreates the task dir cleanly.
    if vision_mp4.exists():
        vision_mp4.unlink()
    pending.append(row)

shard_jsonl.write_text("\n".join(json.dumps(r) for r in pending))
skipped = len(shard_rows) - len(pending)
print(f"Shard {shard_idx}/{num_shards}: {len(pending)} to generate, {skipped} already done")
PY
    done

    # Launch one background Python process per GPU — no torch.distributed.
    local -a _worker_pids=()
    for (( gi=0; gi<_num_gpus; gi++ )); do
        local _shard_jsonl="${_shard_jsonls[$gi]}"
        local _pending_count
        _pending_count="$(wc -l < "$_shard_jsonl" | tr -d ' ')"
        local _gpu_id="${_gpu_ids[$gi]}"

        if [[ "$_pending_count" -eq 0 ]]; then
            log "  GPU $_gpu_id: all tasks done, skipping"
            rm -f "$_shard_jsonl"
            continue
        fi

        log "  GPU $_gpu_id: $_pending_count tasks (starting background worker) ..."
        (
            cd "$COSMOS3_REPO"
            CUDA_VISIBLE_DEVICES="$_gpu_id" LD_LIBRARY_PATH= \
            "$COSMOS3_UV_ENV/bin/python" \
                -m cosmos_framework.scripts.inference \
                --parallelism-preset=latency \
                --no-guardrails \
                -i "$_shard_jsonl" \
                -o "$RAW_DIR" \
                --checkpoint-path "$PAIBENCH_C_CHECKPOINT"
            rm -f "$_shard_jsonl"
        ) &
        _worker_pids+=("$!")
    done

    # Wait for all GPU workers to finish.
    local _fail=0
    for _pid in "${_worker_pids[@]}"; do
        wait "$_pid" || _fail=1
    done
    [[ "$_fail" -eq 0 ]] || die "One or more GPU workers failed for modality=$MODALITY"

    log "  Flattening outputs → $OUTPUT_DIR/videos/ ..."
    # Copy only valid (non-empty) vision.mp4 files.  Stale stub files from
    # earlier failed runs are excluded by the min-size guard.
    _py="${COSMOS3_UV_ENV}/bin/python"
    [[ -x "$_py" ]] || _py="$(command -v python3 || command -v python)"
    PAIBENCH_C_RAW_DIR="$RAW_DIR" \
    PAIBENCH_C_VIDEOS_DIR="$OUTPUT_DIR/videos" \
    PAIBENCH_C_MIN_VALID_BYTES="$_min_valid_bytes" \
    "$_py" - <<'PY'
import os, shutil, pathlib
raw      = pathlib.Path(os.environ["PAIBENCH_C_RAW_DIR"])
vids     = pathlib.Path(os.environ["PAIBENCH_C_VIDEOS_DIR"])
min_sz   = int(os.environ["PAIBENCH_C_MIN_VALID_BYTES"])
vids.mkdir(parents=True, exist_ok=True)
copied = skipped_exists = skipped_empty = 0
for mp4 in sorted(raw.rglob("vision.mp4")):
    dst = vids / f"{mp4.parent.name}.mp4"
    if dst.exists():
        skipped_exists += 1
        continue
    if mp4.stat().st_size < min_sz:
        skipped_empty += 1
        continue
    shutil.copy2(mp4, dst)
    copied += 1
print(f"Collected {copied} video(s) → {vids}  (skipped: {skipped_exists} existing, {skipped_empty} empty)")
PY
    log "  Generation complete: $MODALITY"
}

if [[ "$SKIP_GEN" == "1" ]]; then
    log "SKIP_GEN=1 - skipping generation"
else
    for _mod in $PAIBENCH_C_MODALITIES; do
        build_jsonl_and_run "$_mod"
    done
fi

# -----------------------------------------------------------------------------
# STEP 8 - Evaluation (compute_metrics.py)
# -----------------------------------------------------------------------------
if [[ "$SKIP_EVAL" == "1" ]]; then
    log "SKIP_EVAL=1 - skipping evaluation"
    log "=== PAI-Bench-C generation complete ==="
    exit 0
fi

log "=== Step 8: Evaluation ==="
unset MPLBACKEND  # prevent Jupyter's inline backend from leaking into subprocesses

_ckpt_slug="${PAIBENCH_C_CHECKPOINT//\//-}"

# Use a robust python for printing results - prefer cosmos3 venv if functional,
# fall back to system python3 (only stdlib used).
_py_for_summary() {
    local _p="$COSMOS3_UV_ENV/bin/python"
    if "$_p" -c 'import sys' 2>/dev/null; then
        echo "$_p"
    else
        command -v python3 2>/dev/null || command -v python || die "No python3 found"
    fi
}

for _mod in $PAIBENCH_C_MODALITIES; do
    _videos_parent="$PAIBENCH_C_OUTPUT_ROOT/${_ckpt_slug}/$_mod"
    _videos_dir="$_videos_parent/videos"
    _metrics_out="$_videos_parent/metrics.json"
    _eval_ngpu="$(int_min "$PAIBENCH_C_NUM_SAMPLES" "$COSMOS3_NUM_GPUS")"

    # Verify videos exist before launching torchrun to surface errors early.
    if [[ ! -d "$_videos_dir" ]]; then
        die "Videos directory not found: $_videos_dir
  Run without SKIP_GEN=1, or point PAIBENCH_C_OUTPUT_ROOT at an existing output tree."
    fi
    _nvideos="$(find "$_videos_dir" -name '*.mp4' | wc -l)"
    if [[ "$_nvideos" -eq 0 ]]; then
        die "No .mp4 files found in $_videos_dir
  Re-run generation (remove SKIP_GEN=1) to produce videos."
    fi
    log "  Found $_nvideos video(s) in $_videos_dir"

    _gpu_list=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | head -n "$_eval_ngpu" | tr '\n' ',' | sed 's/,$//')
    log "  Evaluating: modality=$_mod  ngpu=$_eval_ngpu  gpus=$_gpu_list  videos=$_videos_parent ..."

    # Run compute_metrics.py from its own directory so relative imports work.
    # Capture the error file path before entering the subshell.
    _err_file="${_metrics_out%.json}.err"
    (
        cd "$PAIBENCH_EVAL_ROOT"
        CUDA_VISIBLE_DEVICES="$_gpu_list" \
        TORCHELASTIC_ERROR_FILE="$_err_file" \
        "$PAIBENCH_UV_ENV/bin/torchrun" \
            --standalone --nproc_per_node "$_eval_ngpu" \
            compute_metrics.py calculate-metrics \
            --gt_path    "$PAIBENCH_C_DATASET_ROOT" \
            --videos_path "$_videos_parent" \
            --output_path "$_metrics_out" \
            --force_recompute_gt_seg \
            --force_recompute_gt_depth
    ) || {
        log "  ERROR: evaluation failed for modality=$_mod"
        [[ -f "$_err_file" ]] && log "  Error details: $_err_file" && cat "$_err_file" >&2
        die "Evaluation failed for modality=$_mod"
    }

    log "  Results: $_metrics_out"
    _mout="$_metrics_out" \
    "$(_py_for_summary)" - <<'PY'
import json, os, pathlib
m = json.loads(pathlib.Path(os.environ["_mout"]).read_text())
for k, v in m["global"].items():
    print(f"    {k}: {v:.4f}")
PY
done

log "=== PAI-Bench-C complete ==="
