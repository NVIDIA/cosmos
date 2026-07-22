#!/usr/bin/env bash
# Install PAIBench-G scoring into an environment isolated from Cosmos Framework.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCORER_REPO=${PAIBENCH_SCORER_REPO:-$SCRIPT_DIR/scorers/physical-ai-bench}
SCORER_DIR=${PAIBENCH_SCORER_DIR:-$SCORER_REPO/generation}
SCORER_ENV=${PAIBENCH_SCORER_ENV:-$SCORER_DIR/.venv-scorer}
PYTHON_VERSION=${PAIBENCH_SCORER_PYTHON:-3.10}
SCORER_GIT_URL=${PAIBENCH_SCORER_GIT_URL:-https://github.com/SHI-Labs/physical-ai-bench.git}
SCORER_COMMIT=${PAIBENCH_SCORER_COMMIT:-}
DETECTRON2_REF=${PAIBENCH_DETECTRON2_REF:-main}
INSTALL_DETECTRON2=${PAIBENCH_INSTALL_DETECTRON2:-0}
PREPARE_DREAMSIM=${PAIBENCH_PREPARE_DREAMSIM:-1}
PREPARE_AMT=${PAIBENCH_PREPARE_AMT:-1}
DINO_GIT_URL=${PAIBENCH_DINO_GIT_URL:-https://github.com/facebookresearch/dino.git}
DINO_REF=${PAIBENCH_DINO_REF:-main}
INSTALL_LOG=${PAIBENCH_INSTALL_LOG:-$SCRIPT_DIR/paibench_scorer_install.log}
COMPAT_PATCH=${PAIBENCH_COMPAT_PATCH:-$SCRIPT_DIR/patches/physical-ai-bench-vllm-dense-model.patch}
OBJECT_GATHER_PATCH=${PAIBENCH_OBJECT_GATHER_PATCH:-$SCRIPT_DIR/patches/physical-ai-bench-object-gather.patch}

exec > >(tee -a "$INSTALL_LOG") 2>&1

echo "PAIBench scorer installation"
echo "  repository:  $SCORER_REPO"
echo "  project:     $SCORER_DIR"
echo "  environment: $SCORER_ENV"
echo "  Python:      $PYTHON_VERSION"
echo "  log:         $INSTALL_LOG"

command -v git >/dev/null || { echo "ERROR: git is required"; exit 1; }
command -v uv >/dev/null || { echo "ERROR: uv is required"; exit 1; }

if [[ ! -d "$SCORER_REPO/.git" ]]; then
    if [[ -e "$SCORER_REPO" ]]; then
        echo "ERROR: $SCORER_REPO exists but is not a Git checkout"
        exit 1
    fi
    mkdir -p "$(dirname "$SCORER_REPO")"
    git clone "$SCORER_GIT_URL" "$SCORER_REPO"
fi

if [[ -n "$SCORER_COMMIT" ]]; then
    git -C "$SCORER_REPO" checkout --detach "$SCORER_COMMIT"
fi

echo "  scorer SHA:  $(git -C "$SCORER_REPO" rev-parse HEAD)"

if [[ ! -f "$SCORER_DIR/pyproject.toml" ]]; then
    echo "ERROR: missing scorer project: $SCORER_DIR/pyproject.toml"
    exit 1
fi

# vLLM 0.14 rejects expert parallelism for the dense Qwen2.5-VL model. The
# pinned PAIBench scorer enables it unconditionally, so apply the narrow source
# compatibility patch once and fail if the upstream context changes.
if [[ -f "$COMPAT_PATCH" ]]; then
    if git -C "$SCORER_REPO" apply --check "$COMPAT_PATCH" 2>/dev/null; then
        git -C "$SCORER_REPO" apply "$COMPAT_PATCH"
    elif git -C "$SCORER_REPO" apply --reverse --check "$COMPAT_PATCH" 2>/dev/null; then
        echo "vLLM dense-model compatibility patch already applied"
    else
        echo "ERROR: compatibility patch does not match scorer checkout"
        exit 1
    fi
fi

# The pinned scorer manually serializes per-video result dictionaries into CUDA
# byte tensors before gathering them. A corrupted gathered size can therefore
# trigger a multi-terabyte allocation. Use PyTorch's object collective for this
# small metadata payload while leaving the generic tensor gather unchanged.
if [[ ! -f "$OBJECT_GATHER_PATCH" ]]; then
    echo "ERROR: missing object-gather patch: $OBJECT_GATHER_PATCH"
    exit 1
fi
if git -C "$SCORER_REPO" apply --check "$OBJECT_GATHER_PATCH" 2>/dev/null; then
    git -C "$SCORER_REPO" apply "$OBJECT_GATHER_PATCH"
elif git -C "$SCORER_REPO" apply --reverse --check "$OBJECT_GATHER_PATCH" 2>/dev/null; then
    echo "Distributed object-gather patch already applied"
else
    echo "ERROR: object-gather patch does not match scorer checkout"
    exit 1
fi

# Never let a notebook-exported generation environment capture this sync.
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
    generation_env=$(realpath -m "$UV_PROJECT_ENVIRONMENT")
    scorer_env=$(realpath -m "$SCORER_ENV")
    if [[ "$generation_env" == "$scorer_env" ]]; then
        echo "ERROR: scorer and generation environments resolve to the same path"
        exit 1
    fi
fi
unset VIRTUAL_ENV CONDA_PREFIX

if [[ ! -x "$SCORER_ENV/bin/python" ]]; then
    uv venv --python "$PYTHON_VERSION" "$SCORER_ENV"
fi

cd "$SCORER_DIR"
if [[ -f uv.lock ]]; then
    echo "Syncing from existing uv.lock"
    UV_PROJECT_ENVIRONMENT="$SCORER_ENV" uv sync --locked
else
    echo "No upstream lockfile found; resolving once and creating uv.lock"
    UV_PROJECT_ENVIRONMENT="$SCORER_ENV" uv sync
fi

PY="$SCORER_ENV/bin/python"

# The upstream lock contains both GUI and headless OpenCV distributions. On
# headless compute nodes that makes cv2 depend on libGL.so.1. Keep only the
# headless wheel and reinstall NumPy alongside it so uv cannot upgrade NumPy 2.
uv pip uninstall --python "$PY" opencv-python || true
uv pip install \
    --python "$PY" \
    --reinstall \
    "numpy==1.26.4" \
    "opencv-python-headless==4.11.0.86"

# DreamSim 0.2.1 leaves PEFT unconstrained. The lock currently resolves PEFT
# 0.19, whose tensor-parallel imports are incompatible with Transformers 4.57.
uv pip install --python "$PY" --reinstall --no-deps "peft==0.17.1"

# DreamSim changes torch.hub's cache to ./models and otherwise validates DINO
# through GitHub's rate-limited API. Populate the exact cache directory first.
if [[ "$PREPARE_DREAMSIM" == "1" ]]; then
    DINO_CACHE="$SCORER_DIR/models/facebookresearch_dino_main"
    if [[ ! -d "$DINO_CACHE" ]]; then
        git clone --depth 1 --branch "$DINO_REF" "$DINO_GIT_URL" "$DINO_CACHE"
    fi
fi

# Use the scorer's canonical asset resolver so motion smoothness is ready
# before a long evaluation starts. The helper is idempotent when cached.
if [[ "$PREPARE_AMT" == "1" ]]; then
    echo "Preparing AMT-S motion-smoothness assets"
    VBENCH_CACHE_DIR="$SCORER_REPO/.vbench_cache" PYTHONNOUSERSITE=1 "$PY" - <<'PY'
from pathlib import Path

from pbench.utils import init_submodules

assets = init_submodules(["motion_smoothness"], local=True)["motion_smoothness"]
for label in ("config", "ckpt"):
    path = Path(assets[label])
    if not path.is_file():
        raise FileNotFoundError(f"AMT {label} is missing: {path}")
    print(f"AMT {label}: {path}")
PY
fi

# No generation/pbench runtime module imports Detectron2. It is therefore off
# by default, especially because the cluster CUDA toolkit can differ from the
# CUDA version bundled with the locked PyTorch wheel.
if [[ "$INSTALL_DETECTRON2" == "1" ]]; then
    uv pip install --python "$PY" ninja setuptools wheel
    if ! PYTHONNOUSERSITE=1 "$PY" -c 'import detectron2' >/dev/null 2>&1; then
        uv pip install \
            --python "$PY" \
            --no-build-isolation \
            "git+https://github.com/facebookresearch/detectron2.git@$DETECTRON2_REF"
    fi
else
    echo "Skipping optional Detectron2 (set PAIBENCH_INSTALL_DETECTRON2=1 to install)"
fi

echo "Validating scorer environment"
PYTHONNOUSERSITE=1 "$PY" - <<'PY'
import importlib.metadata as metadata
import inspect
import os

from pbench.distributed import gather_list_of_dict
import cv2
import decord
import dreamsim
import numpy
import pbench
import pyiqa
import qwen_vl_utils
import torch
import torchvision
import vllm

print("python:", os.sys.version.split()[0])
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("vLLM:", vllm.__version__)
print("NumPy:", numpy.__version__)
print("PEFT:", metadata.version("peft"))
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())

assert int(numpy.__version__.split(".", 1)[0]) < 2, "PAIBench requires NumPy < 2"
assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
assert torch.cuda.device_count() >= 1, "No GPU is visible"
assert "all_gather_object" in inspect.getsource(gather_list_of_dict), (
    "PAIBench distributed object-gather patch is not active"
)

try:
    torchao_version = metadata.version("torchao")
except metadata.PackageNotFoundError:
    torchao_version = "not installed"
print("torchao:", torchao_version)
PY

if [[ "$INSTALL_DETECTRON2" == "1" ]]; then
    PYTHONNOUSERSITE=1 "$PY" - <<'PY'
from detectron2 import _C
print("Detectron2 compiled extension: OK")
PY
fi

uv pip freeze --python "$PY" > "$SCORER_DIR/scorer_environment.freeze.txt"
echo "Environment ready: $SCORER_ENV"
echo "Manifest: $SCORER_DIR/scorer_environment.freeze.txt"
