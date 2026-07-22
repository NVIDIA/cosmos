#!/usr/bin/env bash
# Install the RBench 5-tasks scorer in an environment isolated from Cosmos Framework.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCORER_REPO=${REVIDGEN_REPO:-$SCRIPT_DIR/scorers/ReVidgen}
SCORER_ENV=${REVIDGEN_SCORER_ENV:-$SCORER_REPO/.venv-rbench-scorer}
PYTHON_VERSION=${RBENCH_SCORER_PYTHON:-3.10}
SCORER_GIT_URL=${REVIDGEN_GIT_URL:-https://github.com/DAGroup-PKU/ReVidgen.git}
SCORER_COMMIT=${REVIDGEN_COMMIT:-b03df27f0376faa148dcd8cd620a1989a32ca979}
TORCH_BACKEND=${RBENCH_TORCH_BACKEND:-cu128}
TORCH_VERSION=${RBENCH_TORCH_VERSION:-2.9.1}
TORCHVISION_VERSION=${RBENCH_TORCHVISION_VERSION:-0.24.1}
INSTALL_LOG=${RBENCH_INSTALL_LOG:-$SCRIPT_DIR/rbench_scorer_install.log}
OPENAI_MODEL_PATCH="$SCRIPT_DIR/patches/revidgen-configurable-openai-model.patch"

exec > >(tee -a "$INSTALL_LOG") 2>&1

echo "RBench 5-tasks scorer installation"
echo "  repository:    $SCORER_REPO"
echo "  environment:   $SCORER_ENV"
echo "  scorer commit: $SCORER_COMMIT"
echo "  Python:        $PYTHON_VERSION"
echo "  Torch backend: $TORCH_BACKEND"
echo "  log:           $INSTALL_LOG"

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
    if ! git -C "$SCORER_REPO" cat-file -e "$SCORER_COMMIT^{commit}" 2>/dev/null; then
        git -C "$SCORER_REPO" fetch origin "$SCORER_COMMIT"
    fi
    git -C "$SCORER_REPO" checkout --detach "$SCORER_COMMIT"
fi

echo "  scorer SHA:    $(git -C "$SCORER_REPO" rev-parse HEAD)"

if [[ ! -f "$SCORER_REPO/requirements_vlm.txt" ]]; then
    echo "ERROR: missing $SCORER_REPO/requirements_vlm.txt"
    exit 1
fi
if [[ ! -d "$SCORER_REPO/eval/5_tasks" ]]; then
    echo "ERROR: missing $SCORER_REPO/eval/5_tasks"
    exit 1
fi
if [[ ! -f "$OPENAI_MODEL_PATCH" ]]; then
    echo "ERROR: missing configurable OpenAI model patch: $OPENAI_MODEL_PATCH"
    exit 1
fi

# The public ReVidgen commit hard-codes the dated GPT-5 snapshot. Preserve that
# value as the default while allowing OpenAI-compatible gateways to select the
# model identifier they expose through OPENAI_MODEL.
if git -C "$SCORER_REPO" apply --unidiff-zero --reverse --check "$OPENAI_MODEL_PATCH" >/dev/null 2>&1; then
    echo "Configurable OpenAI model patch already applied"
elif git -C "$SCORER_REPO" apply --unidiff-zero --check "$OPENAI_MODEL_PATCH"; then
    git -C "$SCORER_REPO" apply --unidiff-zero "$OPENAI_MODEL_PATCH"
    echo "Applied configurable OpenAI model patch"
else
    echo "ERROR: configurable OpenAI model patch does not apply cleanly"
    exit 1
fi

# A notebook may export its generation environment globally. Never allow that
# environment to capture RBench installs.
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
    generation_env=$(realpath -m "$UV_PROJECT_ENVIRONMENT")
    scorer_env=$(realpath -m "$SCORER_ENV")
    if [[ "$generation_env" == "$scorer_env" ]]; then
        echo "ERROR: scorer and generation environments resolve to the same path"
        exit 1
    fi
fi
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV CONDA_PREFIX

if [[ ! -x "$SCORER_ENV/bin/python" ]]; then
    uv venv --python "$PYTHON_VERSION" "$SCORER_ENV"
fi
PY="$SCORER_ENV/bin/python"

# Install an explicit CUDA build. `auto` can retain or select a CPU wheel when
# rerunning against an existing environment, which would make 72B scoring hang
# on host RAM instead of failing early.
uv pip install \
    --python "$PY" \
    --torch-backend "$TORCH_BACKEND" \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION"

# Install the packages from requirements_vlm.txt explicitly so its unconstrained
# `transformers` and GUI `opencv-python` entries cannot transiently upgrade this
# environment to known-bad variants on every rerun. The repository's full
# requirements pull the compiled 4-embodiments stack (GroundingDINO, SAM2,
# CoTracker), which is deliberately out of scope here.
#
# Qwen2.5-VL support landed in Transformers 4.x. Transformers 5 has broken the
# scorer's image_grid_thw path, so constrain it explicitly. Keep GUI OpenCV out
# of headless compute nodes and ensure device_map="auto" has Accelerate.
uv pip uninstall --python "$PY" opencv-python || true
uv pip install \
    --python "$PY" \
    "numpy==1.26.4" \
    "tqdm" \
    "pillow" \
    "opencv-python-headless==4.11.0.86" \
    "openai" \
    "scipy" \
    "scikit-learn" \
    "pandas" \
    "matplotlib" \
    "accelerate>=1,<2" \
    "qwen-vl-utils>=0.0.8" \
    "transformers>=4.49,<5"

SCORER_REPO="$SCORER_REPO" PYTHONNOUSERSITE=1 "$PY" - <<'PY'
import os
from pathlib import Path

source_root = Path(os.environ["SCORER_REPO"]) / "eval" / "5_tasks"
old_import = "from torchvision.io import write_video"
new_import = [
    "try:",
    "    from torchvision.io import write_video",
    "except Exception:",
    "    write_video = None",
]

for source_path in source_root.glob("*.py"):
    lines = source_path.read_text().splitlines()
    import_lines = [i for i, line in enumerate(lines) if line.strip() == old_import]
    if not import_lines:
        continue
    if len(import_lines) != 1:
        raise RuntimeError(f"expected one write_video import in {source_path}")

    # Repair both the original import and blocks generated by older installer runs.
    start = import_lines[0]
    while start > 0 and lines[start - 1].strip() == "try:":
        start -= 1
    end = import_lines[0] + 1
    while end < len(lines) and lines[end].strip() in {
        "except Exception:",
        "write_video = None",
    }:
        end += 1

    if lines[start:end] != new_import:
        lines[start:end] = new_import
        source_path.write_text("\n".join(lines) + "\n")
PY

echo "Validating RBench scorer environment"
PYTHONNOUSERSITE=1 "$PY" - <<'PY'
import importlib.metadata as metadata
import os

import accelerate
import av
import cv2
import numpy
import pandas
import qwen_vl_utils
import torch
import torchvision
import transformers

print("python:", os.sys.version.split()[0])
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("qwen-vl-utils:", metadata.version("qwen-vl-utils"))
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("PyAV:", av.__version__)
print("pandas:", pandas.__version__)
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())

assert torch.version.cuda is not None, "Installed PyTorch is CPU-only"
assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
assert torch.cuda.device_count() >= 1, "No GPU is visible"
assert int(transformers.__version__.split(".", 1)[0]) == 4, "RBench requires Transformers 4.x"
assert int(numpy.__version__.split(".", 1)[0]) < 2, "RBench environment requires NumPy < 2"
PY

for task in \
    common_manipulation \
    long-horizon_planning \
    multi-entity_collaboration \
    spatial_relationship \
    visual_reasoning; do
    PYTHONNOUSERSITE=1 "$PY" -m py_compile "$SCORER_REPO/eval/5_tasks/$task.py"
done

uv pip freeze --python "$PY" > "$SCORER_REPO/rbench_scorer_environment.freeze.txt"
echo "Environment ready: $SCORER_ENV"
echo "Interpreter: $PY"
echo "Manifest: $SCORER_REPO/rbench_scorer_environment.freeze.txt"
