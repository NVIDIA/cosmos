#!/usr/bin/env bash
# Install the low-level operators used by RBench's 4-embodiment evaluation.
# Keep this environment separate from .venv-rbench-scorer: Q-Align requires an
# older Transformers/PEFT stack, while GroundingDINO must compile against CUDA.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REVIDGEN_REPO=${REVIDGEN_REPO:-$SCRIPT_DIR/scorers/ReVidgen}
OPS_ENV=${RBENCH_OPS_ENV:-$REVIDGEN_REPO/.venv-rbench-ops}
PYTHON_VERSION=${RBENCH_OPS_PYTHON:-3.10}
TORCH_INDEX_URL=${RBENCH_OPS_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}
TORCH_VERSION=${RBENCH_OPS_TORCH_VERSION:-2.9.1}
TORCHVISION_VERSION=${RBENCH_OPS_TORCHVISION_VERSION:-0.24.1}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
TORCH_CUDA_ARCH_LIST=${RBENCH_OPS_CUDA_ARCH_LIST:-9.0}
MAX_JOBS=${MAX_JOBS:-16}
DOWNLOAD_CHECKPOINTS=${RBENCH_DOWNLOAD_CHECKPOINTS:-1}
INSTALL_LOG=${RBENCH_OPS_INSTALL_LOG:-$SCRIPT_DIR/rbench_ops_install.log}

export CUDA_HOME TORCH_CUDA_ARCH_LIST MAX_JOBS
exec > >(tee -a "$INSTALL_LOG") 2>&1

echo "RBench 4-embodiment operator installation"
echo "  repository:     $REVIDGEN_REPO"
echo "  environment:    $OPS_ENV"
echo "  Python:         $PYTHON_VERSION"
echo "  Torch index:    $TORCH_INDEX_URL"
echo "  CUDA_HOME:      $CUDA_HOME"
echo "  CUDA arch list: $TORCH_CUDA_ARCH_LIST"
echo "  checkpoints:    $DOWNLOAD_CHECKPOINTS"
echo "  log:            $INSTALL_LOG"

command -v uv >/dev/null || { echo "ERROR: uv is required"; exit 1; }
command -v git >/dev/null || { echo "ERROR: git is required"; exit 1; }
command -v g++ >/dev/null || { echo "ERROR: g++ is required"; exit 1; }
[[ -x "$CUDA_HOME/bin/nvcc" ]] || {
    echo "ERROR: nvcc is missing at $CUDA_HOME/bin/nvcc"
    exit 1
}
[[ -d "$REVIDGEN_REPO/eval/4_embodiments" ]] || {
    echo "ERROR: ReVidgen 4-embodiment sources are missing: $REVIDGEN_REPO"
    echo "Run setup_rbench_scorer.sh first."
    exit 1
}

unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV CONDA_PREFIX

if [[ ! -x "$OPS_ENV/bin/python" ]]; then
    uv venv --python "$PYTHON_VERSION" "$OPS_ENV"
fi
PY="$OPS_ENV/bin/python"

# The active H100 container exposes nvcc 13.0, so use a matching official
# PyTorch build. A cu12 wheel cannot safely build GroundingDINO's CUDA module
# with that compiler.
uv pip install \
    --python "$PY" \
    --index "$TORCH_INDEX_URL" \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION"

# Install runtime and build dependencies explicitly. Vendored projects are
# installed with --no-deps below so they cannot replace the selected Torch or
# pull GUI OpenCV into a headless compute environment.
uv pip uninstall --python "$PY" bitsandbytes opencv-python || true
uv pip install --python "$PY" \
    "setuptools" \
    "wheel" \
    "ninja" \
    "packaging" \
    "numpy==1.26.4" \
    "protobuf==3.20.1" \
    "tqdm" \
    "pillow" \
    "opencv-python-headless==4.11.0.86" \
    "matplotlib" \
    "pandas" \
    "scipy" \
    "scikit-learn==1.2.2" \
    "imageio[ffmpeg]" \
    "decord" \
    "addict" \
    "yapf==0.40.1" \
    "timm==0.6.13" \
    "supervision<1" \
    "pycocotools" \
    "hydra-core>=1.3.2" \
    "iopath>=0.1.10" \
    "transformers==4.36.1" \
    "tokenizers==0.15.0" \
    "huggingface-hub>=0.19.3,<1" \
    "sentencepiece==0.1.99" \
    "shortuuid" \
    "accelerate==0.21.0" \
    "peft==0.4.0" \
    "pydantic>=1,<2" \
    "markdown2[all]" \
    "requests" \
    "httpx==0.24.0" \
    "uvicorn" \
    "fastapi<0.100" \
    "icecream" \
    "einops==0.6.1" \
    "einops-exts==0.0.4"

SEGMENT_ANYTHING="$REVIDGEN_REPO/pkgs/Grounded-Segment-Anything/segment_anything"
GROUNDING_DINO="$REVIDGEN_REPO/pkgs/Grounded-Segment-Anything/GroundingDINO"
GROUNDED_SAM2="$REVIDGEN_REPO/pkgs/Grounded-SAM-2"
Q_ALIGN="$REVIDGEN_REPO/pkgs/Q-Align"
CO_TRACKER="$REVIDGEN_REPO/pkgs/co-tracker"

for package_dir in \
    "$SEGMENT_ANYTHING" \
    "$GROUNDING_DINO" \
    "$GROUNDED_SAM2" \
    "$Q_ALIGN" \
    "$CO_TRACKER"; do
    [[ -d "$package_dir" ]] || {
        echo "ERROR: missing vendored package: $package_dir"
        exit 1
    }
done

# ReVidgen vendors a GroundingDINO kernel written for older PyTorch releases.
# PyTorch 2.9 removed the implicit DeprecatedTypeProperties -> ScalarType
# conversion used by AT_DISPATCH_FLOATING_TYPES. This two-call update is the
# upstream-compatible spelling and is safe to apply repeatedly.
GDINO_CUDA_SOURCE="$GROUNDING_DINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu"
if grep -q 'AT_DISPATCH_FLOATING_TYPES(value.type()' "$GDINO_CUDA_SOURCE"; then
    sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' "$GDINO_CUDA_SOURCE"
fi

uv pip install --python "$PY" --no-deps --editable "$SEGMENT_ANYTHING"
uv pip install --python "$PY" --no-build-isolation --no-deps --editable "$GROUNDING_DINO"
uv pip install --python "$PY" --no-build-isolation --no-deps --editable "$GROUNDED_SAM2"
uv pip install --python "$PY" --no-deps --editable "$Q_ALIGN"
uv pip install --python "$PY" --no-deps --editable "$CO_TRACKER"

# Some transitive metadata still requests GUI OpenCV. Ensure the final
# environment exposes only the headless wheel.
uv pip uninstall --python "$PY" bitsandbytes opencv-python || true
uv pip install --python "$PY" --reinstall-package opencv-python-headless "opencv-python-headless==4.11.0.86"

if [[ "$DOWNLOAD_CHECKPOINTS" == "1" ]]; then
    echo "Downloading RBench operator checkpoints (approximately 21.8 GB)"
    REVIDGEN_REPO="$REVIDGEN_REPO" PYTHONNOUSERSITE=1 "$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="DAGroup-PKU/RBench",
    repo_type="dataset",
    allow_patterns=["checkpoints/**"],
    local_dir=os.environ["REVIDGEN_REPO"],
    token=os.environ.get("HF_TOKEN"),
)
snapshot_download(
    repo_id="DAGroup-PKU/RBench",
    repo_type="dataset",
    allow_patterns=["prompts/*_prompts.json"],
    local_dir=os.path.join(os.environ["REVIDGEN_REPO"], "data"),
    token=os.environ.get("HF_TOKEN"),
)
PY
fi

echo "Validating RBench embodiment operator environment"
REVIDGEN_REPO="$REVIDGEN_REPO" PYTHONNOUSERSITE=1 "$PY" - <<'PY'
import glob
import os
import sys

repo = os.environ["REVIDGEN_REPO"]
sys.path.insert(0, os.path.join(repo, "pkgs", "Grounded-Segment-Anything"))
sys.path.insert(0, os.path.join(repo, "pkgs", "Grounded-Segment-Anything", "GroundingDINO"))
sys.path.insert(0, os.path.join(repo, "pkgs", "co-tracker"))
sys.path.insert(0, os.path.join(repo, "pkgs", "sam2"))

import torch
from GroundingDINO.groundingdino import _C
from GroundingDINO.groundingdino.models import build_model
from cotracker.predictor import CoTrackerPredictor
from q_align.model.builder import load_pretrained_model
from sam2.build_sam import build_sam2

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("GroundingDINO extension:", _C.__file__)
print("SAM2 import:", build_sam2.__module__)
print("CoTracker import:", CoTrackerPredictor.__module__)
print("Q-Align import:", load_pretrained_model.__module__)

assert torch.version.cuda and torch.version.cuda.startswith("13."), torch.version.cuda
assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
assert glob.glob(os.path.join(repo, "pkgs", "Grounded-Segment-Anything", "GroundingDINO", "groundingdino", "_C*.so")), "GroundingDINO _C extension is absent"
PY

if [[ "$DOWNLOAD_CHECKPOINTS" == "1" ]]; then
    for checkpoint in \
        data/prompts/dual_arm_prompts.json \
        data/prompts/humanoid_prompts.json \
        data/prompts/single_arm_prompts.json \
        data/prompts/quad_prompts.json \
        checkpoints/BERT/google-bert/bert-base-uncased/config.json \
        checkpoints/GroundingDino/groundingdino_swinb_cogcoor.pth \
        checkpoints/SAM/sam2.1_hiera_large.pt \
        checkpoints/Cotracker/scaled_offline.pth \
        checkpoints/q-future/one-align/config.json; do
        [[ -e "$REVIDGEN_REPO/$checkpoint" ]] || {
            echo "ERROR: missing checkpoint artifact: $checkpoint"
            exit 1
        }
    done
fi

for script in "$REVIDGEN_REPO"/eval/4_embodiments/*.py; do
    PYTHONNOUSERSITE=1 "$PY" -m py_compile "$script"
done

uv pip freeze --python "$PY" > "$REVIDGEN_REPO/rbench_ops_environment.freeze.txt"
echo "Environment ready: $OPS_ENV"
echo "Interpreter: $PY"
echo "Manifest: $REVIDGEN_REPO/rbench_ops_environment.freeze.txt"
