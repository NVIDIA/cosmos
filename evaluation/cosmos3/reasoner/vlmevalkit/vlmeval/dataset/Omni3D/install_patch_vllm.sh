#!/usr/bin/env bash
set -e

##############################################
# Setup venv_vllm for vLLM-accelerated inference
# This venv does NOT have detectron2/pytorch3d
# Use this for fast inference only
##############################################

cd "$(dirname "$0")"
cd ../../..  # Go to project root

PROJECT_ROOT="$PWD"
echo "[INFO] Project root: $PROJECT_ROOT"

##############################################
# 1. Create venv_vllm
##############################################
if [ -d "venv_vllm" ]; then
    echo "[WARNING] venv_vllm already exists, skipping creation"
else
    echo "[INFO] Creating venv_vllm..."
    python3.11 -m venv venv_vllm
fi

##############################################
# 2. Activate venv_vllm
##############################################
source venv_vllm/bin/activate
echo "[INFO] Activated venv_vllm: $(which python)"

##############################################
# 3. Install PyTorch 2.4.1 (compatible with vLLM)
##############################################
echo "[INFO] Installing PyTorch 2.4.1 + cu121..."
pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

##############################################
# 4. Install vlmeval package structure
##############################################
echo "[INFO] Installing vlmeval package (without deps)..."
pip install -e . --no-deps

##############################################
# 5. Install core dependencies
##############################################
echo "[INFO] Installing core dependencies..."
pip install transformers pillow pandas openpyxl pyyaml "smart-open[s3]" boto3

##############################################
# 6. Install all vlmeval dependencies
##############################################
echo "[INFO] Installing vlmeval dependencies..."
pip install validators datasets imageio loguru matplotlib nltk openai \
    opencv-python portalocker protobuf python-box python-dotenv \
    qwen-vl-utils rich s3fs sty tabulate timeout-decorator xlsxwriter \
    ipdb omegaconf decord

##############################################
# 7. Install vLLM (0.13.0 supports Qwen3-VL)
##############################################
echo "[INFO] Installing vLLM..."
pip install vllm

##############################################
# 8. Verify installation
##############################################
echo "[INFO] Verifying installation..."
python - << 'EOF'
import torch
import vllm
import transformers

print("[CHECK] PyTorch:", torch.__version__)
print("[CHECK] CUDA available:", torch.cuda.is_available())
print("[CHECK] vLLM version:", vllm.__version__)
print("[CHECK] Transformers:", transformers.__version__)

# Verify Qwen3-VL support in vLLM
import importlib
try:
    m = importlib.import_module("vllm.model_executor.models.qwen3_vl")
    print("[CHECK] ✓ Qwen3-VL supported in vLLM")
except ImportError:
    print("[CHECK] ✗ Qwen3-VL NOT supported")

# Check if detectron2 is NOT installed (should fail in venv_vllm)
try:
    import detectron2
    print("[WARNING] detectron2 found - this should be inference-only venv!")
except ImportError:
    print("[CHECK] ✓ detectron2 not installed (inference-only mode)")
EOF

##############################################
# 9. Export environment variables
##############################################
export LMUData=${LMUData:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_compgenai/users/sifeil/LMUData}
export AWS_PROFILE=team-cosmos
export AWS_ENDPOINT_URL=https://pdx.s8k.io

echo ""
echo "=================================================="
echo "venv_vllm Setup Complete!"
echo "=================================================="
echo ""
echo "To use this environment for inference:"
echo ""
echo "  source venv_vllm/bin/activate"
echo "  export VLLM_WORKER_MULTIPROC_METHOD=spawn"
echo "  python run.py --data Omni3D_ARKitScenes --model Qwen3-VL-8B-Instruct"
echo ""
echo "For evaluation, switch to original venv:"
echo ""
echo "  deactivate"
echo "  source venv/bin/activate"
echo "  python vlmeval/dataset/omni3D/eval_omni3d.py \\"
echo "    --result_file outputs/Qwen3-VL-8B-Instruct/<timestamp>/*.xlsx \\"
echo "    --output_dir outputs/omni3d_eval \\"
echo "    --dataset Omni3D_ARKitScenes"
echo ""
echo "[SUCCESS] Installation complete!"
