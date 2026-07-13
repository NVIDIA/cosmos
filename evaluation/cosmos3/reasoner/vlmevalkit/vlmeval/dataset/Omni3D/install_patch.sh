#!/usr/bin/env bash
set -e

##############################################
# 0. Move to project root
##############################################
cd "$(dirname "$0")"

PROJECT_ROOT="$PWD"
OMNI3D_PATH="$PROJECT_ROOT/vlmeval/dataset/Omni3D/3rd_party"
CUBERCNN_PATH="$OMNI3D_PATH/cubercnn"

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Omni3D path:  $OMNI3D_PATH"

# ##############################################
# 1. Need to be in the (venv) already!!
# ##############################################

##############################################
# 2. System dependencies (fix for cv2/libGL)
##############################################
echo "[INFO] Installing system packages (ffmpeg, libsm6, libxext6)..."

if [ "$(id -u)" -eq 0 ]; then
  apt-get update
  apt-get install -y ffmpeg libsm6 libxext6
else
  sudo apt-get update
  sudo apt-get install -y ffmpeg libsm6 libxext6
fi

##############################################
# 3. Install PyTorch (CUDA 12.1)
##############################################
echo "[INFO] Installing PyTorch (2.4.0 + cu121)..."
pip install --upgrade pip
pip install \
  "torch==2.4.0+cu121" \
  "torchvision==0.19.0+cu121" \
  --index-url https://download.pytorch.org/whl/cu121

##############################################
# 4. Install PyTorch3D + Detectron2 (matching build)
##############################################
echo "[INFO] Installing PyTorch3D + Detectron2 (CUDA)..."
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
  "pytorch3d==0.7.9+pt2.4.0cu121" \
  "detectron2==0.6+fd27788pt2.4.0cu121"

##############################################
# 5. Other Python dependencies
##############################################
echo "[INFO] Installing additional Python packages..."
pip install pycocotools fvcore iopath omegaconf hydra-core tabulate scipy
pip install "smart-open[s3]" boto3  # For S3 support

##############################################
# 6. Patch ~/.bashrc (tclsh + Omni3D PYTHONPATH)
##############################################
# echo "[INFO] Updating ~/.bashrc (tclsh + PYTHONPATH)..."

# BASHRC="$HOME/.bashrc"
# touch "$BASHRC"

# # 6.1 Comment out any /usr/bin/tclsh lines to avoid
# #     "bash: /usr/bin/tclsh: No such file or directory"
# if grep -q "/usr/bin/tclsh" "$BASHRC"; then
#   echo "[INFO] Found /usr/bin/tclsh in ~/.bashrc, commenting it out..."
#   sed -i 's|^\(.*\/usr/bin/tclsh.*\)$|# \1  # disabled by install.sh|' "$BASHRC"
# fi

# # 6.2 Remove existing Omni3D PYTHONPATH lines (avoid duplicates)
# sed -i "\|$OMNI3D_PATH|d" "$BASHRC"

# # 6.3 Add new Omni3D PYTHONPATH entry
# echo "export PYTHONPATH=\"$OMNI3D_PATH:\$PYTHONPATH\"" >> "$BASHRC"

# # 6.4 Source ~/.bashrc for current shell (now without tclsh error)
# # shellcheck disable=SC1090
# source "$BASHRC"
export PYTHONPATH=\"$OMNI3D_PATH:\$PYTHONPATH\"
export PYTHONPATH=\"$CUBERCNN_PATH:\$PYTHONPATH\"

##############################################
# 7. Final check
##############################################
echo "[INFO] Verifying installation..."

python - << 'EOF'
import torch, pytorch3d, detectron2
from pytorch3d import _C

print("[CHECK] torch:", torch.__version__, "cuda:", torch.version.cuda)
print("[CHECK] CUDA available:", torch.cuda.is_available())
print("[CHECK] pytorch3d OK")
print("[CHECK] detectron2 OK")
EOF

##############################################
# 8. AWS Configuration Instructions
##############################################
echo ""
echo "=================================================="
echo "AWS S3 Configuration Required:"
echo "=================================================="
echo "Set these environment variables before running:"
echo ""
echo "  export AWS_PROFILE=team-cosmos"
echo "  export AWS_ENDPOINT_URL=https://pdx.s8k.io"
echo ""
echo "Or add to your ~/.bashrc for persistence:"
echo ""
echo "  echo 'export AWS_PROFILE=team-cosmos' >> ~/.bashrc"
echo "  echo 'export AWS_ENDPOINT_URL=https://pdx.s8k.io' >> ~/.bashrc"
echo "  source ~/.bashrc"
echo ""
echo "[SUCCESS] Installation complete!"
