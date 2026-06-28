# Omni3D 3D Detection Evaluation

Evaluate VLM models on Omni3D 3D object detection benchmarks (KITTI, nuScenes, sunrgbd, objectron, arkitscenes, hypersim). More detailed version: https://docs.google.com/document/d/1cZOvbgd10B2-vRGl57X-QlrJ4Y6KQoJUi6hYPuyJBfM/edit?usp=sharing

## Quick Start

### 1. Install Dependencies

#### Option A: Standard Installation (for API models or evaluation)

```bash
# Install evaluation dependencies (PyTorch3D + Detectron2)
bash vlmeval/dataset/Omni3D/install_patch.sh

# Or manually:
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.9+pt2.4.0cu121 detectron2==0.6+fd27788pt2.4.0cu121
pip install pycocotools fvcore iopath omegaconf hydra-core tabulate scipy

# Add 3rd_party to PYTHONPATH
export PYTHONPATH="${PWD}/vlmeval/dataset/Omni3D/3rd_party:${PYTHONPATH}"
```

#### Option B: vLLM Installation (for local model inference with vLLM acceleration)

If you want to use vLLM for faster local model inference (e.g., Qwen2-VL, Qwen3-VL), you need a separate environment because vLLM requires newer PyTorch versions that conflict with PyTorch3D/Detectron2.

```bash
# 1. Install vLLM environment (creates venv_vllm)
bash vlmeval/dataset/Omni3D/install_patch_vllm.sh

# 2. Activate vLLM environment for inference
source venv_vllm/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 3. Run inference with vLLM (see Section 2)
```

> **Important:** Use separate environments for inference and evaluation:
> - **Inference with vLLM**: Use `venv_vllm` (created by `install_patch_vllm.sh`)
> - **Evaluation**: Use your original environment with PyTorch3D/Detectron2 (from `install_patch.sh`)

### 2. Run Inference

#### Option A: Using API Models

```bash
# Using hosted API model (no local GPU needed)
DATASET="Omni3D_KITTI"
nproc=128
model="qwen3_30b_a3b"  # or "qwen3_235b_a22b", "qwen3_8b"
python run.py --data="$DATASET" --api-nproc=$nproc --model="$model"
```

#### Option B: Using Local Models with vLLM

```bash
# 1. Activate vLLM environment (if not already active)
source venv_vllm/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 2. Run inference with vLLM flag
DATASET="Omni3D_KITTI"
model="Qwen3-VL-8B-Instruct"  # or "Qwen3-VL-30B-A3B-Instruct", "Qwen3-VL-235B-A22B-Instruct"
python run.py --data="$DATASET" --model="$model" --use-vllm

# Example with other datasets:
# python run.py --data Omni3D_nuScenes --model Qwen3-VL-8B-Instruct --use-vllm
# python run.py --data Omni3D_ARKitScenes --model Qwen3-VL-30B-A3B-Instruct --use-vllm

# Also works with Qwen2-VL series:
# python run.py --data Omni3D_KITTI --model Qwen2-VL-7B-Instruct --use-vllm
```

**Alternative:** You can also enable vLLM in the model config ([vlmeval/config.py](../../config.py)) by adding `use_vllm=True`:
```python
"Qwen3-VL-8B-Instruct": partial(
    Qwen2VLChat,
    model_path="Qwen/Qwen3-VL-8B-Instruct",
    use_vllm=True,  # Add this
    min_pixels=1280 * 28 * 28,
    max_pixels=16384 * 28 * 28,
    use_custom_prompt=False,
    use_vllm=True,
),
```

**Supported datasets:** `Omni3D_KITTI`, `Omni3D_nuScenes`, `Omni3D_SUNRGBD`, `Omni3D_Objectron`, `Omni3D_ARKitScenes`, `Omni3D_Hypersim`

### 3. Evaluate Results

```bash
# If you used vLLM environment for inference, switch back to evaluation environment
deactivate  # if in venv_vllm
source venv/bin/activate  # your original environment with PyTorch3D/Detectron2
export PYTHONPATH="${PWD}/vlmeval/dataset/Omni3D/3rd_party:${PYTHONPATH}"

# Run official 3D IoU evaluation
python vlmeval/dataset/Omni3D/eval_omni3d.py \
  --result_file outputs/qwen3_30b_a3b/<timestamp>/qwen3_30b_a3b_Omni3D_KITTI.xlsx \
  --output_dir outputs/omni3d_eval \
  --dataset Omni3D_KITTI

# For local models with vLLM (Qwen3-VL example):
# python vlmeval/dataset/Omni3D/eval_omni3d.py \
#   --result_file outputs/Qwen3-VL-8B-Instruct/<timestamp>/Qwen3-VL-8B-Instruct_Omni3D_KITTI.xlsx \
#   --output_dir outputs/omni3d_eval \
#   --dataset Omni3D_KITTI
```

**Outputs:**
- Console: AP2D, AP3D, AR metrics per category
- `outputs/omni3d_eval/KITTI_test/omni_instances_results.json`: Predictions in COCO format

### 4. Visualize (Optional)
Set number of samples to be visualized via `--max_samples`.

```bash
python vlmeval/dataset/Omni3D/visualize_omni3d_results.py \
  --result_file outputs/qwen3_30b_a3b/<timestamp>/qwen3_30b_a3b_Omni3D_KITTI.xlsx \
  --output_dir outputs/visualizations \
  --dataset Omni3D_KITTI \
  --max_samples 10
```

**Example visualization:**

![3D Detection Example](assets/0_lamp_pred.png)

## Configuration

Place config at `vlmeval/dataset/Omni3D/configs/basic_gt.yaml`:

```yaml
dataset:
  raw_image_path: "/path/to/raw/images"
  val:
    kitti:
      pkl_path: "/path/to/KITTI_test.pkl"
      range: {begin: 0, end: -1, interval: 1}
```

## Output Format

Model predictions in JSON:
```json
[{"bbox_3d": [x, y, z, width, height, length, roll, pitch, yaw], "label": "car"}]
```

Where:
- `x,y,z`: Center in camera coords (meters)
- `width,height,length`: Object dimensions (meters)
- `roll,pitch,yaw`: Rotation angles (degrees)

## Troubleshooting

**Import errors:**
```bash
export PYTHONPATH="${PWD}/vlmeval/dataset/Omni3D/3rd_party:${PYTHONPATH}"
```

**Low AP scores (<10%):**
- Normal for rare/specific domains
- Check visualizations to verify predictions are reasonable

**No predictions:**
- Verify API key is set
- Check `outputs/<model>/status.json` for errors
