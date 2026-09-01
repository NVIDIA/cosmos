# Cosmos3 FP8 Quantization Examples

Convert the Cosmos3 checkpoints to static-scale **FP8** with NVIDIA TensorRT Model
Optimizer (ModelOpt). FP8 roughly halves a model's memory footprint and speeds up
inference on NVIDIA GPUs with FP8 tensor cores, while preserving generation quality.
The output is a drop-in diffusers checkpoint (FP8 weights + per-tensor `weight_scale` /
`input_scale` sidecars + `hf_quant_config.json`) that loads on vLLM-Omni.

Everything needed for the FP8 recipe lives in [`src/`](./src); the cookbook loads
Cosmos3 through the `cosmos-framework` inference API, and the notebooks are
thin walkthroughs on top of it. No other quantization pipeline is required.

## What's in this folder

| Path | Role |
| --- | --- |
| [`src/`](./src) | The complete FP8 recipe (framework load, calibrate, export). |
| [`notebooks/`](./notebooks) | Three customer-facing walkthroughs (one per model family). |

### The `src/` package

`import src` exposes a one-call API; the internals are split for readability:

| Module | Contents |
| --- | --- |
| `src/checkpoint_io.py` | Load through `OmniInference`; save sharded safetensors. |
| `src/calibration.py` | Calibration prompts, image conditioning, the skip filter, and the denoising `forward_loop` ModelOpt calibrates against. |
| `src/export.py` | Materialize the FP8 weights + scales into a vLLM-Omni diffusers checkpoint. |
| `src/__init__.py` | The public API: `Sampler` / `Shape` presets and `quantize_fp8_checkpoint(...)`. |

The whole pipeline for one checkpoint is a single call:

```python
import src
from src import quantize_fp8_checkpoint, SHAPE_VIDEO, SAMPLER_VIDEO_BASE

quantize_fp8_checkpoint(
    model_name_or_path="nvidia/Cosmos3-Nano",  # Hugging Face repo ID or local checkpoint
    output_dir="/path/to/nano-fp8",       # the FP8 drop-in written here
    profile="t2v", sampler=SAMPLER_VIDEO_BASE, shape=SHAPE_VIDEO,
    num_samples=8,
)
```

A base model and its distilled student are the *same* network and **differ only by the
`Sampler`** (scheduler class + steps + guidance). That is the whole idea behind the
Text-to-Image and Image-to-Video notebooks: the same call, twice, with a different sampler.

## Setup (once)

Use a Linux machine with an NVIDIA GPU and Hugging Face model access (`uvx hf@latest
auth login` or `HF_TOKEN`). Each notebook includes an install cell (**step 3**) that
builds the environment and registers a Jupyter kernel; to build it from the shell
instead:

```bash
cd cookbooks/cosmos3/quantization
export COSMOS3_QUANTIZE_VENV="$PWD/.venv-cosmos3-quantize"

uv venv "$COSMOS3_QUANTIZE_VENV" --python 3.13 --seed --managed-python
source "$COSMOS3_QUANTIZE_VENV/bin/activate"
uv pip install --torch-backend=cu130 \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  "nvidia-modelopt[torch]" \
  cosmos-framework \
  accelerate datasets huggingface_hub imageio imageio-ffmpeg \
  ipykernel jupyter-client nbconvert \
  numpy pillow safetensors torch torchvision transformers
python -m ipykernel install --user \
  --name cosmos3-quantize --display-name "Cosmos3 Quantize (Python 3.13)"
```

Then either open a notebook in Jupyter and run the cells top to bottom (switching to the
**Cosmos3 Quantize (Python 3.13)** kernel after step 3), or run it headless with the
commands below.

By default the notebooks run in **DEMO** mode — one calibration prompt at a small shape, so
a run takes minutes. Set `DEMO=0` for the shipped recipe (the production shape and 8
calibration prompts). See [Demo vs. full runs](#demo-vs-full-runs).

## Quantize Cosmos3-Nano and Cosmos3-Super

Convert the Nano (8B) and Super (32B) checkpoints. Same recipe, different checkpoint —
Nano first, then Super.

### Run

```bash
cd cookbooks/cosmos3/quantization
DEMO=1 "$COSMOS3_QUANTIZE_VENV/bin/jupyter" nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=cosmos3-quantize \
  --ExecutePreprocessor.timeout=7200 \
  notebooks/quantize_nano_super.ipynb
```

### Notebook walkthrough

[`notebooks/quantize_nano_super.ipynb`](./notebooks/quantize_nano_super.ipynb) walks
through prerequisites, environment setup, and then quantizes Nano and Super in separate
sections, printing a short summary of each FP8 checkpoint it writes.

## Quantize Cosmos3-Super Text-to-Image (base and distilled)

Quantize the full-quality Text-to-Image model, then its 4-step distilled student. The
two are the same network; only the sampler differs (50-step UniPC vs. 4-step
FlowMatchEuler, guidance off), and the notebook shows exactly that.

### Run

```bash
cd cookbooks/cosmos3/quantization
DEMO=1 "$COSMOS3_QUANTIZE_VENV/bin/jupyter" nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=cosmos3-quantize \
  --ExecutePreprocessor.timeout=7200 \
  notebooks/quantize_super_text2image.ipynb
```

### Notebook walkthrough

[`notebooks/quantize_super_text2image.ipynb`](./notebooks/quantize_super_text2image.ipynb)
quantizes the base model in one section, then the distilled model in a second section that
explains the sampler difference.

## Quantize Cosmos3-Super Image-to-Video (base and distilled)

Same as Text-to-Image, but the image-to-video regime: calibration conditions on a clean
first frame (a real image, VAE-encoded) held fixed while the rest of the clip denoises.
Provide the conditioning images through `I2V_COND_DIR` (a local folder); if unset, the
notebook falls back to a public calibration image dataset.

### Run

```bash
cd cookbooks/cosmos3/quantization
DEMO=1 I2V_COND_DIR=/path/to/cond_images \
  "$COSMOS3_QUANTIZE_VENV/bin/jupyter" nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=cosmos3-quantize \
  --ExecutePreprocessor.timeout=7200 \
  notebooks/quantize_super_image2video.ipynb
```

### Notebook walkthrough

[`notebooks/quantize_super_image2video.ipynb`](./notebooks/quantize_super_image2video.ipynb)
sets up conditioning images, then quantizes the base and distilled models in separate
sections.

## Demo vs. full runs

| | `DEMO=1` (default) | `DEMO=0` (shipped recipe) |
| --- | --- | --- |
| Calibration prompts | 1 | 8 |
| Shape | small (e.g. 480×720, 29 frames) | production (720×1280, up to 189 frames) |
| Purpose | quick, runnable end to end | reproduces the shipped FP8 scales |

The FP8 **weights** are identical either way — they are computed directly from the bf16
weights (`weight_scale = max|W| / 448`) and do not depend on calibration. DEMO only affects
the calibrated activation **scales** (fewer samples, smaller shape).

## What gets produced

Each run writes a drop-in checkpoint at `OUTPUT_ROOT/<name>-fp8/`:

```
<name>-fp8/
├── transformer/                 # FP8 weights + weight_scale/input_scale + quantization_config
│   ├── diffusion_pytorch_model-*.safetensors
│   ├── diffusion_pytorch_model.safetensors.index.json
│   └── config.json
├── hf_quant_config.json         # FP8 quantization_config at the root
├── model.safetensors.index.json # regenerated whole-model index
└── vae/ scheduler/ ...          # symlinked back to the source checkpoint
```

Point your serving stack at this directory.

## Inference

The FP8 output is a **drop-in checkpoint** — the same layout as the bf16 source, plus an
`hf_quant_config.json` and a `quantization_config` in `transformer/config.json`. A
Cosmos3-capable server detects the ModelOpt FP8 format automatically. Serve it from a **local
directory** — either the output produced above, or the published `fp8` revision downloaded
with `snapshot_download` (pass `--served-model-name` so clients see the usual model id):

```bash
# A) the local FP8 output produced above
vllm serve /path/to/OUTPUT_ROOT/nano-fp8 --served-model-name nvidia/Cosmos3-Nano ...

# B) download the published fp8 revision to a local dir, then serve that dir
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('nvidia/Cosmos3-Nano', revision='fp8', local_dir='cosmos3-nano-fp8')"
vllm serve cosmos3-nano-fp8 --served-model-name nvidia/Cosmos3-Nano ...
```

> The `--revision fp8` flag alone is **not** honored by the vLLM-Omni Cosmos3 pipeline (it
> loads the default revision), so the FP8 weights must be materialized to a local directory
> first.

**Runnable inference examples live in the other Cosmos3 cookbooks, not here.** Each
vLLM / vLLM-Omni notebook has a **Quantized Checkpoints** section that downloads the FP8
weights and serves them from a local directory:

| Cookbook | Notebook(s) |
| --- | --- |
| Generator · Audiovisual | [`run_with_vllm_omni.ipynb`](../generator/audiovisual/run_with_vllm_omni.ipynb) |
| Generator · Action | [`run_fd_with_vllm_omni.ipynb`](../generator/action/run_fd_with_vllm_omni.ipynb), [`run_id_with_vllm_omni.ipynb`](../generator/action/run_id_with_vllm_omni.ipynb) |
| Generator · Transfer | [`run_video_transfer_with_vllm_omni.ipynb`](../generator/transfer/run_video_transfer_with_vllm_omni.ipynb) |
| Reasoner | [`run_with_vllm.ipynb`](../reasoner/run_with_vllm.ipynb) |

See the shared [Cosmos3 cookbooks environment setup](../README.md) for server install and
launch details ([vLLM](../README.md#vllm), [vLLM-Omni](../README.md#vllm-omni)).
