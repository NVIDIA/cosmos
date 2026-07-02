# Cosmos3 Generator Transfer Examples

Cosmos3 video **transfer** examples — **Nano** (single GPU) and **Super** (multi-GPU, 32B) — on
the native PyTorch (Cosmos Framework) path.
Sample assets under [`assets/`](./assets) cover spatial control signals paired with
`prompt.json` files:

- **Edge (Canny)** — edge map control plus caption.
- **Blur** — blurred-reference control plus caption.
- **Depth** — depth map control plus caption.
- **Segmentation** — segmentation map control plus caption.
- **World scenario (WSM)** — world-scenario map control plus caption.

vLLM-Omni does not expose transfer controls today.

Environment setup is centralized in the shared
[Cosmos3 cookbooks environment setup](../../README.md) guide.

## Transfer Definition

Video transfer generates a target clip from a `prompt.json` caption and a precomputed
control video on the hint block (`control_path`). Inference uses `model_mode` `video2video`;
there is no `vision_path` or source RGB video at run time. Output frame count and geometry
come from the control video; see the spec field reference for how `fps` and
`aspect_ratio` are resolved. All examples share
`assets/negative_prompt.json` for the negative caption.

| Control | Asset folder | Inference input | Generation duration |
| --- | --- | --- | --- |
| Edge (Canny) | `assets/edge/` | `control_edge.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| Blur | `assets/blur/` | `control_blur.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| Depth | `assets/depth/` | `control_depth.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| Segmentation | `assets/seg/` | `control_seg.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| World scenario (WSM) | `assets/wsm/` | `control_wsm.mp4` + `prompt.json` | 101 frames @ 10 FPS |

Transfer inference is selected automatically when any hint key is present in the spec.
The same spec files are used for both Nano and Super — model selection is controlled
entirely by `--checkpoint-path`.

## Run with Cosmos Framework

### Quickstart

Set up the environment: [Cosmos Framework setup](../../README.md#cosmos-framework).
Run the commands below inside the **cosmos container** (e.g. `pytorch:25.09-py3`) — the same
environment used to install the venv and run the notebook. The commands mirror the notebook
exactly: `cd` into the framework repo first, then invoke the venv's Python or torchrun
(the system Python does not have `cosmos_framework`).

```bash
# Set once — the cosmos-framework repo root (contains .venv/ and pyproject.toml).
# In this cosmos checkout: packages/cosmos3 (or packages/cosmos-framework).
export COSMOS_FRAMEWORK=/path/to/cosmos-framework   # e.g. <cosmos_root>/packages/cosmos3
export TRANSFER_ROOT=$(pwd)/cookbooks/cosmos3/generator/transfer

# NGC containers bundle libtorch in LD_LIBRARY_PATH which conflicts with Triton/CUDA.
unset LD_LIBRARY_PATH
```

#### Cosmos3-Nano (single GPU)

```bash
cd "$COSMOS_FRAMEWORK"

# edge — replace edge.json with blur.json / depth.json / seg.json / wsm.json for other controls
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$TRANSFER_ROOT/specs/edge.json" \
  -o "$TRANSFER_ROOT/outputs/Cosmos3-Nano/" \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026
```

#### Cosmos3-Super (multi-GPU)

```bash
cd "$COSMOS_FRAMEWORK"

# edge — replace edge.json with other control specs as needed
CUDA_VISIBLE_DEVICES=0,1,2,3 \
.venv/bin/torchrun --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29500 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$TRANSFER_ROOT/specs/edge.json" \
  -o "$TRANSFER_ROOT/outputs/Cosmos3-Super/" \
  --checkpoint-path Cosmos3-Super \
  --seed 2026
```

| | Cosmos3-Nano | Cosmos3-Super |
|---|---|---|
| `--checkpoint-path` | `Cosmos3-Nano` | `Cosmos3-Super` |
| Launcher | `.venv/bin/python` (from framework root) | `.venv/bin/torchrun --nproc-per-node=<N>` (from framework root) |
| `--parallelism-preset` | `latency` | `latency` |
| GPUs | 1 | 4+ |

The input spec sets `prompt_path` and a hint block with `control_path` pointing at the
checked-in assets under [`assets/`](./assets) via paths relative to [`specs/`](./specs).

Outputs are written under the directory passed to `-o`, with one subdirectory per sample
name, e.g. `outputs/Cosmos3-Nano/transfer_edge/vision.mp4`.

### Notebook (self-contained)

[`run_video_transfer_with_cosmos_framework.ipynb`](./run_video_transfer_with_cosmos_framework.ipynb)
is a self-contained tutorial: it installs all dependencies (system packages, framework
clone, Python venv via `uv`), authenticates with Hugging Face, and runs all five controls
with previews.

1. Open the notebook and edit **§2 (Configure)** — paste your `HF_TOKEN` and optionally
   set cache/output paths.
2. Run **§9–§13** for Cosmos3-Nano (single GPU) or **§14–§18** for Cosmos3-Super (multi-GPU).
   No model flag needed — each section uses its matching checkpoint explicitly.

To execute headlessly:

```bash
cd cookbooks/cosmos3/generator/transfer
jupyter execute run_video_transfer_with_cosmos_framework.ipynb
```

Outputs land under `outputs/notebooks/<model>/transfer_<control>/vision.mp4`.

### Spec field reference

A representative spec (`specs/edge.json`):

```json
{
  "name": "transfer_edge",
  "model_mode": "video2video",
  "resolution": "720",
  "aspect_ratio": "16,9",
  "num_frames": 121,
  "fps": 30,
  "num_video_frames_per_chunk": 121,
  "num_conditional_frames": 1,
  "num_first_chunk_conditional_frames": 0,
  "share_vision_temporal_positions": true,
  "guidance": 3.0,
  "control_guidance": 1.5,
  "negative_prompt_file": "../assets/negative_prompt.json",
  "prompt_path": "../assets/edge/prompt.json",
  "edge": {
    "control_path": "../assets/edge/control_edge.mp4",
    "preset_edge_threshold": "medium"
  }
}
```

Key fields:

- **`resolution`** — target resolution (e.g. `720` for 720p).

- **`aspect_ratio`** — aspect ratio of the control video; together with `resolution` determines the spatial dimensions (e.g. `720` + `16,9` → 1280 × 720).

- **`fps`** — model conditioning signal and playback rate of the saved output video. Should match the native fps of the control video.

- **`num_frames`** — number of video frames.

### Cookbook entrypoints

- [`run_video_transfer_with_cosmos_framework.ipynb`](./run_video_transfer_with_cosmos_framework.ipynb) —
  self-contained notebook: §9–§13 Nano (single GPU), §14–§18 Super (multi-GPU). Edit §2, run top-to-bottom.
- [`specs/`](./specs) — checked-in Framework input JSON per control (paths relative to `specs/`).
  Shared by both Nano and Super.

### Troubleshooting

If inference fails inside attention with a NATTEN/libnatten error, verify that the active Python
environment uses a matching Torch and NATTEN build. Avoid mixing a container-provided Torch/NATTEN
stack with packages from `~/.local` or another venv. In containerized environments,
`PYTHONNOUSERSITE=1` can help prevent user-site packages from shadowing the container stack.
