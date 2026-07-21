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
- **Multi-control** — two or more hints combined with per-hint weights.

vLLM-Omni does not expose transfer controls today.

Environment setup is centralized in the shared
[Cosmos3 cookbooks environment setup](../../README.md) guide.

## Transfer Definition

Video transfer generates a target clip from a `prompt.json` caption and one or more
spatial control signals. Inference uses `model_mode` `video2video`. Control signals can
be supplied as pre-computed videos (`control_path`) or derived on-the-fly from a raw
source video (`vision_path`). Output frame count and geometry come from the control
video; see the spec field reference for how `fps` and `aspect_ratio` are resolved.
All examples share `assets/negative_prompt.json` for the negative caption.

| Control | Asset folder | Inference input | Generation duration |
| --- | --- | --- | --- |
| Edge (Canny) | `assets/edge/` | `control_edge.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| Blur | `assets/blur/` | `control_blur.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| Depth | `assets/depth/` | `control_depth.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| Segmentation | `assets/seg/` | `control_seg.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| World scenario (WSM) | `assets/wsm/` | `control_wsm.mp4` + `prompt.json` | 101 frames @ 10 FPS |
| Multi-control | `assets/multi_control/` | `vision_path` + multiple hints | 121 frames @ 30 FPS |

Transfer inference is selected automatically when any hint key is present in the spec.
The same spec files are used for both Nano and Super — model selection is controlled
entirely by `--checkpoint-path`.

## Run with Cosmos Framework

### Quickstart — Single-control transfer

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
clone, Python venv via `uv`), authenticates with Hugging Face, and runs all six controls
with previews.

1. Open the notebook and edit **§2 (Configure)** — paste your `HF_TOKEN` and optionally
   set cache/output paths.
2. Run **§9–§13** for Cosmos3-Nano single-control (single GPU), **§14–§18** for Cosmos3-Super
   single-control (multi-GPU), or **§19** for multi-control (Nano).
   No model flag needed — each section uses its matching checkpoint explicitly.

To execute headlessly:

```bash
cd cookbooks/cosmos3/generator/transfer
jupyter execute run_video_transfer_with_cosmos_framework.ipynb
```

Outputs land under `outputs/notebooks/<model>/transfer_<control>/vision.mp4`.

---

## Multi-Control Transfer

Multi-control transfer blends two or more spatial hint streams — for example edge + depth
— into a single generation pass. Each active hint receives a `weight` that determines its
relative influence. Weights across all active hints should sum to 1.0 for predictable
behavior, though the model accepts any positive values.

### Concepts

| Field | Description |
| --- | --- |
| `edge` / `blur` / `depth` / `seg` | Hint block; set any subset to activate those controls |
| `weight` | Per-hint blending weight; ratios matter, not absolute values (default `1.0`) |
| `control_path` | Path to a **pre-computed** control video (optional; see below) |
| `vision_path` | Raw source video; the framework derives all active controls on-the-fly |
| `control_guidance` | Global CFG scale across all active control streams (default `1.5`) |

**Two input modes:**

1. **`vision_path` mode** — Provide a raw source video. For `edge` and `blur` hints
   with no `control_path`, the framework computes the control signal on-the-fly
   (Canny for `edge`, downscale/upscale for `blur`). `depth` and `seg` always require
   a pre-computed `control_path` — they depend on DepthAnything and SAM2 which are
   not bundled in the cosmos-framework.

2. **Pre-computed mode** — Provide a `control_path` inside each hint block. All
   control videos must be derived from the **same source video** and share identical
   resolution, fps, and frame count. Use this when you want exact control over the
   pre-processed signals or to reuse cached extractions across runs.

### Quickstart — Multi-control from a source video

Uses the same env vars as the single-control quickstart (`COSMOS_FRAMEWORK` and `TRANSFER_ROOT`).

#### Cosmos3-Nano (single GPU)

```bash
cd "$COSMOS_FRAMEWORK"

# edge + blur computed on-the-fly from vision_path (robot_pouring.mp4)
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$TRANSFER_ROOT/specs/multi_control.json" \
  -o "$TRANSFER_ROOT/outputs/Cosmos3-Nano/" \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026
```

Output lands at `outputs/Cosmos3-Nano/transfer_multi_control/vision.mp4`.

#### Cosmos3-Super (multi-GPU)

```bash
cd "$COSMOS_FRAMEWORK"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
.venv/bin/torchrun --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29500 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$TRANSFER_ROOT/specs/multi_control.json" \
  -o "$TRANSFER_ROOT/outputs/Cosmos3-Super/" \
  --checkpoint-path Cosmos3-Super \
  --seed 2026
```

Output lands at `outputs/Cosmos3-Super/transfer_multi_control/vision.mp4`.

### Spec field reference — multi-control

**`specs/multi_control.json`** (edge dominant + blur secondary, controls derived on-the-fly from `vision_path`):

```json
{
  "name": "transfer_multi_control",
  "model_mode": "video2video",
  "resolution": "720",
  "aspect_ratio": "16,9",
  "num_frames": 121,
  "fps": 30,
  "guidance": 3.0,
  "control_guidance": 1.5,
  "negative_prompt_file": "../assets/negative_prompt.json",
  "prompt_path": "../assets/multi_control/prompt.json",
  "vision_path": "https://.../robot_pouring.mp4",
  "edge": {
    "weight": 0.75,
    "preset_edge_threshold": "medium"
  },
  "blur": {
    "weight": 0.25,
    "preset_blur_strength": "medium"
  }
}
```

Only the **ratio** between weights matters — `edge: 3, blur: 1` is equivalent to
`edge: 0.75, blur: 0.25`. Omitting `weight` defaults to `1.0` (equal contribution).

To use **pre-computed control videos** instead, replace `vision_path` with `control_path`
inside each hint block. All control videos must be derived from the same source and
share identical resolution, fps, and frame count:

```json
{
  "edge": {
    "control_path": "/path/to/control_edge.mp4",
    "weight": 0.75,
    "preset_edge_threshold": "medium"
  },
  "blur": {
    "control_path": "/path/to/control_blur.mp4",
    "weight": 0.25,
    "preset_blur_strength": "medium"
  }
}
```

`control_guidance` scales the influence of all active control streams collectively;
`weight` distributes that influence among individual hints.

---

### Spec field reference — single-control

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
  self-contained notebook: §9–§13 Nano single-control, §14–§18 Super single-control, §19 multi-control (Nano). Edit §2, run top-to-bottom.
- [`specs/`](./specs) — checked-in Framework input JSON per control (paths relative to `specs/`).
  Shared by both Nano and Super.

### Troubleshooting

If inference fails inside attention with a NATTEN/libnatten error, verify that the active Python
environment uses a matching Torch and NATTEN build. Avoid mixing a container-provided Torch/NATTEN
stack with packages from `~/.local` or another venv. In containerized environments,
`PYTHONNOUSERSITE=1` can help prevent user-site packages from shadowing the container stack.
