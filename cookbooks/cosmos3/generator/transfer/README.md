# Cosmos3 Generator Transfer Cookbooks

Cosmos3-Nano video **transfer** cookbooks on the native PyTorch (Cosmos Framework) path.

## Basic Examples

The [`basic_examples/`](./basic_examples/) directory contains the shipped starter
cookbook and sample assets. Community-contributed cookbooks are added as sibling
directories alongside `basic_examples/` — see the
[Contributing Guide](../../../../CONTRIBUTING.md) for the recipe structure.

| Cookbook | Backend | Notebook |
|---------|---------|----------|
| Video transfer (edge, blur, depth, seg, wsm) | Cosmos Framework | [`basic_examples/run_video_transfer_with_cosmos_framework.ipynb`](./basic_examples/run_video_transfer_with_cosmos_framework.ipynb) |

Sample assets under [`basic_examples/assets/`](./basic_examples/assets/) cover spatial control signals paired with
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
| Edge (Canny) | `basic_examples/assets/edge/` | `control_edge.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| Blur | `basic_examples/assets/blur/` | `control_blur.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| Depth | `basic_examples/assets/depth/` | `control_depth.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| Segmentation | `basic_examples/assets/seg/` | `control_seg.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| World scenario (WSM) | `basic_examples/assets/wsm/` | `control_wsm.mp4` + `prompt.json` | 101 frames @ 10 FPS |

Transfer inference is selected automatically when any hint key is present in the spec.

## Run with Cosmos Framework

### Quickstart

Set up the environment: [Cosmos Framework setup](../../README.md#cosmos-framework).
Activate the framework venv, then run inference (checked-in `specs/*.json` use paths
relative to `basic_examples/specs/`). Transfer on Nano looks like:

```bash
cd cookbooks/cosmos3/generator/transfer

# edge
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i basic_examples/specs/edge.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026

# blur
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i basic_examples/specs/blur.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026

# depth
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i basic_examples/specs/depth.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026

# seg
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i basic_examples/specs/seg.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026

# wsm
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i basic_examples/specs/wsm.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026
```

The input spec sets `prompt_path` and a hint block with `control_path` pointing at the
checked-in assets under [`basic_examples/assets/`](./basic_examples/assets/) via paths relative to [`basic_examples/specs/`](./basic_examples/specs/).

Outputs are written under the directory passed to `-o`, with one subdirectory per sample name,
for example `output/transfer_edge/vision.mp4`. Batch size must be 1 for transfer.

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

- [`run_video_transfer_with_cosmos_framework.ipynb`](./basic_examples/run_video_transfer_with_cosmos_framework.ipynb) —
  full tutorial on a **GPU host**: environment setup, `nvidia-smi` check, then five inference blocks
  (edge, blur, depth, seg, wsm) with previews. See [Cosmos3 environment setup](../../README.md).
- [`basic_examples/specs/`](./basic_examples/specs/) — checked-in Framework input JSON per control (paths relative to `basic_examples/specs/`).

### Troubleshooting

If inference fails inside attention with a NATTEN/libnatten error, verify that the active Python
environment uses a matching Torch and NATTEN build. Avoid mixing a container-provided Torch/NATTEN
stack with packages from `~/.local` or another venv. In containerized environments,
`PYTHONNOUSERSITE=1` can help prevent user-site packages from shadowing the container stack.
