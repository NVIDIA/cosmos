<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 RBench Reproduction

End-to-end recipe for generating the RBench robotics video-generation benchmark
with Cosmos3-Super using the native Cosmos Framework PyTorch entrypoint
(`python -m cosmos_framework.scripts.inference`).

RBench is an **Image-to-Video (I2V)** benchmark of 650 cases across 9 categories
(`common_manipulation`, `dual_arm`, `humanoid`, `long-horizon_planning`,
`multi-entity_collaboration`, `quad`, `single_arm`, `spatial_relationship`,
`visual_reasoning`). Each case conditions on a single image and generates a clip.

- **Image-to-Video (I2V)**: condition on the per-case image
  (`imgs/<image_path>`); generate 121 frames (1 conditioning + 120 generated).

Generation is at 24 FPS, 720p, 16:9, and the raw output is kept (no staging).

## Files

- `run_with_cosmos_framework.ipynb` — main notebook (demo case + full-sweep cell).
- `assets/prompts/*.json` — 9 category files, 650 entries total, each with
  `json_upsampled_prompt` and `negative_prompt`.

## Dataset

The condition images come from the Hugging Face dataset
[`DAGroup-PKU/RBench`](https://huggingface.co/datasets/DAGroup-PKU/RBench),
cloned via `git clone` (Git LFS, ~22 GB). Only the condition images are read
from the dataset; the prompts come from the local `assets/prompts/` files.

## Sampling settings

| Setting     | Value         |
| ----------- | ------------: |
| num_frames  |           121 |
| fps         |            24 |
| resolution  |           720 |
| num_steps   |            50 |
| guidance    |           6.0 |
| shift       |          10.0 |
| seed        |             0 |

## Requirements

- 4-GPU Linux node (configurable via `COSMOS3_NUM_GPUS`, default 4)
- `uv >= 0.11.3`
- `git`, `git-lfs`
- Hugging Face access to the Cosmos3 model family
