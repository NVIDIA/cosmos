<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 PAI-Bench (Generation) Reproduction

End-to-end recipe for generating the PAI-Bench (Physical AI Bench) generation
set with Cosmos3-Super using the native Cosmos Framework PyTorch entrypoint
(`python -m cosmos_framework.scripts.inference`).

PAI-Bench covers Physical AI domains (AV driving, robotics, industry, physics,
human, common sense) across 1044 samples. The notebook runs both generation
tasks:

- **Text-to-Video (T2V)**: generate from the prompt only (no condition image);
  generate 189 frames.
- **Image-to-Video (I2V)**: condition on the per-sample image
  (`condition_image/<image_name>`); generate 189 frames.

Both tasks generate at 24 FPS, 720p, 16:9, and keep the raw output (no staging).

## Files

- `run_with_cosmos_framework.ipynb` — main notebook (demos for both tasks + two full-sweep cells).
- `assets/i2v_prompts.json` — 1044 I2V entries with `json_upsampled_prompt` and `negative_prompt`.
- `assets/t2v_prompts.json` — 1044 T2V entries with `json_upsampled_prompt` and `negative_prompt`.

## Dataset

The condition images come from the Hugging Face dataset
[`shi-labs/physical-ai-bench-generation`](https://huggingface.co/datasets/shi-labs/physical-ai-bench-generation),
cloned via `git clone` (Git LFS). Only the condition images are read from the
dataset; the prompts come from the local `assets/` files.

## Sampling settings

| Setting     | Value         |
| ----------- | ------------: |
| num_frames  |           189 |
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
