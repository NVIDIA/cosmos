<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 PhysicsIQ Reproduction

End-to-end recipe for reproducing the PhysicsIQ benchmark with Cosmos3-Super
using the native Cosmos Framework PyTorch entrypoint
(`python -m cosmos_framework.scripts.inference`).

The notebook walks through both PhysicsIQ task formats:

- **Image-to-Video (I2V)**: condition on the single switch-frame JPG; generate
  121 frames (1 conditioning + 120 generated); score the last 120.
- **Video-to-Video (V2V)**: condition on a 3-second clip; pad the official
  72-frame conditioning to 73 frames; generate 193 frames total; score
  frames 73..192.

## Files

- `run_with_cosmos_framework.ipynb` — main notebook.
- `assets/i2v_prompts.json` — 198 I2V upsampled prompts.
- `assets/v2v_prompts.json` — 198 V2V upsampled prompts.

## Reference scores (Cosmos3-Super)

| Task | PhysicsIQ score |
| ---- | --------------: |
| I2V  |            43.8 |
| V2V  |            59.7 |

## Requirements

- 4-GPU Linux node (configurable via `COSMOS3_NUM_GPUS`, default 4)
- `uv >= 0.11.3`
- `ffmpeg`, `git`, `gcloud` (or `gsutil`)
- Hugging Face access to the Cosmos3 model family
