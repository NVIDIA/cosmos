<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 RBench Reproduction

End-to-end recipe for generating the RBench robotics video-generation benchmark
with Cosmos3-Nano or Cosmos3-Super using the native Cosmos Framework PyTorch
entrypoint (`python -m cosmos_framework.scripts.inference`). The notebook
defaults to Cosmos3-Nano; set `RBENCH_MODEL_VARIANT=Super` to run
Cosmos3-Super instead.

RBench is an **Image-to-Video (I2V)** benchmark of 650 cases across 9 categories
(`common_manipulation`, `dual_arm`, `humanoid`, `long-horizon_planning`,
`multi-entity_collaboration`, `quad`, `single_arm`, `spatial_relationship`,
`visual_reasoning`). Each case conditions on a single image and generates a clip.

- **Image-to-Video (I2V)**: condition on the per-case image
  (`imgs/<image_path>`); generate 121 frames (1 conditioning + 120 generated).

Generation is at 24 FPS, 720p, 16:9, and the raw output is kept (no staging).
Generated inputs and videos are stored under `outputs/<model_name>/` so runs
from different model variants remain separate.

## Files

- `run_with_cosmos_framework.ipynb` — main notebook (demo case + full-sweep cell).
- `assets/prompts/*.json` — 9 category files, 650 entries total, each with
  `json_upsampled_prompt` and `negative_prompt`.
- `setup_rbench_scorer.sh` — isolated GPT/local-Qwen VQA environment.
- `setup_rbench_embodiment_scorer.sh` — isolated motion-metric environment.

## Dataset

The condition images come from the Hugging Face dataset
[`DAGroup-PKU/RBench`](https://huggingface.co/datasets/DAGroup-PKU/RBench),
cloned via `git clone` (Git LFS). Only the condition images are read
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

- 8-GPU Linux node (configurable via `COSMOS3_NUM_GPUS`, default 8)
- `uv >= 0.11.3`
- `git`, `git-lfs`
- Hugging Face access to the Cosmos3 model family

## Scoring evaluators

The notebook uses GPT as the default VLM evaluator for both the five-task split
and the three VQA metrics in the four-embodiment split. Configure an
OpenAI-compatible endpoint before enabling scoring:

```bash
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export OPENAI_MODEL=your-endpoint-model-id
export RBENCH_VLM_API_KEY=your-api-key
```

`OPENAI_API_KEY` is accepted as a fallback key variable. If neither key
variable is set, the notebook requests the key through a hidden input prompt;
do not save a literal key in the notebook. The dated public ReVidgen model,
`gpt-5-2025-08-07`, remains the default model when `OPENAI_MODEL` is unset.

To use local Qwen instead:

```bash
export RBENCH_USE_GPT=False
export QWEN_MODEL_PATH=Qwen/Qwen2.5-VL-72B-Instruct
```

GPT results are written under `gpt` and local-Qwen results under
`qwen_local`; their summaries and overall-score files are also kept
separate. GPT outputs are validated strictly against the upstream ReVidgen
schema. Only local Qwen enables the narrow `action_effectiveness` compatibility
mapping for the two task categories whose prompt terminology and example JSON
field names differ.

VQA scoring uses `.venv-rbench-scorer`. The motion amplitude and smoothness
metrics are evaluator-independent and continue to use `.venv-rbench-ops`.
Changing the VLM evaluator does not affect generation or require regenerating
videos.
