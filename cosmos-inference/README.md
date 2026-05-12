<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">🤗 <a href="https://huggingface.co/collections/nvidia-cosmos-ea/cosmos3-ea">Hugging Face</a> | <a href="./docs/Cosmos3.pdf">Paper Draft</a></p>

# Cosmos3

## User Guide

- [Gallery](./docs/gallery.md)
- [Quickstart](#setup)
- [Setup](./docs/setup.md): Installation, environment, checkpoints
- [Prompting](./docs/prompting.md): Prompt engineering, upsampling
- [Inference](./docs/inference.md): Sample arguments, default values, offline batch inference
  - [Online Inference](./docs/inference_online.md): Online serving, web UI
  - [Action Policy Closed-Loop Evaluation on LIBERO](./docs/action_policy_closed_loop_eval.md): Action policy server/client setup and LIBERO evaluation. The client setup clones and installs the external [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) simulator package.
- [Training](./docs/training.md): Post-Training (Supervised Fine-Tuning)
- [FAQ](./docs/faq.md): FAQ, tips, troubleshooting
- [AGENTS.md](./AGENTS.md): AI agent entry point — start here for quick codebase orientation

## Overview

**Cosmos3** is a world foundation model that unifies understanding and generation within a single Mixture-of-Transformer (MoT) architecture. Two tightly coupled towers—a **Reasoner** (vision-language model) and a **Generator** (world simulator)—share latent representations so that structured perception directly grounds realistic, temporally consistent simulation.

<p align="center"><img width="930" height="545" alt="Image" src="https://github.com/user-attachments/assets/81ec0329-a425-4a62-a18b-da0a66672e1f" /></p>

One model, many capabilities:

| Input Modality          | Output Modality | Application           | EA1          |
| ----------------------- | --------------- | --------------------- | ------------ |
| Video \| Text           | Video           | Video Generator       | ✅           |
| Video \| Text           | Text            | Vision Language Model | Coming soon! |
| Action \| Video \| Text | Video           | World Model           | ✅           |
| Video \| Text           | Video & Action  | Policy Model          | ✅           |

## Supported Features (Cosmos3 EA1 — Robotics Backbone)

### User Stories

- **Video Backbone**: Evaluate and benchmark the model’s task understanding and review its architecture to inform codebase decisions.

### Modalities Supported

- Text2Image (t2i), Text2Video (t2v), Image2Video (i2v)

### Base Model Specifications

| Spec             | Value                                                                      |
| ---------------- | -------------------------------------------------------------------------- |
| Model Size       | Nano, Super                                                                |
| Resolution       | 256p / 480p / 720p                                                         |
| Frame Rate (FPS) | 10–30                                                                      |
| Num of Frames    | Default: 189 (max by resolution: `256p → 400`, `480p → 300`, `720p → 200`) |
| Max Duration     | Variable                                                                   |
| View             | Single view only                                                           |

## Setup

For more details and alternative installation methods, see [Setup](./docs/setup.md#installation).

Install system dependencies:

```shell
sudo apt-get update && sudo apt-get install -y --no-install-recommends curl ffmpeg libx11-dev tree wget
```

Install the package with `uv`:

```shell
uv sync --all-extras --group=cu130-train
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

## Prompting

For prompting guidance, see [Prompting](./docs/prompting.md).

## Inference

For more details, see [Inference](./docs/inference.md).

### Offline Batch Inference

Generate a single sample with 1 GPU:

```shell
python -m cosmos3.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/omni/t2v.json" \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

Generate multiple samples with 8 GPUs (~5 mins):

```shell
torchrun --nproc-per-node=8 -m cosmos3.scripts.inference \
    --parallelism-preset=throughput \
    -i "inputs/omni/*.json" \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

**Note:** The progress bar only prints on rank 0.

### Models

| Model         | Arguments                         |
| ------------- | --------------------------------- |
| Cosmos3-Nano  | `--checkpoint-path=Cosmos3-Nano`  |
| Cosmos3-Super | `--checkpoint-path=Cosmos3-Super` |

### Modalities

| Modality         | Example                                                                                            |
| ---------------- | -------------------------------------------------------------------------------------------------- |
| Text2Image       | [`-i "inputs/omni/t2i.json"`](inputs/omni/t2i.json)                                                |
| Text2Video       | [`-i "inputs/omni/t2v.json"`](inputs/omni/t2v.json)                                                |
| Image2Video      | [`-i "inputs/omni/i2v.json"`](inputs/omni/i2v.json)                                                |
| Forward Dynamics | [`-i "inputs/omni/action_forward_dynamics*.json"`](inputs/omni/action_forward_dynamics_robot.json) |
| Inverse Dynamics | [`-i "inputs/omni/action_inverse_dynamics*.json"`](inputs/omni/action_inverse_dynamics_av.json)    |
| Policy           | [`-i "inputs/omni/action_policy*.json"`](inputs/omni/action_policy_robot.json)                     |

To generate all examples, use `-i "inputs/omni/*.json"`.

For more information regarding action inference, please see [Action Inference](./docs/inference.md#action-inference).

### CLI Reference

To see all available arguments:

```shell
python -m cosmos3.scripts.inference --help
```
