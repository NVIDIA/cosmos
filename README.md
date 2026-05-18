<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">🤗 <a href="https://huggingface.co/collections/nvidia-cosmos-ea/cosmos3-ea">Hugging Face</a> | <a href="./cosmos-inference/docs/Cosmos3.pdf">Paper Draft</a></p>

# Cosmos

**Cosmos** is an end-to-end framework for training and serving world foundation models. It is organized as two tightly-integrated halves:

- **Training infrastructure** — the `cosmos/` package and the documentation in [`docs/`](./docs), covering setup, dataset preparation, distributed training, checkpointing, evaluation, and configuration.
- **Inference infrastructure** — the [`cosmos-inference/`](./cosmos-inference) subtree, containing the Diffusers / Transformers / vLLM integrations and end-to-end inference pipelines.

Both halves share the same dependency manifest at the repository root, so a single `uv sync` installs everything needed to train a model and serve its checkpoints.

- [Quickstart](#setup)
- [Training](#training)
- [Inference](#inference)

## Overview

The training side provides:

- A distributed trainer with FSDP / tensor / context / pipeline parallelism (see [`cosmos/trainer/`](./cosmos_training/cosmos/trainer) and [`cosmos/model/`](./cosmos_training/cosmos/model)).
- A worker-based RL / post-training topology with `controller`, `rollout`, `reward`, `reference`, and `simulations` workers (see [`cosmos/workers/`](./cosmos_training/cosmos/workers)).
- A pluggable algorithm layer for losses, reward models, and RL update rules (see [`cosmos/algorithm/`](./cosmos_training/cosmos/algorithm)).
- Native DCP checkpointing with HuggingFace `safetensors` import/export.
- Dataset abstractions for JSONL, WebDataset, and LeRobot formats.

The inference side under [`cosmos-inference/`](./cosmos-inference) exposes ready-to-use pipelines for offline batch generation and online serving (Ray, Gradio), with backends for HuggingFace Transformers, vLLM, and Diffusers.

## Setup

For full instructions and alternative installation methods, see [Setup](./docs/setup.md).

Before installing, make sure your machine meets the [System Requirements](./docs/setup.md#system-requirements). If you want a curated PyTorch + CUDA environment, start from the [recommended base image](./docs/setup.md#recommended-base-image).

Install system dependencies:

```shell
sudo apt-get install -y --no-install-recommends curl ffmpeg libx11-dev tree wget
```

Install the package with `uv`:

```shell
uv sync --all-extras --group=cu130-train
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

If you are starting from the [recommended NVIDIA NGC base image](./docs/setup.md#recommended-base-image) (`nvcr.io/nvidia/pytorch:25.09-py3`), see the [one-shot quickstart](./docs/setup.md#quickstart-from-the-recommended-base-image).

## Training

The training infrastructure lives in [`cosmos/`](./cosmos_training/cosmos), with user-facing documentation in [`docs/`](./docs):

| Topic                                                        | What it covers                                                                                                          |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| [Setup](./docs/setup.md)                                     | Hardware/software prerequisites, `uv` install paths, CUDA variants, Docker base image, and base-checkpoint downloading. |
| [Code Structure](./docs/code_structure.md)                   | Repository layout and a per-subpackage tour of `cosmos/` — where each concern lives and where to add new code.          |
| [Configs](./docs/configs.md)                                 | The LazyConfig / experiment system, CLI and YAML overrides, and how to register a new experiment.                       |
| [Dataset](./docs/dataset.md)                                 | Supported data formats (JSONL, WebDataset, LeRobot), preparation steps, augmentations, and multi-dataset weighting.     |
| [Training](./docs/training.md)                               | Launching single-GPU, multi-GPU, and multi-node runs; parallelism strategies; mixed precision; resuming.                |
| [Checkpoints](./docs/checkpoints.md)                         | DCP vs. HuggingFace `safetensors`, conversion utilities, and resuming a training run from a saved checkpoint.           |
| [Inference (from a trained checkpoint)](./docs/inference.md) | Loading a trained checkpoint into one of the inference backends — points back into `cosmos-inference/`.                 |
| [Examples](./docs/examples.md)                               | End-to-end training, fine-tuning, and inference walkthroughs; runnable scripts in [`examples/`](./examples).            |
| [FAQ](./docs/faq.md)                                         | Troubleshooting (OOM, NCCL hangs, slow training), environment variables, and common pitfalls.                           |

A minimal single-GPU training launch looks like:

```shell
python -m cosmos.scripts.train --config <experiment-config>
```

See [Training](./docs/training.md) for multi-GPU / multi-node launches and the full set of CLI arguments.

### Reference

- [Code Structure](./docs/code_structure.md) — repository layout and a tour of each `cosmos/` subpackage.
- [FAQ](./docs/faq.md) — troubleshooting OOM, NCCL hangs, slow training, and environment variables.
- [AGENTS.md](./AGENTS.md) — contributor-facing guidance for AI agents working in this repo.

## Inference

End-to-end inference — offline batch generation, online serving with Ray and Gradio, and integration with HuggingFace Transformers, vLLM, and Diffusers — is documented in [`cosmos-inference/README.md`](./cosmos-inference/README.md).

Once a checkpoint has been trained in this repo, export it (see [Checkpoints](./docs/checkpoints.md)) and follow the inference README for serving.
