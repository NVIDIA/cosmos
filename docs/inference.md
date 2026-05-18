# Inference

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Overview](#overview)
- [From Training to Inference](#from-training-to-inference)
- [Quickstart](#quickstart)
- [Offline Inference](#offline-inference)
  - [Parallelism Presets](#parallelism-presets)
  - [Modalities](#modalities)
- [Online / Server Inference](#online--server-inference)
- [Backends](#backends)
- [Prompting](#prompting)
- [Closed-Loop Evaluation](#closed-loop-evaluation)
- [Reference](#reference)

______________________________________________________________________

<!--TOC-->

## Overview

This page is a short bridge from the **training side** of the repo (`cosmos/`, `docs/`) into the **inference side** (`cosmos-inference/`). It explains how a checkpoint produced by training is consumed by inference, and points to the detailed inference documentation under [`cosmos-inference/docs/`](../cosmos-inference/docs).

If you only want to run inference and never train, you can read [`cosmos-inference/README.md`](../cosmos-inference/README.md) directly.

## From Training to Inference

A typical end-to-end flow looks like:

1. **Train.** Launch a training run with `python -m cosmos.scripts.train --config <experiment-config>` (see [Training](./training.md)). Training writes a [DCP](./checkpoints.md#dcp-distributed-checkpoint) checkpoint to the output directory at the cadence configured in your experiment.
2. **Export.** Convert the DCP checkpoint into the format expected by the inference backend you intend to use — typically HuggingFace `safetensors` for Transformers / vLLM, or a Diffusers pipeline directory. See [Checkpoints → Conversion](./checkpoints.md#conversion).
3. **Serve.** Load the exported checkpoint into one of the inference entry points described below.

The training and inference halves share the same `uv` environment at the repository root, so steps (1) and (3) can run from the same `.venv` without re-installation.

## Quickstart

Generate a single sample with 1 GPU using a trained Cosmos3-Nano checkpoint:

```shell
python -m cosmos3.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/omni/t2v.json" \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

Generate multiple samples in parallel with 8 GPUs:

```shell
torchrun --nproc-per-node=8 -m cosmos3.scripts.inference \
    --parallelism-preset=throughput \
    -i "inputs/omni/*.json" \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

To see every available argument:

```shell
python -m cosmos3.scripts.inference --help
```

The full argument reference, output layout, and troubleshooting steps live in [`cosmos-inference/docs/inference.md`](../cosmos-inference/docs/inference.md).

## Offline Inference

Offline inference (`cosmos3.scripts.inference`) reads sample arguments from JSON/JSONL/YAML files and writes outputs to disk. It is the right choice for batch generation and evaluation jobs.

### Parallelism Presets

- `--parallelism-preset=latency` — spread one sample across all GPUs to minimize per-sample latency. Used for real-time and interactive jobs.
- `--parallelism-preset=throughput` — run one sample per GPU in parallel. Used for batch jobs.
- `--max-num-seqs` — batch size per GPU.

### Modalities

The same script handles every supported modality; pick one by selecting an input file with the matching prefix:

| Modality           | Example input file                          |
| ------------------ | ------------------------------------------- |
| `text2image`       | `inputs/omni/t2i.json`                      |
| `text2video`       | `inputs/omni/t2v.json`                      |
| `image2video`      | `inputs/omni/i2v.json`                      |
| `forward_dynamics` | `inputs/omni/action_forward_dynamics_*.json`|
| `inverse_dynamics` | `inputs/omni/action_inverse_dynamics_*.json`|
| `policy`           | `inputs/omni/action_policy_*.json`          |

See [`cosmos-inference/docs/inference.md`](../cosmos-inference/docs/inference.md) for sample-argument schemas, action-mode details, and the schema reference under `cosmos-inference/schemas/`.

## Online / Server Inference

For interactive serving — HTTP API, web UI, or Ray Serve deployment — see [`cosmos-inference/docs/inference_online.md`](../cosmos-inference/docs/inference_online.md). It covers the Ray + Gradio entry point, request format, and scaling configuration.

## Backends

The inference side packages three backends under [`cosmos-inference/packages/`](../cosmos-inference/packages), each consuming a converted checkpoint:

- **`transformers-cosmos3`** — HuggingFace Transformers integration; good for single-GPU generation and fine-tuning hand-off.
- **`vllm-cosmos3`** — vLLM-based high-throughput serving for the language/reasoner tower.
- **`diffusers-cosmos3`** — Diffusers-style pipeline for the generator tower; preferred for Diffusers-native workflows.

Each package has its own README in `cosmos-inference/packages/<name>/README.md`.

## Prompting

Prompt structure, system prompts, and prompt-engineering guidelines for each modality live in [`cosmos-inference/docs/prompting.md`](../cosmos-inference/docs/prompting.md).

## Closed-Loop Evaluation

For action-policy checkpoints evaluated against the LIBERO simulator (the closed-loop benchmark referenced from training runs), see [`cosmos-inference/docs/action_policy_closed_loop_eval.md`](../cosmos-inference/docs/action_policy_closed_loop_eval.md). The `libero` uv dependency group required for that flow is already declared at the repository root.

## Reference

- [`cosmos-inference/README.md`](../cosmos-inference/README.md) — top-level inference overview, model index, and quickstart.
- [`cosmos-inference/docs/inference.md`](../cosmos-inference/docs/inference.md) — full offline-inference argument reference, schemas, and troubleshooting.
- [`cosmos-inference/docs/inference_online.md`](../cosmos-inference/docs/inference_online.md) — online serving (Ray, Gradio, request format).
- [`cosmos-inference/docs/prompting.md`](../cosmos-inference/docs/prompting.md) — prompt structure per modality.
- [`cosmos-inference/docs/environment_variables.md`](../cosmos-inference/docs/environment_variables.md) — runtime environment variables that affect inference.
- [`cosmos-inference/docs/gallery.md`](../cosmos-inference/docs/gallery.md) — example outputs for each modality.
- [`cosmos-inference/docs/action_policy_closed_loop_eval.md`](../cosmos-inference/docs/action_policy_closed_loop_eval.md) — closed-loop action eval (LIBERO).
- [Checkpoints](./checkpoints.md) — exporting a trained checkpoint to a format the inference backends understand.
