# AGENTS.md — Cosmos Framework

Read this file first — it is the canonical map for navigating the Cosmos repository and stays up to date.

**Cosmos** is a framework for training and serving world foundation models. It is organized as two halves:

- **Training infrastructure** — the `cosmos/` package and user-facing documentation in `docs/`.
- **Inference infrastructure** — the `cosmos-inference/` subtree (Diffusers / Transformers / vLLM integrations, online serving with Ray + Gradio).

Both halves share the same dependency manifest at the repository root.

> All paths below are relative to the repository root (the directory containing `pyproject.toml`, the `cosmos/` Python package, and the `cosmos-inference/` subtree).

## Commands

| Task                   | Command                                             |
| ---------------------- | --------------------------------------------------- |
| Lint                   | `uv run ruff check .`                               |
| Format check           | `uv run ruff format --check .`                      |
| Auto-fix lint + format | `uv run ruff check --fix . && uv run ruff format .` |
| Type-check             | `uv run pyrefly check`                              |
| Test (all)             | `uv run pytest`                                     |
| Test (single file)     | `uv run pytest --capture=no <path>`                 |

Config files: `.ruff.toml` (ruff), `pyrefly.toml` (pyrefly), `.pytest.toml` (pytest), `conftest.py` (pytest fixtures).

A `justfile` is provided at the root with longer recipes (`just install`, `just lint`, `just test`, `just docker-cu130`).

## Rules

- Always answer questions with references to code or documentation in `file:line` format.
- When unsure, point the user to the closest doc rather than guessing.
- Keep this file short. Link out to skills and docs for detail — this file is included in every prompt.
- Do not duplicate inference behavior into `cosmos/`; it belongs in `cosmos-inference/`. Do not duplicate training behavior into `cosmos-inference/`; it belongs in `cosmos/`.

## Key File Locations

### Training (`cosmos/`)

| What                       | Where                                                                 |
| -------------------------- | --------------------------------------------------------------------- |
| Algorithms (losses, RL, reward) | `cosmos/algorithm/{loss,reward,rl}`                              |
| Training loop              | `cosmos/trainer/`                                                     |
| Models + parallelism       | `cosmos/model/`                                                       |
| Datasets / data loading    | `cosmos/data/`                                                        |
| Checkpoint I/O             | `cosmos/checkpoint/`                                                  |
| Callbacks (logging, eval)  | `cosmos/callbacks/`                                                   |
| RL workers (rollout, reward, reference, simulations) | `cosmos/workers/`                           |
| Controller / orchestrator  | `cosmos/controller/`                                                  |
| Launchers (Slurm, torchrun, k8s) | `cosmos/launcher/`                                              |
| Evaluation harness         | `cosmos/evaluation/`                                                  |
| CLI tools                  | `cosmos/tools/`, `tools/` (repo root)                                 |

For a per-subpackage tour with descriptions, see [`docs/code_structure.md`](./docs/code_structure.md).

### Inference (`cosmos-inference/`)

| What                       | Where                                                                                         |
| -------------------------- | --------------------------------------------------------------------------------------------- |
| CLI entry point            | `cosmos-inference/cosmos3/scripts/inference.py`                                               |
| Args / param definitions   | `cosmos-inference/cosmos3/args.py`                                                            |
| Per-modality defaults      | `cosmos-inference/cosmos3/defaults/<mode>/sample_args.json`                                   |
| Model / inference core     | `cosmos-inference/cosmos3/model.py`, `cosmos-inference/cosmos3/inference.py`                  |
| Ray serving configs        | `cosmos-inference/cosmos3/ray/configs/latency.yaml`, `.../throughput.yaml`                    |
| Backend packages           | `cosmos-inference/packages/{diffusers,transformers,vllm}-cosmos3/`                            |
| Example inputs             | `cosmos-inference/inputs/omni/*.json`                                                         |

## Documentation

### Training (root `docs/`)

| Doc                                                | What it covers                                                       |
| -------------------------------------------------- | -------------------------------------------------------------------- |
| [docs/setup.md](./docs/setup.md)                   | Install, NGC base image, CUDA variants, base-checkpoint download.    |
| [docs/code_structure.md](./docs/code_structure.md) | Repo layout and per-subpackage tour of `cosmos/`.                    |
| [docs/configs.md](./docs/configs.md)               | LazyConfig / experiment system and overrides.                        |
| [docs/dataset.md](./docs/dataset.md)               | JSONL / WebDataset / LeRobot formats and data prep.                  |
| [docs/training.md](./docs/training.md)             | Single- and multi-node launches, parallelism, mixed precision.       |
| [docs/checkpoints.md](./docs/checkpoints.md)       | DCP vs. HuggingFace safetensors, conversion, resume.                 |
| [docs/inference.md](./docs/inference.md)           | Bridge from a trained checkpoint to the inference backends.          |
| [docs/examples.md](./docs/examples.md)             | End-to-end training, fine-tuning, and inference walkthroughs.        |
| [docs/faq.md](./docs/faq.md)                       | Troubleshooting (OOM, NCCL, slow training) + env vars.               |

### Inference (`cosmos-inference/docs/`)

| Doc                                                                          | What it covers                                           |
| ---------------------------------------------------------------------------- | -------------------------------------------------------- |
| [cosmos-inference/docs/setup.md](./cosmos-inference/docs/setup.md)           | Inference-side install/env (subtree-local).              |
| [cosmos-inference/docs/inference.md](./cosmos-inference/docs/inference.md)   | Sample arguments, default values, schemas.               |
| [cosmos-inference/docs/inference_online.md](./cosmos-inference/docs/inference_online.md) | Online serving with Ray Serve and Gradio.    |
| [cosmos-inference/docs/prompting.md](./cosmos-inference/docs/prompting.md)   | Prompt engineering, upsampling with vLLM.                |
| [cosmos-inference/docs/faq.md](./cosmos-inference/docs/faq.md)               | Inference-side FAQ and troubleshooting.                  |

## Common Tasks

### Training

| Task                    | Command                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| Single-GPU train (smoke) | `python -m cosmos.scripts.train --config <experiment-config>`                                |
| Multi-GPU train         | `torchrun --nproc-per-node=8 -m cosmos.scripts.train --config <experiment-config>`            |
| Resume from checkpoint  | `python -m cosmos.scripts.train --config <experiment-config> --resume <path>`                 |
| Export DCP → HF         | `python -m cosmos.scripts.export_checkpoint --src <dcp> --dst <hf>`                           |
| Run a config sweep      | `just run python -m cosmos.scripts.train --config <experiment-config> --overrides "..."`      |

### Inference (in `cosmos-inference/`)

| Task                    | Command                                                                                                              |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Single-GPU inference    | `python -m cosmos3.scripts.inference -i cosmos-inference/inputs/omni/t2v.json -o outputs/ --checkpoint-path Cosmos3-Nano` |
| Multi-GPU inference     | `torchrun --nproc-per-node=4 -m cosmos3.scripts.inference --parallelism-preset=latency -i ... -o outputs/ ...`       |
| Start online Ray server | `python -m cosmos3.ray.serve --parallelism-preset=latency -o outputs/ray_serve --checkpoint-path Cosmos3-Nano`       |
| Launch Gradio UI        | `python -m cosmos3.ray.gradio --port=8080`                                                                           |
| See all CLI flags       | `python -m cosmos3.scripts.inference --help`                                                                         |

## Gotchas

- **NGC / PyTorch containers**: run `export LD_LIBRARY_PATH=''` before any `python` call or you'll hit a `torch._C` import error. See [`docs/setup.md`](./docs/setup.md#pytorch-import-issue).
- **Reproducibility**: always pass `--seed <int>`. Without it a random seed is used each run.
- **JSON paths**: relative paths inside input JSON files resolve relative to the JSON file's directory, not the working directory.
- **Resume**: re-running the same inference command skips already-generated outputs automatically.
- **Don't cross the streams**: training code in `cosmos/` must not import from `cosmos-inference/`. Inference code in `cosmos-inference/` must not import from `cosmos/`.
