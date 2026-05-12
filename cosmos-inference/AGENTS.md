# AGENTS.md — Cosmos3 Package

Read this file first — it is the canonical map for navigating the Cosmos3 codebase and stays up to date.

Cosmos3 is a Mixture-of-Transformer (MoT) world foundation model supporting text-to-image, text-to-video, and image-to-video generation. This package covers inference, online serving, and the public API surface.

> All paths below are relative to the repository root (the directory containing `pyproject.toml` and the `cosmos3/` Python package).

## Commands

| Task                   | Command                                             |
| ---------------------- | --------------------------------------------------- |
| Lint                   | `uv run ruff check .`                               |
| Format check           | `uv run ruff format --check .`                      |
| Auto-fix lint + format | `uv run ruff check --fix . && uv run ruff format .` |
| Type-check             | `uv run pyrefly check`                              |
| Test (all)             | `uv run pytest`                                     |
| Test (single file)     | `uv run pytest --capture=no <path>`                 |

Config files: `.ruff.toml` (ruff), `pyrefly.toml` (pyrefly).

## Rules

- Always answer questions with references to code or documentation in `file:line` format.
- When unsure, point the user to the closest doc rather than guessing.
- Keep this file short. Link out to skills and docs for detail — this file is included in every prompt.

## Key File Locations

| What                     | Where                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| CLI entry point          | `cosmos3/scripts/inference.py`                                                                |
| Args / param definitions | `cosmos3/args.py` → `SamplingArgs`, `SamplingOverrides`, `OmniSampleArgs`, `OmniSetupArgs`    |
| Per-modality defaults    | `cosmos3/defaults/<mode>/sample_args.json` (modes: `text2image`, `text2video`, `image2video`) |
| Model / inference core   | `cosmos3/model.py`, `cosmos3/inference.py`                                                    |
| Feature flags            | `cosmos3/flags.py`                                                                            |
| Ray serving configs      | `cosmos3/ray/configs/latency.yaml`, `cosmos3/ray/configs/throughput.yaml`                     |
| Example inputs           | `inputs/omni/t2i.json`, `inputs/omni/t2v.json`, `inputs/omni/i2v.json`                        |

For the full config-defaults resolution chain, modality tables, and "where to make changes" guidance, see the **cosmos3-codebase-nav** skill (`.agents/skills/cosmos3-codebase-nav/SKILL.md`).

## Documentation

| Doc                                                    | What it covers                                        |
| ------------------------------------------------------ | ----------------------------------------------------- |
| [docs/setup.md](./docs/setup.md)                       | Installation, environment, NGC container, checkpoints |
| [docs/prompting.md](./docs/prompting.md)               | Prompt engineering, upsampling with vLLM              |
| [docs/inference.md](./docs/inference.md)               | Sample arguments, default values, custom defaults     |
| [docs/inference_online.md](./docs/inference_online.md) | Online serving with Ray Serve and Gradio              |
| [docs/faq.md](./docs/faq.md)                           | FAQ, tips, and troubleshooting                        |

## Common Tasks

| Task                    | Command                                                                                                                                                    |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Single-GPU inference    | `python -m cosmos3.scripts.inference -i inputs/omni/t2v.json -o outputs/ --checkpoint-path Cosmos3-Nano`                                                   |
| Multi-GPU inference     | `torchrun --nproc-per-node=4 -m cosmos3.scripts.inference --parallelism-preset=latency -i inputs/omni/t2v.json -o outputs/ --checkpoint-path Cosmos3-Nano` |
| Start online Ray server | `python -m cosmos3.ray.serve --parallelism-preset=latency -o outputs/ray_serve --checkpoint-path Cosmos3-Nano`                                             |
| Launch Gradio UI        | `python -m cosmos3.ray.gradio --port=8080`                                                                                                                 |
| See all CLI flags       | `python -m cosmos3.scripts.inference --help`                                                                                                               |

## Inference

For full parameter reference, input formats, parallelism presets, and online serving, read the **cosmos3-inference** skill (`.agents/skills/cosmos3-inference/SKILL.md`).

Key things not obvious from the CLI help:

- **NGC/PyTorch containers**: run `export LD_LIBRARY_PATH=''` before any `python` call or you'll hit a `torch._C` import error. See `docs/setup.md` § PyTorch Import Issue.
- **Reproducibility**: always pass `--seed <int>`. Without it a random seed is used each run.
- **JSON paths**: relative paths inside input JSON files resolve relative to the JSON file's directory, not the working directory.
- **Resume**: re-running the same command skips already-generated outputs automatically.
- **Parameters / defaults**: `docs/inference.md` is the reference for all sampling args and their defaults.
