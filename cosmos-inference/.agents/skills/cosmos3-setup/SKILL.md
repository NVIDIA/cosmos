---
name: cosmos3-setup
description: >
  Guide users through Cosmos3 installation, environment setup, checkpoint downloading,
  and verification. Use when the user asks "how do I install cosmos3", "how do I set up
  the environment", "how do I download checkpoints", "how do I use Docker", or any
  question about getting the package running for the first time.
---

# Cosmos3 Setup

## When to use this skill

- Use when a user wants to install Cosmos3 or set up a development environment
- Use when a user asks about system requirements, CUDA versions, or GPU compatibility
- Use when a user needs to download model checkpoints or configure HuggingFace auth
- Use when a user wants to run Cosmos3 inside a Docker container or NGC container
- For errors during setup, hand off to the **cosmos3-env-troubleshoot** skill

## Path convention

All paths below are relative to the cosmos3 package root (`../../../` from this skill file). All `uv run` / `python` commands should also be run from there.

## Where to find answers

The canonical setup reference is `docs/setup.md`. The README (`README.md` § Setup) has the shortest quickstart.

| User question                                          | Go to                                                                |
| ------------------------------------------------------ | -------------------------------------------------------------------- |
| What are the system requirements?                      | `docs/setup.md` § System Requirements                                |
| How do I install with uv? (sync, pip venv, pip system) | `docs/setup.md` § Virtual Environment                                |
| How do I install with Docker?                          | `docs/setup.md` § Docker Container                                   |
| Custom torch/CUDA versions or attention backends?      | `docs/setup.md` § Advanced                                           |
| Which CUDA version? (cu130 vs cu128)                   | `docs/setup.md` § CUDA Variants, `docs/faq.md` § Which CUDA version? |
| How do I download checkpoints?                         | `docs/setup.md` § Downloading Checkpoints                            |
| NGC container issues?                                  | `docs/setup.md` § PyTorch Import Issue                               |
| Any installation error                                 | `../cosmos3-env-troubleshoot/SKILL.md`                               |

## Setup steps at a glance

1. **Clone** the repository and `cd` into the project root (the directory containing `pyproject.toml`)
2. **System deps**: `sudo apt-get install -y --no-install-recommends curl ffmpeg libx11-dev tree wget`
3. **Install uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env`
4. **Install package**: `uv sync --all-extras --group=cu130 && source .venv/bin/activate` (see docs for alternative methods)
5. **Optional extras**: `.[serve]` for Ray/Gradio, `.[guardrail]` for safety, `.[train]` for post-training
6. **Checkpoints**: auto-downloaded during inference; requires HuggingFace auth (see docs)
7. **Verify**: `uv run --all-extras --group=cu130 python -c "import cosmos3; print('ok')"`

## Things not obvious from the docs

- **NGC container caveat**: you must run `export LD_LIBRARY_PATH=''` *before* any Python imports when inside an NGC PyTorch container. Easy to miss.
- **CUDA version alignment**: the major CUDA version from `nvidia-smi` must match `torch.version.cuda`. Mismatches cause cryptic shared-library errors.
- **`HF_HOME`**: controls where checkpoints are cached (default: `~/.cache/huggingface`). Set this if disk space is tight or you want a shared cache.
- **Conflicting env vars**: stale `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` env vars can silently override CLI auth. Check with `printenv | grep HF_`.

## Related skills

| Skill                                  | When to use                                    |
| -------------------------------------- | ---------------------------------------------- |
| `../cosmos3-inference/SKILL.md`        | Running inference after setup is complete      |
| `../cosmos3-codebase-nav/SKILL.md`     | Finding files, parameters, and configs in code |
| `../cosmos3-env-troubleshoot/SKILL.md` | Debugging environment and runtime errors       |
