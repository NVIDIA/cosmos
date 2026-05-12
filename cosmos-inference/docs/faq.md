# Cosmos3 FAQ

> **Skills:** `.agents/skills/cosmos3-setup/SKILL.md` · `.agents/skills/cosmos3-inference/SKILL.md`

A catch-all collection of frequently asked questions, tips, and troubleshooting for the Cosmos3 package. Can't find what you need? Check [setup.md](./setup.md) for installation issues or [inference.md](./inference.md) for inference details.

To add a new entry, append it under the most relevant section — or under [Miscellaneous](#miscellaneous) if nothing fits.

---

## Table of Contents

- [Setup and Installation](#setup-and-installation)
- [Configuration and Defaults](#configuration-and-defaults)
- [Inference](#inference)
- [Tips and Tricks](#tips-and-tricks)
- [Miscellaneous](#miscellaneous)

---

## Setup and Installation

### Q: I get `ImportError: cannot import name '_functionalization' from 'torch._C'` inside an NGC container

Clear the library path before running anything:

```shell
export LD_LIBRARY_PATH=''
```

This is needed because the NGC PyTorch container ships its own libraries that conflict with the venv-installed versions. See [setup.md#pytorch-import-issue](./setup.md#pytorch-import-issue).

### Q: `ModuleNotFoundError: No module named 'cosmos3'`

Make sure you installed the package:

```shell
uv sync --all-extras --group=cu130
source .venv/bin/activate
```

If already installed, try `--reinstall` to force a clean state.

### Q: Which CUDA version should I use?

CUDA 13.0 is recommended. CUDA 12.8 is also supported. The major version must match between your system CUDA and the installed PyTorch wheels. Check with:

```shell
nvidia-smi                                    # system CUDA
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA
```

### Q: How do I download model checkpoints?

Checkpoints are downloaded automatically from Hugging Face during inference. You need:

1. A [Hugging Face token](https://huggingface.co/settings/tokens) with Read permission
2. Accepted [NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/Cosmos-Guardrail1)
3. `HF_TOKEN` environment variable set, or `uvx hf auth login`

Control the download location with `HF_HOME` (default: `~/.cache/huggingface`). If downloads fail, the commands are printed to the console — run them manually to debug. See [setup.md#downloading-checkpoints](./setup.md#downloading-checkpoints).

### Q: `fatal error: Python.h: No such file or directory`

Reinstall uv and the venv from scratch:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install --reinstall
rm -rf .venv
uv sync --all-extras --group=cu130 --reinstall
source .venv/bin/activate
```

---

## Configuration and Defaults

### Q: Where are the default inference parameters (guidance, shift, num_steps, etc.)?

Per-modality defaults live in JSON files under `cosmos3/defaults/<mode>/sample_args.json`:

| Mode          | Default file                                    |
| ------------- | ----------------------------------------------- |
| `text2image`  | `cosmos3/defaults/text2image/sample_args.json`  |
| `text2video`  | `cosmos3/defaults/text2video/sample_args.json`  |
| `image2video` | `cosmos3/defaults/image2video/sample_args.json` |

See [AGENTS.md](../AGENTS.md) for the full config defaults chain.

### Q: How do I override a default parameter?

From most temporary to most permanent:

1. **CLI flag**: `--shift 5.0` (per-run, applies to all samples)
2. **Sample argument file**: set the field in your input JSON (per-sample)
3. **Custom defaults file**: pass `"defaults_file": "my_defaults.json"` in your sample argument file (see [inference.md#custom-defaults](./inference.md#custom-defaults))
4. **Built-in default**: edit `cosmos3/defaults/<mode>/sample_args.json` (permanent change)

Fields set in the sample argument file take precedence over defaults. CLI flags override both.

### Q: What is the `shift` parameter and which value should I use?

`shift` controls the time-shift in the UniPC diffusion sampler. Higher values produce more detail but can introduce artifacts. Recommended values:

| Model               | Recommended shift |
| ------------------- | ----------------- |
| Cosmos3-Nano (8B)   | `10.0` (default)  |
| Cosmos3-Super (32B) | `5.0`             |

### Q: How do I add a new parameter to the inference pipeline?

1. Add the field to `SamplingArgs` and `SamplingOverrides` in `cosmos3/args.py`
2. Add its default to each `cosmos3/defaults/<mode>/sample_args.json`
3. Wire it through `OmniSampleOverrides.build_sample()` in `args.py`

### Q: What does `defaults_file` do?

It lets you supply a custom JSON file of default values instead of the built-in presets. The format is the same as the files in `cosmos3/defaults/`. Fields in your sample argument file still take precedence over the custom defaults. See [inference.md#custom-defaults](./inference.md#custom-defaults).

---

## Inference

### Q: How much GPU memory do the models need?

| Model               | GPU Memory |
| ------------------- | ---------- |
| Cosmos3-Nano (8B)   | 32 GB      |
| Cosmos3-Super (32B) | 128 GB     |

### Q: What is the difference between `latency` and `throughput` parallelism presets?

| Preset       | What it does                        | When to use                 |
| ------------ | ----------------------------------- | --------------------------- |
| `latency`    | Spreads each sample across all GPUs | Interactive / real-time use |
| `throughput` | One sample per GPU in parallel      | Large batch jobs            |

### Q: How do I generate images instead of videos?

Use a text2image input file:

```shell
python -m cosmos3.scripts.inference -i inputs/omni/t2i.json -o outputs/ --checkpoint-path Cosmos3-Nano
```

The modality is determined by the input JSON (`num_frames=1` for images), not by a separate flag. See `inputs/omni/t2i.json` for the format.

### Q: How many frames can I generate?

Depends on resolution:

| Resolution | Max frames |
| ---------- | ---------- |
| 256p       | 400        |
| 480p       | 300        |
| 720p       | 200        |

Default is 189 frames at 24 FPS (~7.9 seconds).

### Q: What input formats does image-to-video support?

Provide a `vision_path` pointing to an image (`.jpg`, `.jpeg`, `.png`) or a URL. See `inputs/omni/i2v.json` for the format.

### Q: How do I run online inference with Ray?

Install serve dependencies and start the server:

```shell
uv pip install -e ".[serve]"
python -m cosmos3.ray.serve --parallelism-preset=latency -o outputs/ray_serve --checkpoint-path Cosmos3-Nano
```

Then submit requests via curl, the submit CLI, or the Gradio UI. See [inference_online.md](./inference_online.md) for details.

---

## Tips and Tricks

### Seed reproducibility

Always pass `--seed` when comparing runs. Without it, a random seed is used each time.

### Prompt upsampling

Short prompts produce worse results. Use the built-in prompt upsampler with a vLLM-served Qwen3 model:

```shell
python -m cosmos3.scripts.upsample_prompts -i "inputs/omni/*.json" -o outputs/upsample_prompts
```

See [prompting.md#upsampling](./prompting.md#upsampling) for full setup instructions.

### Batch inference resume

The inference script automatically skips samples whose output files already exist. If a run is interrupted, re-run the same command to resume.

### Generate all modalities at once

```shell
python -m cosmos3.scripts.inference -i "inputs/omni/*.json" -o outputs/ --checkpoint-path Cosmos3-Nano --seed=0
```

### CLI help

All available flags and their current defaults:

```shell
python -m cosmos3.scripts.inference --help
```

---

## Miscellaneous

*This section is a catch-all for tips that don't fit elsewhere. Add new entries freely.*

### Q: What are the example scripts in `examples/` for?

They illustrate how the inference logic works under the hood — `examples/inference.py` shows the low-level model API and `examples/inference_pipeline.py` shows the pipeline API. For production use, prefer `python -m cosmos3.scripts.inference`.

### Q: Where are the Ray Serve config files?

`cosmos3/ray/configs/latency.yaml` and `cosmos3/ray/configs/throughput.yaml`. These configure the Ray Serve deployment with different parallelism strategies.
