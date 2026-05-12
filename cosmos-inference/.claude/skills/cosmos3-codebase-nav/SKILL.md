---
name: cosmos3-codebase-nav
description: >
  Navigate the Cosmos3 package codebase to find where parameters, configs, defaults,
  scripts, and documentation live. Use when the user asks "where is X in cosmos3",
  "how do I find the config for Y", "where are the defaults", "where do I change a
  parameter", or any question about locating files, modules, or settings. Also use
  when the user opens or edits files and needs orientation.
---

# Cosmos3 Codebase Navigation

## When to use this skill

- Use this skill when an agent is navigating the Cosmos3 package
- Use this skill to answer "where is X", "how do I find the config for Y", or any file-location question
- Use this skill when the user opens or edits cosmos3 files and needs orientation

## Path convention

All paths below are relative to this file's location (`.agents/skills/cosmos3-codebase-nav/`).

## Quick Reference

### Where parameters and defaults live

| What you're looking for                                 | File                                                                                              |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Sampling params (num_steps, guidance, shift, fps, etc.) | `../../../cosmos3/args.py` → `SamplingArgs`, `SamplingOverrides`                                  |
| Per-modality default values                             | `../../../cosmos3/defaults/<mode>/sample_args.json`                                               |
| Setup params (parallelism, checkpoints, model path)     | `../../../cosmos3/args.py` → `OmniSetupArgs`, `OmniSetupOverrides`                                |
| Common args base classes                                | `../../../cosmos3/common/args.py` → `ArgsBase`, `OverridesBase`                                   |
| Ray serving parallelism presets                         | `../../../cosmos3/ray/configs/latency.yaml`, `../../../cosmos3/ray/configs/throughput.yaml`       |
| Feature flags                                           | `../../../cosmos3/flags.py`                                                                       |
| Prompt upsampler system prompt                          | `../../../cosmos3/defaults/prompt_upsampler.txt`                                                  |
| Example inputs                                          | `../../../inputs/omni/t2i.json`, `../../../inputs/omni/t2v.json`, `../../../inputs/omni/i2v.json` |

Available modality modes for defaults: `text2image`, `text2video`, `image2video`.

### Config defaults resolution chain

When a user runs inference, default parameter values are resolved in this order:

```
cosmos3/defaults/<mode>/sample_args.json          # 1. Per-modality JSON defaults (num_steps, guidance, shift, fps, etc.)
        ↓
_load_modality_defaults() in cosmos3/args.py      # 2. Loaded and cached at import time
        ↓
SamplingArgs / SamplingOverrides                  # 3. Pydantic models with field-level validation
        ↓
OmniSampleOverrides.build_sample()                # 4. Merges user overrides → final resolved args
        ↓
_RESOLUTION_SHIFT_DEFAULTS[model_size, resolution] # 5. Model+resolution shift override (if user didn't set shift)
        ↓
CLI flags (--guidance, --shift, etc.)             # 6. User overrides from command line
```

The `_RESOLUTION_SHIFT_DEFAULTS` table in `../../../cosmos3/args.py` overrides the default `shift` based on model size and resolution, unless the user explicitly specified `--shift`.

| Mode          | Default file                                             | Key defaults                                   |
| ------------- | -------------------------------------------------------- | ---------------------------------------------- |
| `text2image`  | `../../../cosmos3/defaults/text2image/sample_args.json`  | `num_frames=1`, `guidance=6.0`, `shift=10.0`   |
| `text2video`  | `../../../cosmos3/defaults/text2video/sample_args.json`  | `num_frames=189`, `guidance=6.0`, `shift=10.0` |
| `image2video` | `../../../cosmos3/defaults/image2video/sample_args.json` | `num_frames=189`, `guidance=6.0`, `shift=10.0` |

Users can also supply a custom defaults file per-request via the `defaults_file` field in sample arguments (see `../../../docs/inference.md`).

### Where to make changes

| Task                            | Edit                                                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Change a built-in default value | `../../../cosmos3/defaults/<mode>/sample_args.json`                                                     |
| Add a new CLI parameter         | `SamplingArgs` + `SamplingOverrides` in `../../../cosmos3/args.py`, then add to each `sample_args.json` |
| Change parallelism presets      | `../../../cosmos3/ray/configs/latency.yaml` or `throughput.yaml`                                        |
| Add a new script                | `../../../cosmos3/scripts/` — follow `inference.py` as the pattern                                      |

### Key entry points

| Entry point          | How to run                                             |
| -------------------- | ------------------------------------------------------ |
| Batch inference      | `python -m cosmos3.scripts.inference`                  |
| Online serving (Ray) | `python -m cosmos3.ray.serve`                          |
| Submit to Ray server | `python -m cosmos3.ray.submit`                         |
| Gradio UI            | `python -m cosmos3.ray.gradio`                         |
| Prompt upsampling    | `python -m cosmos3.scripts.upsample_prompts`           |
| Model export         | `python -m cosmos3.scripts.export_model`               |
| Diffusers conversion | `python -m cosmos3.scripts.convert_model_to_diffusers` |

### Documentation

| Doc                                 | Covers                                                |
| ----------------------------------- | ----------------------------------------------------- |
| `../../../AGENTS.md`                | Commands, rules, key file locations (read this first) |
| `../../../README.md`                | Overview, quickstart, examples                        |
| `../../../docs/setup.md`            | Installation, environment, checkpoints                |
| `../../../docs/inference.md`        | Sample args, default values, custom defaults          |
| `../../../docs/inference_online.md` | Ray Serve and Gradio                                  |
| `../../../docs/prompting.md`        | Prompt engineering, upsampling                        |
| `../../../docs/faq.md`              | FAQ, tips, and troubleshooting                        |
