---
name: cosmos3-inference
description: >
  Guide users through running Cosmos3 inference — offline batch generation, online
  serving with Ray and Gradio, parallelism options, input formats, sampling parameters,
  and prompt upsampling. Use when the user asks "how do I run inference",
  "how do I generate a video", "how do I serve the model", "what parameters should I use",
  or any question about running the model to produce outputs.
---

# Cosmos3 Inference

## When to use this skill

- Use when a user wants to generate images or videos with Cosmos3
- Use when a user asks about inference parameters, input formats, or parallelism
- Use when a user wants to set up online serving (Ray Serve, Gradio)
- Use when a user asks about prompt engineering or upsampling
- For environment or import errors, hand off to **cosmos3-env-troubleshoot**

## Path convention

All paths below are relative to the cosmos3 package root (`../../../` from this skill file). All `uv run` / `python` commands should also be run from there.

## Where to find answers

| User question                                                           | Go to                                                                          |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| How do I run inference? (single-GPU, multi-GPU)                         | `README.md` § Offline Batch Inference                                          |
| Which model should I use? (Nano vs Super, memory, shift)                | `README.md` § Models                                                           |
| Which modality? (t2i, t2v, i2v, examples)                               | `README.md` § Modalities                                                       |
| What parallelism preset? (latency vs throughput)                        | `README.md` § Offline Batch Inference                                          |
| What input fields are available? (prompt, vision_path, num_frames, ...) | `docs/inference.md` § Sample Arguments                                         |
| What are the default parameter values?                                  | `docs/inference.md` § Default Values                                           |
| How do I use custom defaults?                                           | `docs/inference.md` § Custom Defaults                                          |
| How do I override a parameter? (precedence)                             | `docs/faq.md` § How do I override a default parameter?                         |
| What is the shift parameter?                                            | `docs/faq.md` § What is the `shift` parameter?                                 |
| How many frames can I generate? (resolution caps)                       | `docs/faq.md` § How many frames can I generate?                                |
| How do I start Ray Serve / Gradio / submit requests?                    | `docs/inference_online.md`                                                     |
| How do I write good prompts?                                            | `docs/prompting.md`                                                            |
| How do I upsample short prompts?                                        | `docs/prompting.md` § Upsampling                                               |
| How do I use the low-level API? (examples/)                             | `docs/inference.md` § Supplementary Examples                                   |
| All CLI flags                                                           | `uv run --all-extras --group=cu130 python -m cosmos3.scripts.inference --help` |

## Things not obvious from the docs

- **Path resolution**: relative paths in input JSON files are resolved relative to the **JSON file's directory**, not the working directory.
- **Seed**: always pass `--seed` for reproducible results. Without it, a random seed is used each time.
- **Resume**: interrupted runs can be resumed by re-running the same command — existing outputs are skipped automatically.
- **`--keep-going`**: continues processing remaining samples after a per-sample failure (e.g. guardrail rejection). Used in online serving by default.
- **Unique names**: every sample in a run must have a unique `name` field, or the script will error.

## Related skills

| Skill                                  | When to use                                    |
| -------------------------------------- | ---------------------------------------------- |
| `../cosmos3-setup/SKILL.md`            | Installation and environment setup             |
| `../cosmos3-codebase-nav/SKILL.md`     | Finding files, parameters, and configs in code |
| `../cosmos3-env-troubleshoot/SKILL.md` | Debugging environment and runtime errors       |
