---
name: cosmos3-post-training
description: >
  Guide users through Cosmos3 supervised fine-tuning (SFT) post-training:
  preparing the example dataset and DCP base checkpoint, editing the experiment
  config, launching distributed training with `torchrun`, running T2V/I2V/V2V
  inference with the trained DCP checkpoint, optionally exporting it to
  Hugging Face safetensors, running **action evaluation** (`cosmos3.scripts.eval`)
  on a held-out dataset for action checkpoints (forward / inverse dynamics, policy)
  to score PSNR / action MSE, and the optional Video Captioning pipeline for
  building custom datasets. Use when the user asks "how do I post-train Cosmos3",
  "how do I fine-tune on my own video dataset", "how do I export a trained
  checkpoint", "how do I evaluate an action checkpoint", "how do I run eval.py /
  cosmos3.scripts.eval", "how do I caption videos for training", or any question
  about SFT, the `cu130-train` / `cu128-train` install groups, the
  `mixed_modality_sft_8b.yaml` config, the `convert_model_to_dcp` / `export_model`
  / `train` / `eval` / `caption_from_video` / `captions_to_sft_jsonl` scripts,
  action-evaluation metrics, or where SFT and action-eval outputs land on disk.
  Note: this skill's evaluation coverage is **action-only** — eval.py does not
  score non-action (T2V/I2V/V2V) checkpoints; for those, use the inference flow.
---

# Cosmos3 Post-Training (SFT)

## When to use this skill

- User wants to fine-tune Cosmos3-Nano on their own video dataset (SFT)
- User asks which fields in `mixed_modality_sft_8b.yaml` to override (lr, FSDP shard, max_iter, jsonl_paths, ...)
- User wants to convert a base Hugging Face checkpoint to DCP, or convert a trained DCP back to safetensors
- User wants to score an **action** checkpoint (forward / inverse dynamics, policy) against a held-out dataset with `cosmos3.scripts.eval` — per-sample PSNR / action MSE plus an aggregate. Eval is action-only; do not invoke this skill's eval guidance for T2V/I2V/V2V checkpoints
- User wants to caption raw videos with a VLM to build a training dataset
- User wants to assemble a JSONL manifest from videos + captions
- For installation, `--group=cu130-train` / `cu128-train`, or LD_LIBRARY_PATH issues, hand off to **cosmos3-setup**
- For inference parameters, parallelism presets, or online serving, hand off to **cosmos3-inference**

## Path convention

All paths below are relative to the cosmos3 package root (`../../../` from this skill file). All `uv run` / `python` / `torchrun` commands should also be run from there.

## Where to find answers

The canonical reference is `docs/training.md`. Use this table to route questions:

| User question                                                      | Go to                                                               |
| ------------------------------------------------------------------ | ------------------------------------------------------------------- |
| Full step-by-step SFT workflow                                     | `docs/training.md`                                                  |
| Which install group? (`cu130-train` vs `cu128-train`)              | `docs/training.md` § Setup, `docs/setup.md` § CUDA Variants         |
| How do I download the example bridge dataset?                      | `docs/training.md` § Step 1 — Prepare data and checkpoint           |
| How do I convert a base HF checkpoint to DCP?                      | `docs/training.md` § Step 1 — Prepare data and checkpoint           |
| Which `mixed_modality_sft_8b.yaml` fields are commonly overridden? | `docs/training.md` § Step 2 — Prepare config                        |
| How do I launch distributed training?                              | `docs/training.md` § Step 3 — Run training                          |
| How do I validate the config without actually training?            | `docs/training.md` § Step 3 (the `--dry-run` flag)                  |
| How do I export the trained DCP back to safetensors?               | `docs/training.md` § Export checkpoint to Hugging Face safetensors  |
| How do I run inference with the trained checkpoint?                | `docs/training.md` § Run inference with trained checkpoint          |
| How do I evaluate an action checkpoint (forward/inverse/policy)?   | `docs/training.md` § Evaluation                                     |
| How do I run `cosmos3.scripts.eval` / `eval.py`?                   | `docs/training.md` § Run action evaluation with trained checkpoint  |
| Latency vs throughput preset for action eval?                      | `docs/training.md` § Run action evaluation with trained checkpoint  |
| What metrics does action eval report (PSNR, action MSE)?           | `docs/training.md` § Evaluation                                     |
| Where do training and action-eval artifacts land?                  | `docs/training.md` § Outputs                                        |
| How do I caption raw videos for SFT?                               | `docs/training.md` § Video Captioning for Training Data Processing  |
| How do I serve the captioning VLM?                                 | `docs/training.md` § Server setup                                   |
| How do I build a JSONL dataset from captions + videos?             | `docs/training.md` § Creating Video Dataset JSONL File for Training |

## Workflow at a glance

1. **Setup** — install the training extras: `uv sync --all-extras --group=cu130-train` (or `cu128-train` on older drivers), then `source .venv/bin/activate && export LD_LIBRARY_PATH=`.
2. **Step 1 — Prepare data and checkpoint** — download the example bridge dataset to `$DATASET_PATH` (Hugging Face cache) and `convert_model_to_dcp` the base checkpoint into `$BASE_CHECKPOINT_PATH` (default: `/tmp/$USER/checkpoints/cosmos3_nano`).
3. **Step 2 — Prepare config** — the provided `cosmos3/configs/experiment/mixed_modality_sft_8b.yaml` runs as-is on the example dataset (~100 iterations); override `model.config.parallelism.data_parallel_shard_degree`, `dataloader_train.dataloader.datasets.*.jsonl_paths`, `optimizer.lr`, `trainer.max_iter`, etc. for custom runs.
4. **Step 3 — Run training** — `torchrun --nproc_per_node=8 -m cosmos3.scripts.train -o outputs/train --config-file cosmos3/configs/experiment/mixed_modality_sft_8b.yaml --config-overrides "checkpoint.load_path=$BASE_CHECKPOINT_PATH" "dataloader_train.dataloader.datasets.video.dataset.jsonl_paths=$DATASET_PATH/train/video_dataset_file.jsonl"` (use `--dry-run` first when iterating on config). DCP checkpoints land in `outputs/train/job/checkpoints/iter_<N>`.
5. **Inference** — read `outputs/train/job/checkpoints/latest_checkpoint.txt`, point `cosmos3.scripts.inference` at the resulting `outputs/train/job/checkpoints/iter_<N>` DCP path with `--config-file outputs/train/config.yaml`. The example input glob `"$DATASET_PATH/val/inference_prompt*/episode_049683_clip000.json"` covers T2V, I2V, and V2V (see `cosmos3-inference` skill for presets / input formats).
6. **Action evaluation (action checkpoints only)** — `torchrun --nproc-per-node=8 -m cosmos3.scripts.eval --parallelism-preset=throughput -o outputs/train_eval --checkpoint-path $CHECKPOINT_PATH --config-file outputs/train/config.yaml --root-override /path/to/eval/dataset`. Resolves the dataloader from the training config (`val` split, falling back to `dataloader_train`), generates each sample through the inference engine, and scores against GT — `psnr` for video modes (`forward_dynamics`, `policy`) and `action_mse` for action modes (`inverse_dynamics`, `policy`). Per-sample `metrics.json` lives next to each `vision.mp4`; rank-0 aggregate is `outputs/train_eval/metrics_aggregate.json`. Skip this step entirely for T2V/I2V/V2V checkpoints — eval.py only computes action-mode metrics.
7. **Export (optional)** — `cosmos3.scripts.export_model` converts the DCP iter to Hugging Face safetensors at `outputs/train/model`. Not required for the standard inference flow above.

## Things not obvious from the docs

- **Training extras are a separate group**: SFT requires the `cu130-train` / `cu128-train` install group, not the inference-only `cu130` / `cu128`. Re-running `uv sync` with the wrong group silently leaves training deps uninstalled.
- **`-o` controls the entire output tree**: passing `-o outputs/train` to `cosmos3.scripts.train` makes everything land under `outputs/train/job/...` (logs, `config.yaml`, `checkpoints/iter_<N>`, callback outputs). Without `-o`, training falls back to `${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/{job.project}/{job.group}/{job.name}/`.
- **Inference uses the DCP checkpoint directly**: the standard flow points `cosmos3.scripts.inference` at `outputs/train/job/checkpoints/iter_<N>` together with `--config-file outputs/train/config.yaml`. The Hugging Face safetensors export (`outputs/train/model`) is optional — only needed if you want a portable single-file checkpoint.
- **Mixed-modality input glob**: the example uses `"$DATASET_PATH/val/inference_prompt*/episode_049683_clip000.json"` with a `*` so a single command runs T2V, I2V, and V2V (the dataset has `inference_prompt/`, `inference_prompt_i2v/`, `inference_prompt_v2v/` siblings under `val/`).
- **`data_parallel_shard_degree` must equal `WORLD_SIZE`**: it has to match `--nproc_per_node` on the `torchrun` command. Mismatch → FSDP init failure.
- **`--dry-run`**: `cosmos3.scripts.train` accepts `--dry-run` to validate the config end-to-end without launching training. Use it whenever iterating on YAML overrides.
- **`eval.py` is action-only**: `cosmos3.scripts.eval` only scores action-mode generations (PSNR for predicted video, MSE for predicted action). It does *not* score the bridge-v2 video SFT walkthrough or any T2V/I2V/V2V checkpoint — those use `cosmos3.scripts.inference` (no GT scoring). Pointing `eval.py` at a non-action dataloader fails with "mode requires GT video/action but data_batch had none".
- **Throughput preset for full-dataset action eval**: `--parallelism-preset` defaults to `latency` (model sharded across all ranks, one sample at a time — required when the checkpoint is too large to fit on a single GPU). For full-dataset action eval when the model fits on one GPU, pass `--parallelism-preset=throughput` so wall-clock scales as `N / num_gpus × per_sample_time` instead of `N × per_sample_time`.
- **Action eval reuses the training dataloader via `--config-file`**: pointing `eval.py` at `outputs/train/config.yaml` resolves the same dataloader the model was trained against (`val` split by default; falls back to `dataloader_train` when there is no `dataloader_val`). Use `--root-override /path/to/eval/dataset` to swap in held-out data without editing the config; alternatives are `--gcs-root-override <s3-uri>` (downloads via `--cache-dir`) and `--gcs-path-map`. Use `--dataset <key-or-name>` when the dataloader has multiple entries.
- **`--model-mode` defaults to `joint`**: every dataset entry is evaluated under all three action modes — total generation count = `len(modes) × ceil(len(val_split) / sample_stride)`, capped by `--num-samples`. Restrict during development with `--num-samples N`, `--sample-stride K`, or `--model-mode <single mode>`. Mode is also encoded in each sample's name (`<dataset>/<mode>/<id>`) and is what the metric dispatcher reads back when scoring.
- **Captioning server flags**: `vllm serve ... --allowed-local-media-path /` is required so the VLM can read the `file://` paths the captioning script sends. Use `Qwen/Qwen3-VL-8B-Instruct-FP8` as the recommended model; first launch downloads weights and may take several minutes (server is ready when you see `Application startup complete.`).
- **Captioning input modes**: `cosmos3.scripts.caption_from_video` accepts `--video <file_or_dir>` (single file or directory of `.mp4`s) or `-i <jsonl>` where each line has a `vision_path` field — same JSONL format used downstream by training.
- **Captioning output layout**: each input video produces a directory containing `caption.txt` and `sample_args.json`; `captions_to_sft_jsonl` then assembles those plus the source videos into a training-ready JSONL.
- **`uv run` over bare `python`**: per repo convention, prefer `uv run` for new commands. The `python -m cosmos3.scripts.caption_from_video ...` snippets in `docs/training.md` are a known pending migration.

## Related skills

| Skill                                  | When to use                                                                  |
| -------------------------------------- | ---------------------------------------------------------------------------- |
| `../cosmos3-setup/SKILL.md`            | Initial install, CUDA variant selection, container/`LD_LIBRARY_PATH` setup   |
| `../cosmos3-inference/SKILL.md`        | Inference parameters, parallelism presets, input JSON format, online serving |
| `../cosmos3-codebase-nav/SKILL.md`     | Locating configs, scripts, and defaults inside the package                   |
| `../cosmos3-env-troubleshoot/SKILL.md` | Debugging environment / runtime errors during training                       |
