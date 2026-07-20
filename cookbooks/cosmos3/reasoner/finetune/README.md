# Cosmos3 Reasoner Fine-Tuning (SFT)

Supervised fine-tuning (SFT) of the Cosmos3 Reasoner on your own data. Tested on 8×H100 (80 GB).

| Recipe | Launch shell | Dataset | Notes |
| --- | --- | --- | --- |
| Alignment SFT (LLaVA-OneVision) | `launch_sft_llava_ov.sh` | [lmms-lab/LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data) | Streams from HF; Cosmos3-Nano Reasoner checkpoint auto-prepared |
| Physical-plausibility SFT (VideoPhy-2) | `launch_sft_videophy2_nano.sh` | [videophysics/videophy2_train](https://huggingface.co/datasets/videophysics/videophy2_train) | 1–5 plausibility scoring; dataset + checkpoint auto-prepared |
| Physical-plausibility SFT (VideoPhy-2, Cosmos3-Super) | `launch_sft_videophy2_super.sh` | [videophysics/videophy2_train](https://huggingface.co/datasets/videophysics/videophy2_train) | Cosmos3-Super tier — Qwen3-VL-32B full fine-tune; dataset + checkpoint auto-prepared |
| Physical-plausibility SFT (VideoPhy-2, Cosmos3-Edge) | `launch_sft_videophy2_edge.sh` | [videophysics/videophy2_train](https://huggingface.co/datasets/videophysics/videophy2_train) | Cosmos3-Edge tier — Nemotron-2B-Dense-VL (SigLIP2 tower, frozen); dataset auto-prepared; reasoner weights load directly from the (ungated) `nvidia/Cosmos3-Edge` snapshot |

All use `[job].task = "vlm"`. The Nano recipes bootstrap from a Cosmos3-Nano Reasoner checkpoint; the Cosmos3-Super recipe bootstraps from a Cosmos3-Super Reasoner checkpoint (Cosmos3-Super LM merged onto the Qwen3-VL-32B visual tower); the Nano/Super checkpoints are auto-prepared on first run. The Cosmos3-Edge recipe loads its reasoner weights (Edge's own Nemotron-2B-Dense-VL LM + SigLIP2 reasoner tower) directly from the public `nvidia/Cosmos3-Edge` snapshot at startup — no conversion step.

## Prerequisites

1. **Install the framework.** These recipes drive `cosmos_framework.scripts.train`, so install a cosmos-framework checkout first — follow the shared [Cosmos Framework setup](../../README.md#cosmos-framework) (clone into `packages/cosmos3`, then `uv sync --all-extras --group=cu130-train`; use `cu128-train` on a CUDA 12.x driver).
2. **Recommended container.** For a curated CUDA + PyTorch base, NVIDIA recommends starting from the NGC PyTorch container **`nvcr.io/nvidia/pytorch:25.09-py3`** (CUDA 13; use **`:25.06-py3`** for a CUDA 12.8 driver). See the framework [setup guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/setup.md#recommended-base-image).
3. **Activate** the framework venv so `cosmos_framework` is importable: `source <path-to>/packages/cosmos3/.venv/bin/activate`.
4. **Hugging Face access.** The Cosmos3-Nano base checkpoint and datasets are fetched from HF — authenticate once with `uvx hf@latest auth login` (or export `HF_TOKEN`); accept any dataset terms first.
5. **Run from this directory** (`cookbooks/cosmos3/reasoner/finetune/`). Any downloads, converted checkpoints, and run outputs default to `data/`, `checkpoints/`, and `outputs/` here.

## Quick start

Each launcher is a complete recipe — just run it from this folder:

```shell
bash launch_sft_llava_ov.sh          # alignment SFT; dataset streams from HF, builds the Cosmos3-Nano Reasoner checkpoint, then trains
# or
bash launch_sft_videophy2_nano.sh    # first run materializes VideoPhy-2 + builds the Cosmos3-Nano Reasoner checkpoint, then trains
# or (Cosmos3-Super tier — Qwen3-VL-32B full fine-tune)
bash launch_sft_videophy2_super.sh   # first run materializes VideoPhy-2 + builds the Cosmos3-Super Reasoner checkpoint, then trains
# or (Cosmos3-Edge tier — Nemotron-2B-Dense-VL, 2B; also fits a 4-GPU node)
bash launch_sft_videophy2_edge.sh    # first run materializes VideoPhy-2, then trains (reasoner weights load directly from nvidia/Cosmos3-Edge)
```

The VideoPhy-2 download/convert steps are skipped once their outputs exist (Edge has no convert step — its weights stream from the HF cache). Paths are fixed at the top of each script — edit them there to relocate data or checkpoints.

These recipes default to 8 GPUs. On a 4-GPU node (e.g. GB200×4), set `--nproc_per_node=4` on the `torchrun` line in the launch script.

## Outputs

Training writes to `outputs/train/<project>/<group>/<name>/`:

- `checkpoints/iter_<N>/` — DCP checkpoint (model / optim / scheduler / trainer state); `checkpoints/latest_checkpoint.txt` names the newest.
- `config.yaml`, launch metadata, logs, and one directory per registered callback.

## Export to Hugging Face safetensors

```shell
RUN_DIR=outputs/train/<project>/<group>/<name>
CKPT=$RUN_DIR/checkpoints/$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")
python -m cosmos_framework.scripts.export_model \
    --checkpoint-path "$CKPT" --config-file "$RUN_DIR/config.yaml" -o "$RUN_DIR/model"
```

Use the exported `$RUN_DIR/model` with the [reasoner inference cookbook](../README.md).

## Advanced configuration

These recipes are intentionally minimal. For the full post-training reference — raw `torchrun`, resuming, every TOML field, and advanced parallelism — see the canonical framework docs:

- [Post-Training (SFT) guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md)
- [SFT structured-TOML config reference](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/sft_config.md)
- [JSONL dataset format](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/dataset_jsonl.md) · [environment variables](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/environment_variables.md) · [FAQ / OOM during SFT](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/faq.md)

> SFT here is a multi-GPU `torchrun` job, so these cookbooks ship as launch scripts + this README rather than a one-click notebook.
