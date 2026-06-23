# Cosmos3 Vision Generator Fine-Tuning (SFT)

Supervised fine-tuning (SFT) of the Cosmos3 video generator on your own captioned video data. Tested on 8×H100 (80 GB).

| Recipe | Launch shell | Base model | Dataset |
| --- | --- | --- | --- |
| Vision SFT (full) | `launch_sft_vision_nano.sh` | Cosmos3-Nano | [BridgeData2-Subset-Synthetic-Captions](https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions) |
| Vision SFT (LoRA) | `launch_sft_vision_super.sh` | Cosmos3-Super | same as above |

Both recipes train on structured-JSON captions (`caption_json`, the model's native prompt format), so training stays aligned with inference.

## Prerequisites

1. **Install the framework.** These recipes drive `cosmos_framework.scripts.train`, so install a cosmos-framework checkout first — follow the shared [Cosmos Framework setup](../../../README.md#cosmos-framework) (clone into `packages/cosmos3`, then `uv sync --all-extras --group=cu130-train`; use `cu128-train` on a CUDA 12.x driver).
2. **Recommended container.** For a curated CUDA + PyTorch base, NVIDIA recommends starting from the NGC PyTorch container **`nvcr.io/nvidia/pytorch:25.09-py3`** (CUDA 13; use **`:25.06-py3`** for a CUDA 12.8 driver). See the framework [setup guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/setup.md#recommended-base-image).
3. **Activate** the framework venv so `cosmos_framework` is importable: `source <path-to>/packages/cosmos3/.venv/bin/activate`.
4. **Hugging Face access.** Some assets are license-gated — accept terms on the dataset/model pages and authenticate once with `uvx hf@latest auth login` (or export `HF_TOKEN`).
5. **Run from this directory** (`cookbooks/cosmos3/generator/audiovisual/finetune/`). Downloads, converted checkpoints, and run outputs default to `data/`, `checkpoints/`, and `outputs/` here (all git-ignored).

## Quick start

Each launcher is a complete recipe — run it from this folder and it downloads the dataset, fetches the Wan2.2 VAE, converts the base checkpoint, then runs 8-GPU training (the download/convert steps are skipped if their outputs already exist):

```shell
bash launch_sft_vision_nano.sh      # full SFT on Cosmos3-Nano
# or
bash launch_sft_vision_super.sh     # LoRA SFT on Cosmos3-Super
```

Paths are fixed at the top of each script (under this git-ignored folder) — edit them there to put data or checkpoints on another filesystem.

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

Use the exported `$RUN_DIR/model` with the [audiovisual inference cookbook](../README.md).

## Advanced configuration

These recipes are intentionally minimal. For the full post-training reference — raw `torchrun`, resuming, every TOML field, parallelism / LoRA / EMA knobs, and the VFM↔VLM remap — see the canonical framework docs:

- [Post-Training (SFT) guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md)
- [SFT structured-TOML config reference](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/sft_config.md)
- [JSONL dataset format](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/dataset_jsonl.md) · [environment variables](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/environment_variables.md) · [FAQ / OOM during SFT](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/faq.md)

> SFT here is a multi-GPU `torchrun` job, so these cookbooks ship as launch scripts + this README rather than a one-click notebook.
