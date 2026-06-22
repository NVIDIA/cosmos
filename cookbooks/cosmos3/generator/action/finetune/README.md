# Cosmos3-Nano-Policy-DROID Fine-Tuning (SFT)

This example demonstrates supervised fine-tuning (SFT) of [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) into an action policy for the DROID robot. It reproduces the post-training recipe used to create [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID), leveraging the public [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) dataset and the action-policy recipe from [cosmos-framework](https://github.com/NVIDIA/cosmos-framework).

| Recipe | Launch shell | Base model | Dataset |
| --- | --- | --- | --- |
| Policy-DROID SFT | `launch_sft_action_policy_droid.sh` | Cosmos3-Nano | [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) success split |

The recipe uses `[job].task = "vfm"` with the registered `action_policy_droid_nano` experiment. It trains a DROID policy model with `joint_pos` 8-D actions, proprioceptive state, `concat_view` 480p video, chunk length 32, episode-shuffle streaming, and the optional `keep_ranges_1_0_1.json` window filter.

## Prerequisites

1. **Install cosmos-framework.** This recipe drives `cosmos_framework.scripts.train`, so install a cosmos-framework checkout first — follow the shared [cosmos-framework setup](../../../README.md#cosmos-framework) (clone into `packages/cosmos3`, then `uv sync --all-extras --group=cu130-train`; use `cu128-train` on a CUDA 12.x driver).
2. **Recommended container.** For a curated CUDA + PyTorch base, NVIDIA recommends starting from the NGC PyTorch container **`nvcr.io/nvidia/pytorch:25.09-py3`** (CUDA 13; use **`:25.06-py3`** for a CUDA 12.8 driver). See the cosmos-framework [setup guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/setup.md#recommended-base-image).
3. **Activate** the cosmos-framework venv so `cosmos_framework` is importable: `source <path-to>/packages/cosmos3/.venv/bin/activate`.
4. **Hugging Face access.** The base model, Wan2.2 VAE, Cosmos3-DROID dataset, and `keep_ranges_1_0_1.json` filter are hosted on Hugging Face — authenticate once with `uvx hf@latest auth login` (or export `HF_TOKEN`) and accept any model/dataset terms first.
5. **Weights & Biases.** The TOML follows the canonical cosmos-framework example and defaults to `wandb_mode="online"`. Export `WANDB_API_KEY` for online logging, or add `job.wandb_mode=disabled` to `EXTRA_TAIL_OVERRIDES` for a local/no-W&B run.
6. **Prepare the Cosmos3-DROID dataset.** The full Cosmos3-DROID dataset is large, so this cookbook expects you to stage it explicitly. By default the launcher looks for `data/Cosmos3-DROID/success/meta/info.json` under this folder. Set `DATASET_PATH=/path/to/Cosmos3-DROID/success` to use another filesystem.
7. **Run from this directory** (`cookbooks/cosmos3/generator/action/finetune/`). Converted checkpoints, the Wan2.2 VAE, the `keep_ranges_1_0_1.json` filter, and run outputs default to `checkpoints/`, `data/`, and `outputs/` here (all git-ignored).

## Quick start

Stage the dataset, then run the launcher:

```shell
# One possible layout. The dataset is large; put it on a filesystem with enough space.
uvx hf@latest download \
    nvidia/Cosmos3-DROID \
    --repo-type dataset \
    --local-dir data/Cosmos3-DROID

bash launch_sft_action_policy_droid.sh
```

The launcher is a complete local wrapper for the public cookbook:

- checks that the Cosmos3-DROID success split exists
- downloads `Wan2.2_VAE.pth` if needed
- converts `Cosmos3-Nano` to a local DCP checkpoint if needed
- downloads `keep_ranges_1_0_1.json` if needed
- launches 8-GPU training with `action_policy_droid_repro.toml`

The script intentionally stays close to the `cosmos-framework` example launcher: `DATASET_PATH`
is bridged to `DROID_ROOT`, `BASE_CHECKPOINT_PATH` and `WAN_VAE_PATH` are exported for the TOML,
`EXTRA_TAIL_OVERRIDES` is appended after `--`, and `NPROC_PER_NODE` / `NNODES` / `NODE_RANK` /
`MASTER_ADDR` / `MASTER_PORT` map directly to `torchrun`. The extra cookbook behavior is the
reasoner-style local prep above and enabling the released `keep_ranges_1_0_1.json` filter by default.

Paths are fixed at the top of the script (under this git-ignored folder) — edit them there or export env vars to relocate data and checkpoints:

```shell
export DATASET_PATH=/scratch/Cosmos3-DROID/success
export BASE_CHECKPOINT_PATH=/scratch/checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=/scratch/checkpoints/wan22_vae/Wan2.2_VAE.pth
export FILTER_PATH=/scratch/droid/keep_ranges_1_0_1.json
bash launch_sft_action_policy_droid.sh
```

To run a short smoke test, keep the same inputs and override the iteration/batch knobs:

```shell
export EXTRA_TAIL_OVERRIDES="job.wandb_mode=disabled trainer.max_iter=10 checkpoint.save_iter=10 dataloader_train.max_samples_per_batch=32"
bash launch_sft_action_policy_droid.sh
```

## Outputs

Training writes to `outputs/train/<project>/<group>/<name>/`:

- `checkpoints/iter_<N>/` — DCP checkpoint (model / optim / scheduler / trainer state); `checkpoints/latest_checkpoint.txt` names the newest.
- `config.yaml`, launch metadata, logs, and one directory per registered callback.

## Export to Hugging Face safetensors

```shell
RUN_DIR=outputs/train/<project>/<group>/<name>
CKPT=$RUN_DIR/checkpoints/$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")
python -m cosmos_framework.scripts.export_model \
    --checkpoint-path "$CKPT" \
    --config-file "$RUN_DIR/config.yaml" \
    -o "$RUN_DIR/model"
```

Use the exported `$RUN_DIR/model` with the [Cosmos3-Nano-Policy-DROID inference cookbook](../run_policy_with_cosmos_framework.md).

## Advanced configuration

These recipes are intentionally minimal. For the full post-training reference — raw `torchrun`, resuming, every TOML field, the `keep_ranges_1_0_1.json` filter, and advanced HSDP/multi-node parallelism — see the canonical cosmos-framework docs:

- [Cosmos3-Nano-Policy-DROID post-training guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_posttrain.md)
- [Post-Training (SFT) guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md)
- [SFT structured-TOML config reference](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/sft_config.md)
- [environment variables](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/environment_variables.md) · [FAQ / OOM during SFT](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/faq.md)

> SFT here is a multi-GPU `torchrun` job, so this cookbook ships as a launch script + this README rather than a one-click notebook.
