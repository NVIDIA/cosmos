# Action FD DROID Post-Training

This document describes how to run the `action_fd_droid_posttrain` experiment.
It trains Cosmos3-DROID forward dynamics in `cosmos_framework`.

## Overview

| Piece                     | Value                                                                                                                  |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Experiment                | `action_fd_droid_posttrain`                                                                                            |
| TOML                      | `toml/sft_config/action_fd_droid_posttrain.toml`                                                                       |
| Launch shell              | `launch_sft_action_fd_droid_posttrain.sh`                                                                              |
| Config module             | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_fd_droid_posttrain.py`                        |
| Dataset wrapper           | `cosmos_framework/data/generator/action/datasets/droid_merged_lerobot_dataset.py`                                      |
| Dataset root              | [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) parent root containing `success/` and `failure/` |
| Task mode                 | `forward_dynamics`                                                                                                     |
| Action space              | `ee_pose`: 10-D `[pos_delta, rot6d_delta, gripper]`                                                                    |
| Chunk length / resolution | `16` frames at `480`                                                                                                   |

## Prerequisites

- Install cosmos-framework as described in the shared
  [setup instructions](../../../README.md), then activate its
  virtual environment.
- Authenticate with Hugging Face (`uvx hf@latest auth login` or `HF_TOKEN`) and
  accept the terms for the required model and dataset repositories.
- The TOML defaults to online Weights & Biases logging. Export `WANDB_API_KEY`,
  or disable it with `EXTRA_TAIL_OVERRIDES="job.wandb_mode=disabled"`.
- Run commands from `cookbooks/cosmos3/generator/action/finetune/`.
- In NGC / PyTorch containers, set `LD_LIBRARY_PATH=''` before Python commands.

## Inputs You Provide

This package ships the training stack — the registered `action_fd_droid_posttrain`
experiment, the dataset class, and the paired TOML/launch shell. Three inputs
are external and must be provided per environment:

1. **[Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) dataset (in LeRobotDataset v3.0 format)** — pre-download the
   dataset and point `DATASET_PATH` at the
   resulting `.../Cosmos3-DROID` parent directory. This experiment trains on both
   `success/` and `failure/` subsets.
2. **DCP base checkpoint** — prepare a base DCP checkpoint and point
   `BASE_CHECKPOINT_PATH` at it. For local smoke runs this can be
   `checkpoints/Cosmos3-Nano`.
3. **Wan2.2 VAE** — download `Wan2.2_VAE.pth` from
   [`Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
   and point `WAN_VAE_PATH` at the file.

## Data Layout

Set `DATASET_PATH` to the Cosmos3-DROID parent directory:

```shell
export DATASET_PATH=/path/to/Cosmos3-DROID
```

The loader uses both subsets:

```text
$DATASET_PATH/
├── success/
│   └── meta/info.json
└── failure/
    └── meta/info.json
```

or institution-sharded split roots:

```text
$DATASET_PATH/
├── success/<institution>/meta/info.json
└── failure/<institution>/meta/info.json
```

The launch shell passes `DATASET_PATH` through to the training config.

## Full Reproduction

The flow mirrors the other SFT recipes:

```shell
# Step 1: prepare Cosmos3-DROID -> $DATASET_PATH.
# If you do not already have the dataset locally, download it from Hugging Face.
uvx hf@latest download --repo-type dataset nvidia/Cosmos3-DROID \
  --local-dir data/Cosmos3-DROID --quiet
export DATASET_PATH=data/Cosmos3-DROID

# Step 2: prepare the base DCP checkpoint and Wan2.2 VAE.
python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o checkpoints/Cosmos3-Nano \
  --checkpoint-path Cosmos3-Nano
uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --local-dir checkpoints/wan22_vae
export BASE_CHECKPOINT_PATH=checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=checkpoints/wan22_vae/Wan2.2_VAE.pth

# Step 3: optionally choose the output root and launch.
export OUTPUT_ROOT=/path/to/output_root
export LD_LIBRARY_PATH=''

bash launch_sft_action_fd_droid_posttrain.sh
```

`BASE_CHECKPOINT_PATH` should point at the base DCP checkpoint. When using the
paired launch shell, it must be a local directory because the shared launcher
validates it before starting `torchrun`.

## Validate The Config

Use `--dryrun` before launching workers:

```shell
PYTHONPATH=. python -m cosmos_framework.scripts.train \
  --sft-toml toml/sft_config/action_fd_droid_posttrain.toml \
  --dryrun
```

## Run Training

Recommended paired launch shell:

```shell
bash launch_sft_action_fd_droid_posttrain.sh
```

Pass short smoke-run overrides through `EXTRA_TAIL_OVERRIDES`:

```shell
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10"
bash launch_sft_action_fd_droid_posttrain.sh
```

Single-node, 8 GPU:

```shell
PYTHONPATH=. torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
  --sft-toml toml/sft_config/action_fd_droid_posttrain.toml
```

Multi-node HSDP:

```shell
PYTHONPATH=. torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=8 \
  -m cosmos_framework.scripts.train \
  --sft-toml toml/sft_config/action_fd_droid_posttrain.toml \
  -- model.parallelism.data_parallel_replicate_degree=$NNODES
```

Keep `data_parallel_shard_degree=8` and set
`model.parallelism.data_parallel_replicate_degree` to the number of nodes.

## Outputs

Training outputs land under:

```text
$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_fd/action_sft/<job.name>/
```

DCP checkpoints are saved under:

```text
$RUN_DIR/checkpoints/iter_<N>/
```

The run is resumable by relaunching with the same output directory and job name.

## Notes

- This recipe uses `mode="forward_dynamics"`, so actions are conditioning and the
  model trains video prediction from the first frame plus action sequence.
- `DROIDMergedLeRobotDataset` uses `action_space="ee_pose"` for the 10-D
  end-effector pose action path.
