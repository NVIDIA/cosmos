# Post-Training (Supervised Fine-Tuning)

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Training](#training)
  - [Step 1 - Prepare data and config](#step-1---prepare-data-and-config)
  - [Step 2 — Prepare checkpoint](#step-2--prepare-checkpoint)
  - [Step 3 — Run training](#step-3--run-training)
    - [VFM Post-Training](#vfm-post-training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Config](#config)

______________________________________________________________________

<!--TOC-->

Fine-tune a pre-trained Cosmos3 model on your own dataset using supervised fine-tuning (SFT).

This guide was tested on the following environments:

- 8x H100 (80 GB)

Prerequisites:

- [Setup](../README.md#setup)
- [Environment Variables](./environment_variables.md)

## Training

### Step 1 - Prepare data and config

Datasets are downloaded to the [Hugging Face cache](https://huggingface.co/docs/datasets/cache).

**Note:** Some datasets are license gated; visit the repository page and accept any license terms. Ensure you are authenticated with `uvx hf@latest auth login`.

Select one of the following recipes:

<details open><summary><b>Vision Generation with JSONL Dataset</b></summary>

Fine-tune vision generation on [nvidia/bridge-v2-subset-synthetic-captions](https://huggingface.co/datasets/nvidia/bridge-v2-subset-synthetic-captions/tree/main).

```shell
# Nano
BASE_CHECKPOINT_NAME=Cosmos3-Nano
CONFIG_FILE="cosmos3/configs/experiment/mixed_modality_sft_nano.yaml"

# Or, Super with LoRA
BASE_CHECKPOINT_NAME=Cosmos3-Super
CONFIG_FILE="cosmos3/configs/experiment/mixed_modality_sft_cosmos3_super.yaml"

export DATASET_PATH=$(uvx hf@latest download --repo-type dataset nvidia/bridge-v2-subset-synthetic-captions --revision 46468e12ac0dd36901e9e3240d4fc7620942b5d7 --quiet)/sft_dataset_bridge
```

For more details, see [JSONL Dataset](./dataset_jsonl.md).

</details>

<details><summary><b>Action Policy with Bridge LeRobot Dataset</b></summary>

Fine-tune action policy on [nvidia/LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3/). This config trains in policy mode with concatenated `agentview` and wrist camera observations, frame-wise relative actions, 6D rotations, and quantile-rotation action normalization.

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Nano
CONFIG_FILE="cosmos3/configs/experiment/action_policy_sft_nano.yaml"

export DATASET_PATH=$(uvx hf@latest download --repo-type dataset nvidia/LIBERO_LeRobot_v3 --revision ddc1edeb6e51e2b7d4d2ba7a1433daaecd37aa64 --quiet)
```

For evaluation, see [LIBERO Closed-Loop Evaluation](./action_policy_closed_loop_eval.md).

</details>

<details><summary><b>Action Forward Dynamics with Bridge LeRobot Dataset</b></summary>

Fine-tune action forward-dynamics model on [nvidia/bridge_lerobot_v3](https://huggingface.co/datasets/nvidia/bridge_lerobot_v3). This config trains in `forward_dynamics` mode with the `ego_view` viewpoint, quantile action normalization, and backward-framewise pose convention.

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Nano
CONFIG_FILE="cosmos3/configs/experiment/action_fdm_sft_nano.yaml"

export DATASET_PATH=$(uvx hf@latest download --repo-type dataset nvidia/bridge_lerobot_v3 --revision b887e193b141f2fe5b6e3d567577aa51c475693b --quiet)
```

</details>

### Step 2 — Prepare checkpoint

Convert base checkpoint to [PyTorch Distributed Checkpoint (DCP)](https://pytorch.org/docs/stable/distributed.checkpoint.html) format:

```shell
BASE_CHECKPOINT_PATH=/tmp/$USER/checkpoints/$BASE_CHECKPOINT_NAME
python -m cosmos3.scripts.convert_model_to_dcp \
  -o $BASE_CHECKPOINT_PATH \
  --checkpoint-path $BASE_CHECKPOINT_NAME
```

### Step 3 — Run training

#### VFM Post-Training

Launch distributed training with `torchrun`:

```shell
torchrun --nproc_per_node=8 -m cosmos3.scripts.train \
    -o outputs/train \
    --config-file $CONFIG_FILE \
    --config-overrides \
    "checkpoint.load_path=$BASE_CHECKPOINT_PATH"
```

The dataset path is read from the `DATASET_PATH` environment variable.

Arguments:

- `-o`: Output directory for checkpoints and logs.
- `--config-file`: Training config YAML from Step 3.
- `--config-overrides`: [Hydra overrides](https://hydra.cc/docs/advanced/override_grammar/basic/). See [Config](#config).
- `--dry-run`: Validate the setup without running training.
- `--resume`: Resume training from the latest checkpoint.

Outputs:

1. `debug.log`, `console.log`: Logs.
1. `config.yaml`: Finalized hydra config.
1. `job/`
    1. `checkpoints/`
        1. `latest_checkpoint.txt`: Text file containing the latest checkpoint iteration.
        1. `iter_<iter>/`: DCP checkpoints saved every `{checkpoint.save_iter}` iterations.
    1. `<callback_name>/`: Callback outputs.

## Inference

Export the DCP checkpoint to a Hugging Face safetensors checkpoint:

```shell
CHECKPOINT_ITER=$(cat outputs/train/job/checkpoints/latest_checkpoint.txt)
CHECKPOINT_PATH=outputs/train/job/checkpoints/$CHECKPOINT_ITER

python -m cosmos3.scripts.export_model \
  --checkpoint-path $CHECKPOINT_PATH \
  --config-file outputs/train/config.yaml \
  -o outputs/train/model
```

The checkpoint can be used in [Inference](../README.md#inference) commands by passing `--checkpoint-path outputs/train/model`.

## Evaluation

**Supported modalities:** Forward Dynamics, Inverse Dynamics, Policy.

Run inference on held-out dataset split and compare to ground truth:

```shell
torchrun --nproc-per-node=8 -m cosmos3.scripts.eval \
    -o outputs/train_eval \
    --checkpoint-path outputs/train/model \
    --dataset.config-file outputs/train/config.yaml
```

Arguments:

- `--dataset.model-mode`: Which modality to evaluate.
- `--dataset.num-samples N`: Maximum number of samples to evaluate.

Outputs:

- `metrics_aggregate.json`: Aggregate metrics.
- `<dataset>/<mode>/<id>/metrics.json`: Per sample metrics.

Metrics:

- Vision: Peak Signal-to-Noise Ratio (PSNR)
- Action: Mean Squared Error (MSE)

## Config

The following config fields are commonly updated:

1. `job`
    1. `project`, `group`, `name`: Job directory path.
    1. `cluster`
        1. `wandb_mode`: Enable Wandb logging (choices: `online`, `disabled`). The `online` mode requires setting environment variable `WANDB_API_KEY`.
1. `model`
    1. `config`
        1. `parallelism`
            1. `data_parallel_shard_degree`: FSDP shard size. `data_parallel_shard_degree * context_parallel_shard_degree` must equal `WORLD_SIZE`.
            1. `context_parallel_shard_degree`: Context-parallel shard size.
            1. `use_torch_compile`: Enable torch compile. Improves speed, but increases memory consumption.
        1. `lora_enabled`: Enable LoRA adapters on the matched linear layers.
        1. `lora_rank`: LoRA inner dimension.
        1. `lora_alpha`: LoRA scaling factor; effective scale is `lora_alpha / lora_rank`.
1. `checkpoint`
    1. `load_path`: Base DCP checkpoint path.
    1. `save_iter`: Save DCP checkpoint every N iterations.
1. `dataloader_train`
    1. `dataloader`
        1. `datasets.*`
            1. `jsonl_paths`: Dataset JSONL file paths.
            1. `num_video_frames`: Number of input video frames to sample (`-1`: all frames).
        1. `num_workers`: Number of dataloader workers.
1. `trainer`
    1. `max_iter`: Total number of training iterations.
    1. `grad_accum_iter`: Number of iterations to accumulate gradients to increase the effective batch size with limited memory.
1. `optimizer`
    1. `lr`: Learning rate.
