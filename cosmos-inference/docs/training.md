# Post-Training (Supervised Fine-Tuning)

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Setup](#setup)
- [Training](#training)
  - [Step 1 — Prepare data](#step-1--prepare-data)
    - [VFM Bridge dataset](#vfm-bridge-dataset)
    - [Action LIBERO dataset](#action-libero-dataset)
  - [Step 2 — Prepare checkpoint](#step-2--prepare-checkpoint)
  - [Step 3 — Prepare config](#step-3--prepare-config)
  - [Step 4 — Run training](#step-4--run-training)
    - [VFM Post-Training](#vfm-post-training)
    - [Action Policy Post-Training](#action-policy-post-training)
- [Inference](#inference)
  - [Run inference with trained checkpoint](#run-inference-with-trained-checkpoint)
    - [Result Comparison](#result-comparison)
  - [Export checkpoint to Hugging Face safetensors](#export-checkpoint-to-hugging-face-safetensors)
- [Evaluation](#evaluation)
  - [Run evaluation with trained checkpoint](#run-evaluation-with-trained-checkpoint)
- [Outputs](#outputs)
- [Video Captioning for Training Data Processing](#video-captioning-for-training-data-processing)
  - [Server setup](#server-setup)
  - [Running Video Captioning](#running-video-captioning)
- [Creating Video Dataset JSONL File for Training](#creating-video-dataset-jsonl-file-for-training)

______________________________________________________________________

<!--TOC-->

Fine-tune a pre-trained Cosmos3 model on your own video dataset using supervised fine-tuning (SFT). The workflow has two stages:

1. **Training** (4 steps): prepare data, prepare the checkpoint, prepare the config, and launch distributed training with `torchrun`.
2. **Inference** (2 steps): run inference with the trained checkpoint, and optionally export it to Hugging Face safetensors.
3. **Evaluation** (1 step): run evaluation with the trained checkpoint. Currently supported modalities are Forward Dynamics, Inverse Dynamics, and Policy.

This guide was tested on the following environments:

- 8x H100 (80 GB)

## Setup

Install training dependencies. Pick the CUDA group that matches your driver — see [Setup](./setup.md#cuda-variants) to check your driver's max CUDA version.

```shell
# CUDA 13.0 (recommended; needs newer driver)
uv sync --all-extras --group=cu130-train

# OR, CUDA 12.8 (use this if your driver does not support CUDA 13.0)
uv sync --all-extras --group=cu128-train

source .venv/bin/activate && export LD_LIBRARY_PATH=
```

Environment variables:

- `IMAGINAIRE_OUTPUT_ROOT=/tmp/imaginaire4-output`: Output directory for training DCP checkpoints. We recommend at least 1 TB of free disk space.
- `COSMOS_SMOKE=1`: Enable smoke test.
- `COSMOS_VERBOSE=1`: Enable verbose console output.

## Training

### Step 1 — Prepare data

#### VFM Bridge dataset

Download example dataset [nvidia/bridge-v2-subset-synthetic-captions](https://huggingface.co/datasets/nvidia/bridge-v2-subset-synthetic-captions/tree/main) (~650 MB) to the [Hugging Face cache](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhubcache):

```shell
DATASET_PATH=$(uvx hf@latest download --repo-type dataset nvidia/bridge-v2-subset-synthetic-captions --include "sft_dataset_bridge/*" --quiet)/sft_dataset_bridge
```

#### Action LIBERO dataset

Download the LIBERO LeRobot v3 dataset from [nvidia/LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3/) into `outputs/libero_datasets`:

```shell
mkdir -p outputs/libero_datasets
uvx hf@latest download \
    --repo-type dataset nvidia/LIBERO_LeRobot_v3 \
    --local-dir outputs/libero_datasets
```

### Step 2 — Prepare checkpoint

Convert base checkpoint to [PyTorch Distributed Checkpoint (DCP)](https://pytorch.org/docs/stable/distributed.checkpoint.html) format:

```shell
BASE_CHECKPOINT_PATH=/tmp/$USER/checkpoints/cosmos3_nano
torchrun -m cosmos3.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Nano \
  -o $BASE_CHECKPOINT_PATH
```

### Step 3 — Prepare config

> **Quick start:** the provided `cosmos3/configs/experiment/mixed_modality_sft_8b.yaml` works as-is for the example dataset. Skip to [Step 4](#step-4--run-training).

A config template for the Cosmos3-Nano model is provided at [`mixed_modality_sft_8b.yaml`](../cosmos3/configs/experiment/mixed_modality_sft_8b.yaml). The following fields are commonly updated:

1. `job`
    1. `project`, `group`, `name`: Job directory path.
    1. `cluster`
        1. `wandb_mode`: Enable Wandb logging (choices: `online`, `disabled`).
1. `model`
    1. `config`
        1. `parallelism`
            1. `data_parallel_shard_degree`: Model FSDP shard size. Should be set to `WORLD_SIZE`.
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

### Step 4 — Run training

#### VFM Post-Training

Launch distributed training with `torchrun`:

```shell
torchrun --nproc_per_node=8 -m cosmos3.scripts.train \
    -o outputs/train \
    --config-file cosmos3/configs/experiment/mixed_modality_sft_8b.yaml \
    --config-overrides \
    "checkpoint.load_path=$BASE_CHECKPOINT_PATH" \
    "dataloader_train.dataloader.datasets.video.dataset.jsonl_paths=$DATASET_PATH/train/video_dataset_file.jsonl"
```

#### Action Policy Post-Training

Use [`action_policy_sft_8b.yaml`](../cosmos3/configs/experiment/action_policy_sft_8b.yaml) to post-train an action policy on LIBERO. This config trains in policy mode with concatenated `agentview` and wrist camera observations, frame-wise relative actions, 6D rotations, and quantile-rotation action normalization.

```shell
BASE_CHECKPOINT_PATH=outputs/checkpoints/action_policy_sft_8b

torchrun --nproc_per_node=8 -m cosmos3.scripts.train \
    -o outputs/libero_sft \
    --config-file cosmos3/configs/experiment/action_policy_sft_8b.yaml \
    --config-overrides \
    "checkpoint.load_path=$BASE_CHECKPOINT_PATH"
```

Flags:

- `-o`: Output directory for checkpoints and logs.
- `--config-file`: Training config YAML from Step 3.
- `--dry-run`: Validate the setup without running training.
- `--config-overrides`: [Hydra overrides](https://hydra.cc/docs/advanced/override_grammar/basic/). See [Config](#step-3--prepare-config).

## Inference

### Run inference with trained checkpoint

Get the path to the latest training checkpoint:

```shell
CHECKPOINT_ITER=$(cat outputs/train/job/checkpoints/latest_checkpoint.txt)
CHECKPOINT_PATH=outputs/train/job/checkpoints/$CHECKPOINT_ITER
```

Run Text2Video (T2V), Image2Video (I2V), and Video2Video (V2V) inference for a single clip with the output checkpoint:

```shell
COSMOS_TRAINING=1 torchrun --nproc-per-node=8 -m cosmos3.scripts.inference \
    --parallelism-preset=latency \
    -i "$DATASET_PATH/val/inference_prompt*/episode_049683_clip000.json" \
    -o outputs/train_inference \
    --checkpoint-path $CHECKPOINT_PATH \
    --config-file outputs/train/config.yaml \
    --seed=0
```

- The ground truth video is in `${DATASET_PATH}/val/videos/`.
- The input image for I2V is in `${DATASET_PATH}/val/images/`.
- The input 5-frame video clip for V2V is in `${DATASET_PATH}/val/videos_5frames/`.

#### Result Comparison

Each example below uses the following layout:

- Row 1 (T2V): ground truth video (left), before SFT (middle), after 500 iterations of SFT (right).
- Row 2 (I2V): input image (left), before SFT (middle), after 500 iterations of SFT (right).
- Row 3 (V2V): 5-frame input clip (left), before SFT (middle), after 500 iterations of SFT (right).

**episode_049683_clip000**

<details><summary><b>Input prompt</b></summary>

> A robotic arm with articulated joints and a gripping mechanism is positioned centrally on a wooden kitchen countertop, manipulating a small silver metal object while also interacting with scattered black coffee beans. The arm moves the metal object slightly, adjusting its position before shifting focus to the coffee beans, scattering and repositioning them with precision. The countertop is surrounded by kitchen elements, including a stove on the right, a microwave on the left, and two canned goods labeled "Tomato Juice" and "Baking Soda" in the background. The scene is illuminated by bright, even indoor lighting, casting minimal shadows, and the camera remains static throughout, offering a top-down perspective that emphasizes the robotic arm's movements. The composition centers on the robotic arm and its interaction with the metal object and coffee beans, with a shallow depth of field keeping the focus sharp on these elements while softly blurring the background. The overall atmosphere is technical and functional, highlighting the precision and control of the robotic manipulation within a domestic kitchen setting.

</details>

<video src="https://github.com/user-attachments/assets/d85ccc4f-7eea-46e0-afda-955ab9e19cbe" controls width="100%"></video>

**episode_009171_clip000**

<details><summary><b>Input prompt</b></summary>

> A robotic arm with a black and metallic gripper, accented with blue near its base, extends over a white rectangular tray filled with scattered brown almonds, methodically picking up and placing each almond in a precise line across the tray's surface. The arm moves with deliberate, controlled motion, shifting its position to reach different almonds while maintaining a top-down perspective that captures the entire workspace. The background reveals an indoor setting with a wooden table and various kitchen items, including a metal bowl and utensils, subtly visible behind the tray. The lighting is bright and evenly distributed, casting minimal shadows and highlighting the contrast between the white tray, the brown almonds, and the metallic sheen of the robotic arm. The camera remains static throughout, offering a wide-angle view that emphasizes the robotic arm's precision and the systematic rearrangement of the almonds, creating a clean, minimalist aesthetic that underscores the technical nature of the task. The scene unfolds as a continuous, uninterrupted sequence, showcasing the robotic arm's efficiency in organizing the almonds without any cuts or transitions.

</details>

<video src="https://github.com/user-attachments/assets/50623895-c49d-4dbb-a0c0-5281aaac5c23" controls width="100%"></video>

**Note:** For action policy inference and evaluation, please refer to [LIBERO Closed-Loop Evaluation](./action_policy_closed_loop_eval.md).

### Export checkpoint to Hugging Face safetensors

Optionally, convert the DCP checkpoint to a Hugging Face model:

```shell
torchrun -m cosmos3.scripts.export_model \
  --checkpoint-path $CHECKPOINT_PATH \
  --config-file outputs/train/config.yaml \
  -o outputs/train/model
```

## Evaluation

Supported modalities: Forward Dynamics, Inverse Dynamics, Policy.

`cosmos3.scripts.eval` scores checkpoints on a held-out dataset — `psnr` for predicted video, `action_mse` for predicted action.

### Run evaluation with trained checkpoint

```shell
torchrun --nproc-per-node=8 -m cosmos3.scripts.eval \
    -o outputs/train_eval \
    --checkpoint-path $CHECKPOINT_PATH \
    --config-file outputs/train/config.yaml \
    --root-override /path/to/eval/dataset
```

- `--config-file` reuses the training dataloader (`val` split, falling back to `dataloader_train`).
- `--root-override` swaps the dataset's baked-in `root` for a local path. Alternatives: `--gcs-root-override <s3-uri>` + `--cache-dir` to download, or `--dataset <name>` to pick one entry from a multi-dataset config.
- `--model-mode` defaults to `joint` (every entry evaluated under all three modes); pass a single mode (e.g. `forward_dynamics`) to evaluate only that mode.
- `--num-samples N` caps the number of dataset entries evaluated; `--sample-stride K` evaluates every K-th entry (defaults to `1`, i.e. every entry).
- For a quick debug run, shrink the workload with `--num-samples N`, `--sample-stride K`, or `--model-mode <single mode>`.
- Per-sample scores land in `outputs/train_eval/<dataset>/<mode>/<id>/metrics.json`; rank-0 aggregate is `outputs/train_eval/metrics_aggregate.json`.
- Optional flags: `--no-compute-metrics` (generation only), `--benchmark` (per-stage timings), `--debug` (raw tensors + pickled debug data).

## Outputs

The training output directory contains:

1. `debug.log`, `console.log`: Logs.
1. `config.yaml`: Finalized hydra config.
1. `job/`
    1. `checkpoints/`
        1. `latest_checkpoint.txt`: Text file containing the latest checkpoint iteration.
        1. `iter_<iter>/`: DCP checkpoints saved every `{checkpoint.save_iter}` iterations.
    1. `<callback_name>/`: Callback outputs.

| Artifact                                                  | Path                                            |
| --------------------------------------------------------- | ----------------------------------------------- |
| Example dataset                                           | `$DATASET_PATH`                                 |
| Base checkpoint (DCP)                                     | `$BASE_CHECKPOINT_PATH`                         |
| Trained checkpoint (DCP)                                  | `outputs/train/job/checkpoints/iter_<N>`        |
| Trained checkpoint (Hugging Face safetensors, optional)   | `outputs/train/model`                           |
| Inference video from trained model                        | `outputs/train_inference`                       |
| Action evaluation per-sample outputs (action checkpoints) | `outputs/train_eval/<dataset>/<mode>/<id>/`     |
| Action evaluation aggregate metrics (action checkpoints)  | `outputs/train_eval/metrics_aggregate.json`     |

______________________________________________________________________

## Video Captioning for Training Data Processing

If you have video sources and would like to synthesize caption annotations to build video–text pairs for training, follow this section for data preprocessing. The script sends each video directly to a Vision-Language Model (VLM), which analyzes the visual content and produces a dense narrative caption following a two-phase process (scene analysis → narrative rewrite) — the same format expected by the Cosmos3 training pipeline.

The captioning prompt template is available at [`cosmos3/defaults/video_captioner.txt`](../cosmos3/defaults/video_captioner.txt).

### Server setup

The captioning script passes video files to vLLM via `video_url` content parts using `file://` paths, so the server must be able to read files from the local filesystem. We recommend [Qwen/Qwen3-VL-8B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8) as the VLM. Start the server — this may take a couple of minutes:

```shell
uvx --with nvidia-cuda-runtime-cu12 \
    vllm@0.19.0 serve Qwen/Qwen3-VL-8B-Instruct-FP8 \
    --tensor-parallel-size 1 \
    --allowed-local-media-path /
```

The server is ready when you see `Application startup complete.`

### Running Video Captioning

Caption a single video:

```shell
python -m cosmos3.scripts.caption_from_video \
    --video /path/to/video.mp4 -o outputs/captions \
    --server http://localhost:8000/v1
```

Caption all `.mp4` files in a directory:

```shell
python -m cosmos3.scripts.caption_from_video \
    --video /path/to/videos/ -o outputs/captions \
    --server http://localhost:8000/v1
```

Caption videos listed in a JSONL manifest (each line must have a `vision_path` field pointing to a video):

```shell
python -m cosmos3.scripts.caption_from_video \
    -i samples.jsonl -o outputs/captions \
    --server http://localhost:8000/v1
```

Options:

| Flag                     | Default  | Description                      |
| ------------------------ | -------- | -------------------------------- |
| `--max_workers`          | `16`     | Concurrent API requests          |
| `--prompt_template_path` | built-in | Path to a custom prompt template |
| `--debug`                | `False`  | Save raw API responses           |

Each video produces an output directory containing `caption.txt` (the plain-text caption) and `sample_args.json` (metadata).

## Creating Video Dataset JSONL File for Training

After generating the captions, you will have videos and captions stored in the following file structure:

```
path/to/dataset/
└── captions/
└── videos/
```

To create a video dataset JSONL file for post-training, run the following command:

```
python -m cosmos3.scripts.captions_to_sft_jsonl \
    --captions-dir outputs/sft_dataset/train/captions \
    --videos-dir outputs/sft_dataset/train/videos \
    -o outputs/sft_dataset/train/video_dataset_file.jsonl
```

It will create a dataset JSONL file containing captions and their corresponding paths to video files.
