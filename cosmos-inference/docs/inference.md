# Inference

> **Skill:** `.agents/skills/cosmos3-inference/SKILL.md`

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Setup Arguments](#setup-arguments)
  - [Parallelism](#parallelism)
- [Sample Arguments](#sample-arguments)
  - [Condition](#condition)
    - [Text](#text)
    - [Vision (Image/Video)](#vision-imagevideo)
    - [Action](#action)
  - [Generation](#generation)
    - [Vision (Image/Video)](#vision-imagevideo-1)
    - [Action](#action-1)
- [Action Inference](#action-inference)
  - [Action Modes](#action-modes)
  - [Action Configuration](#action-configuration)
- [Default Values](#default-values)
  - [Custom Defaults](#custom-defaults)
- [Schema Reference](#schema-reference)
- [Troubleshooting](#troubleshooting)
  - [Checkpoint Issue](#checkpoint-issue)
  - [Torch CUDA Out of Memory Error](#torch-cuda-out-of-memory-error)
  - [NCCL Issue](#nccl-issue)
    - [NCCL Plugin Issue](#nccl-plugin-issue)
- [Supplementary Examples](#supplementary-examples)

______________________________________________________________________

<!--TOC-->

This guide applies to the following:

- [Offline Batch Inference](../README.md#offline-batch-inference)
- [Online Inference](./inference_online.md)

## Setup Arguments

### Parallelism

| Parallelism | Arguments                         | Description                                                                                                    |
| ----------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Latency     | `--parallelism-preset=latency`    | Generates each sample as fast as possible by spreading work across GPUs. Use for interactive or real-time use. |
| Throughput  | `--parallelism-preset=throughput` | Generates many samples in parallel, one per GPU. Use for large batch jobs.                                     |

## Sample Arguments

Each inference run takes one or more **sample argument files** (JSON, JSONL, or YAML) that describe what to generate:

```shell
python -m cosmos3.scripts.inference \
    -i "inputs/omni/t2i.json" \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

| CLI Argument          | Description                                                                        |
| --------------------- | ---------------------------------------------------------------------------------- |
| `-i`, `--input-files` | Path to the sample argument file(s). Accepts glob patterns (e.g. `inputs/*.json`). |
| `-o`, `--output-dir`  | Output directory.                                                                  |

General sample arguments:

| Argument     | Description                                                                                                                                                                                                                                                                                                           |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_mode` | Generation modality. One of `text2image`, `text2video`, `image2image`, `image2video`, `video2video`, `forward_dynamics`, `inverse_dynamics`, `policy`. Also selects the matching default preset. When omitted, the VFM modality is inferred from `vision_path` and `num_frames`; action modes must be set explicitly. |
| `name`       | Output subfolder name (inside `--output-dir`).                                                                                                                                                                                                                                                                        |
| `seed`       | Random seed for reproducibility.                                                                                                                                                                                                                                                                                      |

### Condition

Condition fields control the inputs to generation. Paths are relative to the input file.

#### Text

Provide a text prompt inline via `prompt`, or point to a `.txt` file via `prompt_path`. If both are provided, `prompt` takes precedence. See [inputs/omni/t2v.json](../inputs/omni/t2v.json) for an example.

| Argument          | Description                                                      |
| ----------------- | ---------------------------------------------------------------- |
| `prompt`          | Inline text prompt.                                              |
| `prompt_path`     | Path to a `.txt` file with the prompt (alternative to `prompt`). |
| `negative_prompt` | Describes what to avoid in the output.                           |

#### Vision (Image/Video)

Provide an image or video via `vision_path`.

| Argument      | Description                                         |
| ------------- | --------------------------------------------------- |
| `vision_path` | Path to an image or video file (local path or URL). |

- Image conditioning: see [inputs/omni/i2v.json](../inputs/omni/i2v.json)

#### Action

Action inference is enabled by setting `model_mode` to one of the action tasks in the sample argument file. The examples live in [`inputs/omni/`](../inputs/omni/).

| Argument            | Description                                                                                                                               |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `model_mode`        | Selects the action task: `forward_dynamics`, `inverse_dynamics`, or `policy`. This also selects the matching default preset.              |
| `vision_path`       | Observation image or video. URLs are downloaded into the sample output's `inputs/` directory before generation.                           |
| `prompt`            | Text instruction or scene/task description used as the action sample caption.                                                             |
| `domain_name`       | Domain name passed to the action domain registry, such as `libero` or `av`. Use the domain used by the checkpoint's action training data. |
| `image_size`        | Action input resize bucket. The value is passed as the action media resolution bucket; examples use `256` for LIBERO and `480` for AV.    |
| `fps`               | Conditioning FPS and output video FPS.                                                                                                    |
| `action_chunk_size` | Number of action steps in the chunk. The action media loader reads at most `action_chunk_size + 1` observation frames.                    |
| `action_path`       | JSON action sequence. Required for `forward_dynamics`; each row is one action step and each column is one raw action dimension.           |
| `raw_action_dim`    | Raw action width to return for generated actions. Required for `inverse_dynamics` and `policy`.                                           |

`action_path` files are plain JSON arrays with shape `action_chunk_size x raw_action_dim`, for example `[[...], [...], ...]`.

For example, [`inputs/omni/action_forward_dynamics_robot.json`](../inputs/omni/action_forward_dynamics_robot.json) conditions on an observation image, a task prompt, and an action JSON to generate a future rollout. [`inputs/omni/action_inverse_dynamics_av.json`](../inputs/omni/action_inverse_dynamics_av.json) conditions on a video and predicts an action sequence with `raw_action_dim`.

### Generation

#### Vision (Image/Video)

Outputs `vision.jpg` or `vision.mp4` depending on `num_frames`.

| Argument       | Type                                                  | Description                                          |
| -------------- | ----------------------------------------------------- | ---------------------------------------------------- |
| `num_frames`   | `int`                                                 | Number of output frames. `1` = image; `≥24` = video. |
| `fps`          | `int`                                                 | Frames per second.                                   |
| `resolution`   | `"256"` \| `"480"` \| `"720"`                         | Output resolution (height in pixels).                |
| `aspect_ratio` | `"1,1"` \| `"4,3"` \| `"3,4"` \| `"16,9"` \| `"9,16"` | Output aspect ratio. Defaults to `16:9`.             |

#### Action

Action samples still write the generated visual output as `vision.mp4` in each sample subdirectory. When the model returns predicted actions, `sample_outputs.json` also contains `outputs[0].content.action` as a JSON list of action rows.

Use `--debug` to save raw generated tensors in `output.safetensors` and non-tensor debug data in `output.pickle`.

Typical action output layout:

```text
outputs/omni_nano/action_forward_dynamics_robot/
+-- inputs/
+-- sample_args.json
+-- sample_outputs.json
+-- vision.mp4
```

## Action Inference

Run one action sample:

```shell
python -m cosmos3.scripts.inference \
    -i inputs/omni/action_forward_dynamics_robot.json \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano\
    --seed=0
```

Run the bundled forward dynamics examples:

```shell
python -m cosmos3.scripts.inference \
    -i "inputs/omni/action_forward_dynamics_*.json" \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

For standalone `policy` or `inverse_dynamics` runs, include `raw_action_dim` in the sample JSON.

For S3 or other DCP checkpoints, use the same checkpoint arguments as regular inference, for example `--credential-path`, `--checkpoint-cache-dir`, `--config-file`, and `--experiment` when they are needed by the checkpoint.

### Action Modes

| Mode               | Inputs                                                | Outputs                                                                     | Required action fields |
| ------------------ | ----------------------------------------------------- | --------------------------------------------------------------------------- | ---------------------- |
| `forward_dynamics` | Observation image/video, text prompt, action sequence | Future visual rollout in `vision.mp4` or `vision.jpg`                       | `action_path`          |
| `inverse_dynamics` | Observation video, text prompt                        | Predicted action sequence in `sample_outputs.json`                          | `raw_action_dim`       |
| `policy`           | Current observation image/video, text prompt          | Predicted action sequence, and any visual output returned by the checkpoint | `raw_action_dim`       |

### Action Configuration

The action sample fields control input preprocessing and action tensor shape. `action_chunk_size` should match the chunk length used by the checkpoint. `image_size` should match the action training/evaluation resolution bucket. `domain_name` must be compatible with the checkpoint's action domain registry.

Action tensors are padded to `model.config.max_action_dim` before generation. Set it with `--experiment_overrides "[model.config.max_action_dim=<D>]"` when the checkpoint config does not already define the desired padded width. Use a value greater than or equal to the raw action width in `action_path` or `raw_action_dim`.

## Default Values

Each modality ships with a built-in preset that supplies sampling parameters (`num_steps`, `guidance`, `shift`, ...), negative prompts, and output settings. These presets are applied automatically. Any field you set in your sample argument file takes precedence over the preset.

The built-in presets live in the package under `cosmos3/defaults/`:

| Modality       | Preset File                                                                                         |
| -------------- | --------------------------------------------------------------------------------------------------- |
| Text-to-Image  | [`cosmos3/defaults/text2image/sample_args.json`](../cosmos3/defaults/text2image/sample_args.json)   |
| Text-to-Video  | [`cosmos3/defaults/text2video/sample_args.json`](../cosmos3/defaults/text2video/sample_args.json)   |
| Image-to-Video | [`cosmos3/defaults/image2video/sample_args.json`](../cosmos3/defaults/image2video/sample_args.json) |

Action presets use the same sample argument format:

| Modality         | Preset File                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| Forward Dynamics | [`cosmos3/defaults/forward_dynamics/sample_args.json`](../cosmos3/defaults/forward_dynamics/sample_args.json) |
| Inverse Dynamics | [`cosmos3/defaults/inverse_dynamics/sample_args.json`](../cosmos3/defaults/inverse_dynamics/sample_args.json) |
| Policy           | [`cosmos3/defaults/policy/sample_args.json`](../cosmos3/defaults/policy/sample_args.json)                     |

> **Tip:** Only the parameters listed in `python -m cosmos3.scripts.inference --help` are recommended to change. The remaining fields in the preset files are internal and may break generation if altered.

### Custom Defaults

To use your own default values instead of the built-in presets, pass a JSON file via the `defaults_file` field in your sample arguments:

```json
{
    "defaults_file": "my_defaults.json",
    "prompt": "..."
}
```

The custom defaults file has the same format as the built-in presets. Fields you set explicitly in the sample argument file still take precedence over the custom defaults file.

## Schema Reference

The `schemas/` directory contains auto-generated reference files listing every available argument with types, constraints, and descriptions. These files are the authoritative reference for field names and valid values.

| File                                                                                    | Description                                                      |
| --------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [`schemas/OmniSampleOverrides.yaml`](../schemas/OmniSampleOverrides.yaml)               | All sample arguments with default values and inline comments.    |
| [`schemas/OmniSampleOverrides.schema.json`](../schemas/OmniSampleOverrides.schema.json) | JSON Schema with types, enums, and validation constraints.       |
| [`schemas/OmniSetupOverrides.yaml`](../schemas/OmniSetupOverrides.yaml)                 | All setup/CLI arguments with default values and inline comments. |
| [`schemas/OmniSetupOverrides.schema.json`](../schemas/OmniSetupOverrides.schema.json)   | JSON Schema with types, enums, and validation constraints.       |

## Troubleshooting

### Checkpoint Issue

If you encounter failures downloading checkpoints, refer to [Downloading Checkpoints](./setup.md#downloading-checkpoints).

Checkpoint download commands are printed to the console. You can run them manually to debug issues.

### Torch CUDA Out of Memory Error

Error: `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X MiB`

[Optimize memory allocation](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf):

```shell
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### NCCL Issue

Error:

```shell
[rank0]:[W415 18:57:09.249883195 ProcessGroupNCCL.cpp:5138] Guessing device ID based on global rank. This can cause a hang if rank to GPU mapping is heterogeneous. You can specify device_id in init_process_group()

Fatal Python error: Segmentation fault
```

Re-run with debugging enabled:

```shell
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1
```

#### NCCL Plugin Issue

Error:

```shell
NCCL INFO Failed to initialize NET plugin Libfabric

Fatal Python error: Segmentation fault
```

Fix:

```shell
export NCCL_NET_PLUGIN=none
```

## Supplementary Examples

The following scripts are provided for learning purposes - they illustrate how the inference logic works under the hood. For production use, prefer `cosmos3.scripts.inference` as shown above.

Inference [example](../examples/inference.py) (low-level model API):

```shell
python examples/inference.py
```

Inference [example](../examples/inference_pipeline.py) (pipeline API):

```shell
python examples/inference_pipeline.py
```
