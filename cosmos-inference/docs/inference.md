# Inference

> **Skill:** `.agents/skills/cosmos3-inference/SKILL.md`

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Parallelism Arguments](#parallelism-arguments)
- [Sample Arguments](#sample-arguments)
  - [Text](#text)
  - [Vision (Image/Video)](#vision-imagevideo)
  - [Action](#action)
    - [Action Modes](#action-modes)
    - [Action Configuration](#action-configuration)
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

Prerequisites:

- [Setup](../README.md#setup)
- [Environment Variables](./environment_variables.md)

For example commands, see [README](../README.md#inference).

Arguments:

- `-i`, `--input-files`: Path to the sample argument file(s) (JSON, JSONL, YAML). Accepts quoted glob patterns (e.g. `"inputs/*.json"`).
- `-o`, `--output-dir`: Output directory.

Outputs:

- `<sample_name>/`
  - `sample_args.json`: Sample arguments.
  - `sample_outputs.json`: Generation status, action (if enabled).
  - `vision.jpg`, `vision.mp4`: Vision output (if enabled).

To see all available arguments:

```shell
python -m cosmos3.scripts.inference --help
```

## Parallelism Arguments

- `--parallelism-preset`
  - `latency`: Generate each sample as fast as possible by spreading work across GPUs. Used for real-time jobs.
  - `throughput`: Generate many samples in parallel, one per GPU. Used for batch jobs.
- `--max-num-seqs`: Batch size per GPU.

## Sample Arguments

Sample arguments are read from multiple sources (in priority order):

- CLI overrides (e.g. `--model-mode=text2video`): Overrides for all samples.
- Input files (e.g. `--input-files "inputs/omni/*t2i*.json"`): Single sample per input.
- Defaults: `cosmos3/defaults/<model_mode>`: Defaults for all samples.

For debugging, the full set of sample arguments is saved to `<output_dir>/<sample_name>/sample_args.json`.

Common arguments:

- `model_mode`: Generation modality. See [Modalities](../README.md#modalities) for all options.
- `seed`: Random seed for reproducibility.

**Note:** Condition file paths are relative to the input file.

### Text

- `prompt`: Inline text prompt.

### Vision (Image/Video)

Common arguments:

- `fps`: Condition and output frames per second.
- `resolution` (`"256"`, `"480"`, `"720"`): Condition and output resolution (height in pixels).
- `aspect_ratio` (`1,1`, `4,3`, `"3,4`, `16,9`, `9,16`): Condition and output aspect ratio. Defaults to `16,9`.

Condition arguments:

- `vision_path`: Path to an image or video file (local path or URL).

Generation arguments:

- `num_frames`: Number of output frames. `1` = image; `≥24` = video.

Outputs `vision.jpg` or `vision.mp4` depending on `num_frames`.

### Action

Common arguments:

- `action_chunk_size`: Number of action steps in the chunk. The action media loader reads at most `action_chunk_size + 1` observation frames.
- `domain_name`: Domain name passed to the action domain registry, such as `libero` or `av`.

Condition arguments:

- `action_path`: JSON action sequence. Required for `forward_dynamics`; each row is one action step and each column is one raw action dimension.
- `image_size`: Action input resize bucket. The value is passed as the action media resolution bucket; examples use `256` for LIBERO and `480` for AV.

Generation arguments:

- `raw_action_dim`: Raw action width to return for generated actions. Required for `inverse_dynamics` and `policy`.

The action output is written to `sample_outputs.json`.

#### Action Modes

| Mode               | Inputs                                                | Outputs                                                                     | Required action fields |
| ------------------ | ----------------------------------------------------- | --------------------------------------------------------------------------- | ---------------------- |
| `forward_dynamics` | Observation image/video, text prompt, action sequence | Future visual rollout in `vision.mp4` or `vision.jpg`                       | `action_path`          |
| `inverse_dynamics` | Observation video, text prompt                        | Predicted action sequence in `sample_outputs.json`                          | `raw_action_dim`       |
| `policy`           | Current observation image/video, text prompt          | Predicted action sequence, and any visual output returned by the checkpoint | `raw_action_dim`       |

#### Action Configuration

The action sample fields control input preprocessing and action tensor shape. `action_chunk_size` should match the chunk length used by the checkpoint. `image_size` should match the action training/evaluation resolution bucket. `domain_name` must be compatible with the checkpoint's action domain registry.

Action tensors are padded to `model.config.max_action_dim` before generation. Set it with `--experiment_overrides "[model.config.max_action_dim=<D>]"` when the checkpoint config does not already define the desired padded width. Use a value greater than or equal to the raw action width in `action_path` or `raw_action_dim`.

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

- [`OmniSetupOverrides.yaml`](../schemas/OmniSetupOverrides.yaml): All setup/CLI arguments with default values and inline comments.
- [`OmniSetupOverrides.schema.json`](../schemas/OmniSetupOverrides.schema.json): JSON Schema with types, enums, and validation constraints.
- [`OmniSampleOverrides.yaml`](../schemas/OmniSampleOverrides.yaml): All sample arguments with default values and inline comments.
- [`OmniSampleOverrides.schema.json`](../schemas/OmniSampleOverrides.schema.json): JSON Schema with types, enums, and validation constraints.

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
