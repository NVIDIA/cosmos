<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Generator Distillation

This cookbook demonstrates short DMD2 distillation smoke runs for the released
Cosmos3-Super Text2Image and Image2Video models with Cosmos Framework.

> This is a functional training recipe, not a production reproduction recipe.
> It uses a small public dataset and six optimizer iterations to validate model
> loading, loss and gradient computation, checkpoint resume, student-only
> export, and inference. It does not promise production-quality samples or
> reproduction of NVIDIA training results.

## Supported recipes

| Launcher | TOML | Teacher | Initial student | Data view |
| --- | --- | --- | --- | --- |
| `launch_distillation_t2i.sh` | `toml/distillation_t2i.toml` | [`nvidia/Cosmos3-Super-Text2Image`](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image) | [`nvidia/Cosmos3-Super-Text2Image-4Step`](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image-4Step) | One 768-class frame per BridgeData2 video, no visual conditioning |
| `launch_distillation_i2v.sh` | `toml/distillation_i2v.toml` | [`nvidia/Cosmos3-Super-Image2Video`](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video) | [`nvidia/Cosmos3-Super-Image2Video-4Step`](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video-4Step) | 61 frames at 480p, first-frame conditioning |

Both use
[`nvidia/BridgeData2-Subset-Synthetic-Captions`](https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions).
The fake-score network has the same 32B architecture and is initialized from
the teacher by the existing DMD2 training path.

## Prerequisites

1. Follow the shared [Cosmos Framework environment setup](../../../README.md#cosmos-framework).
   Activate the resulting training environment before launching.
2. Accept the licenses for the four model repositories and BridgeData2, then
   authenticate with `uvx hf@latest auth login` or set `HF_TOKEN`.
3. Allocate **8 GB200 nodes with 4 GPUs per node** for the provided canonical
   profile. A 4-node x 4-GPU GB200 profile has also completed end-to-end
   validation; see [Validated topology scope](#validated-topology-scope) for
   the coordinated configuration changes it requires.
4. Choose a high-capacity shared filesystem path visible at the same location
   on every node. The launchers store the dataset, Hugging Face materialization,
   converted DCP checkpoints, training state, and exports beneath it.
5. Invoke the same launcher once on every allocated node. Your scheduler is
   responsible for assigning `NODE_RANK` and starting one launcher process per
   node.

The launchers never upload weights, checkpoints, or outputs to Hugging Face.

## Topology variables

Each node must receive these values:

| Variable | Provided canonical profile |
| --- | --- |
| `DISTILL_ROOT` | Shared writable directory |
| `NNODES` | `8` |
| `NPROC_PER_NODE` | `4` |
| `NODE_RANK` | Unique integer from `0` through `7` |
| `MASTER_ADDR` | Host name or IP of node rank 0 |
| `MASTER_PORT` | One free TCP port, identical on every node |

The scripts use standard multi-node `torchrun`; they do not contain Slurm,
Kubernetes, or another scheduler-specific submission layer.

### Validated topology scope

Both T2I and I2V have completed short training, iteration-5 checkpoint save,
strict resume through iteration 6, student-only export, and inference on these
GB200 profiles:

| Profile | GPUs | FSDP data-parallel shard degree | Cookbook status |
| --- | ---: | ---: | --- |
| 8 nodes x 4 GPUs | 32 | 32 | Provided canonical profile |
| 4 nodes x 4 GPUs | 16 | 16 | Validated custom profile |

The checked-in launchers and TOMLs remain pinned to the canonical 8-node
profile; a separate 4-node TOML is intentionally not shipped. To use the
validated 4-node profile, make these coordinated local changes:

1. Set `NNODES=4`, keep `NPROC_PER_NODE=4`, and assign `NODE_RANK` from `0`
   through `3`.
2. Update the topology validation in `distill/common.sh` to accept 4 nodes.
3. Set `model.parallelism.data_parallel_shard_degree=16` in the selected T2I
   or I2V TOML.

The 16-node x 4-GPU profile and other topology or parallelism combinations
have not been validated by this cookbook.

## Training configuration

The two TOML files are the user-facing training configuration. They expose the
mode-specific data view, four-step sampling schedule, DMD2 loss settings,
student and fake-score optimizer settings, strict checkpoint policy, and the
provided canonical 32-GPU parallelism shape. The validated 16-GPU custom
profile uses the coordinated changes described above.

`build_distillation_config.py` strictly validates the selected TOML and
constructs framework objects that TOML cannot represent directly, including
`DMD2RFModel`, `DistillationTrainer`, the two-optimizer LazyDict, the nested
BridgeData2 dataloader, callbacks, and the distillation checkpointer. The
launchers supply the runtime dataset and converted teacher/student DCP paths,
then serialize and round-trip validate the resulting Cosmos Framework YAML.
Unknown TOML keys are rejected instead of being silently ignored.

## Launch T2I

Run the following on every node after the scheduler fills in `NODE_RANK`:

```bash
cd cookbooks/cosmos3/generator/audiovisual

export DISTILL_ROOT=/shared/path/cosmos3-distillation
export NNODES=8
export NPROC_PER_NODE=4
export NODE_RANK=<0-through-7>
export MASTER_ADDR=<rank-0-hostname>
export MASTER_PORT=29500

bash distill/launch_distillation_t2i.sh
```

## Launch I2V

Use a different free rendezvous port if T2I is still running:

```bash
cd cookbooks/cosmos3/generator/audiovisual

export DISTILL_ROOT=/shared/path/cosmos3-distillation
export NNODES=8
export NPROC_PER_NODE=4
export NODE_RANK=<0-through-7>
export MASTER_ADDR=<rank-0-hostname>
export MASTER_PORT=29501

bash distill/launch_distillation_i2v.sh
```

Use `DRY_RUN=1` to print asset-preparation, training, resume, and export commands
without downloading models or starting GPU work:

```bash
DRY_RUN=1 bash distill/launch_distillation_t2i.sh
```

## What each launcher does

Node rank 0 prepares shared assets while the other nodes wait for completed
files. Existing validated files are reused.

1. Download the pinned BridgeData2 dataset revision and Wan2.2 VAE.
2. Materialize the selected full teacher and four-step student from Hugging
   Face and convert both to DCP.
3. Load the selected TOML and round-trip validate the generated Cosmos
   Framework YAML.
4. Start a 32-rank training process with `--no-resume`, train through iteration
   5, and write `iter_000000005`.
5. Start a new 32-rank process with `--resume` and train from iteration 5 through
   iteration 6. Because iteration 6 is not a `save_iter` multiple, the trainer's
   terminal-checkpoint behavior writes `iter_000000006` before shutdown.
6. On node rank 0, export the iteration-6 student without teacher or fake-score
   weights. Set `EXPORT_AFTER_TRAIN=0` to skip this local export.

The DMD2 settings intentionally follow the validated four-step shape: backward
SDE simulation with timesteps `[1.0, 15/16, 5/6, 5/8]`, teacher guidance 6.0,
student update frequency 5, VSD mean reduction, fake-score active-mean
reduction, and gradient clipping. The public dataset and short duration are
deliberate smoke-test deviations from production training.

## Outputs

For T2I, the key paths are:

```text
$DISTILL_ROOT/
|-- configs/distillation_t2i.yaml
|-- checkpoints/t2i/teacher.dcp/model/
|-- checkpoints/t2i/student.dcp/model/
`-- outputs/
    |-- train/cosmos3/distillation/cosmos3_super_t2i_dmd2_smoke/checkpoints/
    `-- invocations/t2i/cosmos3_super_t2i_dmd2_smoke/
        |-- config.yaml
        |-- job -> $DISTILL_ROOT/outputs/train/cosmos3/distillation/cosmos3_super_t2i_dmd2_smoke
        `-- student/
```

Replace `t2i` with `i2v` for the I2V run. DCP checkpoints include model,
optimizer, scheduler, trainer, and dataloader state so the second process tests
a real training resume rather than only loading model weights.

## Inference with the published four-step students

The audiovisual
[`run_with_cosmos_framework.ipynb`](../run_with_cosmos_framework.ipynb)
contains the published distilled T2I and I2V inference examples, including
payload creation, four-GPU model sharding, the fixed-step sampling arguments,
and artifact display. These are functional smoke examples, not
production-quality reproduction claims.

## Troubleshooting

- **Topology validation fails:** the checked-in launchers intentionally accept
  only the provided 8-node x 4-GPU profile. The 4-node x 4-GPU profile is also
  validated, but requires the three coordinated local changes in
  [Validated topology scope](#validated-topology-scope). The 16-node x 4-GPU
  profile and other configurations remain unvalidated.
- **Nodes time out waiting for assets or the generated config:** node rank 0
  performs the downloads, checkpoint conversion, and config generation while
  the other nodes wait for files under `DISTILL_ROOT`. Confirm that every node
  sees the same shared path, then inspect the rank-0 process for the original
  failure. `ASSET_WAIT_TIMEOUT_SECONDS` defaults to 7200 seconds.
- **Hugging Face 401/403 on node rank 0:** accept the licenses for gated model
  and dataset repositories, and confirm that rank 0 can access the teacher,
  student, BridgeData2, and Wan VAE repositories with its `HF_TOKEN` or saved
  Hugging Face login.
- **Python, CUDA, or libtorch mismatch:** confirm that `PYTHON_BIN` and
  `TORCHRUN_BIN` resolve to the same activated Cosmos Framework environment.
  The launchers already clear `LD_LIBRARY_PATH`; no extra manual clearing is
  required.
- **Out of memory after changing the recipe:** both documented profiles were
  validated on GB200, while the checked-in TOMLs provide the 8-node x 4-GPU
  profile. Changes to GPU count, hardware, resolution, sequence length, batch
  size, or parallelism outside those profiles may require coordinated retuning
  and separate validation.
