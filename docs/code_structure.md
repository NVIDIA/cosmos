# Code Structure

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Repository Layout](#repository-layout)
- [The `cosmos/` Package](#the-cosmos-package)
  - [`cosmos/algorithm/`](#cosmosalgorithm)
  - [`cosmos/callbacks/`](#cosmoscallbacks)
  - [`cosmos/checkpoint/`](#cosmoscheckpoint)
  - [`cosmos/communicator/`](#cosmoscommunicator)
  - [`cosmos/controller/`](#cosmoscontroller)
  - [`cosmos/data/`](#cosmosdata)
  - [`cosmos/evaluation/`](#cosmosevaluation)
  - [`cosmos/inference/`](#cosmosinference)
  - [`cosmos/launcher/`](#cosmoslauncher)
  - [`cosmos/model/`](#cosmosmodel)
  - [`cosmos/tools/`](#cosmostools)
  - [`cosmos/trainer/`](#cosmostrainer)
  - [`cosmos/utils/`](#cosmosutils)
  - [`cosmos/workers/`](#cosmosworkers)
- [Supporting Directories](#supporting-directories)
- [Where to Add New Code](#where-to-add-new-code)

______________________________________________________________________

<!--TOC-->

## Repository Layout

```
Cosmos/
├── cosmos/             # Main training-infra package (this repo's import root)
├── cosmos-inference/   # Inference-side sibling subtree (transformers/vLLM/diffusers integrations)
├── docs/               # User documentation (you are here)
├── docker/             # Dockerfiles for reproducible environments
├── examples/           # Runnable training / fine-tuning / inference examples
├── tests/              # Unit and integration tests
├── tools/              # Standalone CLI utilities (e.g. checkpoint conversion)
├── pyproject.toml      # uv-managed dependency manifest
├── uv.lock             # Pinned dependency graph (do not edit by hand)
└── .python-version     # Python version pin (used by uv)
```

`cosmos/` is the entry point for training. `cosmos-inference/` is kept in-tree as a synced subtree containing the inference-side packages, served as a reference and for end-to-end checkpoint export workflows; it is not a build dependency of `cosmos/`.

## The `cosmos/` Package

The `cosmos/` package is organized around the workflow of a large-scale, distributed training run — particularly post-training and reinforcement-learning regimes — with each subpackage owning one concern.

```
cosmos/
├── algorithm/      # Loss functions, reward models, RL algorithms
│   ├── loss/
│   ├── reward/
│   └── rl/
├── callbacks/      # Lifecycle hooks (logging, profiling, eval triggers, checkpoint cadence)
├── checkpoint/     # Saving, loading, conversion (DCP ↔ HF safetensors)
├── communicator/   # Inter-process / inter-worker communication primitives
├── controller/     # Top-level orchestration of multi-worker training jobs
├── data/           # Dataset loading, batching, augmentation, sharding
├── evaluation/     # Eval harness for trained checkpoints
├── inference/      # Inference helpers used during training (rollouts, validation)
├── launcher/       # Job launching (Slurm, torchrun, k8s)
├── model/          # Model definitions and parallelism wrappers
├── tools/          # Standalone CLI tools surfaced from the package
├── trainer/        # Training loop, optimizer step, gradient accumulation
├── utils/          # Shared low-level utilities (logging, config, distributed helpers)
└── workers/        # Specialized roles in a distributed RL job
    ├── reference/  # Reference / frozen-policy worker (KL anchor)
    ├── reward/     # Reward-model worker
    ├── rollout/    # On-policy rollout generation worker
    └── simulations/# Simulator-driven environment worker
```

### `cosmos/algorithm/`

Algorithmic primitives that are independent of the model and trainer.

- `loss/` — supervised and distillation losses (cross-entropy, flow-matching, KL, etc.).
- `reward/` — reward functions and learned reward heads.
- `rl/` — RL update rules (PPO, GRPO, DPO-family) that consume losses and rewards.

Add new objectives here, not inside the trainer.

### `cosmos/callbacks/`

Pluggable lifecycle hooks invoked by the trainer at well-defined points (step begin/end, epoch boundary, eval, save, exception). Use callbacks for cross-cutting concerns such as wandb/W&B logging, gradient clipping, MoE stability monitoring, dataloader-state checkpointing, and learning-rate logging.

### `cosmos/checkpoint/`

All checkpoint I/O lives here:

- DCP (PyTorch Distributed Checkpoint) save/load
- HuggingFace `safetensors` import/export
- Schema migration and resume-from-step logic

See also `docs/checkpoints.md`.

### `cosmos/communicator/`

Communication primitives between processes — point-to-point send/recv, broadcast helpers, and any RPC-style channels used between the controller and workers. Keep raw `torch.distributed` / NCCL calls out of business logic; route them through this layer.

### `cosmos/controller/`

The orchestrator for a multi-worker job. The controller drives the training loop, hands batches to rollout/reward workers, collects gradients, and decides when to checkpoint or evaluate. Think "head node logic" — there is one controller per job.

### `cosmos/data/`

Datasets, samplers, collators, augmentations, and data-side parallelism (e.g. sequence packing, multi-aspect batching). New dataset formats and new augmentations both live here. See `docs/dataset.md`.

### `cosmos/evaluation/`

Evaluation harnesses run against trained checkpoints — metrics, dataset-driven eval loops, and reporting. Distinct from `inference/`: evaluation is offline and metric-oriented.

### `cosmos/inference/`

Inference utilities used **during training** (e.g. rollout generation, validation-time sampling). Production inference engines live in `cosmos-inference/` and are not pulled in here.

### `cosmos/launcher/`

Job launching back-ends: Slurm, `torchrun`, and Kubernetes adapters. Selects the launch path based on the environment and forwards process rank/world-size to the controller.

### `cosmos/model/`

Model architectures and the parallelism wrappers around them (FSDP, tensor parallel, context parallel, pipeline parallel). The trainer is model-agnostic; everything the trainer touches goes through this layer.

### `cosmos/tools/`

CLI entry points surfaced from the package (as opposed to standalone scripts in the top-level `tools/`). Use this for utilities that need to import `cosmos.*` internals.

### `cosmos/trainer/`

The training loop itself — gradient accumulation, optimizer step, scheduler step, mixed-precision policy, and the dispatcher that fires callbacks. Stays narrow on purpose: model details live in `model/`, algorithm details in `algorithm/`.

### `cosmos/utils/`

Shared low-level helpers (logging, config loading, distributed setup, profiling). Keep this folder *thin* — anything substantial should grow into its own subpackage.

### `cosmos/workers/`

Specialized worker roles for distributed RL jobs. Each worker is a long-running process the controller talks to:

- `reference/` — frozen reference policy (for KL anchoring in PPO/GRPO/DPO).
- `reward/` — reward-model worker; computes scalar rewards for rollouts.
- `rollout/` — on-policy generation worker; samples trajectories from the current policy.
- `simulations/` — simulator-backed environment worker (used when reward comes from a sim rather than a learned model).

Add new worker types as sibling subpackages — each owns its own startup, message loop, and shutdown.

## Supporting Directories

- `tests/` — pytest tests, mirroring the `cosmos/` package layout.
- `examples/` — runnable end-to-end examples; see `examples/README.md`.
- `docker/` — Dockerfiles and image build helpers; see `docker/README.md`.
- `tools/` (repo root) — standalone CLI utilities not requiring `cosmos.*` imports; see `tools/README.md`.
- `cosmos-inference/` — synced subtree of the inference-side codebase, used for checkpoint export and reference integrations.

## Where to Add New Code

| You want to add… | Put it in… |
| --- | --- |
| A new loss function | `cosmos/algorithm/loss/` |
| A new RL update rule | `cosmos/algorithm/rl/` |
| A new reward function or head | `cosmos/algorithm/reward/` |
| A new model architecture | `cosmos/model/` |
| A new dataset format / augmentation | `cosmos/data/` |
| A new training callback | `cosmos/callbacks/` |
| A new checkpoint format or converter | `cosmos/checkpoint/` |
| A new launcher back-end (Slurm flavor, k8s) | `cosmos/launcher/` |
| A new RL worker role | `cosmos/workers/<new_role>/` |
| A new evaluation suite | `cosmos/evaluation/` |
| A new runnable example | `examples/` |
| A new standalone CLI tool | `tools/` (repo root) for non-cosmos imports, otherwise `cosmos/tools/` |
| A new test | `tests/` mirroring the package path |
