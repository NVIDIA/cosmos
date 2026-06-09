# Cosmos3 Generator Action Examples

Cosmos3-Nano action-generation examples across two inference backends — native
PyTorch (Cosmos Framework) and vLLM-Omni. Both backends use the sample assets
under [`assets/`](./assets) and cover two tasks:

- **Forward dynamics (`fd`)** — predict future observations from a start image
  plus an action trajectory (AV, DROID, and UMI robotics examples) using the Cosmos3-Nano.
- **Inverse dynamics (`id`)** — predict ego-motion trajectories from input AV
  videos using the Cosmos3-Nano.
- **Policy (`policy`)** — predict future observations and action trajectories from a start image, task instruction, and state for DROID robot using the Cosmos3-Nano-Policy-DROID.

Environment setup for both backends is centralized in the shared
[Cosmos3 cookbooks environment setup](../../README.md) guide; each backend below
links to the section you need.

Generator requires the Guardrail. Request access to the gated
[nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail)
HF repository before running these examples. To disable the guardrail, set
`enable_safety_checker=False` (Diffusers), `guardrails: false` (vLLM-Omni
`extra_params`/`extra_args`), or `--no-guardrails` (Cosmos Framework).

## Table of Contents

- [Run with Cosmos Framework](#run-with-cosmos-framework)
  - [Quickstart](#quickstart)
  - [Cosmos Framework Walkthrough](#cosmos-framework-walkthrough)
- [Run with vLLM-Omni](#run-with-vllm-omni)
  - [Quickstart](#quickstart-1)
  - [Notebook walkthrough](#notebook-walkthrough)


## Action Definition

Cosmos3 treats action as a modality whose tokens represent transitions between
consecutive visual states. This cookbook follows the unified action interface
defined in the paper: ego and end-effector poses are represented as 9D pose
deltas, consisting of 3D translation and a 6D continuous rotation representation.
Grasp state encodes the current manipulation state, using a 1D open-close value
for grippers or a 15D human-hand representation with 3D state for each of 5
fingers.

| Embodiment | Representation | Dimensionality | Unit | Post-processing | Generation duration |
| --- | --- | --- | --- | --- | --- |
| Autonomous vehicle | Ego pose (9D) | 9D | Meter | Normalization | 60 frames @ 10FPS |
| [DROID](https://arxiv.org/abs/2403.12945) | End-effector pose (9D) + gripper grasp state (1D) | 10D | Meter | Multiview concatenation, `to-OpenCV`, normalization | 16 frames @ 15FPS |
| UMI | End-effector pose (9D) + gripper grasp state (1D) | 10D | Meter | Normalization | 16 frames @ 20FPS |

Action data samples across different embodiments can be inspected interactively in the [Cosmos3 Action Viewer](https://huggingface.co/spaces/nvidia/Cosmos3-Action-Viewer) Hugging Face Space.

## Run with Cosmos Framework

### Quickstart

Set up the environment: [Cosmos Framework setup](../../README.md#cosmos-framework).
Activate the framework venv, then run the native inference entrypoint. Forward
dynamics on Nano looks like:

```bash
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i <forward-dynamics input spec>.json \
  -o /tmp/cosmos3_action_fd \
  --checkpoint-path Cosmos3-Nano \
  --seed 0
```

The input spec pairs a start image with an action trajectory. The notebooks
assemble ready-to-run specs for AV, DROID, and UMI examples from the checked-in
assets under [`assets/`](./assets). Outputs are written under the framework
checkout.

### Cosmos Framework Walkthrough

The Cosmos Framework build their input spec, run inference, and
visualize the generated videos:

- [`run_fd_with_cosmos_framework.ipynb`](./run_fd_with_cosmos_framework.ipynb) —
  forward dynamics for AV, DROID, and UMI robotics examples using Cosmos3-Nano.
- [`run_id_with_cosmos_framework.ipynb`](./run_id_with_cosmos_framework.ipynb) —
  inverse dynamics, predicting ego-motion trajectories from input AV videos using Cosmos3-Nano.
- [`run_policy_with_cosmos_framework.md`](./run_policy_with_cosmos_framework.md) - policy, predicting future observations and action trajectories for DROID robot using Cosmos3-Nano-Policy-DROID.


## Run with vLLM-Omni

### Quickstart

Set up the environment and start the server:
[vLLM-Omni setup](../../README.md#vllm-omni) (Docker recommended). From the
`cosmos` repo root, set `export COSMOS3_WORKDIR=$PWD` and
`export COSMOS3_HOST_PORT=8001`, then run the Docker command from the env setup
guide. Wait until this succeeds:

```bash
curl http://localhost:8001/v1/models
```

Forward-dynamics requests are multipart `POST`s to `/v1/videos` — a start image
under `files={"input_reference": ...}` plus an `extra_params` payload carrying the
action trajectory. The vLLM notebooks use these diffusion defaults for action
generation (see [`run_fd_with_vllm.ipynb`](./run_fd_with_vllm.ipynb) and
[`run_id_with_vllm.ipynb`](./run_id_with_vllm.ipynb)):

| Field | Value |
| --- | --- |
| `num_inference_steps` | `30` |
| `guidance_scale` | `1.0` |
| `flow_shift` | `10.0` |

The notebooks build the full request body for AV, DROID, and UMI examples,
including autoregressive chunked generation for the robotics examples.

### VLLM-Omni Notebook Walkthrough

The vLLM-Omni notebooks send requests through the OpenAI-compatible video API and
write outputs under `outputs/cosmos3_action_vllm/`:

- [`run_fd_with_vllm.ipynb`](./run_fd_with_vllm.ipynb) — forward dynamics for AV,
  DROID, and UMI robotics examples.
- [`run_id_with_vllm.ipynb`](./run_id_with_vllm.ipynb) — inverse dynamics,
  predicting ego-motion trajectories from input AV videos.



## TODO

- [ ] Add additional embodiment examples.
