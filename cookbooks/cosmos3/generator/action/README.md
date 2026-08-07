# Cosmos3 Generator Action

Cosmos3-Nano action-generation supports three distinct tasks:

- **Forward dynamics (`fd`)** — predict future observations from a start image plus an action trajectory.
- **Inverse dynamics (`id`)** — predict ego-motion trajectories from input AV or camera videos using the Cosmos3-Nano.
- **Policy (`policy`)** — predict future observations and action trajectories from a start image, task instruction, and state.

Each of the modes can be explored interactively using the [Action Viewer](https://huggingface.co/spaces/nvidia/Cosmos3-Action-Viewer) Hugging Face space.
The rest of this doc shows how to run these modes on selected embodiments directly.

## Table of Contents

- [Overview](#overview)
- [Run with Cosmos Framework](#run-with-cosmos-framework)
  - [Quickstart](#quickstart)
  - [Cosmos Framework Walkthrough](#cosmos-framework-walkthrough)
- [Run with Diffusers](#run-with-diffusers)
  - [Quickstart](#quickstart-1)
  - [Notebook walkthrough](#diffusers-notebook-walkthrough)
- [Run with vLLM-Omni](#run-with-vllm-omni)
  - [Quickstart](#quickstart-2)
  - [Notebook walkthrough](#vllm-omni-notebook-walkthrough)
- [Post-Train for Cosmos3-Nano-Policy-DROID](#post-train-for-cosmos3-nano-policy-droid)

## Overview

All examples are shown across three different inference backends — native
PyTorch (Cosmos Framework), Diffusers, and vLLM-Omni. Every backend uses the sample
assets under [`assets/`](./assets) and covers three tasks:

Environment setup for all backends is centralized in the shared
[Cosmos3 cookbooks environment setup](../../README.md) guide; each backend below
links to the section you need.

Generator requires the Guardrail. Request access to the gated
[nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail)
HF repository before running these examples. To disable the guardrail, set
`enable_safety_checker=False` (Diffusers), `guardrails: false` (vLLM-Omni
`extra_params`/`extra_args`), or `--no-guardrails` (Cosmos Framework).

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
| Human hand pose | Ego pose (9D) + right wrist (9D) + right fingertips (15D) + left wrist (9D) + left fingertips (15D) | 57D | Meter | Wrist-frame alignment, normalization | 16 frames @ 15FPS |

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
assemble ready-to-run specs for AV, DROID, UMI, and human hand-pose examples from the checked-in
assets under [`assets/`](./assets). Outputs are written under the framework
checkout.

### Cosmos Framework Walkthrough

The Cosmos Framework build their input spec, run inference, and
visualize the generated videos:

- [`run_fd_with_cosmos_framework.ipynb`](./run_fd_with_cosmos_framework.ipynb) —
  forward dynamics for AV, DROID, UMI, and human hand-pose examples using Cosmos3-Nano.
- [`run_id_with_cosmos_framework.ipynb`](./run_id_with_cosmos_framework.ipynb) —
  inverse dynamics, predicting ego-motion trajectories from input AV videos using Cosmos3-Nano.
- [`run_policy_with_cosmos_framework.md`](./run_policy_with_cosmos_framework.md) - policy, predicting future observations and action trajectories for DROID robot using Cosmos3-Nano-Policy-DROID and Cosmos3-Edge-Policy-DROID.

## Run with Diffusers

### Quickstart

Set up the environment: [Diffusers setup](../../README.md#diffusers). Action inputs are
grouped into a `CosmosActionCondition`. The pipeline derives the frame count from
`chunk_size + 1` and the conditioning canvas from `resolution_tier`, so `height`, `width`,
and `num_frames` stay unset:

```python
import json
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers.utils import export_to_video, load_image

action_root = Path("cookbooks/cosmos3/generator/action")
raw_actions = torch.as_tensor(
    json.load(open(action_root / "assets/actions/av_traj_forward.json")), dtype=torch.float32
)

pipe = Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16)
pipe.to("cuda")

result = pipe(
    prompt="You are an autonomous vehicle planning system.",
    action=CosmosActionCondition(
        mode="forward_dynamics",
        chunk_size=60,
        domain_name="av",
        resolution_tier=480,
        raw_actions=raw_actions,
        image=load_image(str(action_root / "assets/images/av_0.jpg")),
        view_point="ego_view",
    ),
    fps=10,
    num_inference_steps=30,
    guidance_scale=1.0,
    use_system_prompt=False,
    generator=torch.Generator(device="cuda").manual_seed(0),
)
export_to_video(result.video, "/tmp/cosmos3_action_fd.mp4", fps=10, macro_block_size=1)
```

For inverse dynamics, pass `mode="inverse_dynamics"` with `video=` instead of `image=` and
no `raw_actions`; the predicted trajectory comes back as `result.action`. `mode="policy"`
conditions on a first frame like forward dynamics but takes no `raw_actions`, predicting the
action chunk alongside the video. Cosmos3-Super also ships `action_gen=True`, so the forward
and inverse calls work against that checkpoint too.

### Diffusers Notebook Walkthrough

The Diffusers notebooks provision a dedicated venv with the LeRobot readers, pose helpers, and
plotting extras, register it as the `Cosmos3 Diffusers Action` kernel, then run each task with
`Cosmos3OmniPipeline` and write outputs under `outputs/notebooks/diffusers/`:

- [`run_fd_with_diffusers.ipynb`](./run_fd_with_diffusers.ipynb) — forward dynamics for
  three AV ego trajectories and three camera-pose trajectories, an autoregressive multiview
  DROID rollout, an autoregressive UMI rollout, and a bimanual hand-pose chunk.
- [`run_id_with_diffusers.ipynb`](./run_id_with_diffusers.ipynb) — inverse dynamics for the
  checked-in AV clips plus the Bridge, AgiBotWorld-Beta, RoboMIND (Franka, Franka dual-arm, UR),
  UMI, and Fractal episodes, plotting each prediction as a camera or end-effector trajectory.
- [`run_policy_with_diffusers.ipynb`](./run_policy_with_diffusers.ipynb) — policy, jointly
  predicting future observations and an action chunk for DROID with Cosmos3-Nano-Policy-DROID,
  with the predicted chunk plotted as an end-effector trajectory.

The examples that read the checked-in LeRobot episodes use the readers from the Cosmos framework,
and the trajectory plots use its pose helpers, so set `COSMOS3_REPO` to your framework checkout
before running them. The DROID reader resolves its feature layout from the name of the directory
it is given, so also set `COSMOS3_DROID_ROOT` to a directory named after the DROID release you
have available.

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
generation (see [`run_fd_with_vllm_omni.ipynb`](./run_fd_with_vllm_omni.ipynb) and
[`run_id_with_vllm_omni.ipynb`](./run_id_with_vllm_omni.ipynb)):

| Field | Value |
| --- | --- |
| `num_inference_steps` | `30` |
| `guidance_scale` | `1.0` |
| `flow_shift` | `10.0` |

The notebooks build the full request body for AV, DROID, UMI and human hand-pose examples,
including autoregressive chunked generation for the robotics examples. Policy
inference uses async `POST /v1/videos` to retrieve a rollout video plus
top-level `action` metadata.

### VLLM-Omni Notebook Walkthrough

The vLLM-Omni notebooks send requests through the OpenAI-compatible video API and
write outputs under `outputs/cosmos3_action_vllm/`:

- [`run_fd_with_vllm_omni.ipynb`](./run_fd_with_vllm_omni.ipynb) — forward dynamics for AV,
  DROID, UMI, and human hand-pose examples.
- [`run_id_with_vllm_omni.ipynb`](./run_id_with_vllm_omni.ipynb) — inverse dynamics,
  predicting ego-motion trajectories from input AV videos.
- [`run_policy_with_vllm_omni.ipynb`](./run_policy_with_vllm_omni.ipynb) — policy
  inference for DROID through the async video API.

## Post-Train for Cosmos3-Nano-Policy-DROID

To reproduce our post-training recipe for [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID), use the
[Cosmos3-Nano-Policy-DROID SFT cookbook](./finetune/README.md). It follows the same
launch-script pattern as the other Cosmos3 finetune cookbooks while delegating
the canonical training implementation to Cosmos Framework.

The same [action-policy SFT cookbook](./finetune/README.md) also covers **LIBERO-10**
(`launch_sft_action_policy_libero_10_nano.sh`) — fine-tuning Cosmos3-Nano on the `libero_10`
simulation benchmark with the same launch-script pattern.

## TODO

- [ ] Add additional embodiment examples.
