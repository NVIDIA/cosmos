<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Use Cosmos3 action capabilities through the Certified NIM

Use this page for forward dynamics, policy, and inverse dynamics requests. All
three use a compatible **Generator** model and synchronous JSON
`POST /v1/infer`; top-level `model_mode` selects the task.

> Confirm model and domain availability for the selected image in the
> [Support matrix](support-matrix.md).

See [Deployment](deployment.md) to start the NIM. There are two distinct
contracts:

- general Action models produce visual rollouts for forward dynamics, policy,
  and inverse dynamics; and
- the Nano-DROID specialist implements policy only and returns actions without
  visual media.

## Understand an action sequence

Cosmos3 models an action as a transition between consecutive visual
observations:

```text
o0 --a0--> o1 --a1--> o2 ... --a(T-1)--> oT
```

An action trajectory with shape `[T,D]` therefore describes `T` transitions
across `T+1` observations. `T` is `action_chunk_size`; `D` is the action width
for the selected domain. For example, the included AV cases provide 60 rows of
9 values and produce a 61-frame rollout.

Clients must omit top-level `num_frames` for every Action request. The service
derives it as `action_chunk_size + 1`, preserving the Generator's `4k+1` video
cadence. This relationship is the same whether actions are supplied to forward
dynamics or predicted by policy or inverse dynamics.

## Choose an Action mode

| `model_mode` | Input | Output |
| --- | --- | --- |
| `forward_dynamics` | Conditioning image plus action trajectory `[T,D]` | Rollout video; `action` response is null |
| `policy` | Conditioning image, task prompt, and optional state | Rollout video plus predicted action trajectory, or action only for a compatible specialist profile |
| `inverse_dynamics` | Conditioning video and optional task prompt | Video plus predicted action trajectory |

Action requests must not set top-level `resolution`. The conditioning media and
Action settings establish the visual request shape.

## Domains and representations

| `domain_name` | Domain ID | General `raw_action_dim` | Example chunk |
| --- | ---: | ---: | ---: |
| `av` | 1 | 9 | 60 actions at 10 action FPS |
| `umi` | 6 | 10 | 16 actions at 20 action FPS |
| `bridge_orig_lerobot` | 7 | 10 | 16 actions at 5 action FPS |
| `droid_lerobot` | 8 | 10 | 16 actions in the general contract |

The AV examples use 9D actions with translation and a 6D continuous rotation
representation. The UMI and DROID examples combine a 9D end-effector pose
representation with a gripper value. The API validates the domain and numeric
width, but checkpoint-specific units, normalization statistics, coordinate
frames, gripper conventions, and denormalization come from the selected model.

Do not send returned values directly to physical hardware based only on this
width table. Use the representation and transformations published for the
active checkpoint. In particular, Nano-DROID's 8-wide specialist output is a
distinct checkpoint contract, not a truncated general DROID 10D action.

`action_chunk_size` must be a positive multiple of 4. If `raw_action_dim` is
supplied for a general request, it must match the selected domain. A specialist
profile can own a different fixed representation; clients omit the width when
that profile requires it.

## Cadence and visual geometry

Action requests have several related but distinct controls:

- top-level `fps` is the cadence of the generated visual rollout;
- `action_params.action_fps` is the cadence represented by action rows;
- `action_params.image_size` selects an Action model image-size setting; and
- the actual conditioning image or video still has its own decoded dimensions
  and aspect ratio.

The included cases align `fps` and `action_fps`, but the API treats them as
separate fields. `image_size` is not a replacement for top-level `resolution`;
Action requests forbid `resolution`.

## Run the examples

Install the [client tooling](prerequisites.md#client-tooling). Then, from the
repository root, enter the cookbook directory. `uv` reads the pinned client
environment in that directory:

```bash
cd cookbooks/cosmos3/nim
export NIM_URL=${NIM_URL:-http://localhost:8000}
```

The executable cases include AV, UMI, and robot examples:

| Case | Conditioning and expected result | Run |
| --- | --- | --- |
| `av_forward` | Supplied AV straight trajectory; rollout visualizes that trajectory | `uv run python examples/action.py --case av_forward` |
| `av_left` | Supplied AV left trajectory; ego rollout turns left | `uv run python examples/action.py --case av_left` |
| `av_right` | Supplied AV right trajectory; ego rollout turns right | `uv run python examples/action.py --case av_right` |
| `umi_forward` | First 16-action UMI chunk; rollout visualizes that chunk | `uv run python examples/action.py --case umi_forward` |
| `av_inverse_0` | Observed `av_0.mp4`; predicts its AV action trajectory | `uv run python examples/action.py --case av_inverse_0` |
| `av_inverse_1` | Observed `av_1.mp4`; predicts its AV action trajectory | `uv run python examples/action.py --case av_inverse_1` |
| `bridge_inverse` | Observed pinned robot video; predicts its Bridge action trajectory | `uv run python examples/action.py --case bridge_inverse` |
| `av_policy_left` | `av_0.jpg` plus a left-turn goal; predicts a left-turn trajectory and rollout | `uv run python examples/action.py --case av_policy_left` |
| `av_policy_right` | `av_0.jpg` plus a right-turn goal; predicts a right-turn trajectory and rollout | `uv run python examples/action.py --case av_policy_right` |

The script rejects unknown command-line arguments before contacting the
endpoint. Before loading case assets or submitting inference, it requires
`/v1/metadata` to report the Generator runtime, `/v1/infer`, and a selected
profile ID. The AV policy cases require a general-purpose `nano` variant; the
other cases accept a general-purpose `nano` or `super` variant. These variant
checks do not establish image-specific Action or domain availability; confirm
those separately in the support matrix. The `av_policy` and `policy` aliases
select `av_policy_right`.

The script writes `action_<case>.mp4` for the visual rollout and, for policy or
inverse dynamics, `action_<case>.json` for the validated predicted trajectory.
It validates input trajectories and response shape, metadata, domain, and
finite numeric values. For the policy cases, it also prints the expected
qualitative direction. These structural checks do not establish whether a
rollout follows its task goal; review the saved video for directional
compliance. Use the
[Cosmos3 Action Viewer](https://huggingface.co/spaces/nvidia/Cosmos3-Action-Viewer)
to inspect supported action data interactively; it does not replace the
checkpoint-specific transformations required for execution.

## Forward dynamics

Forward dynamics receives an initial observation and a complete action chunk,
then predicts the resulting visual observations. The AV comparison cases share
`images/av_0.jpg` and differ only in the forward, left, or right
trajectory:

```python
request = {
    "model_mode": "forward_dynamics",
    "prompt": "You are an autonomous vehicle planning system.",
    "input_reference": image_data_url,
    "action_params": {
        "domain_name": "av",
        "action_chunk_size": 60,
        "action": trajectory,  # 60 rows × 9 finite numeric values
        "raw_action_dim": 9,
        "action_space": "joint_pos",
        "image_size": "480",
        "action_fps": 10.0,
    },
    "fps": 10.0,
    "num_inference_steps": 30,
    "guidance_scale": 1.0,
    "flow_shift": 10.0,
    "seed": 0,
}
```

Every action row must have the same width. The number of rows must equal
`action_chunk_size`; the width must equal `raw_action_dim`; and values must be
finite JSON numbers. The generic role prompt does not select the maneuver in
these cases: the supplied straight, left, or right action trajectory is the
behavioral condition. Forward dynamics does not accept `history_length`,
`use_state`, or `observation`. Its response contains a rollout video and no
predicted action because the trajectory was an input.

The `umi_forward` case uses `images/umi.png` and the first 16 rows of the
included 32-row `actions/umi.json` trajectory. It demonstrates one synchronous
chunk. A longer autoregressive rollout is client orchestration: extract the
last generated observation, use it to condition the next action chunk, and
remove duplicate boundary observations when joining output videos.

## Policy

Policy predicts an action chunk instead of receiving one:

```json
{
  "model_mode": "policy",
  "prompt": "You are an autonomous vehicle planning system. Turn right onto the road and continue driving in the rightmost lane.",
  "input_reference": "data:image/jpeg;base64,<BASE64_IMAGE>",
  "action_params": {
    "domain_name": "av",
    "action_chunk_size": 60,
    "raw_action_dim": 9,
    "action_space": "joint_pos",
    "image_size": "480",
    "action_fps": 10.0
  },
  "fps": 10.0,
  "num_inference_steps": 30,
  "guidance_scale": 1.0,
  "seed": 0
}
```

The `av_policy_left` and `av_policy_right` cases use the same `av_0.jpg`
observation, seed, and inference controls. Only the language goal changes:

| Case | Task goal | Expected qualitative rollout |
| --- | --- | --- |
| `av_policy_left` | Turn left onto the road and continue in the leftmost legal lane | The ego viewpoint turns left and proceeds along the roadway |
| `av_policy_right` | Turn right onto the road and continue in the rightmost lane | The ego viewpoint turns right and proceeds along the roadway |

Both cases return a predicted `[60,9]` action and a 61-frame rollout. The client
validates the response structure but cannot infer directional correctness from
the raw action values without checkpoint-specific coordinate transformations.
Review the saved video and confirm that the ego starts from the conditioning
scene, turns in the requested direction, and continues consistently after
entering the road. Directional agreement in a synthetic rollout does not
establish traffic-rule compliance, collision avoidance, or suitability for
physical execution.

Confirm that the selected image supports the Nano AV policy model and domain
before running either case. Policy must omit `action` and may also use:

- `history_length`: number of state-history steps;
- `use_state`: whether to condition on supplied state; and
- `observation`: free-form state passed to the pipeline without a public nested
  schema.

Use state conditioning only with a model/profile and observation shape that
has been validated for the selected deployment.

### Integrate a policy loop

`POST /v1/infer` is synchronous and stateless; it is not the streaming policy
server used by the RoboLab workflow in the general Action cookbook. A client
controls the loop:

1. Capture the current visual observation and checkpoint-required state.
2. Compose or transform views exactly as required by the checkpoint.
3. Send one policy request.
4. Validate the response and apply the checkpoint-specific denormalization and
   coordinate transforms.
5. Execute only an approved portion of the returned horizon.
6. Capture a new observation and replan.

Before physical execution, independently enforce joint/workspace bounds,
collision constraints, timing limits, emergency stops, and task-specific safety
checks. A valid API response is not an execution authorization.

## Nano-DROID policy

Nano-DROID is a specialist policy checkpoint selected with:

```bash
-e NIM_MODEL_TYPE=generator \
-e NIM_MODEL_VARIANT=nano-droid
```

Its request uses the same `POST /v1/infer` API as other Generator tasks:

```json
{
  "model_mode": "policy",
  "prompt": "Remove the crumpled paper from the sink and place it in the trash can.",
  "input_reference": "data:image/jpeg;base64,<COMPOSED_DROID_OBSERVATION>",
  "action_params": {
    "domain_name": "droid_lerobot",
    "action_chunk_size": 32,
    "observation": {
      "observation/joint_position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "observation/gripper_position": 0.0
    }
  },
  "seed": 0
}
```

The observation must contain exactly seven finite joint positions and one
finite gripper position. The current fixture is a 640×540 composition with the
wrist view above two exterior views. Use the exact observation composition
required by the selected checkpoint and a task prompt that describes the
desired manipulation.

The client must omit model-owned fields:

- `raw_action_dim`, `action_space`, `image_size`, `action_fps`,
  `history_length`, and `use_state` inside `action_params`; and
- top-level `fps`, `num_frames`, and `negative_prompt`.

The model returns 32 actions with width 8. It defaults to four inference steps,
guidance `3.0`, and flow shift `5.0`; `num_inference_steps`, `guidance_scale`,
`flow_shift`, and `seed` remain overridable.

A successful response is action-only:

```json
{
  "action": {
    "data": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    "shape": [32, 8],
    "dtype": "float32",
    "raw_action_dim": 8,
    "action_mode": "policy",
    "domain_id": 8
  }
}
```

The shortened `data` shows the row width only. The full response has 32 rows
and no `b64_image` or `b64_video`. A runnable cookbook case remains deferred
until an approved public composed DROID observation and matching state are
available.

## Inverse dynamics

Inverse dynamics receives an observed video and estimates the action transitions
between its visual states. The included local cases use `videos/av_0.mp4` and
`videos/av_1.mp4`:

```json
{
  "model_mode": "inverse_dynamics",
  "prompt": "You are an autonomous vehicle planning system.",
  "input_reference": "data:video/mp4;base64,<BASE64_AV_VIDEO>",
  "action_params": {
    "domain_name": "av",
    "action_chunk_size": 60,
    "raw_action_dim": 9,
    "action_space": "joint_pos",
    "image_size": "480",
    "action_fps": 10.0
  },
  "fps": 10.0,
  "num_inference_steps": 30,
  "guidance_scale": 1.0,
  "seed": 0
}
```

The additional `bridge_inverse` case uses a pinned public fixture. It requires
URL input to be enabled and network access from the container; use a data URL
instead for reproducible offline execution. Inverse dynamics does not accept
`action` or the policy-only state fields.

## Response action object

General visual policy and inverse-dynamics profiles return a video and an action
object:

```json
{
  "b64_video": "<BASE64_MP4>",
  "action": {
    "data": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    "shape": [16, 10],
    "dtype": "float32",
    "raw_action_dim": 10,
    "action_mode": "inverse_dynamics",
    "domain_id": 7
  }
}
```

The shortened `data` shows one row; a real response has every row reported by
`shape`. Before consuming a predicted trajectory, verify:

- `shape == [len(data), len(data[0])]`;
- the row count and width match the request contract;
- `raw_action_dim`, `action_mode`, and `domain_id` match the request;
- `dtype` is the documented value; and
- every value is numeric and finite.

The runnable script performs these checks before saving predicted action JSON.
A specialist policy can instead return the same action envelope with both media
fields absent, as described under [Nano-DROID policy](#nano-droid-policy).

Reasoner prompts can also produce text or JSON describing a “next action” or a
2D trajectory. Those normalized visual coordinates are semantic Reasoner
output, not Generator Action tensors, and are not interchangeable with the
`[T,D]` data documented here. See
[Reasoner prompting patterns](reasoning.md#prompting-patterns).

## Action parameter reference

| Field | Type/default | Contract |
| --- | --- | --- |
| `domain_name` | required enum | `av`, `bridge_orig_lerobot`, `droid_lerobot`, or `umi` |
| `action_chunk_size` | required integer | Positive multiple of 4; output frames equal this value plus 1 |
| `action` | number array `[T,D]` or null | Required only for forward dynamics; `T=action_chunk_size` and `D=raw_action_dim` |
| `raw_action_dim` | integer or null | Defaults to the selected domain width; an explicit value must match. Specialist profiles can replace an omitted value. |
| `action_space` | enum; `joint_pos` | `joint_pos` or `midtrain` |
| `image_size` | enum; `480` | `256`, `480`, `704`, or `720`; distinct from top-level `resolution` |
| `action_fps` | number or null | Optional range `[1.0,60.0]` |
| `history_length` | integer or null | Policy-only and `>= 1` |
| `use_state` | boolean or null | Policy-only |
| `observation` | object or null | Policy-only free-form state without a public nested schema |

## Defaults and request validation

For general Action profiles, omitted fields use `num_inference_steps=30`,
`guidance_scale=1.0`, and `fps=10.0`. The service always derives `num_frames`
as `action_chunk_size + 1`; Action requests reject an explicit `num_frames`.
Specialist profiles can replace these defaults and own additional fields.
Explicit values are validated against the selected profile's contract and
shared Generator ranges.

Action mode rejects:

- top-level `resolution`;
- top-level `transfer`;
- `condition_frame_indexes_vision` and `condition_video_keep`;
- video for forward dynamics or policy;
- image for inverse dynamics; and
- unknown action fields.

Request-level `guardrails` is not part of `/v1/infer`. Guardrails are operator
configuration; see [operations.md](operations.md#guardrails).

## Common failures

| Failure | Fix |
| --- | --- |
| `action_chunk_size` is not a positive multiple of 4 | Use 60 for AV or 16 for the general UMI and robot examples |
| Action rows or width do not match | Validate `[T,D]` and finite values before sending; use the domain table above |
| `num_frames` is rejected | Omit it; the service always derives `action_chunk_size + 1` |
| Wrong conditioning media | Image for forward/policy; video for inverse dynamics |
| Configuration cannot run the action case | Confirm the selected image's action-capable model variant and supported domain |
| Bridge URL input fails | Enable allowed URL input and container network access, or replace it with a data URL |
| Nano-DROID rejects model-owned fields | Omit fixed action dimensions, cadence, state flags, and top-level media-output controls |
| Client fails on missing `b64_video` | Handle action-only specialist responses separately from general visual Action responses |

For startup, OOM, and service diagnostics, see
[operations.md](operations.md#troubleshooting).
