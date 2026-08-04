<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Use Cosmos3 action capabilities through the Certified NIM

Use this page for forward dynamics, policy, and inverse dynamics requests. All
three use a compatible **Generator** profile and synchronous JSON
`POST /v1/infer`; the nested `action_params.mode` selects the task.

> **Release status:** The request/response contracts are source-derived.
> Published profile availability and the final supported model/domain matrix
> remain **TBD (release-dependent)**.

See [deployment.md](deployment.md) to start the NIM. This page is the canonical
contract for `action_params`.

## Action modes

| Mode | Input | Output |
| --- | --- | --- |
| `forward_dynamics` | Conditioning image plus action trajectory `[T,D]` | Rollout video; `action` response is null |
| `policy` | Conditioning image, task prompt, and optional state | Rollout video plus predicted action trajectory, or action only for a compatible specialist profile |
| `inverse_dynamics` | Conditioning video and optional task prompt | Video plus predicted action trajectory |

Action requests must not set top-level `resolution`. The backend derives the
shape from the conditioning media and `action_params.image_size`.

## Domains and dimensions

| `domain_name` | Domain ID | `raw_action_dim` | Typical `action_chunk_size` |
| --- | ---: | ---: | ---: |
| `av` | 1 | 9 | 60 |
| `umi` | 6 | 10 | 16 |
| `bridge_orig_lerobot` | 7 | 10 | 16 |
| `droid_lerobot` | 8 | 10 | 16 |

`action_chunk_size` must be a positive multiple of 4. For visual Action
profiles, video frame count is `action_chunk_size + 1`, which preserves the
`4k+1` cadence. If `raw_action_dim` is supplied, it must match the selected
domain.

A specialist profile can own a different fixed representation. In particular,
Nano-DROID supplies an 8-wide action representation even though the general
`droid_lerobot` domain default is 10. Clients must omit `raw_action_dim` for
that specialist contract.

## Action parameter reference

| Field | Type/default | Contract |
| --- | --- | --- |
| `mode` | required enum | `forward_dynamics`, `policy`, or `inverse_dynamics` |
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

## Run the examples

From the repository root:

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
python -m pip install requests

python cookbooks/cosmos3/nim/examples/action.py --case forward_dynamics
python cookbooks/cosmos3/nim/examples/action.py --case policy
python cookbooks/cosmos3/nim/examples/action.py --case inverse_dynamics
```

The script runs only the selected case. It reuses the AV image/trajectory
already tracked by the action cookbook and uses a commit-pinned public input for
the Bridge inverse-dynamics case. It writes `action_<case>.mp4` when visual
media is present and `action_<case>.json` when the server predicts actions.

## Forward dynamics

Forward dynamics requires an image and a complete numeric action trajectory.
The public example constructs this request after reading
`generator/action/assets/actions/av_traj_forward.json`:

```python
request = {
    "prompt": "You are an autonomous vehicle planning system.",
    "image": image_data_url,
    "action_params": {
        "mode": "forward_dynamics",
        "domain_name": "av",
        "action_chunk_size": 60,
        "action": trajectory,  # 60 rows × 9 numeric values
        "raw_action_dim": 9,
        "action_space": "joint_pos",
        "image_size": "480",
        "action_fps": 10.0,
    },
    "num_output_frames": 61,
    "fps": 10.0,
    "steps": 30,
    "guidance_scale": 1.0,
    "flow_shift": 10.0,
    "seed": 0,
}
```

Every action row must have the same width. The number of rows must equal
`action_chunk_size`; the width must equal `raw_action_dim`. Forward dynamics
does not accept `history_length`, `use_state`, or `observation`.

The response contains a generated rollout video and `"action": null` because
the trajectory was an input.

## Policy

Policy predicts the action instead of receiving it:

```json
{
  "prompt": "You are an autonomous vehicle planning system.",
  "image": "data:image/jpeg;base64,<BASE64_IMAGE>",
  "action_params": {
    "mode": "policy",
    "domain_name": "av",
    "action_chunk_size": 60,
    "raw_action_dim": 9,
    "action_space": "joint_pos",
    "image_size": "480",
    "action_fps": 10.0
  },
  "num_output_frames": 61,
  "fps": 10.0,
  "steps": 30,
  "guidance_scale": 1.0,
  "seed": 0
}
```

Do not send `action`. Policy alone may also accept:

- `history_length`: number of state-history steps;
- `use_state`: whether to condition on supplied state; and
- `observation`: free-form state passed to the pipeline without a public nested
  schema.

Use state conditioning only with a model/profile and observation shape that
has been validated for the released deployment.

## Nano-DROID policy

Nano-DROID is a specialist policy checkpoint selected with:

```bash
-e NIM_MODEL_TYPE=generator \
-e NIM_MODEL_VARIANT=nano-droid
```

Its request uses the same `POST /v1/infer` API, not a separate WebSocket or
vLLM-Omni endpoint:

```json
{
  "prompt": "Remove the crumpled paper from the sink and place it in the trash can.",
  "image": "data:image/jpeg;base64,<COMPOSED_DROID_OBSERVATION>",
  "action_params": {
    "mode": "policy",
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
required by the released checkpoint and a task prompt that describes the
desired manipulation.

The client must omit profile-owned fields:

- `raw_action_dim`, `action_space`, `image_size`, `action_fps`,
  `history_length`, and `use_state` inside `action_params`; and
- top-level `fps`, `num_output_frames`, and `negative_prompt`.

Current source supplies an 8-wide action representation, 32 action steps,
33 internal cadence frames, and 15 FPS. It defaults to four inference steps,
guidance `3.0`, and flow shift `5.0`; `steps`, `guidance_scale`, `flow_shift`,
and `seed` remain overridable.

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
and no `b64_image` or `b64_video`. A runnable cookbook case is deferred until
an approved public composed DROID observation asset is available.

## Inverse dynamics

Inverse dynamics uses a video instead of an image:

```json
{
  "prompt": "Put the pot to the left of the purple item.",
  "video": "https://example.com/bridge-task.mp4",
  "action_params": {
    "mode": "inverse_dynamics",
    "domain_name": "bridge_orig_lerobot",
    "action_chunk_size": 16,
    "raw_action_dim": 10,
    "action_space": "joint_pos",
    "image_size": "480",
    "action_fps": 5.0
  },
  "num_output_frames": 17,
  "fps": 5.0,
  "steps": 30,
  "guidance_scale": 1.0,
  "seed": 0
}
```

Replace the illustrative URL with an allowed URL or data URL. Inverse dynamics
does not accept `action` or the policy-only state fields.

## Response action object

General visual Policy and inverse dynamics profiles return:

```json
{
  "b64_video": "<BASE64_MP4>",
  "action": {
    "data": [[0.0, 0.0]],
    "shape": [16, 10],
    "dtype": "float32",
    "raw_action_dim": 10,
    "action_mode": "inverse_dynamics",
    "domain_id": 7
  }
}
```

The shortened `data` above illustrates the envelope only; a real response has
the full shape reported by `shape`. A specialist policy can instead return the
same `action` envelope with both media fields absent, as described under
[Nano-DROID policy](#nano-droid-policy). The public script writes available
media and action artifacts independently.

## Defaults and validation

For general Action profiles, omitted fields use `steps=30`,
`guidance_scale=1.0`, and `fps=10.0`, and the service derives
`num_output_frames`. Specialist profiles can replace these defaults and own
additional fields. Explicit values are validated against the selected
profile's contract and shared Generator ranges.

Action mode rejects:

- top-level `resolution`;
- top-level `transfer`;
- `condition_frame_indexes_vision` and `condition_video_keep`;
- video for forward dynamics or policy;
- image for inverse dynamics; and
- unknown action fields such as vLLM-Omni-only `view_point`.

Request-level `guardrails` is not part of `/v1/infer`. Guardrails are operator
configuration; see [operations.md](operations.md#guardrails).

## Common failures

| Failure | Fix |
| --- | --- |
| `action_chunk_size` is not a positive multiple of 4 | Use 60 for the AV example or 16 for the robot examples |
| Action rows or width do not match | Validate `[T,D]` before sending; use the domain table above |
| `num_output_frames` mismatch | Omit it or set exactly `action_chunk_size + 1` |
| Wrong conditioning media | Image for forward/policy; video for inverse dynamics |
| Profile/model cannot run the action case | Confirm the released image's action-capable model variant and supported domain |
| Nano-DROID rejects profile-owned fields | Omit fixed action dimensions, cadence, state flags, and top-level media-output controls |
| Client fails on missing `b64_video` | Handle action-only specialist responses and save the `action` object independently |

For startup, OOM, and service diagnostics, see
[operations.md](operations.md#troubleshooting).
