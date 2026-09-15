<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM API reference

Use this compact reference for runtime routing, the shared Generator request
envelope, and the common Generator response. Detailed fields, constraints, and
examples live with each task workflow.

> Check `/openapi.json` on the running Generator or Reasoner for routes supplied
> by the active runtime and selected image.

## Runtime and primary endpoints

One selected profile starts one backend. A Generator profile does not serve
Reasoner completion APIs, and a Reasoner profile does not serve `/v1/infer`.

| Runtime | API | Guide |
| --- | --- | --- |
| Generator | `POST /v1/infer` for T2I, T2V, I2V, and V2V | [Generation](generation.md) |
| Generator | `POST /v1/infer` with `action_params` | [Action](action.md) |
| Generator | `POST /v1/infer` with `transfer` | [Transfer](transfer.md) |
| Reasoner | `POST /v1/chat/completions` | [Reasoning](reasoning.md) |
| Reasoner | `POST /v1/responses` and optional state routes | [Reasoning](reasoning.md#use-the-responses-api) |

Reasoner Chat Completions supports media, parsed-reasoning controls, developer
instructions, and OpenAI tool calls as described in
[Reasoning](reasoning.md#final-answers-instructions-and-tool-calls).
`/v1/metadata` reports the active `model_type` and primary
`inference_endpoint`; check those fields before treating `/v1/models` as
Reasoner model discovery.

The NIM framework also exposes health, model, metadata, manifest, version,
license, metrics, and OpenAPI endpoints. See
[Inspect the running service](operations.md#inspect-the-running-service) for
their paths and operational meaning.

## Generator: `POST /v1/infer`

The Generator accepts one synchronous JSON object and rejects unknown fields.
Every request must set `model_mode` to identify the task.

`NIM_MODEL_VARIANT` and `model_mode` select different things:

- `NIM_MODEL_VARIANT` chooses the checkpoint when the container starts.
- `model_mode` chooses the task for one request.

| `model_mode` | Task | `input_reference` |
| --- | --- | --- |
| `text2image` | T2I | Forbidden |
| `text2video` | T2V | Forbidden |
| `image2video` | I2V | Required image |
| `video2video` | V2V | Required video |
| `video2video` with `transfer` | Transfer | Optional for precomputed controls; required for derived edge/blur |
| `forward_dynamics` | Forward dynamics | Required image |
| `policy` | Policy | Required image |
| `inverse_dynamics` | Inverse dynamics | Required video |

The task guides define the remaining required inputs and invalid combinations.

### Common Generator request fields

| Field | Type | Applies to |
| --- | --- | --- |
| `model_mode` | required enum | Selects one of the seven modes above |
| `prompt` | string or null; maximum 20,000 characters | Required and non-empty for T2I/T2V; task-dependent otherwise |
| `negative_prompt` | string or null; maximum 20,000 characters | Tasks and models that permit negative prompting |
| `input_reference` | encoded image/video string or null | I2V, V2V, Action, and derived Transfer; media type follows `model_mode` |
| `seed` | non-negative integer or null | Reproducibility; the service generates one when omitted |
| `guidance_scale` | finite number in `[1.0,7.0]` | Sampling when not owned by a specialist model |
| `num_inference_steps` | integer in `[1,100]` | Sampling when not owned by a specialist model |
| `flow_shift` | finite number | Sampling when not owned by a specialist model |
| `resolution` | enum | Generation and Transfer; see [resolution keys](generation.md#resolution-keys) |
| `num_frames` | integer | Generation and Transfer; Action derives it and rejects the field |
| `fps` | finite number in `[1.0,60.0]` | Video tasks; retained but not encoded for T2I |
| `condition_frame_indexes_vision` | integer array or null | V2V only |
| `condition_video_keep` | `first`, `last`, or null | V2V only |
| `action_params` | object | Action modes only; [Action](action.md#action-parameter-reference) |
| `transfer` | object | `video2video` only; [Transfer](transfer.md) |

Empty or whitespace-only media strings are treated as absent. Media
representations and codec boundaries are documented under
[Generation media representations](generation.md#media-representations) and
the [Support matrix](support-matrix.md#media-and-codecs).

### Strict JSON types

Integer and finite-number fields use strict JSON types. For example, `"35"`,
`35.0`, and `true` are not accepted spellings of integer
`num_inference_steps=35`. Unknown top-level and nested fields are rejected
rather than silently ignored.

## Generator response

A successful Generator response contains an image, a video, or—on a compatible
specialist policy profile—an action without visual media. T2I returns:

```json
{
  "b64_image": "<RAW_BASE64_JPEG>"
}
```

Video modes return:

```json
{
  "b64_video": "<RAW_BASE64_MP4>",
  "action": null
}
```

Both media fields are raw base64, not data URLs or file URLs. Inactive fields
can be omitted or null depending on response serialization. T2I cannot return
non-null `action` metadata. Ordinary video generation, Transfer, and forward
dynamics return no predicted action; general Policy and inverse dynamics return
video plus the trajectory envelope documented in
[Response action object](action.md#response-action-object).

A specialist action-only policy can return:

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

In that case both `b64_image` and `b64_video` are absent or null. Clients must
branch on the fields actually present rather than assuming every non-T2I
request has `b64_video`. See [Nano-DROID policy](action.md#nano-droid-policy).

The Generator emits JPEG for T2I and a VP9 video track in an MP4 container for
video modes. See the documented media boundaries in the [Support
matrix](support-matrix.md#media-and-codecs).

## Errors and live schema

A request for the other runtime's primary route returns HTTP 404. The response
uses the common NIM error envelope and identifies the active runtime, its
primary endpoint, and the runtime required by the request.
For the representative envelope, other HTTP status guidance, and symptom-based
diagnosis, see [Errors](operations.md#errors) and
[Troubleshooting](operations.md#troubleshooting). Do not build automation
around exact mutable error-message text.

Save the active runtime's OpenAPI document:

```bash
curl -fsS http://localhost:8000/openapi.json -o openapi.json
python3 -m json.tool openapi.json >/dev/null
```

The local deployment guide publishes either selected runtime on this same host
port. Remove the active example container, launch the other runtime, and repeat the
capture. Treat the running NIM's schema as authoritative for available routes
and constraints in the deployed image.
