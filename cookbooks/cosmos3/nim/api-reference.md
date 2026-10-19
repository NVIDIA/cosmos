<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 NIM API reference

Use this compact reference for runtime routing, the shared Generator request
envelope, media inputs, and the common Generator response. Task-specific rules
and examples live with each task workflow.

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

Reasoner features are in
[Reasoning](reasoning.md#final-answers-instructions-and-tool-calls). Check the
active `model_type` and `inference_endpoint` in `/v1/metadata`; other framework
endpoints are in
[Inspect the running service](operations.md#inspect-the-running-service).

## Generator: `POST /v1/infer`

The Generator accepts one synchronous JSON object and rejects unknown fields.
Every request must set `model_mode` to identify the task.

`NIM_MODEL_VARIANT` chooses the checkpoint at container start; `model_mode`
chooses the task per request.

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

Defaults are for video / T2I; Transfer and Action defaults are on their pages.

| Field | Type | Default (video / T2I) | Applies to |
| --- | --- | --- | --- |
| `model_mode` | required enum | None | Selects one of the seven modes above |
| `prompt` | string or null; maximum 20,000 characters | None | Required and non-empty for T2I/T2V; task-dependent otherwise |
| `negative_prompt` | string or null; maximum 20,000 characters | Built-in / empty | Tasks and models that permit negative prompting |
| `input_reference` | encoded image/video string or null | None | I2V, V2V, Action, and derived Transfer; media type follows `model_mode` |
| `seed` | non-negative integer or null | Service-generated | Reproducibility |
| `guidance_scale` | finite number in `[1.0,7.0]` | `6.0` / `4.0` | Sampling when not owned by a specialist model |
| `num_inference_steps` | integer in `[1,100]` | `35` / `50` | Sampling when not owned by a specialist model |
| `flow_shift` | finite number | `10.0` / `3.0` | Sampling when not owned by a specialist model |
| `resolution` | enum | `720` / `720_1_1` | Generation and Transfer; see [resolution keys](generation.md#resolution-keys) |
| `num_frames` | integer | `189` / `1` | Generation and Transfer; Action derives it and rejects the field |
| `fps` | finite number in `[1.0,60.0]` | `24.0` | Video tasks; retained but not encoded for T2I |
| `condition_frame_indexes_vision` | integer array or null | `[0, 1]` | V2V only |
| `condition_video_keep` | `first`, `last`, or null | `first` | V2V only |
| `action_params` | object | None | Action modes only; [Action](action.md#action-parameter-reference) |
| `transfer` | object | None | `video2video` only; [Transfer](transfer.md) |

Empty or whitespace-only media strings are treated as absent.

### Strict JSON types

Integer and finite-number fields use strict JSON types. For example, `"35"`,
`35.0`, and `true` are not accepted spellings of integer
`num_inference_steps=35`. Unknown nested fields are rejected too.

## Media inputs

Generator image and video inputs (`input_reference` and Transfer control
videos) accept:

- raw base64;
- a MIME-aware data URL such as `data:image/jpeg;base64,...` or
  `data:video/mp4;base64,...`; and
- an HTTP(S) URL when `NIM_ALLOW_URL_INPUT` is enabled (the default).

A data URL must use `;base64` and an `image/*` or `video/*` MIME type matching
the field. An HTTP(S) URL is fetched with a 30-second timeout and must respond
with a `Content-Type` of `image/*` or `video/*`. Generic
`application/octet-stream` is accepted for video only when the URL or
`Content-Disposition` names an `.mp4` file. Prefer data URLs; remote inputs
need container network access and can time out or change.

Input videos must use H.264, H.265/HEVC, MPEG-4, VP8, VP9, or AV1; other codecs
return HTTP 422. The container can be any format the decoder opens. Image
formats are summarized in the
[support matrix](support-matrix.md#media-and-codecs).

Size caps: I2V images 20,000,000 encoded characters; V2V decoded video 75 MB;
Transfer limits are in [Transfer](transfer.md#media-and-output).

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

Media fields are raw base64 (JPEG for T2I, VP9-in-MP4 for video), not data or
file URLs; inactive fields are omitted or null. Only general Policy and inverse
dynamics return video plus an
[action object](action.md#response-action-object).

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

## Errors and live schema

A request for the other runtime's primary route returns HTTP 404 with the
common NIM error envelope naming the active runtime and its endpoint. See
[Errors](operations.md#errors) and
[Troubleshooting](operations.md#troubleshooting); do not automate on exact
error-message text.

Save the active runtime's OpenAPI document:

```bash
curl -fsS http://localhost:8000/openapi.json -o openapi.json
python3 -m json.tool openapi.json >/dev/null
```

Treat the running NIM's schema as authoritative for available routes and
constraints in the deployed image.
