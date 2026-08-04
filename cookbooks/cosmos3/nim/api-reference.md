<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM API reference

Use this compact reference for runtime routing, the shared Generator request
envelope, and the common Generator response. Detailed task contracts live with
their workflows so fields, constraints, and examples have one canonical owner.

> **Release status:** The Generator envelope is derived from current request
> models and tests. NIMlib- and vLLM-supplied routes must be checked against
> `/openapi.json` from the released image under each runtime.

## Runtime and primary endpoints

One selected profile starts one backend. A Generator profile does not serve
Reasoner completion APIs, and a Reasoner profile does not serve `/v1/infer`.

| Runtime | API | Canonical documentation |
| --- | --- | --- |
| Generator | `POST /v1/infer` for T2I, T2V, I2V, and V2V | [Generation](generation.md) |
| Generator | `POST /v1/infer` with `action_params` | [Action](action.md) |
| Generator | `POST /v1/infer` with `transfer` | [Transfer](transfer.md) |
| Reasoner | `POST /v1/chat/completions`, including streaming | [Reasoning](reasoning.md) |
| Reasoner | `POST /v1/responses` and optional state routes | [Reasoning](reasoning.md#use-the-responses-api) |
| Reasoner | `POST /v1/completions` legacy-compatible route | Verify the released schema before use |

The NIM framework also exposes health, model, metadata, manifest, version,
license, metrics, and OpenAPI endpoints. See
[Inspect the running service](operations.md#inspect-the-running-service) for
their paths and operational meaning.

## Generator: `POST /v1/infer`

The Generator accepts one synchronous JSON object and rejects unknown fields.
Its shape selects the task:

| Shape | Task |
| --- | --- |
| Non-empty `prompt`, no conditioning inputs, and `num_output_frames=1` | T2I |
| Non-empty `prompt` without media and with 25 or more output frames | T2V |
| Top-level `image` | I2V, or Action when paired with `action_params` |
| Top-level `video` | V2V, inverse dynamics, or a derived Transfer control depending on nested fields |
| Non-empty `action_params` | Forward dynamics, policy, or inverse dynamics |
| Non-empty `transfer` | Edge, blur, depth, segmentation, WSM, or mixed Transfer |

The task guides define required inputs and invalid combinations.

### Common Generator request fields

Defaults below describe ordinary generation. T2I applies the mode-specific
defaults identified in the table. Action and Transfer apply the defaults
documented on their canonical pages.

| Field | Type and ordinary default | Contract |
| --- | --- | --- |
| `prompt` | string or null; `null` | Maximum 20,000 characters; required when no media or nested task establishes the request |
| `negative_prompt` | string or null; `null` | Maximum 20,000 characters; omission becomes `""` for T2I and selects the bundled default for video modes; explicit `""` disables negative-prompt conditioning |
| `image` | string or null; `null` | Raw base64, image data URL, or allowed HTTP(S) URL; maximum 20,000,000 encoded characters |
| `video` | string or null; `null` | Raw base64, video data URL, or allowed HTTP(S) URL; maximum 100,000,000 encoded characters and 75 MB decoded |
| `seed` | integer or null; `null` | Must be `>= 0`; the service generates one when omitted |
| `guidance_scale` | finite number; video `6.0`, T2I `4.0` | Range `[1.0,7.0]`; Action and Transfer override the default |
| `steps` | integer; video `35`, T2I `50` | Range `[1,100]`; Action and Transfer override the default |
| `flow_shift` | finite number; video `10.0`, T2I `3.0` | No additional range constraint in the current request model |
| `resolution` | enum; video `720`, T2I `720_1_1` | Ordinary Generation and Transfer only; see [resolution keys](generation.md#resolution-keys) |
| `num_output_frames` | integer; video `189` | `1` selects T2I; video uses the `4k+1` cadence from 25 upward. See [generation cadence](generation.md#frame-cadence-and-limits), [Action](action.md#domains-and-dimensions), or [Transfer](transfer.md#transfer-tuning) |
| `fps` | finite number; `24.0` | Range `[1.0,60.0]`; retained in T2I requests but not encoded in the JPEG |
| `condition_frame_indexes_vision` | integer array or null | V2V-only latent-frame indexes; see [Video-to-video](generation.md#video-to-video) |
| `condition_video_keep` | `first`, `last`, or null | V2V-only frame-selection direction; defaults to `first` |
| `action_params` | object or null | Canonical nested contract: [Action parameter reference](action.md#action-parameter-reference) |
| `transfer` | object or null | Canonical nested contract: [Transfer](transfer.md) |

Empty or whitespace-only media strings are treated as absent. Media
representations and release codec boundaries are documented under
[Generation media representations](generation.md#media-representations) and
the [Support matrix](support-matrix.md#supported-media-and-codecs).

### Strict JSON types

Integer and finite-number fields use strict JSON types. For example, `"35"`,
`35.0`, and `true` are not accepted spellings of integer `steps=35`. Unknown
top-level and nested fields are rejected rather than silently ignored.

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

The current source encoder emits JPEG for T2I and a VP9 video track in an MP4
container for video modes. Released output codec support belongs to the
[Support matrix](support-matrix.md#supported-media-and-codecs).

## Errors and live schema

For HTTP status guidance, the common error envelope, and symptom-based
diagnosis, see [Errors](operations.md#errors) and
[Troubleshooting](operations.md#troubleshooting). Do not build automation
around exact mutable error-message text.

Save the active runtime's OpenAPI document:

```bash
curl -fsS http://localhost:8000/openapi.json -o openapi.json
python -m json.tool openapi.json >/dev/null
```

Repeat this under Generator and Reasoner profiles. Treat the released image's
live schema as authoritative for generated routes and release-specific
constraints; report documentation conflicts instead of silently choosing one.
