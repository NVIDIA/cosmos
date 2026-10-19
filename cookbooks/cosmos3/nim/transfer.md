<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Control video generation with Cosmos3 Transfer

Guide Generator output with edge, blur, depth, segmentation, or
world-space-map (WSM) video controls: send `POST /v1/infer` with
`model_mode=video2video` and a top-level `transfer` object.

> Confirm control and model availability for the selected image in the
> [Support matrix](support-matrix.md).

Start a compatible Generator with [Deployment](deployment.md) and begin with
one precomputed control. To choose between V2V, Transfer, and inverse
dynamics, see
[Choose a video-conditioned workflow](generation.md#choose-a-video-conditioned-workflow).

## Control types

| Control | Precomputed video | Server-derived from `input_reference` | Preset |
| --- | --- | --- | --- |
| `edge` | Yes | Yes | `very_low`, `low`, `medium`, `high`, `very_high` |
| `blur` | Yes | Yes | `none`, `very_low`, `low`, `medium`, `high`, `very_high` |
| `depth` | Required | No | None |
| `seg` | Required | No | None |
| `wsm` | Required | No | None |

At least one control must be enabled. Control media uses the same
[media input rules](api-reference.md#media-inputs) as `input_reference`.

## Run the examples

The script has five precomputed cases with matching prompts, control media,
geometry, and seeds; run one at a time, as each is an expensive generation.

Transfer is compute-intensive: use a high-throughput discrete GPU such as an
RTX PRO 6000 Blackwell 96 GB or H100 80 GB. DGX Spark can meet the memory
floors but is much slower and not recommended. Check memory against
[Transfer headroom](support-matrix.md#transfer-headroom).

The client allows 60 minutes per request, but the backend defaults to 30. To
match, add `-e NIM_TRITON_REQUEST_TIMEOUT=3600000000` (microseconds) to the
Generator launch and restart; see
[long-running requests](operations.md#long-running-requests).

Install the [client tooling](prerequisites.md#client-tooling). Then, from the
repository root, enter the cookbook directory before running a case:

```bash
cd cookbooks/cosmos3/nim
export NIM_URL=${NIM_URL:-http://localhost:8000}
uv run python examples/transfer.py --case precomputed_edge
```

For another precomputed control, replace `precomputed_edge` with
`precomputed_blur`, `precomputed_depth`, `precomputed_seg`, or
`precomputed_wsm`.

The NIM can also derive edge or blur controls from a source video. Run one
derived case:

```bash
uv run python examples/transfer.py --case derived_edge
```

Replace `derived_edge` with `derived_blur` to use a derived blur control. The
script needs a general-purpose `nano` or `super` variant and writes
`examples/outputs/transfer_<case>.mp4`.

## Precomputed control

Put the control media inside the matching nested object. A server-local file
path is not accepted as control media:

```json
{
  "model_mode": "video2video",
  "prompt": "<compact contents of assets/edge/prompt.json>",
  "negative_prompt": "<compact contents of assets/negative_prompt.json>",
  "transfer": {
    "edge": {
      "video": "data:video/mp4;base64,<BASE64_EDGE_CONTROL>"
    },
    "control_guidance": 1.5,
    "num_conditional_frames": 1,
    "num_first_chunk_conditional_frames": 0,
    "num_video_frames_per_chunk": 121
  },
  "resolution": "720_16_9",
  "num_frames": 121,
  "fps": 30.0,
  "num_inference_steps": 50,
  "guidance_scale": 3.0,
  "flow_shift": 10.0,
  "seed": 2026
}
```

Replace `edge` with `blur`, `depth`, `seg`, or `wsm` and provide the matching
control.

## Advanced: derive edge or blur

For server-derived controls, send the source video in `input_reference` and
omit the nested control video:

```json
{
  "model_mode": "video2video",
  "prompt": "A red sports car drives through a dramatic snowy landscape.",
  "negative_prompt": "<compact contents of assets/negative_prompt.json>",
  "input_reference": "data:video/mp4;base64,<BASE64_SOURCE_VIDEO>",
  "transfer": {
    "edge": {
      "preset_edge_threshold": "medium"
    },
    "control_guidance": 1.5,
    "num_conditional_frames": 1,
    "num_first_chunk_conditional_frames": 0,
    "num_video_frames_per_chunk": 121
  },
  "resolution": "720_16_9",
  "num_frames": 121,
  "fps": 30.0,
  "num_inference_steps": 50,
  "guidance_scale": 3.0,
  "seed": 2026
}
```

`"edge": true` and `"blur": true` select the corresponding derived control
with preset `medium`; they require `input_reference`. A preset applies
only to a derived control and cannot accompany nested `video`.

Depth, segmentation, and WSM cannot be derived by the server; supply a
precomputed control video.

## Advanced tuning

| Field | Constraint | Current effective default |
| --- | --- | --- |
| `control_guidance` | Number `[0.0,10.0]` | 1.5 generally, 2.0 segmentation-only, 3.0 WSM-only |
| `num_video_frames_per_chunk` | Integer `>= 1` | 93 generally, 101 WSM-only |
| `num_conditional_frames` | Integer `>= 0` and smaller than chunk size | 1 |
| `num_first_chunk_conditional_frames` | Integer `>= 0`; bounded by chunk and output | 0 |

`num_first_chunk_conditional_frames > 0` requires `input_reference`.

Current defaults by family (explicit valid values override them; recheck on
another image):

| Request family | Frames | FPS | Steps | Guidance | Flow shift | Control guidance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Edge, blur, depth, or mixed | 121 | 30 | 50 | 3.0 | 10.0 | 1.5 |
| Segmentation-only | 121 | 30 | 50 | 3.0 | 10.0 | 2.0 |
| WSM-only | 101 | 10 | 50 | 1.0 | 10.0 | 3.0 |

The example script uses a 121-frame chunk, or 101 for WSM.

## Advanced multiple controls

A `transfer` object can enable several controls; mixed requests use the edge
family defaults. This guide does not validate any combination, so use one
control per request unless yours is validated, and only combine controls that
are spatially and temporally aligned.

## Transfer VRAM admission

A ready deployment can serve generation but reject Transfer when GPU headroom
is too small; use a larger GPU or lower-VRAM configuration (see
[Transfer headroom](support-matrix.md#transfer-headroom), including the
`NIM_ALLOW_UNSAFE_TRANSFER` diagnostic override).

## Media and output

Control videos and video `input_reference` values accept up to 100,000,000
encoded characters; see [media inputs](api-reference.md#media-inputs) for
formats, codecs, and URL rules.

- The source video and each control video must have at least `num_frames`
  frames (WSM control: `num_frames - 1`).
- Their aspect ratio must be within 5% of the requested output shape.
- The decoded source and control videos together are capped at 225,000,000
  bytes; each single video is also capped at 75,000,000 bytes.

The response is `b64_video` plus `action: null`.

## Validation and common failures

Besides the failures below, Transfer rejects V2V fields
(`condition_frame_indexes_vision`, `condition_video_keep`) and `action_params`,
and a nested precomputed edge/blur video combined with its preset.

| Failure | Fix |
| --- | --- |
| HTTP 422 says derived control needs `video` | Add top-level source video or provide a nested precomputed edge/blur video |
| HTTP 422 says control video required | Add nested video for depth, segmentation, or WSM |
| Control/output lengths are incompatible | Recheck the output frame count, chunk size, and conditional frame values |
| Transfer is disabled while T2V works | Selected configuration fits T2V but GPU headroom is below Transfer overhead; use a larger GPU or lower-VRAM configuration |

For service-level failures, see [operations.md](operations.md#troubleshooting).
