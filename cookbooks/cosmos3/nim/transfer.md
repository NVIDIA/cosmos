<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Control video generation with Cosmos3 Transfer

Use this page to guide Generator output with edge, blur, depth, segmentation,
or world-space-map (WSM) video controls. Transfer uses synchronous JSON
`POST /v1/infer` with `model_mode=video2video` and a top-level `transfer`
object.

> Confirm control and model availability for the selected image in the
> [Support matrix](support-matrix.md).

See [Deployment](deployment.md) to start a compatible Generator model. Begin
with one precomputed control, then use derived or multiple controls only when
the workload requires them. If you are choosing between an ordinary V2V,
Transfer, or inverse-dynamics request, see
[Choose a video-conditioned workflow](generation.md#choose-a-video-conditioned-workflow).

## Control types

| Control | Precomputed video | Server-derived from `input_reference` | Preset |
| --- | --- | --- | --- |
| `edge` | Yes | Yes | `very_low`, `low`, `medium`, `high`, `very_high` |
| `blur` | Yes | Yes | `none`, `very_low`, `low`, `medium`, `high`, `very_high` |
| `depth` | Required | No | None |
| `seg` | Required | No | None |
| `wsm` | Required | No | None |

At least one control must be enabled. Precomputed control media uses the same
base64/data-URL/allowed-public-URL contract as `input_reference`.

## Run the examples

The script includes five precomputed control cases with matching prompts,
control media, geometry, and seeds. Run one case at a time so one command does
not start several expensive generations.

Transfer is particularly compute-intensive. For practical evaluation
turnaround, use a high-throughput discrete GPU such as an NVIDIA RTX PRO 6000
Blackwell with 96 GB, an H100 with 80 GB, or a higher-throughput discrete GPU
that meets the selected profile's Transfer floor. DGX Spark can satisfy the
memory floors for compatible profiles, but it is not recommended for Transfer
because requests can take substantially longer. This operational recommendation
does not change profile compatibility: continue to check total and currently
free memory against the [Transfer headroom
requirements](support-matrix.md#transfer-headroom).

Transfer inference is synchronous. The connection can remain open without a
response body while the request is running. The example client allows up to 60
minutes for the request; that timeout is a ceiling, not an expected completion
time. The Generator backend has a separate 30-minute default. To permit a
Transfer request to use the full client ceiling, add
`-e NIM_TRITON_REQUEST_TIMEOUT=3600000000` to the Generator Docker launch
command and restart the container before inference. The value is in
microseconds; changing the client does not change the backend setting. See
[Generator configuration](configuration.md#generator-configuration).

Do not retry only because the connection is quiet. Check the active request
using the [long-running request
guidance](operations.md#long-running-requests) first.

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

Replace `derived_edge` with `derived_blur` to use a derived blur control.

The script rejects unknown command-line arguments before contacting the
endpoint. Before loading case media or submitting inference, it requires
`/v1/metadata` to report the Generator runtime, `/v1/infer`, a selected profile
ID, and a general-purpose `nano` or `super` variant. This metadata check does
not establish Transfer headroom; startup evaluates that requirement separately,
and the service can still reject Transfer as unavailable. The decoded result is
written to `examples/outputs/transfer_<case>.mp4`.

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
control. The complete script converts the existing local control video to the
data URL automatically.

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
without an explicit preset; they require `input_reference`. A preset applies
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

Current defaults by family:

| Request family | Frames | FPS | Guidance | Control guidance |
| --- | ---: | ---: | ---: | ---: |
| Edge, blur, depth, or mixed | 121 | 30 | 3.0 | 1.5 |
| Segmentation-only | 121 | 30 | 3.0 | 2.0 |
| WSM-only | 101 | 10 | 1.0 | 3.0 |

All current transfer families default to 50 denoising steps and flow shift
10.0. The comparison script explicitly uses a 121-frame chunk for edge, blur,
depth, and segmentation, and a 101-frame chunk for the 101-frame WSM case.
Explicit request values override defaults when valid. These defaults apply to
the pinned release image; recheck them when moving to another image.

## Advanced multiple controls

The request schema can enable more than one control in the same `transfer`
object. Current default selection deliberately uses the general edge-family
defaults for mixed requests rather than combining defaults from each control.

This guide does not validate a multi-control combination. Use one control per
request unless the exact combination has been validated for the deployment.
Controls in a multi-control smoke test must be spatially and temporally aligned;
do not combine unrelated single-control fixtures and interpret the result as a
quality comparison.

## Transfer VRAM admission

Transfer has a higher peak-memory requirement than ordinary text-to-video on
the same profile. Startup compares the visible GPU's headroom above the
selected profile floor with the measured Transfer overhead. A deployment can
therefore be ready and serve generation without Transfer while rejecting it.

If Transfer is unavailable, use a larger GPU or an available lower-VRAM
configuration.
`NIM_ALLOW_UNSAFE_TRANSFER=1` bypasses the admission check, but the request can
run out of memory and the deployment does not become supported. Do not use the
override as a normal production setting. See
[Transfer headroom](support-matrix.md#transfer-headroom).

## Media and output

Control videos and video `input_reference` values accept raw base64, a video
data URL, or an allowed HTTP(S) URL, with a 100,000,000-character encoded
ceiling. Prefer data URLs and the included MP4 fixtures; verify any other
container, codec, or remote-fetch path against the deployed image before
relying on it.

The response uses the Generator video shape: `b64_video` plus
`action: null`. The inactive image field is omitted. The public script decodes
it to `transfer_<case>.mp4`.

## Validation and common failures

Transfer rejects:

- an empty `transfer` object;
- a `model_mode` other than `video2video`;
- `action_params`;
- V2V fields `condition_frame_indexes_vision` and `condition_video_keep`;
- derived edge/blur without `input_reference`;
- depth/seg/WSM without a nested non-empty control video;
- a nested precomputed edge/blur video combined with its preset; and
- chunk/conditioning values that violate their bounds.

| Failure | Fix |
| --- | --- |
| HTTP 422 says derived control needs `video` | Add top-level source video or provide a nested precomputed edge/blur video |
| HTTP 422 says control video required | Add nested video for depth, segmentation, or WSM |
| Unknown field | Remove fields that are not defined in the Transfer request object |
| Control/output lengths are incompatible | Recheck the output frame count, chunk size, and conditional frame values |
| Transfer is disabled while T2V works | Selected configuration fits T2V but GPU headroom is below Transfer overhead; use a larger GPU or lower-VRAM configuration |

For service-level failures, see [operations.md](operations.md#troubleshooting).
