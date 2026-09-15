<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Generate images and videos with the Cosmos3 Certified NIM

Use this page for text-to-image (T2I), text-to-video (T2V), image-to-video
(I2V), and video-to-video (V2V) requests. These workflows require a running
**Generator** model and use synchronous JSON `POST /v1/infer`.

For launch instructions, see [deployment.md](deployment.md). The
[API reference](api-reference.md#common-generator-request-fields) defines the
shared request envelope; this page defines T2I, T2V, I2V, and V2V rules.

## Prerequisites

Verify that the Generator is ready:

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -f "$NIM_URL/v1/health/ready"
```

Generator inference is synchronous. In particular, a video request can keep the
connection open without returning a response body until generation completes.
The examples allow up to 30 minutes for the HTTP request; that timeout is a
ceiling, not an expected completion time. Do not retry only because the
connection is quiet. Use the [long-running request
guidance](operations.md#long-running-requests) to check the active deployment
first.

Install the [client tooling](prerequisites.md#client-tooling). Then, from the
repository root, enter the cookbook directory before running examples. They use
the pinned `requests` dependency, use the included prompts and media, and
decode responses under
`examples/outputs/`:

```bash
cd cookbooks/cosmos3/nim
uv run python examples/t2i.py
```

## Choose a modality

Choose an example based on the input and output you need:

| Modality | Example scenario | Conditioning input | Response field | Run |
| --- | --- | --- | --- | --- |
| T2I | Robot draping satin over a mannequin | None | `b64_image` | `uv run python examples/t2i.py` |
| T2V | Robot cleaning a kitchen | None | `b64_video` | `uv run python examples/t2v.py` |
| I2V | Car traveling along a coastal road | `car_driving.jpg` | `b64_video` | `uv run python examples/i2v.py` |
| V2V | Continue or transform a car-driving video | `car_driving_plain.mp4` | `b64_video` | `uv run python examples/v2v.py` |

The scripts parse their command line before doing endpoint or media work, so
`--help` exits without inference and unexpected arguments are rejected. Before
submitting a request, each script requires `/v1/metadata` to report the
Generator runtime, `/v1/infer`, a selected profile ID, and a model variant
compatible with that example. They then load prompts and media from the
repository, build the Generator request, and save decoded output under
`examples/outputs/`. The shorter prompts below emphasize the request shape; use
the scripts for complete editable examples.

A standard script can run against an active general-purpose `nano` or `super`
variant when the selected image includes that task. T2I and I2V also accept
their matching full-step specialist variants. Specialist four-step variants
require their exact matching scripts and request contract.

## Text-to-image

A T2I request has a non-empty `prompt`, no conditioning inputs, and exactly
one output frame:

```bash
curl -fsS -X POST "$NIM_URL/v1/infer" \
  -H 'Content-Type: application/json' \
  -d '{
    "model_mode": "text2image",
    "prompt": "A white robotic arm draping sapphire satin over a dress mannequin in a softly lit fashion studio.",
    "negative_prompt": "",
    "resolution": "720_1_1",
    "num_frames": 1,
    "fps": 24.0,
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "flow_shift": 3.0,
    "seed": 0
  }' \
  -o /tmp/cosmos3-t2i-response.json

python3 - <<'PY'
import base64
import json
from pathlib import Path

response = json.loads(Path("/tmp/cosmos3-t2i-response.json").read_text())
Path("t2i_robot_draping.jpg").write_bytes(
    base64.b64decode(response["b64_image"])
)
PY
```

Run the complete editable comparison case:

```bash
uv run python examples/t2i.py
```

It uses the included `assets/prompts/text2image/robot_draping.json` prompt and
saves `examples/outputs/t2i_robot_draping.jpg`.

Set `model_mode` to `text2image`. T2I requires a non-empty prompt, forbids
`input_reference`, and uses exactly `num_frames=1`. Image-to-image is not
supported; use `image2video` when conditioning on an image.

When their fields are omitted, T2I uses:

| Field | T2I default |
| --- | --- |
| `resolution` | `720_1_1` (960 × 960) |
| `negative_prompt` | Empty string |
| `num_inference_steps` | `50` |
| `guidance_scale` | `4.0` |
| `flow_shift` | `3.0` |
| `fps` | `24.0`; retained in the request but not encoded in the JPEG |

All resolution keys listed under [Resolution keys](#resolution-keys) are
accepted for T2I. The supported keys stop at the 720 tier.

## Text-to-video

A T2V request sets `model_mode` to `text2video`, has a non-empty `prompt`, and
omits `input_reference`:

```bash
curl -fsS -X POST "$NIM_URL/v1/infer" \
  -H 'Content-Type: application/json' \
  -d '{
    "model_mode": "text2video",
    "prompt": "A modern industrial robotic arm cleans a kitchen counter with a green sponge.",
    "resolution": "720_16_9",
    "num_frames": 189,
    "fps": 24.0,
    "seed": 0
  }' \
  -o /tmp/cosmos3-response.json

python3 - <<'PY'
import base64
import json
from pathlib import Path

response = json.loads(Path("/tmp/cosmos3-response.json").read_text())
Path("t2v_robot_kitchen.mp4").write_bytes(
    base64.b64decode(response["b64_video"])
)
PY
```

Run the complete editable comparison case:

```bash
uv run python examples/t2v.py
```

It uses `assets/prompts/text2video/robot_kitchen.json`, the shared T2V negative
prompt, and saves `examples/outputs/t2v_robot_kitchen.mp4`. The default negative
prompt is supplied by the service when omitted from a custom request. Pass
`"negative_prompt": ""` only when you intentionally want to disable it.

## Image-to-video

I2V accepts the conditioning image as raw base64, a MIME-aware data URL, or an
allowed public HTTP(S) URL. A data URL preserves the media type and avoids
shell-specific base64 flags:

```python
import base64
import json
import mimetypes
import urllib.request
from pathlib import Path

nim_url = "http://localhost:8000"
image_path = Path(
    "../generator/audiovisual/assets/images/image2video/car_driving.jpg"
)
mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
image = f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode()}"

request = {
    "model_mode": "image2video",
    "prompt": "A car travels along a coastal mountain road with natural motion.",
    "input_reference": image,
    "resolution": "720",
    "num_frames": 189,
    "fps": 24.0,
    "seed": 0,
}
payload = json.dumps(request).encode()
http_request = urllib.request.Request(
    f"{nim_url}/v1/infer",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(http_request, timeout=1800) as response:
    result = json.load(response)
Path("i2v_car_driving.mp4").write_bytes(base64.b64decode(result["b64_video"]))
```

Run the complete editable comparison case:

```bash
uv run python examples/i2v.py
```

It pairs `assets/images/image2video/car_driving.jpg` with
`assets/prompts/image2video/car_driving.json`, uses the shared I2V negative
prompt, and saves `examples/outputs/i2v_car_driving.mp4`.

`input_reference` is required and interpreted as an image because
`model_mode=image2video`. The encoded-image ceiling is 20,000,000 characters.
Use JPEG, PNG, or WebP input as covered by the cookbook fixtures. Verify any
other image format against the deployed image before depending on it.

## Choose a video-conditioned workflow

Use the intent and distinguishing fields—not only the presence of media—to
choose the request shape:

| Intent | `model_mode` | Distinguishing input | Guide |
| --- | --- | --- | --- |
| Animate a still image | `image2video` | Image `input_reference` | [Image-to-video](#image-to-video) |
| Continue or transform a source video | `video2video` | Video `input_reference` plus optional V2V conditioning fields | [Video-to-video](#video-to-video) |
| Follow an edge, blur, depth, segmentation, or WSM control | `video2video` | Non-empty `transfer`; top-level video only for a derived control | [Transfer](transfer.md) |
| Estimate actions from an observed video | `inverse_dynamics` | Video `input_reference` plus `action_params` | [Action](action.md#inverse-dynamics) |

Plain V2V, Transfer, and inverse dynamics are separate contracts. Do not combine
V2V conditioning fields with `transfer`, or combine either with
`action_params`; conflicting shapes return HTTP 422.

## Video-to-video

V2V sets `model_mode` to `video2video`, supplies a video in
`input_reference`, and omits `transfer` and `action_params`:

```json
{
  "model_mode": "video2video",
  "prompt": "Keep the camera motion and change the environment to a snowy valley.",
  "input_reference": "data:video/mp4;base64,<BASE64_VIDEO>",
  "condition_frame_indexes_vision": [0, 1],
  "condition_video_keep": "first",
  "resolution": "720",
  "num_frames": 93,
  "fps": 24.0,
  "seed": 0
}
```

Run the complete local-media example:

```bash
uv run python examples/v2v.py
```

`condition_frame_indexes_vision` indexes latent frames, not pixel frames. The
service sorts and deduplicates it. Its largest value must fit the requested
output latent length. `condition_video_keep` selects frames from the beginning
or end of the input and defaults to `first`.

The decoded-video ceiling is 75 MB. Data URLs expand binary media by roughly
one third, so large videos can make the JSON request substantially larger. Use
the MP4 fixtures included with the examples, and verify any other container,
codec, or remote-fetch path against the deployed image before depending on it.

## Specialist T2I and I2V variants

General-purpose `nano` and `super` models support the tasks included by the
selected image. Super also provides task-specific variants:

| Variant | Accepted request | Sampling behavior |
| --- | --- | --- |
| `super-t2i` | T2I only | Full-step T2I; omitted `flow_shift` becomes `3.0` |
| `super-t2i-4step` | T2I only | Fixed four-step scheduler |
| `super-i2v` | I2V only | Full-step I2V; omitted `flow_shift` becomes `1.0` |
| `super-i2v-4step` | I2V only | Fixed four-step scheduler |

A specialist rejects other Generator tasks. Select the exact matching variant
before launch; `NIM_MODEL_VARIANT=super` selects the general-purpose `super`
contract.

Launch T2I with `NIM_MODEL_VARIANT=super-t2i-4step`, then run:

```bash
uv run python examples/t2i_4step.py
```

Launch I2V with `NIM_MODEL_VARIANT=super-i2v-4step`, then run:

```bash
uv run python examples/i2v_4step.py
```

Each script must run against the matching active model. Four-step requests must
omit `num_inference_steps`, `guidance_scale`, and `flow_shift`; the model owns
those values.
`seed` and other ordinary fields remain available.

## Choose resolution, frames, and FPS

### Frame counts and limits

T2I requires exactly one output frame. Video generation accepts any integer
frame count from 25 through the resolution-tier maximum. The video VAE's native
pixel-frame counts follow `1 + 4k` because it has temporal compression factor 4
and a causal first frame. For a non-native count, the service generates at the
next native count and trims the decoded result before encoding, so the response
contains exactly the requested `num_frames`.

Video output limits are:

| Resolution tier | Maximum frames |
| --- | ---: |
| `256` | 397 |
| `480` | 297 |
| `720` | 197 |

The largest V2V `condition_frame_indexes_vision` value must fit the internal
output latent-frame range. The service calculates that range from the next
native `1 + 4k` count. For example, `num_frames=26` uses 29 frames internally,
so it has 8 latent frames and the largest valid conditioning index is 7. The
response is still trimmed to exactly 26 pixel frames. For the native
`num_frames=93`, there are 24 latent frames and the largest valid index is 23.

### Resolution keys

Bare keys are aliases for the 16:9 shape in the same tier. Shapes are width ×
height from the model's resolution table, not mathematical resizing of the tier
number.

| Aspect | `256` tier | `480` tier | `720` tier |
| --- | --- | --- | --- |
| Bare / `_16_9` | `320 × 192` | `832 × 480` | `1280 × 720` |
| `_1_1` | `256 × 256` | `640 × 640` | `960 × 960` |
| `_9_16` | `192 × 320` | `480 × 832` | `720 × 1280` |
| `_4_3` | `320 × 256` | `736 × 544` | `1104 × 832` |
| `_3_4` | `256 × 320` | `544 × 736` | `832 × 1104` |

Examples `480` and `480_16_9` resolve to the same shape. The other explicit
suffixes select distinct shapes.

### FPS and denoising steps

- `fps` accepts finite values from 1 through 60; 10–30 is recommended.
- T2I retains `fps` for the shared request model, but JPEG output has no frame
  rate.
- For video, approximate duration is `num_frames / fps` seconds.
- More `num_inference_steps` usually costs more latency. Start with 50 for T2I
  and 35 for standard video generation unless a validated recipe calls for another
  value.

## Media representations

Generator image and video inputs recognize:

- raw base64;
- a MIME-aware data URL such as `data:image/jpeg;base64,...` or
  `data:video/mp4;base64,...`; and
- an HTTP(S) URL when `NIM_ALLOW_URL_INPUT` is enabled.

Prefer data URLs for portable local-file examples. Remote inputs require
container network access and introduce download, timeout, and content-change
risks. The documented image formats and video codec are summarized in the
[support matrix](support-matrix.md#media-and-codecs); verify other media paths
against the deployed image.

## Reproducibility

Always set a non-negative integer `seed` when comparing prompts, profiles, or
sampling changes. Omission asks the service to generate a seed. Reusing a seed
improves repeatability but does not guarantee bit-identical results across
different NIM releases, model artifacts, precisions, or hardware layouts.

## Optional prompt upsampling

Operators can enable prompt upsampling for T2I, T2V, and I2V. The NIM sends the
input prompt—and the conditioning image for I2V—to an operator-supplied
OpenAI-compatible Chat Completions endpoint. T2I pins resolution and aspect
back to the original request and removes video-only duration and FPS fields.
T2V and I2V pin resolution, aspect, duration, and FPS.

Prompt upsampling does not apply to V2V, action, or transfer. If the external
request times out, fails, or returns an invalid result, generation continues
with the original prompt and the NIM logs a warning.

Configuration, including the separate
`NIM_PROMPT_UPSAMPLING_API_KEY`, is documented in
[configuration.md](configuration.md#prompt-upsampling). Do not reuse
`NGC_API_KEY` as
the external-service credential.

## Output and playback

T2I returns raw JPEG base64 in `b64_image`; the standard comparison example
decodes it to `t2i_robot_draping.jpg`. Video modes return raw base64 for a VP9
video track in an MP4 container under `b64_video`; the standard examples decode
it to `t2v_robot_kitchen.mp4`, `i2v_car_driving.mp4`, or `v2v.mp4` under the
examples `outputs/` directory. Specialist examples add `_4step` to the
corresponding scenario name. The inactive media field is omitted.

VP9-in-MP4 is not supported by every browser or stock player. The declared
FFmpeg client package includes `ffplay` for direct playback:

```bash
ffplay examples/outputs/t2v_robot_kitchen.mp4
```

For a broadly compatible H.264 copy:

```bash
ffmpeg -i examples/outputs/t2v_robot_kitchen.mp4 -c:v libx264 -crf 18 \
  -pix_fmt yuv420p examples/outputs/t2v_robot_kitchen-h264.mp4
```

## Common failures

| Symptom | Likely cause | Action |
| --- | --- | --- |
| HTTP 422, extra field | The request includes a field that is not part of the selected task | Remove unsupported fields using the [API reference](api-reference.md); unknown fields are rejected |
| HTTP 422, frame count | `num_frames` is not an integer, is fewer than 25 for video, or exceeds its tier ceiling | Pick an integer within the resolution-tier range |
| HTTP 422, missing or invalid mode | `model_mode` is absent or conflicts with fields in the request | Set the explicit mode and include only its fields |
| HTTP 422, reference mismatch | `input_reference` is missing, forbidden, or does not match `model_mode` | Omit it for text modes; provide the required image/video for conditioned modes |
| URL media fails | URL inputs are disabled, unreachable from the container, or rejected by the decoder | Use a data URL and verify `NIM_ALLOW_URL_INPUT` |
| Request times out in the client | Generation exceeded the client timeout, not necessarily the server timeout | Use the examples' 30-minute timeout and inspect NIM logs |
| MP4 does not play | Player lacks VP9-in-MP4 support | Use `ffplay` or re-encode to H.264 with the declared FFmpeg tooling |

For service-level diagnosis, see [operations.md](operations.md#troubleshooting).
