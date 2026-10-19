<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Generate images and videos with the Cosmos3 NIM

Send text-to-image (T2I), text-to-video (T2V), image-to-video (I2V), and
video-to-video (V2V) requests to a running **Generator** with synchronous JSON
`POST /v1/infer`. Launch it with [deployment.md](deployment.md); the shared
request envelope is in the
[API reference](api-reference.md#common-generator-request-fields).

## Prerequisites

Verify that the Generator is ready:

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -f "$NIM_URL/v1/health/ready"
```

Requests are synchronous and can stay quiet until done; the examples allow 30
minutes, so do not retry a quiet connection (see
[long-running requests](operations.md#long-running-requests)).

Install the [client tooling](prerequisites.md#client-tooling), then run the
examples from the cookbook directory. They use the included prompts and media
and write output under `examples/outputs/`:

```bash
cd cookbooks/cosmos3/nim
uv run python examples/t2i.py
```

## Choose a modality

| Modality | Example scenario | Conditioning input | Response field | Run |
| --- | --- | --- | --- | --- |
| T2I | Robot draping satin over a mannequin | None | `b64_image` | `uv run python examples/t2i.py` |
| T2V | Robot cleaning a kitchen | None | `b64_video` | `uv run python examples/t2v.py` |
| I2V | Car traveling along a coastal road | `car_driving.jpg` | `b64_video` | `uv run python examples/i2v.py` |
| V2V | Continue or transform a car-driving video | `car_driving_plain.mp4` | `b64_video` | `uv run python examples/v2v.py` |

Standard scripts run on a general-purpose `nano` or `super` variant whose image
includes the task; T2I and I2V also run on their full-step specialists.
Four-step specialists need their own scripts (see
[Specialist T2I and I2V variants](#specialist-t2i-and-i2v-variants)).

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

Full example: `uv run python examples/t2i.py`.

Set `model_mode=text2image`, omit `input_reference`, and use `num_frames=1`.
Image-to-image is not supported; use `image2video` to condition on an image.

When their fields are omitted, T2I uses:

| Field | T2I default |
| --- | --- |
| `resolution` | `720_1_1` (960 × 960) |
| `negative_prompt` | Empty string |
| `num_inference_steps` | `50` |
| `guidance_scale` | `4.0` |
| `flow_shift` | `3.0` |
| `fps` | `24.0` (ignored for JPEG) |

T2V, I2V, and V2V use these defaults when fields are omitted:

| Field | Video default |
| --- | --- |
| `resolution` | `720` (1280 × 720) |
| `num_frames` | `189` |
| `num_inference_steps` | `35` |
| `guidance_scale` | `6.0` |
| `flow_shift` | `10.0` |
| `fps` | `24.0` |

When `negative_prompt` is omitted, the default depends on the request:

| Request | Default `negative_prompt` |
| --- | --- |
| T2I | Empty string |
| T2V, I2V, V2V, Transfer, and Action | Built-in structured negative prompt |

Send `"negative_prompt": ""` to disable the built-in prompt. The `nano-droid`
variant rejects `negative_prompt`.

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

Full example: `uv run python examples/t2v.py`.

## Image-to-video

I2V accepts the conditioning image as raw base64, a MIME-aware data URL, or an
allowed public HTTP(S) URL. A data URL preserves the media type and avoids
shell-specific base64 flags:

```python
import base64, json, urllib.request
from pathlib import Path

image = Path("../generator/audiovisual/assets/images/image2video/car_driving.jpg")
data_url = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
body = {
    "model_mode": "image2video",
    "prompt": "A car travels along a coastal mountain road with natural motion.",
    "input_reference": data_url,
    "seed": 0,
}
req = urllib.request.Request("http://localhost:8000/v1/infer", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1800) as resp:
    video = base64.b64decode(json.load(resp)["b64_video"])
Path("i2v_car_driving.mp4").write_bytes(video)
```

Full example: `uv run python examples/i2v.py`.

`input_reference` is required (an image, up to 20,000,000 encoded characters).
Use JPEG, PNG, or WebP; verify other formats on the deployed image.

## Choose a video-conditioned workflow

| Intent | `model_mode` | Distinguishing input | Guide |
| --- | --- | --- | --- |
| Animate a still image | `image2video` | Image `input_reference` | [Image-to-video](#image-to-video) |
| Continue or transform a source video | `video2video` | Video `input_reference` plus optional V2V conditioning fields | [Video-to-video](#video-to-video) |
| Follow an edge, blur, depth, segmentation, or WSM control | `video2video` | Non-empty `transfer`; top-level video only for a derived control | [Transfer](transfer.md) |
| Estimate actions from an observed video | `inverse_dynamics` | Video `input_reference` plus `action_params` | [Action](action.md#inverse-dynamics) |

Don't mix V2V fields, `transfer`, and `action_params` (HTTP 422).

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

Full example: `uv run python examples/v2v.py`.

`condition_frame_indexes_vision` indexes latent frames, not pixel frames,
defaults to `[0, 1]`, and is sorted and deduplicated. The source video needs at
least `max(index) * 4 + 1` frames, and the largest index must be below the
output latent count (for `num_frames=93`: 24 latent frames, so at most 23). An
empty list or a negative index returns HTTP 422. `condition_video_keep` takes
frames from the `first` (default) or `last` part of the input.

Decoded video is capped at 75 MB; data URLs add about one third to the JSON
size. See [media inputs](api-reference.md#media-inputs) for codecs.

## Specialist T2I and I2V variants

Super also ships task-specific variants. Each accepts only its task; select it
before launch (`NIM_MODEL_VARIANT=super` is the general-purpose contract).

| Variant | Accepted request | Sampling behavior | Script |
| --- | --- | --- | --- |
| `super-t2i` | T2I only | Full-step; omitted `flow_shift` becomes `3.0` | `examples/t2i.py` |
| `super-t2i-4step` | T2I only | Fixed four-step: steps `4`, guidance `1.0`, flow shift `1.0` | `examples/t2i_4step.py` |
| `super-i2v` | I2V only | Full-step; omitted `flow_shift` becomes `10.0` | `examples/i2v.py` |
| `super-i2v-4step` | I2V only | Fixed four-step: steps `4`, guidance `1.0`, flow shift `1.0` | `examples/i2v_4step.py` |

Launch with the matching `NIM_MODEL_VARIANT`, then run the script with
`uv run python`. Four-step requests must omit `num_inference_steps`,
`guidance_scale`, and `flow_shift` (HTTP 422 otherwise); `seed` and other
fields remain available.

## Choose resolution, frames, and FPS

### Frame counts and limits

T2I uses exactly one frame. Video accepts any integer from 25 through the tier
maximum; the service generates the next native `1 + 4k` count (VAE temporal
compression 4) and trims to exactly `num_frames`.

| Resolution tier | Maximum frames |
| --- | ---: |
| `256` | 397 |
| `480` | 297 |
| `720` | 197 |

### Resolution keys

Bare keys alias the 16:9 shape of their tier (width × height):

| Aspect | `256` tier | `480` tier | `720` tier |
| --- | --- | --- | --- |
| Bare / `_16_9` | `320 × 192` | `832 × 480` | `1280 × 720` |
| `_1_1` | `256 × 256` | `640 × 640` | `960 × 960` |
| `_9_16` | `192 × 320` | `480 × 832` | `720 × 1280` |
| `_4_3` | `320 × 256` | `736 × 544` | `1104 × 832` |
| `_3_4` | `256 × 320` | `544 × 736` | `832 × 1104` |

### FPS and denoising steps

`fps` accepts 1–60 (10–30 recommended); video duration is about
`num_frames / fps` seconds. Fractional `fps` reaches the model and encoder
unchanged; only prompt-upsampling text rounds it. More `num_inference_steps`
costs more latency.

## Media inputs

Data-URL and URL rules, size caps, and accepted codecs are in the
[API reference](api-reference.md#media-inputs).

## Reproducibility

Set a non-negative integer `seed` when comparing runs; omitted, the service
picks one. The same seed is not guaranteed bit-identical across NIM releases,
model artifacts, precisions, or hardware layouts.

## Optional prompt upsampling

Operators can enable prompt upsampling for T2I, T2V, and I2V (not V2V, Action,
or Transfer); it keeps the request's resolution, aspect, duration, and FPS,
and falls back to the original prompt on failure. See
[configuration.md](configuration.md#prompt-upsampling).

## Output and playback

T2I returns raw JPEG base64 in `b64_image`; video modes return raw base64
VP9-in-MP4 in `b64_video`. The inactive media field is omitted. Examples write
`t2i_robot_draping.jpg`, `t2v_robot_kitchen.mp4`, `i2v_car_driving.mp4`, or
`v2v.mp4` under `examples/outputs/` (`_4step` suffix for specialists).

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
| HTTP 422, field or mode | Unknown field, missing `model_mode`, or fields that conflict with the mode | Set the explicit mode and send only its fields ([API reference](api-reference.md)) |
| HTTP 422, frame count | `num_frames` is not an integer, is fewer than 25 for video, or exceeds its tier ceiling | Pick an integer within the resolution-tier range |
| HTTP 422, reference mismatch | `input_reference` is missing, forbidden, or does not match `model_mode` | Omit it for text modes; provide the required image/video for conditioned modes |
| URL media fails | URL inputs are disabled, unreachable from the container, or rejected by the decoder | Use a data URL and verify `NIM_ALLOW_URL_INPUT` |
| Request times out in the client | Generation exceeded the client timeout, not necessarily the server timeout | Use the examples' 30-minute timeout and inspect NIM logs |
| MP4 does not play | Player lacks VP9-in-MP4 support | Use `ffplay` or re-encode to H.264 with the declared FFmpeg tooling |

For service-level diagnosis, see [operations.md](operations.md#troubleshooting).
