<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Generate video with the Cosmos3 Certified NIM

Use this page for text-to-video (T2V), image-to-video (I2V), and
video-to-video (V2V) requests. These workflows require a running **Generator**
profile and use synchronous JSON `POST /v1/infer`.

For launch instructions, see [deployment.md](deployment.md). The
[API reference](api-reference.md#common-generator-request-fields) defines the
shared request envelope; this page owns ordinary generation rules.

## Prerequisites

Verify that the Generator is ready:

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -f "$NIM_URL/v1/health/ready"
```

Run the examples from the repository root. They use `requests`, reuse media
already tracked by this cookbook, and decode the response under
`cookbooks/cosmos3/nim/examples/outputs/`:

```bash
python -m pip install requests
```

## Text-to-video

A T2V request has a non-empty `prompt` and no `image` or `video`:

```bash
curl -fsS -X POST "$NIM_URL/v1/infer" \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A storm trooper vacuuming the beach.",
    "resolution": "720_16_9",
    "num_output_frames": 189,
    "fps": 24.0,
    "seed": 0
  }' \
  -o /tmp/cosmos3-response.json

python - <<'PY'
import base64
import json
from pathlib import Path

response = json.loads(Path("/tmp/cosmos3-response.json").read_text())
Path("t2v.mp4").write_bytes(base64.b64decode(response["b64_video"]))
PY
```

Or run the complete editable example:

```bash
python cookbooks/cosmos3/nim/examples/t2v.py
```

The default negative prompt is supplied by the service when omitted. Pass
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
image_path = Path("input.jpg")
mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
image = f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode()}"

request = {
    "prompt": (
        "A photorealistic red sports car drives through a modern city at "
        "golden hour, with cinematic lighting and smooth camera motion."
    ),
    "image": image,
    "resolution": "720",
    "num_output_frames": 189,
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
Path("i2v.mp4").write_bytes(base64.b64decode(result["b64_video"]))
```

The cookbook version uses an existing image asset:

```bash
python cookbooks/cosmos3/nim/examples/i2v.py
```

`image` and `video` are mutually exclusive. The current encoded-image ceiling
is 20,000,000 characters. Exact released decoder formats remain
**TBD (release-dependent)**; JPEG, PNG, and WebP are the source-tested baseline.

## Video-to-video

V2V uses top-level `video` without `transfer` or `action_params`:

```json
{
  "prompt": "Keep the camera motion and change the environment to a snowy valley.",
  "video": "data:video/mp4;base64,<BASE64_VIDEO>",
  "condition_frame_indexes_vision": [0, 1],
  "condition_video_keep": "first",
  "resolution": "720",
  "num_output_frames": 93,
  "fps": 24.0,
  "seed": 0
}
```

Run the complete local-media example:

```bash
python cookbooks/cosmos3/nim/examples/v2v.py
```

`condition_frame_indexes_vision` indexes latent frames, not pixel frames. The
service sorts and deduplicates it. Its largest value must fit the requested
output latent length. `condition_video_keep` selects frames from the beginning
or end of the input and defaults to `first`.

The current decoded-video ceiling is 75 MB. Data URLs expand binary media by
roughly one third, so large videos can make the JSON request substantially
larger. Exact released container/codec support and remote-fetch behavior remain
validation-gated.

## Choose resolution, frames, and FPS

### Frame cadence and limits

The video VAE has temporal compression factor 4 and a causal first frame, so
pixel-frame counts follow:

```text
num_output_frames = 1 + 4k
```

Ordinary generation accepts at least 25 frames. Current source ceilings are:

| Resolution tier | Maximum frames |
| --- | ---: |
| `256` | 397 |
| `480` | 297 |
| `720` | 197 |

The largest V2V `condition_frame_indexes_vision` value must fit the output
latent-frame range. For `num_output_frames=93`, there are 24 latent frames and
the largest valid conditioning index is 23.

### Resolution keys

Bare keys are aliases for the 16:9 shape in the same tier. Shapes are width ×
height from the model's canonical table, not mathematical resizing of the tier
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
- Approximate duration is `num_output_frames / fps` seconds.
- More `steps` usually costs more latency. Start with the ordinary-generation
  default of 35 unless a validated recipe calls for another value.

## Media representations

Generator image and video inputs recognize:

- raw base64;
- a MIME-aware data URL such as `data:image/jpeg;base64,...` or
  `data:video/mp4;base64,...`; and
- an HTTP(S) URL when `NIM_ALLOW_URL_INPUT` is enabled.

Prefer data URLs for portable local-file examples. Remote inputs require
container network access and introduce download, timeout, and content-change
risks. Exact released image formats, video containers, codecs, and remote-fetch
limits belong to the [support matrix](support-matrix.md).

## Reproducibility

Always set a non-negative integer `seed` when comparing prompts, profiles, or
sampling changes. Omission asks the service to generate a seed. Reusing a seed
improves repeatability but does not guarantee bit-identical results across
different NIM releases, model artifacts, precisions, or hardware layouts.

## Optional prompt upsampling

Operators can enable prompt upsampling for T2V and I2V. The NIM sends the input
prompt—and the conditioning image for I2V—to an operator-supplied
OpenAI-compatible Chat Completions endpoint, then pins resolution, aspect,
duration, and FPS back to the original request.

Prompt upsampling does not apply to V2V, action, or transfer. If the external
request times out, fails, or returns an invalid result, generation continues
with the original prompt and the NIM logs a warning.

Configuration, including the separate
`NIM_PROMPT_UPSAMPLING_API_KEY`, is documented in
[configuration.md](configuration.md#prompt-upsampling). Do not reuse
`NGC_API_KEY` as
the external-service credential.

## Output and playback

The response contains raw base64 for a VP9 video track in an MP4 container.
The public examples decode it to `t2v.mp4`, `i2v.mp4`, or `v2v.mp4` under the
examples `outputs/` directory. VP9-in-MP4 is not supported by every browser or
stock player; `mpv` and `ffplay` are reliable choices.

For a broadly compatible H.264 copy:

```bash
ffmpeg -i t2v.mp4 -c:v libx264 -crf 18 -pix_fmt yuv420p t2v-h264.mp4
```

## Common failures

| Symptom | Likely cause | Action |
| --- | --- | --- |
| HTTP 422, extra field | A vLLM-Omni field such as `input_reference`, `extra_params`, or request-level `guardrails` was copied into `/v1/infer` | Translate the request using the [API reference](api-reference.md); unknown fields are rejected |
| HTTP 422, frame cadence | `num_output_frames` is not `1 + 4k` or exceeds its tier ceiling | Pick the nearest valid count and recheck the tier |
| HTTP 422, image and video conflict | Both top-level media fields are present | Send only the field required by the intended mode |
| URL media fails | URL inputs are disabled, unreachable from the container, or rejected by the decoder | Use a data URL and verify `NIM_ALLOW_URL_INPUT` |
| Request times out in the client | Generation exceeded the client timeout, not necessarily the server timeout | Use the examples' 30-minute timeout and inspect NIM logs |
| MP4 does not play | Player lacks VP9-in-MP4 support | Use `mpv`/`ffplay` or re-encode to H.264 |

For service-level diagnosis, see [operations.md](operations.md#troubleshooting).
