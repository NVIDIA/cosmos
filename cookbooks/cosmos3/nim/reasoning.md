<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Reason over images and video with the Cosmos3 NIM

Use this page for Cosmos3 Reasoner requests through OpenAI-compatible Chat
Completions and Responses APIs. These workflows require a running **Reasoner**
model; `/v1/infer` is a Generator endpoint and is not used here. See
[Deployment](deployment.md) to select and launch a Reasoner model.

## Prepare the client and verify readiness

Install the [client tooling](prerequisites.md#client-tooling). The runnable
examples use the pinned OpenAI Python client from the `uv` project in the
cookbook directory:

```bash
cd cookbooks/cosmos3/nim
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -f "$NIM_URL/v1/health/ready"
curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
curl -fsS "$NIM_URL/v1/models" | python3 -m json.tool
```

Confirm that metadata reports `model_type` as `reasoner` and
`inference_endpoint` as `/v1/chat/completions`; health and `/v1/models` alone do
not prove that the URL reaches the Reasoner. The NIM needs no API key on
localhost, but the OpenAI client requires a non-empty placeholder such as
`not-used`.

## Discover the served model

Do not hard-code the served model ID:

```python
import requests
from openai import OpenAI

nim_url = "http://localhost:8000"
metadata_response = requests.get(f"{nim_url}/v1/metadata", timeout=30)
metadata_response.raise_for_status()
metadata = metadata_response.json()
if (
    metadata.get("model_type") != "reasoner"
    or metadata.get("inference_endpoint") != "/v1/chat/completions"
):
    raise RuntimeError(f"NIM_URL does not target a Reasoner runtime: {metadata}")

client = OpenAI(base_url=f"{nim_url}/v1", api_key="not-used")
models = client.models.list()
model = models.data[0].id
print(model)
```

## Run the task catalog

[`examples/reasoner_cases.yaml`](examples/reasoner_cases.yaml) defines the
media, prompt, sampling, output contract, and review criteria for each case,
and `examples/reasoner.py` sends them to the NIM.

List or inspect cases without a running endpoint (`--format json` emits
machine-readable output; the default is YAML):

```bash
uv run python examples/reasoner.py --list-cases
uv run python examples/reasoner.py --describe trajectory_2d
uv run python examples/reasoner.py --describe trajectory_2d --format json
```

| Category | Cases |
| --- | --- |
| Captioning | `image_caption`, `video_caption` |
| Temporal localization | `temporal_localization`, `event_timeline`, `timestamp_query`, `interval_question` |
| Embodied reasoning | `robotics_next_action`, `drive_scene_next_action`, `robot_planning`, `assisted_task_next_action` |
| Common sense | `common_sense_reasoning` |
| Spatial reasoning | `grounding_2d`, `describe_anything` |
| Action CoT | `trajectory_2d`, `flower_trajectory_2d`, `driving_scene_action_cot` |
| Physical and situation reasoning | `physical_plausibility`, `situation_understanding` |

Run one case (`--case` is a runner option, not a NIM API field):

```bash
uv run python examples/reasoner.py --case image_caption
```

Every video case requests 4 FPS sampling, and all cases read the answer from
`message.content`. `--case all` sends all 18 requests sequentially and takes
substantial time; use it for deliberate catalog validation, not as a first
request:

```bash
uv run python examples/reasoner.py --case all
```

### Super BF16 example baseline

The catalog was validated on Super BF16 (hardware floor in the
[support matrix](support-matrix.md#reasoner-configurations)) with
`NIM_USE_DFLASH=0` and `NIM_VIDEO_PRUNING_RATE=0`. Every case explicitly sends
`temperature=0.7`, `top_p=0.8`, `top_k=20`, `presence_penalty=0`, and
`repetition_penalty=1` (plus `seed=0` where the vLLM notebook sets it) unless
it records an override; the runner only warns on another variant, so do not
present a Nano result as Super-validated.

### Artifacts and validation boundaries

Each run prints the answer and writes `examples/outputs/reasoner_<case>/`.
Validation only checks that JSON parses; review semantics yourself.

## Image reasoning with Chat Completions

Place the media item before the text instruction:

```python
import base64
from pathlib import Path
from openai import OpenAI

nim_url = "http://localhost:8000"
image_path = Path("../reasoner/assets/robot_153.jpg")
image_url = "data:image/jpeg;base64," + base64.b64encode(
    image_path.read_bytes()
).decode()

with OpenAI(base_url=f"{nim_url}/v1", api_key="not-used") as client:
    model = client.models.list().data[0].id
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Caption the image in detail."},
                ],
            }
        ],
        max_tokens=4096,
        seed=0,
    )
print(response.choices[0].message.content)
```

Run the equivalent cookbook example:

```bash
uv run python examples/reasoner.py --case image_caption
```

For direct HTTP, send the same object to `POST /v1/chat/completions` and keep
large data URLs in a request file, not shell arguments.

## Video reasoning with Chat Completions

Use `video_url` content and pass NIM request extensions through `extra_body`.
Input videos can be at most 3840 pixels wide and 2160 pixels tall, so portrait
4K is also rejected; larger videos return HTTP 422, so downscale them first:

```python
import base64
from pathlib import Path
from openai import OpenAI

nim_url = "http://localhost:8000"
video_path = Path("../reasoner/assets/video_caption.mp4")
video_url = "data:video/mp4;base64," + base64.b64encode(
    video_path.read_bytes()
).decode()

with OpenAI(base_url=f"{nim_url}/v1", api_key="not-used") as client:
    model = client.models.list().data[0].id
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": "Describe the video in detail."},
                ],
            }
        ],
        max_tokens=4096,
        extra_body={"media_io_kwargs": {"video": {"fps": 4.0}}},
    )
print(response.choices[0].message.content)
```

Run the complete example:

```bash
uv run python examples/reasoner.py --case video_caption
```

Prefer data URLs; a client-side `file://` URL does not reach the container.

## Use the Responses API

The Responses create route is enabled unless the deployment sets
`NIM_DISABLE_RESPONSES_ROUTE=true`. For image input, put `input_image` before
`input_text`:

```bash
uv run python examples/reasoner_responses.py
```

Requests need a non-empty model and use the Chat Completions sampling defaults.
Read the answer from `response.output_text`. `store=false` is the default;
retrieval, cancellation, background responses, and `previous_response_id` need
response storage, which is disabled by default. Use Chat Completions for video.

## Final answers, instructions, and tool calls

Both APIs map a `developer` turn to a system instruction. Chat Completions
supports OpenAI tool definitions with automatic tool choice in the Hermes
format, and requires `top_logprobs` to be an integer or null (one is requested
when `logprobs=true` and it is omitted). Some prompts ask for literal `<think>`
tags, which appear in `message.content` as ordinary text.

## Reasoner DFlash

Nano and Super Reasoner use their bundled DFlash drafts by default; set
`NIM_USE_DFLASH=0` to run target-only. Requests do not change. See
[Speculative decoding](configuration.md#speculative-decoding) and
[Reasoner checkpoint](bring-your-own-checkpoint.md#reasoner-checkpoint).

## Advanced sampling and request extensions

The service supplies these values when omitted:

| Field | Current default | Current validation |
| --- | ---: | --- |
| `temperature` | 0.7 | `[0, 2]` |
| `top_p` | 0.8 | `(0, 1]` |
| `top_k` | 20 | `-1` or integer `>= 1` |

The service does not inject a default `max_tokens`; send `max_tokens`
explicitly.

`top_k`, `media_io_kwargs`, `structured_outputs`, guided-output fields, and
`nvext` are request extensions; with the OpenAI client, put them in
`extra_body`. Set video sampling per request with `media_io_kwargs`; media
limits and video-token pruning (off by default) are under
[Reasoner configuration](configuration.md#reasoner-configuration).

### Text-only requests

This guide covers image and video input only; validate text-only workflows
yourself.

## Structured output

Prefer the standard OpenAI `response_format` JSON-schema shape; the service
enforces it with guided decoding:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "temporal_events",
      "schema": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "start": {"type": "number"},
            "end": {"type": "number"},
            "caption": {"type": "string"}
          },
          "required": ["start", "end", "caption"],
          "additionalProperties": false
        }
      }
    }
  }
}
```

Responses requests use the native `text.format` shape instead:

```json
{
  "text": {
    "format": {
      "type": "json_schema",
      "name": "temporal_events",
      "schema": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "start": {"type": "number"},
            "end": {"type": "number"},
            "caption": {"type": "string"}
          },
          "required": ["start", "end", "caption"],
          "additionalProperties": false
        }
      },
      "strict": true
    }
  }
}
```

By default the catalog extracts the first JSON value from `message.content`;
add `--guided-output` to use JSON Schema enforcement. Do not recover structured
results with a regular expression over prose or Markdown fences.

## Prompting patterns

All task families use the same Chat Completions endpoint; task names do not
select separate routes:

| Task | Prompt intent |
| --- | --- |
| Captioning and VQA | Describe entities, actions, environment, or answer a focused question |
| Temporal localization | Return timestamps or intervals for a named event |
| Physical plausibility | Judge whether observed dynamics are physically plausible and state observable evidence |
| Planning and next action | Propose the next safe action from the current scene and task |
| Situation understanding | Summarize agents, interactions, risks, and likely near-term evolution |
| 2D grounding | Return normalized coordinates for named objects or regions |
| Action trajectories | Return ordered normalized points or poses for a requested path |

For 2D grounding and trajectory points, each axis is normalized to `[0,1000]`,
with the origin at the upper-left, X increasing rightward, and Y increasing
downward. Convert a validated point to pixels with:

```python
pixel_x = normalized_x / 1000 * image_width
pixel_y = normalized_y / 1000 * image_height
```

A Reasoner next-action answer or 2D trajectory is text/JSON describing a
semantic proposal in visual coordinates. It is not a domain-specific Generator
Action tensor and must not be sent directly to a robot. See
[Generator Action representations](action.md#domains-and-representations).

The catalog reuses the exact user prompts from the
[vLLM Reasoner notebook](../reasoner/run_with_vllm.ipynb). Answers in the
[Reasoner Prompt Guide](../reasoner/reasoner_prompt_guide.md) come from another
run and are not golden NIM output.

## Errors

| Status/symptom | Meaning | Action |
| --- | --- | --- |
| Chat Completions or Responses route 404 | `NIM_URL` reaches Generator, Responses is disabled, or the route is absent | Inspect `/v1/metadata`; start Reasoner or check `NIM_DISABLE_RESPONSES_ROUTE` |
| HTTP 422 | Media validation or preprocessing failed | Check the data URL, media ordering, and media limits |

See [Reasoner troubleshooting](operations.md#reasoner) for other errors, and
check the [live schema](api-reference.md#errors-and-live-schema) before
depending on additional fields. To serve a local or Hugging Face Reasoner
checkpoint, see
[Bring your own checkpoint](bring-your-own-checkpoint.md#reasoner-checkpoint).
