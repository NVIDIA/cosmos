<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Reason over images and video with the Cosmos3 Certified NIM

Use this page for Cosmos3 Reasoner requests through OpenAI-compatible Chat
Completions and Responses APIs. These workflows require a running **Reasoner**
model; `/v1/infer` is a Generator endpoint and is not used here.

See [Deployment](deployment.md) to select and launch a Reasoner model. This
page covers Reasoner routes, media, sampling, and responses.

## Prepare the client and verify readiness

Install the [client tooling](prerequisites.md#client-tooling). Then, from the
repository root, enter the cookbook directory. The runnable examples use the
pinned OpenAI Python client from the `uv` project in that directory:

```bash
cd cookbooks/cosmos3/nim
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -f "$NIM_URL/v1/health/ready"
curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
curl -fsS "$NIM_URL/v1/models" | python3 -m json.tool
```

Before model discovery, confirm that metadata reports `model_type` as
`reasoner` and `inference_endpoint` as `/v1/chat/completions`. A successful
health check and `/v1/models` response do not by themselves prove that the URL
reaches the Reasoner runtime.

The NIM does not require a request API key on localhost, but the OpenAI client
requires a non-empty value. Use a clearly non-secret placeholder such as
`not-used`.

## Discover the served model

Do not assume the served model ID in reusable code:

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

Use metadata for runtime discovery and `/v1/models` for served-model discovery;
do not hard-code a model ID from another image or deployment.

## Run the task catalog

The endpoint-independent catalog in
[`examples/reasoner_cases.yaml`](examples/reasoner_cases.yaml) defines the
media, prompt, sampling, output contract, native-thinking setting, and
qualitative review criteria for every Reasoner example. Its user-prompt strings
exactly match the runtime strings in the nearby vLLM notebook. The Python runner
keeps the NIM-specific API adaptation in one place: it sends local media as data
URLs, uses `media_io_kwargs` for video sampling, discovers the served model,
consumes the final answer from `message.content`, and records whether
prompt-requested JSON can be extracted with the standard JSON decoder.

List or inspect cases without a running endpoint:

```bash
uv run python examples/reasoner.py --list-cases
uv run python examples/reasoner.py --describe trajectory_2d
uv run python examples/reasoner.py --describe trajectory_2d --format json
```

The JSON description is suitable for tooling and AI assistants; the default
YAML description is intended for direct reading. The catalog covers the same
image and video task families as the general Cosmos3 Reasoner examples:

| Category | Cases |
| --- | --- |
| Captioning | `image_caption`, `video_caption` |
| Temporal localization | `temporal_localization`, `event_timeline`, `timestamp_query`, `interval_question` |
| Embodied reasoning | `robotics_next_action`, `drive_scene_next_action`, `robot_planning`, `assisted_task_next_action` |
| Common sense | `common_sense_reasoning` |
| Spatial reasoning | `grounding_2d`, `describe_anything` |
| Action CoT | `trajectory_2d`, `flower_trajectory_2d`, `driving_scene_action_cot` |
| Physical and situation reasoning | `physical_plausibility`, `situation_understanding` |

Run one case:

```bash
uv run python examples/reasoner.py --case image_caption
```

The `--case` option belongs to this cookbook runner, not the NIM API. Every
video case requests 4 FPS sampling. All catalog cases consume the response
from `message.content`. Six vLLM prompt strings contain literal `<think>`
formatting instructions. These tags are ordinary user-prompt text; text cases
can therefore return visible tags in `message.content`, while the runner
extracts structured final answers after a closing `</think>` tag. The service
does not require a separate reasoning field.

Running `--case all` sends all 18 requests sequentially and can take substantial
time and resources. Use it for deliberate catalog validation, not as a first
request:

```bash
uv run python examples/reasoner.py --case all
```

The runner checks `/v1/metadata` before model discovery and fails with an
actionable message if `NIM_URL` reaches a Generator runtime.

### Super BF16 example baseline

The retained catalog validation baseline uses Super BF16 with target-only
decoding and video-token pruning disabled:

```text
NIM_USE_DFLASH=0
NIM_VIDEO_PRUNING_RATE=0
```

The one-GPU Super BF16 configuration requires the hardware floor in the
[support matrix](support-matrix.md#reasoner-configurations).
After readiness, verify the selected model and profile through `/v1/metadata`
and `/v1/manifest` before running the examples.

Every catalog request explicitly sends `temperature=0.7`, `top_p=0.8`,
`top_k=20`, `presence_penalty=0`, and `repetition_penalty=1` unless a case
records an override. These values match the effective Super checkpoint and raw
vLLM example defaults instead of relying on server-side default injection. The
catalog preserves vLLM user-prompt text byte for byte while adapting media
transport to the NIM API.

Cases that set `seed=0` in the vLLM notebook retain that seed while using the
same explicit effective sampling values as the other non-reasoning cases.

Task-level quality remains case-specific and must be reviewed against the
qualitative criteria recorded with each case. The runner warns rather than
fails when the endpoint serves another Reasoner variant so the catalog remains
usable for comparison; do not present a Nano result as a Super-validated
example.

### Artifacts and validation boundaries

For each run, the script prints the final answer and writes an ignored
`examples/outputs/reasoner_<case>/` directory containing:

- `request.json`: the resolved case, selected model/profile, review criteria,
  and request with the embedded media payload omitted;
- `response.json`: the raw OpenAI client response;
- `output.txt`: final `message.content`;
- `output.json`: parsed output, written when JSON extraction succeeds for a
  structured case;
- `validation.json`: separate API, JSON-extraction, annotation, and
  qualitative-review status;
- `annotated.png`: best-effort normalized boxes or trajectories overlaid on
  image cases when the parsed shape supports annotation; and
- `report.md`: a human-readable prompt, answer, validation summary, and review
  checklist.

Format validation records whether the response contains a complete JSON value.
It does not check JSON shape, field formats, timestamp order, coordinate bounds,
or task semantics. Review structured output against the case's qualitative
criteria; those criteria remain `not_performed` in `validation.json` until a
human or application-specific review performs them.

The Responses example uses the image-caption case so API transport differences
do not multiply the task matrix.

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

Use the OpenAI client for normal applications. For direct HTTP integration,
inspect the active `/openapi.json` and send the same request object to
`POST /v1/chat/completions`; keep large data URLs in a request file rather than
shell arguments.

## Video reasoning with Chat Completions

Use `video_url` content and pass NIM request extensions through `extra_body`:

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

Data URLs are the portable baseline. Verify public HTTP(S) media fetching and
formats beyond the included fixtures against the deployed image before relying
on them. A `file://` URL on the client does not identify a file inside the NIM
container.

## Use the Responses API

The Responses create route is enabled unless the deployment sets
`NIM_DISABLE_RESPONSES_ROUTE=true`. For image input, put `input_image` before
`input_text`:

```bash
uv run python examples/reasoner_responses.py
```

The request uses `store=false`. Responses create requests require a non-empty
model, apply the same `temperature`, `top_p`, and `top_k` defaults as Chat
Completions, and map a `developer` input turn to a system instruction. An ordinary answer is
returned as a message item and exposed by the OpenAI client through
`response.output_text` rather than appearing only as a reasoning item.

Persisted retrieval, cancellation, background responses, and
`previous_response_id` require response storage, which is disabled by default.
Use Chat Completions for the documented video-request workflow.

## Final answers, instructions, and tool calls

The task catalog consumes responses from `message.content`. To maintain prompt
parity, it preserves literal `<think>` formatting instructions in the six vLLM
prompts that contain them. Those visible tags are response text and should not
be parsed as a stable separate explanation. Responses requests use the same
chat-template and sampling defaults while retaining their Responses-specific
token-limit and structured-output field names.

Both API styles map a `developer` turn to a system instruction. Chat
Completions also:

- enables standard OpenAI tool definitions and automatic tool choice with the
  Hermes tool-call format; and
- requires `top_logprobs` to be an integer or null. When `logprobs=true` and
  `top_logprobs` is omitted, the service requests one top log probability.

Check the running NIM's `/openapi.json` and the client response model before
depending on reasoning or tool-call fields.

## Reasoner DFlash

Nano and Super Reasoner use their bundled DFlash speculative-decoding drafts by
default. The request routes and payloads do not change. The task catalog's
Super BF16 baseline explicitly sets `NIM_USE_DFLASH=0` and runs the selected
target model without DFlash.
Startup rejects DFlash for Generator or when the required variant-specific
draft artifact is unavailable. An independent local draft path, a
hardware-derived BF16 KV-cache default, and advanced JSON configuration are
also supported.

Compare DFlash and target-only operation for memory, latency, throughput,
correctness, and output quality on representative requests before production
use. See [Reasoner configuration](configuration.md#speculative-decoding) and
[Reasoner checkpoint](bring-your-own-checkpoint.md#reasoner-checkpoint).

## Advanced sampling and request extensions

The service supplies these values when omitted:

| Field | Current default | Current validation |
| --- | ---: | --- |
| `temperature` | 0.7 | `[0, 2]` |
| `top_p` | 0.8 | `(0, 1]` |
| `top_k` | 20 | `-1` or integer `>= 1` |

`top_k`, `media_io_kwargs`, `structured_outputs`, guided-output fields, and
`nvext` are request extensions. With the OpenAI client, put them explicitly in
`extra_body`, as the video example does.

When the operator leaves the image and video limits unset, the NIM does not
override the runtime's modality limits. Use request-level `media_io_kwargs` for
workload-specific video sampling; the example requests 4 FPS. Video-token
pruning is disabled by default. Set a rate greater than `0` to enable the
selected pruning method, which defaults to `vidcom2`; the catalog baseline sets
`0` explicitly. Operator-wide media limits, preprocessing, and pruning are
documented under
[Reasoner configuration](configuration.md#reasoner-configuration). Verify
additional request fields against the active `/openapi.json`.

### Text-only requests

This guide documents image- and video-conditioned Reasoner requests. Use image
or video input with the committed examples, and validate any text-only workflow
against the deployed service before depending on it.

## Structured output

Prefer the standard OpenAI `response_format` JSON-schema shape in portable
client code:

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

The service normalizes the standard Chat Completions `response_format` shape
and enables guided-decoding enforcement for Reasoner output. Responses requests
use their native
`text.format` shape instead:

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

The runnable task catalog follows the vLLM prompt-constrained path by default:
it extracts the first complete JSON value from `message.content` with the
standard JSON decoder and records whether that succeeded. It does not check the
shape or the field formats of the result; review the output against the case's
qualitative criteria. Add `--guided-output` to opt into the NIM-specific JSON
Schema path for structured cases. Do not recover structured results with a
regular expression over prose or Markdown fences.

Validate the running NIM's schema in `/openapi.json`, especially when upgrading
the OpenAI client.

## Prompting patterns

The Reasoner supports several task families through the same Chat Completions
endpoint; task names do not select separate server routes:

| Task | Prompt intent |
| --- | --- |
| Captioning and VQA | Describe entities, actions, environment, or answer a focused question |
| Temporal localization | Return timestamps or intervals for a named event |
| Physical plausibility | Judge whether observed dynamics are physically plausible and state observable evidence |
| Planning and next action | Propose the next safe action from the current scene and task |
| Situation understanding | Summarize agents, interactions, risks, and likely near-term evolution |
| 2D grounding | Return normalized coordinates for named objects or regions |
| Action trajectories | Return ordered normalized points or poses for a requested path |

Use `response_format` when an application requires guided decoding rather than
the catalog's prompt-constrained parity path. For 2D grounding and trajectory
points, the existing cookbook
convention independently normalizes each axis to `[0,1000]`, with the origin at
the upper-left, X increasing rightward, and Y increasing downward. Convert a
validated point to pixels with:

```python
pixel_x = normalized_x / 1000 * image_width
pixel_y = normalized_y / 1000 * image_height
```

A Reasoner next-action answer or 2D trajectory is text/JSON describing a
semantic proposal in visual coordinates. It is not a domain-specific Generator
Action tensor and must not be sent directly to a robot. See
[Generator Action representations](action.md#domains-and-representations).

The catalog preserves the exact user prompts from the
[vLLM Reasoner notebook](../reasoner/run_with_vllm.ipynb) while adapting media
transport and structured output to the NIM API. Treat recorded answers in the
[Reasoner Prompt Guide](../reasoner/reasoner_prompt_guide.md) as examples from
another run, not as golden NIM output. Literal `<think>` blocks requested by
those prompts are visible response formatting, not a guarantee that the service
exposes a stable hidden reasoning trace; downstream logic must not depend on
them.

## Errors

| Status/symptom | Meaning | Action |
| --- | --- | --- |
| HTTP 400 | Sampling or request-shape validation commonly failed | Check model, sampling ranges, extension placement, and strict `top_logprobs` types |
| HTTP 422 | Media validation or preprocessing commonly failed | Check data URL, media ordering, prompt media limits, and selected-image format support |
| Chat Completions route 404 | `NIM_URL` reaches Generator or the route is absent from the selected image | Inspect `/v1/metadata`, start Reasoner, and verify the live OpenAPI document |
| Empty/no choices | Backend did not return a normal Chat Completion | Preserve response/log details and check the selected Reasoner profile |
| Responses route 404 | The deployment disabled Responses, the selected image does not expose it, or `NIM_URL` reaches Generator | Verify metadata first; then use Chat Completions or inspect `NIM_DISABLE_RESPONSES_ROUTE` and live OpenAPI |
| Context or KV-cache failure | Request/media exceeded runtime limits | Reduce media sampling, token budget, concurrency, or adjust operator limits carefully |

See [operations.md](operations.md#troubleshooting) for deployment-level
diagnostics. To serve a local or Hugging Face Reasoner checkpoint, see
[Bring your own checkpoint](bring-your-own-checkpoint.md#reasoner-checkpoint).
