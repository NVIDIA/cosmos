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

## Run representative tasks

The task runner provides representative image and video requests, including
structured-output cases:

| Case | Media | Result |
| --- | --- | --- |
| `image_caption` | `robot_153.jpg` | Detailed text caption |
| `video_caption` | `video_caption.mp4` | Detailed text caption |
| `temporal_localization` | `temporal_localization_1.mp4` | Validated JSON event intervals |
| `robotics_next_action` | `robotics_next_action.mp4` | Concise proposed next action |
| `robot_planning` | `robot_planning.png` | Ordered text plan |
| `grounding_2d` | `grounding_2d.png` | Validated JSON bounding boxes |
| `trajectory_2d` | `action_cot_trajectory.png` | Validated JSON image points |
| `physical_plausibility` | `physical_plausibility.mp4` | Possible/impossible assessment |
| `situation_understanding` | `situation_understanding.mp4` | Current and likely next event |

Run a representative case:

```bash
uv run python examples/reasoner.py --case image_caption
```

The Reasoner scripts check `/v1/metadata` before model discovery and fail with
an actionable message if `NIM_URL` reaches a Generator runtime.

The `--case` option belongs to this cookbook runner, not the NIM API. Substitute
any other exact case name from the table. Every video case requests 4 FPS
sampling. Task quality can differ between Nano and Super; these cases
demonstrate the API and output
contract rather than guaranteeing a particular answer.

For each run, the script prints the final answer and writes an ignored
`examples/outputs/reasoner_<case>/` directory containing:

- `request.json`: case, model, asset, prompt, and request-option metadata without
  the embedded media data URL;
- `response.json`: the raw OpenAI client response;
- `output.txt`: final `message.content`;
- `output.json`: parsed and validated output for structured cases; and
- `reasoning.txt`: parsed reasoning when explicitly requested and returned.

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

Data URLs are the portable baseline. During pre-release evaluation, do not rely
on public HTTP(S) media fetching or formats beyond the included fixtures. A
`file://` URL on the client does not identify a file inside the NIM container.

## Use the Responses API

The Responses create route is enabled unless the deployment sets
`NIM_DISABLE_RESPONSES_ROUTE=true`. For image input, put `input_image` before
`input_text`:

```bash
uv run python examples/reasoner_responses.py
```

The request uses `store=false`. Responses create requests require a non-empty
model, apply the same `temperature`, `top_p`, `top_k`, and
`chat_template_kwargs.enable_thinking=false` defaults as Chat Completions, and
map a `developer` input turn to a system instruction. An ordinary answer is
returned as a message item and exposed by the OpenAI client through
`response.output_text` rather than appearing only as a reasoning item.

Persisted retrieval, cancellation, background responses, and
`previous_response_id` require response storage, which is disabled by default.
Use Chat Completions for video requests in this pre-release version.

## Reasoning, instructions, and tool calls

Chat requests default to `chat_template_kwargs.enable_thinking=false`, so
ordinary untagged output remains in `message.content`. Responses requests use
the same chat-template and sampling defaults while retaining their
Responses-specific token-limit and structured-output field names. To enable
thinking and request parsed reasoning in Chat Completions, pass the controls
through `extra_body`:

```python
extra_body = {
    "chat_template_kwargs": {"enable_thinking": True},
    "include_reasoning": True,
    "thinking_token_budget": 512,
}
```

Pass that object as `extra_body=extra_body` in a normal Chat Completions call,
or run a task with explicit reasoning:

```bash
uv run python examples/reasoner.py \
  --case robotics_next_action \
  --reasoning \
  --thinking-token-budget 512
```

For Chat Completions, `include_reasoning` must be a JSON boolean. The task runner does not add
prompt-authored `<think>` formatting instructions. When the response includes
parsed reasoning, it saves the dedicated `reasoning_content` field separately
and keeps the final
answer in `message.content`; it never parses `<think>` tags. Reasoning text is
not a stable machine-readable explanation and should not be required by
downstream logic.

Both API styles map a `developer` turn to a `system` instruction. Chat
Completions also:

- enables standard OpenAI tool definitions and automatic tool choice with the
  Hermes tool-call format; and
- requires `top_logprobs` to be an integer or null. When `logprobs=true` and
  `top_logprobs` is omitted, the service requests one top log probability.

Check the running NIM's `/openapi.json` and the client response model before
depending on reasoning or tool-call fields.

## Reasoner DFlash

Nano and Super Reasoner use their bundled DFlash speculative-decoding drafts by
default. The request routes and payloads do not change. Set
`NIM_USE_DFLASH=0` at launch to run the selected target model without DFlash.
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
pruning defaults to a `0.6` rate with `vidcom2`; set the operator rate to `0` to
disable it. Operator-wide media limits, preprocessing, and pruning are
documented under
[Reasoner configuration](configuration.md#reasoner-configuration). Verify
additional request fields against the active `/openapi.json`.

### Text-only requests

Text-only Reasoner requests are not part of the pre-release evaluation scope.
Use image or video input with the Reasoner examples.

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
and enables guided-decoding enforcement for Reasoner output, including output
processed by the reasoning parser. Responses requests use their native
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

The runnable task catalog uses JSON schemas for temporal localization, 2D
grounding, and 2D trajectory proposals. It parses `message.content` with the
standard JSON parser and then validates semantic invariants that a schema alone
does not establish, such as ordered timestamps, ordered box corners, non-empty
labels, and coordinates in `[0,1000]`. Do not recover structured results with a
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

Use `response_format` rather than prompt-only formatting when machine parsing
matters. For 2D grounding and trajectory points, the existing cookbook
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

For a broader gallery, see the existing
[Reasoner Prompt Guide](../reasoner/reasoner_prompt_guide.md). Treat its task
ideas and output schemas as guidance, not as a guarantee that the service
exposes hidden reasoning traces. Ask for concise justifications or structured
final answers; do not depend on `<think>` blocks or hidden chain-of-thought.

## Errors

| Status/symptom | Meaning | Action |
| --- | --- | --- |
| HTTP 400 | Sampling or request-shape validation commonly failed | Check model, sampling ranges, extension placement, and strict `include_reasoning`/`top_logprobs` types |
| HTTP 422 | Media validation or preprocessing commonly failed | Check data URL, media ordering, prompt media limits, and selected-image format support |
| Chat Completions route 404 | `NIM_URL` reaches Generator or the route is absent from the selected image | Inspect `/v1/metadata`, start Reasoner, and verify the live OpenAPI document |
| Empty/no choices | Backend did not return a normal Chat Completion | Preserve response/log details and check the selected Reasoner profile |
| Responses route 404 | The deployment disabled Responses, the selected image does not expose it, or `NIM_URL` reaches Generator | Verify metadata first; then use Chat Completions or inspect `NIM_DISABLE_RESPONSES_ROUTE` and live OpenAPI |
| Context or KV-cache failure | Request/media exceeded runtime limits | Reduce media sampling, token budget, concurrency, or adjust operator limits carefully |

See [operations.md](operations.md#troubleshooting) for deployment-level
diagnostics. To serve a local or Hugging Face Reasoner checkpoint, see
[Bring your own checkpoint](bring-your-own-checkpoint.md#reasoner-checkpoint).
