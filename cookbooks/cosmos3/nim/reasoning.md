<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Reason over images and video with the Cosmos3 Certified NIM

Use this page for Cosmos3 Reasoner requests through OpenAI-compatible Chat
Completions, streaming, and Responses APIs. These workflows require a running
**Reasoner** model; `/v1/infer` is a Generator endpoint and is not used here.

See [Deployment](deployment.md) to select and launch a Reasoner model. This
page covers Reasoner routes, media, sampling, and responses.

## Install the client and verify readiness

The runnable examples use the OpenAI Python client:

```bash
python -m pip install openai
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -f "$NIM_URL/v1/health/ready"
curl -sS "$NIM_URL/v1/models" | python -m json.tool
```

The NIM does not require a request API key on localhost, but the OpenAI client
requires a non-empty value. Use a clearly non-secret placeholder such as
`not-used`.

## Discover the served model

Do not assume the selected model size in reusable code:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")
models = client.models.list()
model = models.data[0].id
print(model)
```

Runtime discovery is the preferred contract; do not hard-code a model ID from
another image or deployment.

## Image reasoning with Chat Completions

Place the media item before the text instruction:

```python
import base64
from pathlib import Path
from openai import OpenAI

nim_url = "http://localhost:8000"
image_path = Path("cookbooks/cosmos3/reasoner/assets/robot_153.jpg")
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
python cookbooks/cosmos3/nim/examples/reasoner.py --case image
```

Use the OpenAI client for normal applications. For direct HTTP integration,
inspect the active `/openapi.json` and send the same request object to
`POST /v1/chat/completions`; keep large data URLs in a request file rather than
shell arguments.

## Video reasoning with Chat Completions

Use `video_url` content and pass NIM/vLLM extensions through `extra_body`:

```python
import base64
from pathlib import Path
from openai import OpenAI

nim_url = "http://localhost:8000"
video_path = Path("cookbooks/cosmos3/reasoner/assets/video_caption.mp4")
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
python cookbooks/cosmos3/nim/examples/reasoner.py --case video
```

Data URLs are the portable baseline. Public HTTP(S) media URLs may work when
the container has network access, but exact released fetch, timeout, format,
container, and codec behavior remains **TBD (release-dependent)**. Local
`file://` URLs from standalone vLLM examples are not a portable NIM contract.

## Stream Chat Completions

Set `stream=True`, print each non-empty delta, and retain the assembled output
if downstream code needs the complete result:

```bash
python cookbooks/cosmos3/nim/examples/reasoner_stream.py
```

The example reads `choices[0].delta.content`; chunks without choices or content
are skipped.

## Use the Responses API

The current NIM provides a Responses create route unless the operator sets
`NIM_DISABLE_RESPONSES_ROUTE=true`. For image input, put `input_image` before
`input_text`:

```bash
python cookbooks/cosmos3/nim/examples/reasoner_responses.py
```

The request uses `store=false`. Persisted retrieval, cancellation, background
responses, and `previous_response_id` require response storage. Storage is off
by default in the current source; exact feature support in the released image
is **TBD (release-dependent)**. Keep video requests on Chat Completions until
the released Responses video path is validated.

## Reasoning, instructions, and tool calls

Chat requests default to `chat_template_kwargs.enable_thinking=false`, so
ordinary untagged output remains in `message.content`. To enable thinking and
request parsed reasoning, pass the controls through `extra_body`:

```python
extra_body = {
    "chat_template_kwargs": {"enable_thinking": True},
    "include_reasoning": True,
    "thinking_token_budget": 512,
}
```

Pass that object as `extra_body=extra_body` in a normal Chat Completions call.
`include_reasoning` must be a JSON boolean. When the released response schema
includes parsed reasoning, read its dedicated message field and keep the final
answer in `message.content`; do not parse `<think>` tags. Reasoning text is not
a stable machine-readable explanation and should not be required by downstream
logic.

The current Chat Completions middleware also:

- maps a `developer` message to a `system` instruction;
- enables standard OpenAI tool definitions and automatic tool choice with the
  Hermes tool-call format; and
- requires `top_logprobs` to be an integer or null. When `logprobs=true` and
  `top_logprobs` is omitted, the service requests one top log probability.

Check the released `/openapi.json` and client response model before depending
on reasoning or tool-call fields.

## Optional Nano Reasoner DFlash

Set `NIM_USE_DFLASH=1` at launch to enable DFlash speculative decoding for a
Nano Reasoner. The request routes and payloads do not change. Startup rejects
the option for Generator and Super Reasoner, or when the required draft
artifact is unavailable.

Treat DFlash as an advanced performance option. Compare latency, throughput,
and output quality on representative requests before production use. See
[Reasoner configuration](configuration.md#speculative-decoding).

## Advanced sampling and request extensions

Current normalization supplies these values when omitted:

| Field | Current default | Current validation |
| --- | ---: | --- |
| `temperature` | 0.7 | `[0, 2]` |
| `top_p` | 0.8 | `(0, 1]` |
| `top_k` | 20 | `-1` or integer `>= 1` |

`top_k`, `media_io_kwargs`, `structured_outputs`, guided-output fields, and
`nvext` are request extensions. With the OpenAI client, put them explicitly in
`extra_body`, as the video example does.

The default service limit is five images and one video per prompt. Use
request-level `media_io_kwargs` for workload-specific video sampling; the
example requests 4 FPS. Operator-wide media limits, preprocessing, and optional
video-token pruning are documented under
[Reasoner configuration](configuration.md#reasoner-configuration). Verify
additional request fields against the active `/openapi.json`.

### Text-only requests

The OpenAI-compatible message schema can represent a text-only request, but the
reviewed release evidence does not include a text-only smoke test. Treat
text-only Reasoner support as **TBD (release-dependent)** and verify it against
the target image before relying on it in production.

## Structured output

The Reasoner middleware normalizes OpenAI `response_format`, vLLM
`structured_outputs`, and legacy guided-decoding fields. A Chat Completions
request can ask for a small JSON schema, for example:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "scene_summary",
      "schema": {
        "type": "object",
        "properties": {
          "summary": {"type": "string"},
          "objects": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["summary", "objects"],
        "additionalProperties": false
      }
    }
  }
}
```

Combine this field with a normal request. Validate the exact released schema in
`/openapi.json`, especially when upgrading the OpenAI client or vLLM runtime.

## Prompting patterns

The Reasoner supports several task families without separate endpoints:

| Task | Prompt intent |
| --- | --- |
| Captioning and VQA | Describe entities, actions, environment, or answer a focused question |
| Temporal localization | Return timestamps or intervals for a named event |
| Physical plausibility | Judge whether observed dynamics are physically plausible and state observable evidence |
| Planning and next action | Propose the next safe action from the current scene and task |
| Situation understanding | Summarize agents, interactions, risks, and likely near-term evolution |
| 2D grounding | Return normalized coordinates for named objects or regions |
| Action trajectories | Return ordered normalized points or poses for a requested path |

Put the output contract in the prompt when machine parsing matters. For 2D
grounding, the existing cookbook convention uses coordinates normalized to
`[0,1000]`; convert them to pixels only after validating the model's output.

For a broader gallery, see the existing
[Reasoner Prompt Guide](../reasoner/reasoner_prompt_guide.md). Treat its task
ideas and output schemas as guidance, not as a guarantee that the service
exposes hidden reasoning traces. Ask for concise justifications or structured
final answers; do not depend on `<think>` blocks or hidden chain-of-thought.

## Errors

| Status/symptom | Meaning | Action |
| --- | --- | --- |
| HTTP 400 | Sampling or request-shape validation commonly failed | Check model, sampling ranges, extension placement, and strict `include_reasoning`/`top_logprobs` types |
| HTTP 422 | Media validation or preprocessing commonly failed | Check data URL, media ordering, prompt media limits, and release format support |
| Empty/no choices | Backend did not return a normal Chat Completion | Preserve response/log details and check the selected Reasoner profile |
| Responses route 404 | Operator disabled Responses or the release does not expose it | Use Chat Completions or inspect `NIM_DISABLE_RESPONSES_ROUTE` and live OpenAPI |
| Context or KV-cache failure | Request/media exceeded runtime limits | Reduce media sampling, token budget, concurrency, or adjust operator limits carefully |

See [operations.md](operations.md#troubleshooting) for deployment-level
diagnostics. To serve a local or Hugging Face Reasoner checkpoint, see
[Bring your own checkpoint](bring-your-own-checkpoint.md#reasoner-checkpoint).
