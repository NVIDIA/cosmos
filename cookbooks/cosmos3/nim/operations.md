<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Operate and troubleshoot the Cosmos3 Certified NIM

Use this page to interpret health, inspect the selected profile, configure
logging and guardrails, integrate metrics, and diagnose deployment and request
failures. See [deployment.md](deployment.md) for launch configuration and
[configuration.md](configuration.md) for environment variables.

> Endpoint availability and operational limits can vary with the selected
> image and runtime. Inspect the running service before configuring automation.

## Health and startup

| Probe | Healthy meaning | It does not prove |
| --- | --- | --- |
| `/v1/health/live` | HTTP process is alive | Model artifacts are loaded or inference can run |
| `/v1/health/ready` | Selected backend reports ready after startup work | Every capability/media case has passed a smoke test |

Check both:

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -i "$NIM_URL/v1/health/live"
curl -i "$NIM_URL/v1/health/ready"
```

Generator and Reasoner use the same successful health-response format.
Readiness returns:

```json
{
  "object": "health.response",
  "message": "Service is ready",
  "status": "ready"
}
```

Liveness uses the same schema with `"message": "Service is live"` and
`"status": "live"`. Pollers should treat a successful HTTP status as the
primary result. If a poller also parses the body, compare `status` with the
specific probe: `ready` for readiness and `live` for liveness. Failed backend
probes remain non-2xx and do not use the successful response contract.

Cold startup can include NGC download, cache materialization, engine build,
model load, and warmup. Before materialization, profile selection captures the
current free memory of each visible GPU once and commits to the selected
layout. A competing allocation can cause a lower-memory fallback or a startup
failure; free memory that changes afterward does not trigger reselection. A
live-but-not-ready interval is expected. If you use an orchestrator, configure
its startup and readiness budgets for this interval.

## Inspect the running service

Use the active runtime rather than launch-time assumptions:

```bash
for endpoint in metadata models manifest version license; do
  curl -fsS "$NIM_URL/v1/$endpoint" -o "$endpoint.json"
  python3 -m json.tool "$endpoint.json" >/dev/null
done

curl -fsS "$NIM_URL/openapi.json" -o openapi.json
```

| Endpoint | Operational question |
| --- | --- |
| `/v1/metadata` | Which `model_type`, primary `inference_endpoint`, profile/checkpoint, and Generator `model_variant` did the NIM report? |
| `/v1/models` | Which model ID should the Reasoner client send? This endpoint alone does not identify the active runtime. |
| `/v1/manifest` | Which profile/artifact information is present in this image? |
| `/v1/version` | Which release/server API is running? |
| `/v1/license` | Where is the bundled product license information? |
| `/openapi.json` | Which routes and schemas does this active runtime expose? |

Capture these outputs with deployment evidence, but review them before sharing:
paths, repository overrides, or other fields can reveal internal operational
details.

The manifest endpoint returns a JSON object whose `manifest_file` field is the
complete YAML manifest. JSON formatting validates the outer response but does
not decode that YAML or identify the active profile. To match the profile ID
from `/v1/metadata` to its manifest tags without printing the artifact
inventory, initialize the
[pinned client environment](prerequisites.md#initialize-the-example-environment),
then run:

```bash
uv run python examples/inspect_profile.py
```

The helper requests both endpoints, parses the embedded YAML, and prints the
manifest model and release plus the selected profile's ID, tags, and workspace
hash. Treat exact profile IDs and low-level tags as image-specific diagnostic
data; do not copy them to another image or host.

## Logging

Common controls:

| Variable | Default | Use |
| --- | --- | --- |
| `NIM_LOG_LEVEL` | `INFO` | Service log threshold |
| `NIM_LOGGING_JSONL` | false | JSON-line logs for aggregation |
| `NIM_TRITON_LOG_VERBOSE` | 0 | Generator backend verbosity |
| `NIM_DISABLE_LOG_REQUESTS` | true | Reasoner request-body logging control |

Use `DEBUG` or verbose backend logging only during diagnosis; it can increase
volume, expose request content, and affect performance. Never log
`NGC_API_KEY`, `HF_TOKEN`, `NIM_PROMPT_UPSAMPLING_API_KEY`, raw media data
URLs, or full prompts containing sensitive data.

For a Docker container:

```bash
docker logs --since 10m cosmos3-generator
docker logs --since 10m cosmos3-reasoner
```

## Distributed diagnostics

For an active multi-GPU hang or crash, reproduce with a bounded workload and
enable only the diagnostics needed:

```bash
-e TORCH_NCCL_TRACE_BUFFER_SIZE=10000 \
-e TORCH_NCCL_DESYNC_DEBUG=1 \
-e TORCH_DISTRIBUTED_DEBUG=DETAIL \
-e NCCL_DEBUG=INFO
```

`NCCL_DEBUG=INFO` is verbose. `CUDA_LAUNCH_BLOCKING=1` can map asynchronous GPU
faults to a host stack but serializes launches and carries a significant
performance cost. Remove diagnostic settings after the incident.

## Metrics

Verify the selected runtime before configuring a scraper:

```bash
curl -fsS "$NIM_URL/v1/metrics" -o metrics.txt
head metrics.txt
```

Metric names can vary by image. Inspect the output before creating queries,
dashboards, or alerts.

A minimal Prometheus job, if `/v1/metrics` is enabled:

```yaml
scrape_configs:
  - job_name: cosmos3-certified-nim
    metrics_path: /v1/metrics
    static_configs:
      - targets: ["cosmos3-nim:8000"]
```

## Long-running requests

Generator `/v1/infer` requests are synchronous. The client receives the image
or video response only after inference and response serialization complete, so
a connection can remain open without a response body while work is active. A
quiet connection by itself does not prove that the request is hung.

Most Generator cookbook clients set a 30-minute HTTP request timeout; the
compute-intensive Transfer client uses 60 minutes. The Generator backend has a
separately configurable queue-plus-execution timeout with a 30-minute default;
see
[`NIM_TRITON_REQUEST_TIMEOUT`](configuration.md#generator-configuration).
A longer client timeout does not extend the backend ceiling. For Transfer, set
the backend to 60 minutes at container launch when the request needs the full
client window. All of these values are timeout ceilings, not expected latency
or a service-level objective.

Before retrying a quiet request:

1. confirm that the client connection is still open and has not reported an
   HTTP or network error;
2. follow the active container logs and correlate them with the request start;
3. inspect approved GPU telemetry and runtime metrics for activity; and
4. wait for the configured client or backend timeout unless the logs identify a
   failure that requires intervention.

Do not use readiness as request progress: readiness reports whether the backend
can serve, not how far an individual request has advanced. Avoid blind retries,
which can submit duplicate expensive work.

Complete the pinned [client environment
setup](prerequisites.md#initialize-the-example-environment) before running an
example. The elapsed time of `uv run python examples/...` includes local client
startup, media preparation, response decoding, and output-file writes, so do
not interpret the complete command time as NIM inference latency.

## Unified-memory diagnostics

On DGX Spark, Jetson AGX Thor, and other unified-memory systems, use the
[`MemFree`, `MemAvailable`, and cache
report](prerequisites.md#inspect-unified-memory-capacity-and-current-state) to
distinguish total-capacity failures from temporary current-memory failures.
`MemAvailable` already accounts for memory the kernel expects to reclaim; do
not add cached memory to it or size a profile from `MemFree` alone.

Host page-cache clearing, persistent NIM model-cache deletion, and rebooting
are not normal recovery steps for a current-memory failure. Page-cache clearing
affects the whole host and should be reserved for an approved diagnostic after
its impact has been reviewed; retain before-and-after memory and preflight
output. Deleting the disk-backed NIM model cache is a different destructive
operation that forces artifact download or materialization and is not a normal
way to make shared memory available. Stop an identified competing workload and
rerun preflight instead.

After a successful cold start, verify what actually ran rather than assuming a
profile from the hardware calculation:

```bash
curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
uv run python examples/inspect_profile.py
```

The helper joins the selected profile from metadata to the YAML embedded in
`/v1/manifest`.

## Guardrails

Generator profiles run input/output guardrails controlled by operator
configuration:

| Variable | Current default | Effect |
| --- | --- | --- |
| `NIM_ENABLE_TEXT_GUARDRAILS` | true | Input-prompt blocklist and text classifier path |
| `NIM_ENABLE_VIDEO_GUARDRAILS` | true | Output image/video face-privacy guardrail path |
| `NIM_ENABLE_SIGLIP_GUARDRAILS` | true | Per-frame safety classifier when output visual guardrails are enabled |

A blocked request returns HTTP 422 and no usable partial output. Text checks run
on the prompt that reaches generation; when prompt upsampling is enabled, that
is the upsampled prompt.

Disabling controls can reduce safety/privacy protections and may violate
deployment policy. Do so only for an approved, isolated diagnostic:

```bash
-e NIM_ENABLE_TEXT_GUARDRAILS=0 \
-e NIM_ENABLE_VIDEO_GUARDRAILS=0 \
-e NIM_ENABLE_SIGLIP_GUARDRAILS=0
```

`NIM_ENABLE_VIDEO_GUARDRAILS` controls the output visual path for both generated
images and videos. Disabling it also bypasses
the dependent SigLIP path. Disabling SigLIP alone can retain the rest of the
image/video face path. Generator BYOC does not replace the NIM-owned guardrail
artifacts.

Low-VRAM Generator profiles can independently sleep the text guard and output
visual guardrails during diffusion. Profile tags own the normal policy;
`NIM_OFFLOAD_TEXT_GUARDRAIL` and `NIM_OFFLOAD_VIDEO_GUARDRAIL` are advanced
overrides. Change one dimension at a time and measure memory, latency, and
safety behavior.

## Prompt-upsampling diagnostics

Prompt upsampling is designed to fail open to the original prompt at request
time:

1. Generator startup fails if the feature is enabled but endpoint, model, key,
   or required templates are missing.
2. External timeouts, HTTP failures, or malformed responses log a warning.
3. The generation request continues with the original prompt.

When debugging, confirm:

- the configured URL is OpenAI-compatible and reaches `/v1/chat/completions`;
- the model supports image input when upsampling I2V; T2I and T2V send text
  only;
- the external secret is present but not logged;
- timeout and token limits suit the endpoint; and
- provider-specific fields are placed in
  `NIM_PROMPT_UPSAMPLING_EXTRA_BODY` only when that endpoint accepts them.

Do not claim native Anthropic, Gemini, or another provider protocol is
supported merely because an OpenAI-compatible gateway for that provider works.

## Production checks

Before accepting traffic:

- require readiness, not only liveness;
- start with enough free memory on every participating GPU and review any
  profile-fallback warning;
- record version, active model, profile/checkpoint metadata, and image digest;
- verify the intended Generator or Reasoner route exists in live OpenAPI;
- send one small representative request for each enabled capability;
- confirm output decoding and artifact storage permissions;
- confirm logs do not contain secrets or unbounded media/request bodies;
- scrape metrics and test alerts if metrics are part of the SLO.

## Errors

Generator schema, media, and guardrail validation generally returns HTTP 422.
Reasoner sampling or request-shape failures commonly return 400, while invalid
media can return 422. Unexpected backend failures return 500. Readiness can
remain non-200 while model artifacts are downloaded, materialized, compiled,
loaded, or warmed.

Use the HTTP status and stable error-object fields, and retain the returned
message for diagnosis. Do not make automation depend on exact error wording. A
typical NIM error envelope is:

```json
{
  "error": {
    "message": "<description>",
    "type": "<error class>",
    "code": 422
  }
}
```

Missing routes return a runtime-aware 404. For example, sending Chat
Completions to Generator returns this shape:

```json
{
  "error": {
    "message": "The selected runtime is 'generator', which serves POST /v1/infer. POST /v1/chat/completions requires a reasoner deployment.",
    "type": "NotFoundError",
    "code": 404
  }
}
```

The inverse error identifies Reasoner, `/v1/chat/completions`, and the need for
a Generator deployment when a client sends `POST /v1/infer`. Use the status,
error fields, and runtime guidance for diagnosis; continue to treat exact
message wording as mutable.

Task-specific validation belongs to [Generation](generation.md),
[Action](action.md), [Transfer](transfer.md), and [Reasoning](reasoning.md).

## Troubleshooting

### Startup and deployment

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Image pull unauthorized | Docker is not logged in or key lacks repository access | Review the complete [NGC authentication and credential-storage workflow](deployment.md#authenticate-to-ngc), then log in with `NGC_API_KEY` and literal `$oauthtoken` |
| Artifact download fails | Container lacks `NGC_API_KEY`, entitlement, DNS/network, or storage | Verify key injection and NGC connectivity; inspect cache capacity/ownership |
| Cache permission denied | Host mount is not writable by the container | Fix ownership/ACLs and retain a persistent writable `/opt/nim/.cache` |
| No compatible profile | GPU compute capability, total or current free per-device VRAM, or effective system memory does not satisfy the selected model/precision/offload requirements | Run the [pre-download profile preflight](deployment.md#run-the-pre-download-profile-preflight), inspect the exact image manifest, free unrelated GPU memory, remove unnecessary pins, choose a smaller model or compatible precision, or use supported hardware |
| Preflight passes but cold start fails | The 16-GiB non-offload admission floor passed but practical host RAM is too low, or artifact download/materialization or runtime initialization failed | Compare discrete-host RAM with the practical requirement in the support matrix, preserve the failed container, and inspect its logs; do not treat preflight as full host compatibility |
| Discrete-GPU availability probe fails | NVML or the NVIDIA utility driver capability is unavailable, or CUDA-visible devices cannot be matched to NVML | Verify the driver, Container Toolkit configuration, GPU visibility, and utility capability before retrying |
| Integrated GPU has no compatible profile | The host reserve lowers usable shared memory, and Generator offload profiles are not eligible | Choose a resident profile that fits after the reserve; change `NIM_UNIFIED_MEMORY_HOST_RESERVE_GIB` only from validated host-memory measurements |
| Profile filtered by system memory | The container cgroup or host exposes less RAM than the profile admission tag requires | Raise the container memory limit or choose a compatible profile; admission floors are 16 GiB for non-offloading Generator/Reasoner, 64 GiB for Nano offload, and 150 GiB for Super offload, while discrete hosts must also meet the higher practical RAM requirements in the support matrix |
| Conflicting selectors | A shorthand disagrees with `NIM_TAGS_SELECTOR` | Set each selector in one place and inspect the full launch environment |
| Visible GPUs sit idle | Selected profile uses fewer GPUs than Docker exposed | Restrict `--gpus` or intentionally pin an available layout matching the desired count |
| Compute-capability precision failure | Requested precision needs a newer GPU architecture | Select an available precision compatible with the hardware |
| Container live but never ready | Cold materialization/build/warmup is still running or failed | Follow logs, cache/NGC/VRAM errors, and extend startup probe budget; do not send inference |
| Port already allocated | Another container uses the host port | Remove the active example container before reusing the default `-p 8000:8000`, or choose another host port and update that client's `NIM_URL` |
| Container name already in use | A previous example container still exists | Inspect its logs if startup failed, then remove it with `docker rm -f cosmos3-generator` or `docker rm -f cosmos3-reasoner` |
| Wrong-runtime 404 | `NIM_URL` reaches Generator for a Reasoner request, or Reasoner for `/v1/infer` | Inspect `/v1/metadata`, then start the intended runtime or correct the URL |
| `/dev/shm` or resource error | Shared memory/ulimits are too small | Apply the documented `--shm-size` and ulimits |
| Kubernetes Pod stays Pending | GPU request, scheduling constraints, or quota cannot be satisfied | Inspect Pod events and match the GPU count to an available configuration in the [support matrix](support-matrix.md) |
| Kubernetes volume mount fails | PVC, access mode, ownership, or storage class is incompatible | Inspect Pod and PVC events; verify that the cache mount is writable |
| Kubernetes startup probe fails | Cold startup exceeds the probe budget or startup has failed | Increase the startup budget and inspect container logs, cache, and NGC access |

### Generator and media

| Symptom | Likely cause | Action |
| --- | --- | --- |
| HTTP 422, missing or invalid `model_mode` | The request does not identify a valid Generator task | Set one of the documented top-level modes |
| Selected variant rejects request mode or sampling fields | A specialist received another mode, or a four-step request supplied model-owned controls | Select a compatible `NIM_MODEL_VARIANT`; omit `num_inference_steps`, `guidance_scale`, and `flow_shift` for four-step variants |
| HTTP 422, media decode/fetch | Invalid base64/data URL, URL disabled/unreachable, or unsupported media | Prefer a MIME-aware data URL; check the selected image's codec and format support |
| HTTP 422, frame or resolution | Request violates T2I/video cadence rules or supplies `num_frames` for Action | Recompute with the task tables; Action derives frames from its chunk size |
| Content-policy 422 | Text or generated frames triggered guardrails | Rephrase and review content; disable only under approved diagnostic policy |
| Backend 500/OOM | Profile fit or runtime workload exceeded available memory | Reduce workload/concurrency, choose Nano/offload, or use a larger supported GPU; retain logs |
| Request/client timeout | Image or video generation exceeded client/backend timeout | Use the documented timeout ceiling, inspect NIM logs and runtime metrics, and tune only after measurement |
| MP4 will not play | Player lacks VP9-in-MP4 support | Use `mpv`/`ffplay` or re-encode to H.264 |

### Action and transfer

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Action trajectory shape error | Rows, width, domain dimension, or chunk size disagree | Validate `[T,D]`, domain table, positive multiple-of-4 chunk, and `T+1` frames |
| Wrong action media error | Forward/policy received video or inverse received image | Use image for forward/policy and video for inverse dynamics |
| Derived transfer control needs video | Edge/blur has no nested control and no `input_reference` | Add `input_reference` or a nested precomputed control video |
| Transfer control video required | Depth, segmentation, or WSM lacks nested video | Supply precomputed control media |
| Multi-control result unsupported/poor | Combination is not validated for the selected image | Return to one control and smoke-test aligned controls on the intended configuration |
| Nano-DROID action-only response breaks client | Client assumes every policy response has `b64_video` | Save `action` independently and treat both media fields as optional |
| Transfer disabled while T2V works | GPU fits the profile's ordinary-generation floor but lacks Transfer headroom | Use a larger GPU or lower-VRAM profile; use the unsafe override only for diagnostics |

### Reasoner

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Model not found | Client hard-coded the wrong served ID | Discover it through `/v1/models` |
| HTTP 400 request error | Sampling, thinking controls, `top_logprobs`, or extension placement is invalid | Check current ranges, use strict JSON types, and place NIM extensions in `extra_body` |
| HTTP 422 media error | Media content/order/count/preprocessing failed | Put media before text, use data URLs, and check operator media limits |
| Chat Completions route 404 | Client reached Generator or the selected image lacks the route | Confirm Reasoner through `/v1/metadata` and inspect live OpenAPI |
| Responses route 404 | Route disabled, absent, or requested from Generator | Confirm Reasoner metadata, then use Chat Completions and inspect live OpenAPI/operator setting |
| Retrieval/cancel does not work | Response storage/background support is not enabled | Use `store=false` create flow or validate storage configuration for the selected image |
| Structured output is prose or invalid JSON | The request uses the wrong schema shape or the backend did not return a constrained result | Use Chat Completions `response_format` or Responses `text.format`, inspect live OpenAPI and logs, and preserve the response for diagnosis |
| Host OOM or near-zero `MemAvailable` on unified memory | Reasoner used the `0.93` default or a target too large for the shared host/device pool | Restart with `NIM_GPU_MEMORY_UTILIZATION=0.80` for image-only or `0.70` for video or mixed media; lower it further only from host measurements |
| KV-cache/context OOM | Context, media tokens, batching, or concurrency is too large | Reduce media FPS and concurrency; the default unchunked multimodal path enforces a 16,384 batched-token minimum, so enable multimodal chunking with `NIM_DISABLE_CHUNKED_MM_INPUT=0` only after validation before lowering that budget |
| DFlash startup is rejected | DFlash was enabled for Generator, the selected Nano/Super draft is missing, or an independent path/configuration is invalid | Use the matching bundled draft or a valid absolute local `NIM_DFLASH_MODEL_PATH`; otherwise set `NIM_USE_DFLASH=0` for target-only Reasoner operation |
| Reasoner checkpoint source fails | Local layout, `hf://` URI, revision, token, or inferred profile properties are invalid | Validate `NIM_MODEL_PATH`, `HF_TOKEN`, cache/network access, and matching selectors |

### BYOC

See [Bring your own checkpoint](bring-your-own-checkpoint.md) for layout,
mount, selector, and verification procedures.

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Checkpoint/profile mismatch | Inferred model variant or precision disagrees with selected profile | Pin compatible `NIM_MODEL_VARIANT`/`NIM_PRECISION` and use a BYOC configuration available in the selected image |
| Required file missing | BYOC directory does not match the runtime-specific layout | Generator: check transformer, weights, VAE, scheduler, and model index. Reasoner: check config, safetensors, tokenizer, and processor files |
| Path/mount failure | Local `NIM_MODEL_PATH` is not the exact absolute container mount | Align the read-only bind target and environment path |
| `hf://` source fails offline | Remote source requires download while model download is disabled | Pre-download the Reasoner checkpoint and use an absolute local path |
| Generator rejects disabled download | NIM-provided guardrail artifacts still require materialization | Remove `NIM_DISABLE_MODEL_DOWNLOAD=1` and provide cache/NGC access |
| Long first start | A new engine or remote checkpoint is being downloaded/materialized | Keep a writable persistent cache and wait for readiness |
