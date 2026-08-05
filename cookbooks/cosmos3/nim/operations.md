<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Operate and troubleshoot the Cosmos3 Certified NIM

Use this page to interpret health, inspect the selected profile, configure
logging and guardrails, integrate metrics, and diagnose deployment and request
failures. See [deployment.md](deployment.md) for launch configuration and
[configuration.md](configuration.md) for environment variables.

> Validate endpoint output, metrics, logs, and operational limits against the
> released image used by the deployment.

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

Cold startup can include NGC download, cache materialization, engine build,
model load, and warmup. A live-but-not-ready interval is expected. Kubernetes
startup/readiness budgets must accommodate the slowest supported cold start.

## Inspect the running service

Use the active runtime rather than launch-time assumptions:

```bash
for endpoint in metadata models manifest version license; do
  curl -fsS "$NIM_URL/v1/$endpoint" -o "$endpoint.json"
  python -m json.tool "$endpoint.json" >/dev/null
done

curl -fsS "$NIM_URL/openapi.json" -o openapi.json
```

| Endpoint | Operational question |
| --- | --- |
| `/v1/metadata` | Which profile/checkpoint did the NIM report, and which `model_variant` is active for Generator? |
| `/v1/models` | Which model ID should the Reasoner client send? |
| `/v1/manifest` | Which profile/artifact information is present in this image? |
| `/v1/version` | Which release/server API is running? |
| `/v1/license` | Where is the bundled product license information? |
| `/openapi.json` | Which routes and schemas does this active runtime expose? |

Capture these outputs with deployment evidence, but review them before sharing:
paths, repository overrides, or other fields can reveal internal operational
details.

## Logging

Common controls:

| Variable | Current source default | Use |
| --- | --- | --- |
| `NIM_LOG_LEVEL` | `INFO` | Service/NIMlib log threshold |
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

Verify the released runtime before configuring a scraper:

```bash
curl -fsS "$NIM_URL/v1/metrics" -o metrics.txt
head metrics.txt
```

Do not assume metric names from a previous NIM. Once the final image has been
scraped, document the observed request, latency, error, process, and GPU metric
families here.

A minimal Prometheus job, if `/v1/metrics` is enabled:

```yaml
scrape_configs:
  - job_name: cosmos3-certified-nim
    metrics_path: /v1/metrics
    static_configs:
      - targets: ["cosmos3-nim:8000"]
```

The final Helm `ServiceMonitor`/OpenTelemetry values and any recommended Grafana
dashboard are **TBD (release-dependent)**.

## Guardrails

Generator profiles run input/output guardrails controlled by operator
configuration:

| Variable | Current default | Effect |
| --- | --- | --- |
| `NIM_ENABLE_TEXT_GUARDRAILS` | true | Input prompt/negative-prompt blocklist and text classifier path |
| `NIM_ENABLE_VIDEO_GUARDRAILS` | true | Output image/video face-privacy guardrail path |
| `NIM_ENABLE_SIGLIP_GUARDRAILS` | true | Per-frame safety classifier when output visual guardrails are enabled |

A blocked request returns HTTP 422 and no usable partial output. Current source
runs text checks on the prompt that reaches generation; when prompt upsampling
is enabled, that is the upsampled prompt.

Disabling controls can reduce safety/privacy protections and may violate
deployment policy. Do so only for an approved, isolated diagnostic:

```bash
-e NIM_ENABLE_TEXT_GUARDRAILS=0 \
-e NIM_ENABLE_VIDEO_GUARDRAILS=0 \
-e NIM_ENABLE_SIGLIP_GUARDRAILS=0
```

Despite its historical name, `NIM_ENABLE_VIDEO_GUARDRAILS` controls the output
visual path for both generated images and videos. Disabling it also bypasses
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
- record version, active model, profile/checkpoint metadata, and image digest;
- verify the intended Generator or Reasoner route exists in live OpenAPI;
- send one small representative request for each enabled capability;
- confirm output decoding and artifact storage permissions;
- confirm logs do not contain secrets or unbounded media/request bodies;
- scrape metrics and test alerts if metrics are part of the SLO; and
- document which release-dependent TBDs remain unresolved.

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

Task-specific validation belongs to [Generation](generation.md),
[Action](action.md), [Transfer](transfer.md), and [Reasoning](reasoning.md).

## Troubleshooting

### Startup and deployment

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Image pull unauthorized | Docker is not logged in or key lacks repository access | Re-run password-stdin login with `NGC_API_KEY` and literal `$oauthtoken` |
| Artifact download fails | Container lacks `NGC_API_KEY`, entitlement, DNS/network, or storage | Verify key injection and NGC connectivity; inspect cache capacity/ownership |
| Cache permission denied | Host mount is not writable by the container | Fix ownership/ACLs and retain a persistent writable `/opt/nim/.cache` |
| No compatible profile | Visible GPUs or system memory do not satisfy the selected model/precision/offload requirements | Remove unnecessary pins, choose a smaller model or compatible precision, or use supported hardware |
| Integrated GPU has no compatible profile | The host reserve lowers usable shared memory, and Generator offload profiles are not eligible | Choose a resident profile that fits after the reserve; change `NIM_UNIFIED_MEMORY_HOST_RESERVE_GIB` only from validated host-memory measurements |
| Offload profile requires more system memory | The container cgroup or host exposes less RAM than the profile requires | Raise the container memory limit or choose a compatible precision/profile; current Super BF16 offload profiles require 150 GiB |
| Conflicting selectors | A shorthand disagrees with `NIM_TAGS_SELECTOR` | Set each selector in one place and inspect the full launch environment |
| Visible GPUs sit idle | Selected profile uses fewer GPUs than Docker exposed | Restrict `--gpus` or intentionally pin a released layout matching the desired count |
| Compute-capability precision failure | Requested precision needs a newer GPU architecture | Select a released precision compatible with the hardware |
| Container live but never ready | Cold materialization/build/warmup is still running or failed | Follow logs, cache/NGC/VRAM errors, and extend startup probe budget; do not send inference |
| Port already allocated | Another container uses the host port | Choose a different `-p HOST:8000` mapping and update `NIM_URL` |
| `/dev/shm` or resource error | Shared memory/ulimits are too small | Apply the release-recommended `--shm-size` and documented ulimits |
| Kubernetes Pod stays Pending | GPU resource request, node selector, taint/toleration, or quota cannot be satisfied | Inspect Pod events and GPU Operator/device-plugin status; match a released profile's GPU count in the [support matrix](support-matrix.md) |
| Kubernetes volume mount fails | PVC, access mode, ownership, or storage class is incompatible | Inspect Pod/PVC events and verify the cache mount is writable by the container |
| Kubernetes startup probe fails | Cold materialization exceeds the probe budget or startup has failed | Increase the release-validated startup budget and inspect container logs/cache/NGC access |
| Helm values are rejected or ignored | Values were copied from another NIM/chart version | Follow [Deploy with Helm](helm.md), use the final Cosmos3 chart schema, and pin the chart version |

### Generator and media

| Symptom | Likely cause | Action |
| --- | --- | --- |
| HTTP 422, missing or invalid `model_mode` | Generator tasks are no longer inferred from request shape | Set one of the documented top-level modes |
| HTTP 422, deprecated Generator field | Request uses `image`, `video`, `num_output_frames`, `steps`, or `action_params.mode` | Use `input_reference`, `num_frames`, `num_inference_steps`, and top-level `model_mode` |
| Selected variant rejects request mode or sampling fields | A specialist received another mode, or a four-step request supplied profile-owned controls | Select a compatible `NIM_MODEL_VARIANT`; omit `num_inference_steps`, `guidance_scale`, and `flow_shift` for four-step variants |
| HTTP 422, media decode/fetch | Invalid base64/data URL, URL disabled/unreachable, or unsupported media | Prefer a MIME-aware data URL; check release codec/format support |
| HTTP 422, frame or resolution | Request violates T2I/video cadence rules or supplies `num_frames` for Action | Recompute with the task tables; Action derives frames from its chunk size |
| Content-policy 422 | Text or generated frames triggered guardrails | Rephrase and review content; disable only under approved diagnostic policy |
| Backend 500/OOM | Profile fit or runtime workload exceeded available memory | Reduce workload/concurrency, choose Nano/offload, or use a larger supported GPU; retain logs |
| Request/client timeout | Image or video generation exceeded client/backend timeout | Use a long client timeout, inspect server progress, and tune only after measurement |
| MP4 will not play | Player lacks VP9-in-MP4 support | Use `mpv`/`ffplay` or re-encode to H.264 |

### Action and transfer

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Action trajectory shape error | Rows, width, domain dimension, or chunk size disagree | Validate `[T,D]`, domain table, positive multiple-of-4 chunk, and `T+1` frames |
| Wrong action media error | Forward/policy received video or inverse received image | Use image for forward/policy and video for inverse dynamics |
| Derived transfer control needs video | Edge/blur has no nested control and no `input_reference` | Add `input_reference` or a nested precomputed control video |
| Transfer control video required | Depth, segmentation, or WSM lacks nested video | Supply precomputed control media |
| Multi-control result unsupported/poor | Combination is not validated for the release | Return to one control and smoke-test aligned controls on the intended profile |
| Nano-DROID action-only response breaks client | Client assumes every policy response has `b64_video` | Save `action` independently and treat both media fields as optional |
| Transfer disabled while T2V works | GPU fits the profile's ordinary-generation floor but lacks Transfer headroom | Use a larger GPU or lower-VRAM profile; use the unsafe override only for diagnostics |

### Reasoner

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Model not found | Client hard-coded the wrong served ID | Discover it through `/v1/models` |
| HTTP 400 request error | Sampling, `include_reasoning`, `top_logprobs`, or extension placement is invalid | Check current ranges, use strict JSON types, and place NIM/vLLM extensions in `extra_body` |
| HTTP 422 media error | Media content/order/count/preprocessing failed | Put media before text, use data URLs, and check operator media limits |
| Responses route 404 | Route disabled or absent in this release | Use Chat Completions and inspect live OpenAPI/operator setting |
| Retrieval/cancel does not work | Response storage/background support is not enabled | Use `store=false` create flow or validate storage configuration for the release |
| KV-cache/context OOM | Context, media tokens, batching, or concurrency is too large | Reduce media FPS/token budget/concurrency before raising memory utilization |
| DFlash startup is rejected | DFlash was enabled for Generator/Super Reasoner or its Nano draft artifact is missing | Use Nano Reasoner with a released DFlash artifact, or set `NIM_USE_DFLASH=0` |
| Reasoner checkpoint source fails | Local layout, `hf://` URI, revision, token, or inferred profile properties are invalid | Validate `NIM_MODEL_PATH`, `HF_TOKEN`, cache/network access, and matching selectors |

### BYOC

See [Bring your own checkpoint](bring-your-own-checkpoint.md) for layout,
mount, selector, and verification procedures.

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Checkpoint/profile mismatch | Inferred model variant or precision disagrees with selected profile | Pin compatible `NIM_MODEL_VARIANT`/`NIM_PRECISION` and use a released BYOC-supported profile |
| Required file missing | BYOC directory does not match the runtime-specific layout | Generator: check transformer, weights, VAE, scheduler, and model index. Reasoner: check config, safetensors, tokenizer, and processor files |
| Path/mount failure | Local `NIM_MODEL_PATH` is not the exact absolute container mount | Align the read-only bind target and environment path |
| `hf://` source fails offline | Remote source requires download while model download is disabled | Pre-download the Reasoner checkpoint and use an absolute local path |
| Generator rejects disabled download | Profile-owned guardrail artifacts still require materialization | Remove `NIM_DISABLE_MODEL_DOWNLOAD=1` and provide cache/NGC access |
| Long first start | A new engine or remote checkpoint is being downloaded/materialized | Keep a writable persistent cache and wait for readiness |
