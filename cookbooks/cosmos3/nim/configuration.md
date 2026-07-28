<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Configure the Cosmos3 Certified NIM

This page is the canonical reference for launch-time environment variables.
Use [deployment.md](deployment.md) for complete Docker launch commands and
[operations.md](operations.md) for runtime diagnostics.

> **Release status:** Defaults and availability below are derived from the
> current source. Recheck them against the released image and its live metadata
> before treating them as a compatibility contract.

## Shared variables

### Authentication and model artifacts

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NGC_API_KEY` | Conditional | Empty | NGC credentials used to download model artifacts on first boot. It is not required when all required artifacts are already present in the cache. |
| `NIM_CACHE_PATH` | No | `/opt/nim/.cache` | In-container path for downloaded model artifacts. Mount a writable host directory here to share the cache across runs. |

### Server and logging

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_HTTP_API_PORT` | No | `8000` | Port used by the Uvicorn HTTP server. The container-side port in the Docker publish mapping must match this value. |
| `NIM_LOG_LEVEL` | No | `INFO` | Logging threshold for the NIM service. |
| `NIM_LOGGING_JSONL` | No | `false` | Enable JSON-formatted logs. Readable text logs are enabled by default. |

### Runtime and profile selection

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_MODEL_TYPE` | No | `generator` | Select the active runtime: `generator` or `reasoner`. A running container serves only the selected runtime. |
| `NIM_MODEL_SIZE` | No | Soft preference for Nano | Select the model-size preference used by automatic profile selection. |
| `NIM_PRECISION` | No | Compatible soft preference | Select the precision preference used by automatic profile selection. Available values depend on the released profiles and hardware. |
| `NIM_TAGS_SELECTOR` | No | Empty | Comma-separated `key=value` filters used by automatic profile selection. |
| `NIM_MODEL_PROFILE` | No | Empty | Pin an exact profile ID from the image manifest instead of using automatic selection. |
| `NIM_MANIFEST_PATH` | No | Image default | Override the model-manifest path. |

Shorthands and the corresponding keys in `NIM_TAGS_SELECTOR` must not
conflict. Use `NIM_MODEL_PROFILE` only to pin a reviewed profile from the exact
image release.

## Generator variables

### Generator profile selection

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_PERF_PROFILE` | No | Latency preference | Select the Generator performance scenario used by automatic profile selection. |
| `NIM_OFFLOAD_MODE` | No | Compatible preference | Select the Generator offload preference used by automatic profile selection. |

### Input, output, and request handling

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_ALLOW_URL_INPUT` | No | `true` | Allow HTTP(S) URLs for image, video, and Transfer control-video inputs. Set to `false` to require base64-encoded inputs. |
| `NIM_VIDEO_SAVE_QUALITY` | No | `7` | Set VP9 output-video compression quality from `1` through `9`. The values map linearly to FFmpeg CRF `[63, 0]`; higher values produce higher quality and lower CRF. This does not affect diffusion quality. |
| `NIM_TRITON_REQUEST_TIMEOUT` | No | 30 minutes | Queue-plus-execution timeout in microseconds. The source default is `1800000000`. |

### Startup

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_ENABLE_WARMUP` | No | `false` | Run a synthetic inference pass during startup before the service becomes ready. |

### Execution and compilation

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_ATTENTION_BACKEND` | No | `VANILLA` | Select the Generator attention implementation. |
| `NIM_ENABLE_CUDAGRAPH` | No | `false` | Enable Generator CUDA graph execution. |
| `NIM_ENABLE_TORCH_COMPILE` | No | `true` | Enable the Generator `torch.compile` path. |
| `NIM_ENABLE_FULLGRAPH` | No | `false` | Require full-graph compilation when compilation is enabled. |
| `NIM_ENABLE_AUTOTUNE` | No | `true` | Enable Generator compilation and kernel autotuning. |
| `NIM_MAX_SEQUENCE_LENGTH` | No | `5120` | Set the Generator prompt-token sequence length at startup. This is not a per-request control. |

### Diffusion caching

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_CACHE_BACKEND` | No | Empty | Enable an optional diffusion-step cache. Current source accepts only `cache_dit`. This is separate from the artifact cache configured by `NIM_CACHE_PATH`. |
| `NIM_CACHE_CONFIG` | No | Empty JSON object | Set backend-specific diffusion-cache configuration as a JSON object. |

### Guardrails

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_ENABLE_TEXT_GUARDRAILS` | No | `true` | Enable input-text policy checks. |
| `NIM_ENABLE_VIDEO_GUARDRAILS` | No | `true` | Enable the output face and video guardrail path. |
| `NIM_ENABLE_SIGLIP_GUARDRAILS` | No | `true` | Enable the output-frame safety classifier when video guardrails run. |

### Bring your own checkpoint

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_FT_CHECKPOINT` | No | Empty | Absolute path inside the container to a Generator BYOC diffusion checkpoint. Only the diffusion model is replaced; profile-managed components continue to come from the selected workspace. |

See [Bring your own checkpoint](bring-your-own-checkpoint.md) before setting
`NIM_FT_CHECKPOINT`.

### Prompt upsampling

Prompt upsampling is optional, off by default, and consumed only by the
Generator. It applies to T2V and I2V; V2V, action, and transfer bypass it.

```bash
-e NIM_ENABLE_PROMPT_UPSAMPLING=1 \
-e NIM_PROMPT_UPSAMPLING_ENDPOINT_URL='https://openai-compatible.example/v1' \
-e NIM_PROMPT_UPSAMPLING_MODEL='<UPSAMPLER_MODEL>' \
-e NIM_PROMPT_UPSAMPLING_API_KEY="$UPSAMPLER_API_KEY" \
-e NIM_PROMPT_UPSAMPLING_TEMPLATE_STYLE=external_api
```

The endpoint is normalized to an OpenAI-compatible
`/v1/chat/completions` route and uses Bearer authorization. Do not assume a
provider's native non-OpenAI endpoint is compatible.

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_ENABLE_PROMPT_UPSAMPLING` | No | `false` | Enable prompt upsampling for Generator T2V and I2V requests. |
| `NIM_PROMPT_UPSAMPLING_ENDPOINT_URL` | When enabled | Empty | OpenAI-compatible endpoint base or Chat Completions route. |
| `NIM_PROMPT_UPSAMPLING_MODEL` | When enabled | Empty | Model name sent to the prompt-upsampling endpoint. |
| `NIM_PROMPT_UPSAMPLING_API_KEY` | When enabled | Empty | Bearer credential for the external prompt-upsampling service. Keep it separate from `NGC_API_KEY`. |
| `NIM_PROMPT_UPSAMPLING_TEMPLATE_STYLE` | No | `external_api` | Select `external_api` or `reasoner` prompt templates. |
| `NIM_PROMPT_UPSAMPLING_TIMEOUT_S` | No | `120` | Per-call timeout in seconds. |
| `NIM_PROMPT_UPSAMPLING_MAX_TOKENS` | No | `8192` | Maximum output-token budget requested from the upsampler. |
| `NIM_PROMPT_UPSAMPLING_TEMPERATURE` | No | Omitted | Optional external sampling temperature. |
| `NIM_PROMPT_UPSAMPLING_TOP_P` | No | Omitted | Optional external nucleus-sampling value. |
| `NIM_PROMPT_UPSAMPLING_TOP_K` | No | Omitted | Optional external top-k value. Not every OpenAI-compatible provider accepts it. |
| `NIM_PROMPT_UPSAMPLING_EXTRA_BODY` | No | Empty object | Additional JSON object merged into the external request. |

Enabling the feature without endpoint, model, or key fails Generator startup.
At request time, external timeout, error, or invalid output logs a warning and
falls back to the original prompt. Never log or reuse the upsampler key as
`NGC_API_KEY`.

## Reasoner variables

### Context and scheduling

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_MAX_MODEL_LEN` | No | `262144` | Maximum model context length configured in vLLM. |
| `NIM_MAX_NUM_BATCHED_TOKENS` | No | `8192` | Scheduler token budget. Set an empty value to omit this override. |
| `NIM_MAX_NUM_SEQS` | No | `256` | Maximum number of sequences scheduled in parallel. |
| `NIM_STREAM_INTERVAL` | No | `10` | Streaming update interval. |

### GPU memory and caching

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_GPU_MEMORY_UTILIZATION` | No | `0.90` | vLLM GPU-memory utilization target. Accepted values are greater than `0` and no greater than `1`. |
| `NIM_ENABLE_KV_CACHE_REUSE` | No | `true` | Enable prefix caching and KV-cache reuse. |
| `NIM_ENABLE_CHUNKED_PREFILL` | No | `true` | Enable vLLM chunked prefill. |
| `NIM_DISABLE_CHUNKED_MM_INPUT` | No | `false` | Disable chunking of multimodal input. |
| `NIM_DISABLE_MM_PREPROCESSOR_CACHE` | No | `false` | Disable the multimodal preprocessor cache. |

### Multimodal limits and preprocessing

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_MAX_IMAGES_PER_PROMPT` | No | `5` | Maximum number of images allowed in one request. |
| `NIM_MAX_VIDEOS_PER_PROMPT` | No | `1` | Maximum number of videos allowed in one request. |
| `NIM_MEDIA_IO_KWARGS` | No | Unset | Complete operator-level media preprocessing JSON object. When unset, the Cosmos3 wrapper does not override the underlying media-I/O defaults. |
| `NIM_VIDEO_PRUNING_RATE` | No | `0` (off) | Optional video-token pruning rate from `0` through `1`. |

### Decoding and structured output

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_GUIDED_DECODING_BACKEND` | No | `xgrammar` | Structured-output and guided-decoding backend. |

### vLLM execution and compilation

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_MM_ENCODER_ATTN_BACKEND` | No | `FLASH_ATTN` | Multimodal-encoder attention backend. |
| `NIM_MM_ENCODER_TP_MODE` | No | Empty | Optional vLLM multimodal-encoder tensor-parallel mode. |
| `NIM_COMPILATION_CONFIG` | No | Source JSON object | vLLM compilation configuration. The current source supplies CUDA graph and multimodal-encoder compilation settings. |
| `VLLM_ATTENTION_BACKEND` | No | Empty | Optional vLLM attention-backend override. |

### Request logging and Responses API

| Name | Required? | Default | Notes |
| --- | --- | --- | --- |
| `NIM_DISABLE_LOG_REQUESTS` | No | `true` | Disable request-body logging. Request content may be sensitive. |
| `NIM_DISABLE_RESPONSES_ROUTE` | No | `false` in current route tests | Remove Responses create, retrieve, and cancel routes when set to `true`. |
| `VLLM_ENABLE_RESPONSES_API_STORE` | No | Off | Enable state required by persisted and background Responses features. |

These are operator controls, not a recommendation to override every default.
Change one at a time and validate memory, quality, latency, and correctness on
the target release.

## Secret handling

Keep `NGC_API_KEY` and `NIM_PROMPT_UPSAMPLING_API_KEY` separate. Do not place
either value in source control, image layers, saved requests, notebooks, or
logs. Prefer secret injection supported by the deployment environment instead
of literal command-line values.
