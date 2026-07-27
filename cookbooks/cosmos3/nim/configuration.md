<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Configure the Cosmos3 Certified NIM

This page is the canonical reference for launch-time environment variables.
Use [deployment.md](deployment.md) for complete Docker launch commands and
[operations.md](operations.md) for runtime diagnostics.

> **Release status:** Defaults and availability below are derived from the
> current source. Recheck them against the released image and its live metadata
> before treating them as a compatibility contract.

## Shared and profile-selection variables

| Variable | Current source default | Purpose |
| --- | --- | --- |
| `NGC_API_KEY` | empty | Cold-start NGC artifact authentication |
| `NIM_CACHE_PATH` | `/opt/nim/.cache` | In-container model cache path |
| `NIM_HTTP_API_PORT` | `8000` | HTTP listen port |
| `NIM_HTTP_MAX_WORKERS` | `1` | Uvicorn worker count supplied by the NIM framework |
| `NIM_MODEL_TYPE` | `generator` | Active runtime |
| `NIM_MODEL_SIZE` | soft preference for Nano | Model-size selector |
| `NIM_PRECISION` | compatible soft preference | Precision selector |
| `NIM_PERF_PROFILE` | Generator latency preference | Generator scenario selector |
| `NIM_OFFLOAD_MODE` | compatible preference | Generator offload selector |
| `NIM_TAGS_SELECTOR` | empty | Additional manifest tag filters |
| `NIM_MODEL_PROFILE` | empty | Exact profile ID pin |
| `NIM_MANIFEST_PATH` | image default | Override manifest path |
| `NIM_DISABLE_MODEL_DOWNLOAD` | false | Disable startup download |
| `NIM_IGNORE_MODEL_DOWNLOAD_FAIL` | false | Continue despite download failure; use cautiously |
| `NIM_SKIP_MATERIALIZE` | false | Force materialization rather than reuse the SDK result |
| `NIM_LOG_LEVEL` | `INFO` | Service logging threshold |
| `NIM_LOGGING_JSONL` | false | JSON-formatted logs |

Shorthands and the corresponding keys in `NIM_TAGS_SELECTOR` must not
conflict. Use `NIM_MODEL_PROFILE` only to pin a reviewed profile from the exact
image release.

## Generator variables

| Variable | Current source default | Purpose |
| --- | --- | --- |
| `NIM_ALLOW_URL_INPUT` | true | Permit HTTP(S) media fetching |
| `NIM_ENABLE_WARMUP` | true | Run startup warmup before readiness |
| `NIM_VIDEO_SAVE_QUALITY` | source-defined | VP9 encoder quality; does not change diffusion quality |
| `NIM_TRITON_REQUEST_TIMEOUT` | 30 minutes | Backend queue-plus-execution timeout in microseconds |
| `TLLM_LOG_LEVEL` | `ERROR` | TRT-LLM/backend log level |
| `NIM_ATTENTION_BACKEND` | `VANILLA` | Generator attention implementation |
| `NIM_ENABLE_CUDAGRAPH` | false | Enable Generator CUDA graph execution |
| `NIM_ENABLE_TORCH_COMPILE` | true | Enable Generator `torch.compile` path |
| `NIM_ENABLE_FULLGRAPH` | false | Require full-graph compilation when enabled |
| `NIM_ENABLE_AUTOTUNE` | true | Enable Generator compile/kernel autotuning |
| `NIM_CACHE_BACKEND` | empty | Optional diffusion-step cache; current source accepts only `cache_dit` |
| `NIM_CACHE_CONFIG` | empty JSON object | Backend-specific diffusion-cache configuration |
| `NIM_MAX_SEQUENCE_LENGTH` | 5120 | Generator prompt token sequence length; startup-level, not per request |
| `NIM_FT_CHECKPOINT` | empty | Absolute in-container Generator BYOC path |
| `NIM_ENABLE_TEXT_GUARDRAILS` | true | Input text policy checks |
| `NIM_ENABLE_VIDEO_GUARDRAILS` | true | Output face/video guardrail path |
| `NIM_ENABLE_SIGLIP_GUARDRAILS` | true | Output frame safety classifier when video guardrails run |
| `NIM_QWEN3GUARD_MAX_INPUT_TOKENS` | 8192 | Text-guardrail input ceiling; minimum 512 |

See [Bring your own checkpoint](bring-your-own-checkpoint.md) before setting
`NIM_FT_CHECKPOINT`.

## Reasoner variables

| Variable | Current source default | Purpose |
| --- | --- | --- |
| `NIM_MAX_MODEL_LEN` | 262144 | Maximum model context configured in vLLM |
| `NIM_MAX_NUM_BATCHED_TOKENS` | 8192 | Scheduler token budget; empty can omit the override |
| `NIM_MAX_NUM_SEQS` | 256 | Maximum scheduled sequences |
| `NIM_STREAM_INTERVAL` | 10 | Streaming update interval |
| `NIM_GPU_MEMORY_UTILIZATION` | 0.90 | vLLM GPU memory target `(0,1]` |
| `NIM_ENABLE_KV_CACHE_REUSE` | true | Prefix/KV-cache reuse |
| `NIM_ENABLE_CHUNKED_PREFILL` | true | Chunked prefill |
| `NIM_DISABLE_CHUNKED_MM_INPUT` | false | Disable chunking of multimodal input |
| `NIM_DISABLE_MM_PREPROCESSOR_CACHE` | false | Disable multimodal preprocessor cache |
| `NIM_MAX_IMAGES_PER_PROMPT` | 5 | Image count limit |
| `NIM_MAX_VIDEOS_PER_PROMPT` | 1 | Video count limit |
| `NIM_MEDIA_IO_KWARGS` | source video default | Complete operator-level media preprocessing object |
| `NIM_VIDEO_PRUNING_RATE` | 0/off | Optional video token pruning rate `[0,1]` |
| `NIM_GUIDED_DECODING_BACKEND` | `xgrammar` | Structured-output backend |
| `NIM_MM_ENCODER_ATTN_BACKEND` | `FLASH_ATTN` | Multimodal encoder attention backend |
| `NIM_MM_ENCODER_TP_MODE` | empty | Optional vLLM multimodal-encoder tensor-parallel mode |
| `NIM_COMPILATION_CONFIG` | source JSON object | vLLM compilation configuration |
| `VLLM_ATTENTION_BACKEND` | empty | Optional vLLM attention backend override |
| `NIM_DISABLE_LOG_REQUESTS` | true | Disable request-body logging |
| `NIM_DISABLE_RESPONSES_ROUTE` | false in current route tests | Remove Responses create/retrieve/cancel routes when true |
| `VLLM_ENABLE_RESPONSES_API_STORE` | off by default | Enable state needed by persisted/background Responses features |

These are operator controls, not a recommendation to override every default.
Change one at a time and validate memory, quality, latency, and correctness on
the target release.

## Prompt upsampling

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

| Variable | Current default | Purpose |
| --- | --- | --- |
| `NIM_ENABLE_PROMPT_UPSAMPLING` | false | Enable the feature |
| `NIM_PROMPT_UPSAMPLING_ENDPOINT_URL` | empty | OpenAI-compatible endpoint base/route |
| `NIM_PROMPT_UPSAMPLING_MODEL` | empty | Model sent to the endpoint |
| `NIM_PROMPT_UPSAMPLING_API_KEY` | empty | Separate external-service Bearer token |
| `NIM_PROMPT_UPSAMPLING_TEMPLATE_STYLE` | `external_api` | `external_api` or `reasoner` templates |
| `NIM_PROMPT_UPSAMPLING_TIMEOUT_S` | 120 | Per-call timeout |
| `NIM_PROMPT_UPSAMPLING_MAX_TOKENS` | 8192 | Upsampler output token budget |
| `NIM_PROMPT_UPSAMPLING_TEMPERATURE` | omitted | Optional external sampling parameter |
| `NIM_PROMPT_UPSAMPLING_TOP_P` | omitted | Optional external sampling parameter |
| `NIM_PROMPT_UPSAMPLING_TOP_K` | omitted | Optional; not accepted by every OpenAI-compatible provider |
| `NIM_PROMPT_UPSAMPLING_EXTRA_BODY` | empty object | Additional JSON merged into the external request |

Enabling the feature without endpoint, model, or key fails Generator startup.
At request time, external timeout, error, or invalid output logs a warning and
falls back to the original prompt. Never log or reuse the upsampler key as
`NGC_API_KEY`.

## Secret handling

Keep `NGC_API_KEY` and `NIM_PROMPT_UPSAMPLING_API_KEY` separate. Do not place
either value in source control, image layers, saved requests, notebooks, or
logs. Prefer secret injection supported by the deployment environment instead
of literal command-line values.
