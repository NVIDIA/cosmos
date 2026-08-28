<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Configure the Cosmos3 Certified NIM

This page lists launch-time environment variables intended for users and
operators. Start with the model-selection variables and keep automatic defaults
unless the workload requires an override. See [Deployment](deployment.md) for
complete Docker commands.

## Essential configuration

### Authentication and cache

| Name | Default | Use |
| --- | --- | --- |
| `NGC_API_KEY` | Empty | Download required model and runtime artifacts when they are not already cached |
| `NIM_CACHE_PATH` | `/opt/nim/.cache` | Set the writable in-container artifact cache |

### Choose the runtime and model

| Name | Default | Use |
| --- | --- | --- |
| `NIM_MODEL_TYPE` | `generator` | Select `generator` or `reasoner` |
| `NIM_MODEL_VARIANT` | `nano` preference | Select `nano` or `super` for either runtime; Generator also accepts `nano-droid`, `super-t2i`, `super-t2i-4step`, `super-i2v`, and `super-i2v-4step` |
| `NIM_PRECISION` | Generator: FP8 preference; Reasoner: GPU-derived preference | Optionally pin a precision available in the selected image; when omitted, Reasoner prefers BF16 on compute capability 8.0 through 8.8, FP8 on 8.9 through 9.x, and NVFP4 on 10.0 or newer when compatible |
| `NIM_PERF_PROFILE` | `latency` | Generator only: choose `latency` or `throughput` |

`NIM_MODEL_VARIANT` selects the checkpoint contract for either runtime.
Reasoner accepts `nano` or `super` and does not use `NIM_PERF_PROFILE`.
Nano-DROID currently has BF16 profiles only.

The NIM chooses the best compatible profile for these settings and the visible
GPUs. A normal deployment does not need a profile ID.

### Bring your own checkpoint

| Name | Default | Use |
| --- | --- | --- |
| `NIM_MODEL_PATH` | Empty | Generator: absolute local directory. Reasoner: absolute local directory or `hf://owner/repository[:revision]` |
| `NIM_DFLASH_MODEL_PATH` | Empty | Nano or Super Reasoner: independently override the DFlash draft with an absolute local directory |
| `NIM_DISABLE_MODEL_DOWNLOAD` | `false` | Disable profile download for a completely local Reasoner override; incompatible with Reasoner `hf://` and rejected for Generator |
| `HF_TOKEN` | Empty | Authenticate to a private Reasoner Hugging Face repository |

See [Bring your own checkpoint](bring-your-own-checkpoint.md) for layouts,
mounts, and compatibility checks.

### Server and logging

| Name | Default | Use |
| --- | --- | --- |
| `NIM_HTTP_API_PORT` | `8000` | Set the container HTTP port |
| `NIM_LOG_LEVEL` | `INFO` | Set the service logging threshold |
| `NIM_LOGGING_JSONL` | `false` | Emit JSON-line logs |

## Advanced profile controls

Use these only when automatic model selection is not sufficient:

| Name | Default | Use |
| --- | --- | --- |
| `NIM_OFFLOAD_MODE` | Automatic compatible preference | Request `none`, `model`, or `layer` when the selected model provides that mode |
| `NIM_UNIFIED_MEMORY_HOST_RESERVE_GIB` | `16` | Reserve shared memory for the host before profile selection on integrated GPUs; use a nonnegative binary-GiB value validated for the system |
| `NIM_GPU_MEMORY_HEADROOM_GIB` | `2` | Reasoner only: reserve this many binary GiB of free memory per discrete GPU during profile selection and runtime sizing |
| `NIM_TAGS_SELECTOR` | Empty | Filter by comma-separated exact manifest tags |
| `NIM_MODEL_PROFILE` | Empty | Pin an exact profile ID from the current image |

Do not combine shorthand variables with conflicting values in
`NIM_TAGS_SELECTOR`. Use the `model_variant` tag when filtering by model. Exact
tags and profile IDs are tied to an image release and are less portable than
automatic selection. Leave the unified-memory host reserve at its default unless
host measurements establish a safe system-specific value; it does not affect
discrete GPUs.

## Generator configuration

### Input and output

| Name | Default | Use |
| --- | --- | --- |
| `NIM_ALLOW_URL_INPUT` | `true` | Allow HTTP(S) image, video, and Transfer inputs; set `false` to require encoded input |
| `NIM_VIDEO_SAVE_QUALITY` | `7` | Set VP9 output quality from 1 through 9; this affects encoding, not diffusion quality |
| `NIM_TRITON_REQUEST_TIMEOUT` | 30 minutes (`1800000000` microseconds) | Set the queue-plus-execution timeout in microseconds |

Client HTTP timeouts do not change this backend ceiling. The Transfer example
uses a 60-minute client timeout; to allow the backend the same ceiling, add
`-e NIM_TRITON_REQUEST_TIMEOUT=3600000000` to the Generator Docker launch
command and restart the container. See the [Transfer hardware and timeout
guidance](transfer.md#run-the-examples).

### Startup and execution

| Name | Default | Use |
| --- | --- | --- |
| `NIM_ENABLE_WARMUP` | `false` | Run synthetic inference before readiness |
| `NIM_ENABLE_TORCH_COMPILE` | `true` | Enable the Generator compilation path |
| `NIM_LINEAR_BACKEND` | Quantized profiles: `cutlass`; BF16: unset | Select `auto`, `cutlass`, `flashinfer_cutlass`, `flashinfer_cutedsl`, or `torch` for quantized Generator DiT linear layers |
| `NIM_TRITON_LOG_VERBOSE` | `0` | Increase Generator backend logging during diagnosis |
| `NIM_MAX_SEQUENCE_LENGTH` | `5120` | Set the startup prompt-token sequence length |

Leave the quantized linear backend at its default unless the exact
hardware/precision pairing has been validated. An incompatible backend fails
during model load rather than silently falling back.

### Guardrails

| Name | Default | Use |
| --- | --- | --- |
| `NIM_ENABLE_TEXT_GUARDRAILS` | `true` | Enable input-prompt policy checks |
| `NIM_ENABLE_VIDEO_GUARDRAILS` | `true` | Enable output image/video face and visual guardrails |
| `NIM_ENABLE_SIGLIP_GUARDRAILS` | `true` | Enable the output-frame safety classifier |
| `NIM_OFFLOAD_TEXT_GUARDRAIL` | Profile policy | Override whether the text guard sleeps on CPU during diffusion |
| `NIM_OFFLOAD_VIDEO_GUARDRAIL` | Profile policy | Override whether output guardrail sessions sleep during diffusion |

The selected profile owns normal guardrail residency. Treat offload variables as
advanced overrides, not routine model selection.

### Transfer diagnostic override

| Name | Default | Use |
| --- | --- | --- |
| `NIM_ALLOW_UNSAFE_TRANSFER` | `false` | Bypass Transfer's VRAM admission check for an approved diagnostic; the request can fail with OOM |

### Prompt upsampling

Prompt upsampling is optional and applies only to Generator T2I, T2V, and I2V.
It sends the request to an operator-provided OpenAI-compatible endpoint.

| Name | Default | Use |
| --- | --- | --- |
| `NIM_ENABLE_PROMPT_UPSAMPLING` | `false` | Enable prompt upsampling |
| `NIM_PROMPT_UPSAMPLING_ENDPOINT_URL` | Empty | Set the OpenAI-compatible endpoint base or Chat Completions route |
| `NIM_PROMPT_UPSAMPLING_MODEL` | Empty | Set the external model name |
| `NIM_PROMPT_UPSAMPLING_API_KEY` | Empty | Set the external Bearer credential |
| `NIM_PROMPT_UPSAMPLING_TEMPLATE_STYLE` | `external_api` | Select `external_api` or `reasoner` templates |
| `NIM_PROMPT_UPSAMPLING_TIMEOUT_S` | `120` | Set the external call timeout in seconds |
| `NIM_PROMPT_UPSAMPLING_MAX_TOKENS` | `8192` | Set the requested output-token limit |
| `NIM_PROMPT_UPSAMPLING_TEMPERATURE` | Omitted | Optionally set external sampling temperature |
| `NIM_PROMPT_UPSAMPLING_TOP_P` | Omitted | Optionally set external nucleus sampling |
| `NIM_PROMPT_UPSAMPLING_TOP_K` | Omitted | Optionally set top-k when the provider accepts it |
| `NIM_PROMPT_UPSAMPLING_EXTRA_BODY` | Empty object | Merge additional JSON into the external request |

Endpoint, model, and key are required when the feature is enabled. A request-time
external failure logs a warning and continues with the original prompt. Keep
the external key separate from `NGC_API_KEY`.

## Reasoner configuration

Most of these are advanced workload controls. The unified-memory utilization
setting is required for Reasoner on DGX Spark and Jetson AGX Thor. Change other
controls one at a time and validate memory, latency, quality, and correctness.

### Speculative decoding

| Name | Default | Use |
| --- | --- | --- |
| `NIM_USE_DFLASH` | `true` for Reasoner | Use the bundled Nano or Super DFlash draft; set `false` to run the target model without speculative decoding |
| `NIM_DFLASH_MODEL_PATH` | Empty | Use an independent absolute local DFlash directory containing `config.json` and `model.safetensors` |
| `NIM_DFLASH_BF16_KV_CACHE` | `false` on Hopper; `true` otherwise | Override whether DFlash uses a BF16 KV cache instead of the profile-derived cache dtype |
| `NIM_DFLASH_CONFIG` | Empty object | Add or override advanced vLLM DFlash speculative-configuration fields as JSON |

DFlash does not change the Reasoner request API. Nano and Super Reasoner
profiles include variant-specific drafts and enable them by default; Generator
rejects an explicit DFlash enable. Confirm that the selected image contains the
draft artifact. `NIM_DFLASH_MODEL_PATH` accepts only an absolute local path,
not an `hf://` source. `NIM_DFLASH_CONFIG` requires DFlash to remain enabled and
cannot set the reserved `method` or `model` keys.

The BF16 KV-cache default is disabled on Hopper and enabled on every other
architecture. BF16 can improve DFlash compatibility and acceptance length, but
it uses more KV-cache memory. Treat the independent draft path, KV-cache
override, and `NIM_DFLASH_CONFIG` as advanced controls; measure memory,
correctness, latency, and throughput on the exact image before production use.

### Context and scheduling

| Name | Default | Use |
| --- | --- | --- |
| `NIM_MAX_MODEL_LEN` | `-1` (auto) | Let the runtime choose a context length bounded by the model |
| `NIM_MAX_NUM_BATCHED_TOKENS` | `16384` effective with the default unchunked multimodal input; otherwise `8192` | Set the scheduler token budget; unchunked multimodal input enforces a minimum of `16384` |
| `NIM_MAX_NUM_SEQS` | `256` | Set maximum scheduled sequences |
| `NIM_GPU_MEMORY_UTILIZATION` | `0.93` | Set the Reasoner GPU-memory target in `(0,1]`; startup warns when the target exceeds current free memory but does not reduce it. On DGX Spark/GB10 and Jetson AGX Thor, set `0.80` for image-only workloads or `0.70` for video or mixed-media workloads, and pass it from the first preflight through service launch. |

On DGX Spark and Jetson AGX Thor, this target is a share of the unified
host/device memory pool. Do not rely on the `0.93` default: it can leave too
little memory for the host and media processing. The complete deployment flow
sets and passes the value before the first applicable Docker command; see
[Set the Reasoner memory share on unified-memory
systems](deployment.md#set-the-reasoner-memory-share-on-unified-memory-systems).

### Caching and multimodal processing

| Name | Default | Use |
| --- | --- | --- |
| `NIM_ENABLE_KV_CACHE_REUSE` | `true` | Enable prefix/KV-cache reuse |
| `NIM_ENABLE_CHUNKED_PREFILL` | `true` | Enable chunked prefill |
| `NIM_DISABLE_CHUNKED_MM_INPUT` | `true` | Require one complete multimodal item to fit in a scheduler iteration; this raises `NIM_MAX_NUM_BATCHED_TOKENS` to at least `16384` |
| `NIM_DISABLE_MM_PREPROCESSOR_CACHE` | `false` | Disable the multimodal preprocessor cache |
| `NIM_MAX_IMAGES_PER_PROMPT` | Unset | Optionally set a nonnegative image limit; when unset, do not override the runtime limit |
| `NIM_MAX_VIDEOS_PER_PROMPT` | Unset | Optionally set a nonnegative video limit; when unset, do not override the runtime limit |
| `NIM_MEDIA_IO_KWARGS` | Video backend `pynvvc` | Replace the complete operator-level media I/O object with a JSON object |
| `NIM_MM_PROCESSOR_KWARGS` | Unset | Set operator-level multimodal processor options as a JSON object |
| `NIM_VIDEO_PRUNING_RATE` | `0` (disabled) | Set video-token pruning from 0 through 1; values greater than 0 enable pruning |
| `NIM_VIDEO_PRUNING_METHOD` | `vidcom2` | Select `vidcom2` or `evs` when pruning is enabled |

Prefer request-level `media_io_kwargs` for one workload rather than changing the
operator-wide media object. Leave `NIM_MM_PROCESSOR_KWARGS` unset unless the
selected runtime has been validated with the complete processor-options object.
Both operator-level options require JSON objects. Disabling multimodal-input
chunking is the default and can increase peak GPU memory use because one
complete item must fit in a scheduler iteration; set
`NIM_DISABLE_CHUNKED_MM_INPUT=0` only after validating memory, latency, and
throughput. Leave Reasoner GPU-memory headroom at its default unless system
measurements establish another safe reserve; reducing it increases startup and
runtime OOM risk.

### API behavior

| Name | Default | Use |
| --- | --- | --- |
| `NIM_GUIDED_DECODING_BACKEND` | `xgrammar` | Select the structured-output backend |
| `NIM_DISABLE_LOG_REQUESTS` | `true` | Keep Reasoner request bodies out of logs |
| `NIM_DISABLE_RESPONSES_ROUTE` | `false` | Remove Responses routes when set to `true` |

## Secret handling

Keep `NGC_API_KEY`, `HF_TOKEN`, and `NIM_PROMPT_UPSAMPLING_API_KEY` separate.
Inject them through the deployment environment. Do not put them in source
control, model-source URIs, image layers, requests, notebooks, or logs.
