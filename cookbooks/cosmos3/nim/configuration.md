<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Configure the Cosmos3 NIM

This page lists launch-time environment variables. Start with the
model-selection variables and keep automatic defaults unless the workload
requires an override. See [Deployment](deployment.md) for complete Docker
commands.

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
| `NIM_PRECISION` | Generator: FP8 preference; Reasoner: GPU-derived preference | Optionally pin a precision available in the image; see [Precision selection](deployment.md#precision-selection) |
| `NIM_PERF_PROFILE` | `latency` | Generator only: choose `latency` or `throughput` |

Reasoner rejects `NIM_PERF_PROFILE`, including `profile=` in
`NIM_TAGS_SELECTOR`, with a startup error. Nano-DROID has BF16 profiles only.
The NIM chooses the best compatible profile for these settings and the visible
GPUs; a normal deployment does not need a profile ID.

### GPU selection

The GPUs exposed by Docker `--gpus` are the visible GPUs that automatic
selection considers. List host GPUs with their indexes and UUIDs:

```bash
nvidia-smi -L
```

Expose specific GPUs instead of `--gpus all`, for example
`--gpus '"device=0,1"'` or `--gpus '"device=GPU-<uuid>"'`. Expose a GPU count
listed in the [Generator](support-matrix.md#generator-configurations) or
[Reasoner](support-matrix.md#reasoner-configurations) configurations, all of a
single model ([mixed GPUs are not
supported](support-matrix.md#gpu-architecture-and-topology)); extra GPUs sit
idle.

Do not pass `-e CUDA_VISIBLE_DEVICES`: the NIM sets it for its worker
processes, so a passed value can place work outside the intended GPUs.

### Bring your own checkpoint

| Name | Default | Use |
| --- | --- | --- |
| `NIM_MODEL_PATH` | Empty | Generator: absolute local directory. Reasoner: absolute local directory or `hf://owner/repository[:revision]` |
| `NIM_DISABLE_MODEL_DOWNLOAD` | `false` | Disable profile download for a completely local Reasoner override; incompatible with Reasoner `hf://` and rejected for Generator |
| `HF_TOKEN` | Empty | Authenticate to a private Reasoner Hugging Face repository |

To override the DFlash draft, see `NIM_DFLASH_MODEL_PATH` under [Speculative
decoding](#speculative-decoding). See [Bring your own
checkpoint](bring-your-own-checkpoint.md) for layouts, mounts, and
compatibility checks.

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
| `NIM_TAGS_SELECTOR` | Empty | Filter by comma-separated exact manifest tags as `key=value` pairs; see the valid keys below |

`NIM_TAGS_SELECTOR` accepts only these keys; any other key is an error. Values
are lowercased.

| Key | Meaning | Runtime |
| --- | --- | --- |
| `model_type`, `model_variant`, `precision` | Runtime, model, and precision | Both |
| `profile` | `latency` or `throughput` | Generator |
| `offload`, `text_guard_offload`, `video_guard_offload` | Offload modes | Generator |
| `n_gpus` | GPUs used by one NIM; the product of `nim_dp`, `nim_gp`, `nim_up`, and `nim_tp` | Both |
| `nim_dp` | Data-parallel model replicas; throughput profiles use `n_gpus` | Generator |
| `nim_gp` | CFG (guidance) parallel factor; latency profiles on 2 or more GPUs use `2` | Generator |
| `nim_up` | Ulysses sequence-parallel factor; latency profiles use `n_gpus / 2` | Generator |
| `nim_tp` | Tensor-parallel factor; Super BF16 also has a 2-GPU `nim_tp=2` layout | Reasoner |

`gp`, `cp`, and `dp` are accepted as aliases for `nim_gp`, `nim_up`, and
`nim_dp`.

A shorthand variable that conflicts with `NIM_TAGS_SELECTOR` is a startup
error; use the `model_variant` tag to filter by model. Exact tags and profile
IDs are tied to an image release. For the offload, memory-reserve, and
profile-pin variables, see [Advanced settings](#advanced-settings).

## Generator configuration

### Input and output

| Name | Default | Use |
| --- | --- | --- |
| `NIM_ALLOW_URL_INPUT` | `true` | Allow HTTP(S) image, video, and Transfer inputs; set `false` to require encoded input |
| `NIM_TRITON_REQUEST_TIMEOUT` | 30 minutes (`1800000000` microseconds) | Set the queue-plus-execution timeout in microseconds |

Client HTTP timeouts do not change this backend ceiling; see [Long-running
requests](operations.md#long-running-requests).

### Startup and execution

| Name | Default | Use |
| --- | --- | --- |
| `NIM_ENABLE_WARMUP` | `false` | Run synthetic inference before readiness at the `256_1_1`, `480_16_9`, and `720_16_9` resolutions; Nano-DROID uses only `480_16_9`, and 4-step variants use their fixed step count |

### Guardrails

| Name | Default | Use |
| --- | --- | --- |
| `NIM_ENABLE_TEXT_GUARDRAILS` | `true` | Enable input-prompt policy checks |
| `NIM_ENABLE_VIDEO_GUARDRAILS` | `true` | Enable output image/video face and visual guardrails |
| `NIM_ENABLE_SIGLIP_GUARDRAILS` | `true` | Enable the output-frame safety classifier |

See [Guardrails](operations.md#guardrails) for what each layer does.

### Prompt upsampling

Prompt upsampling is optional and applies only to Generator T2I, T2V, and I2V.
It sends the request to an operator-provided OpenAI-compatible endpoint.

| Name | Default | Use |
| --- | --- | --- |
| `NIM_ENABLE_PROMPT_UPSAMPLING` | `false` | Enable prompt upsampling |
| `NIM_PROMPT_UPSAMPLING_ENDPOINT_URL` | Empty | Set the OpenAI-compatible endpoint base or Chat Completions route |
| `NIM_PROMPT_UPSAMPLING_MODEL` | Empty | Set the external model name |
| `NIM_PROMPT_UPSAMPLING_API_KEY` | Empty | Set the external Bearer credential |
| `NIM_PROMPT_UPSAMPLING_TIMEOUT_S` | `120` | Set the external call timeout in seconds |
| `NIM_PROMPT_UPSAMPLING_TEMPLATE_STYLE`, `NIM_PROMPT_UPSAMPLING_MAX_TOKENS`, `NIM_PROMPT_UPSAMPLING_TEMPERATURE`, `NIM_PROMPT_UPSAMPLING_TOP_P`, `NIM_PROMPT_UPSAMPLING_TOP_K`, `NIM_PROMPT_UPSAMPLING_EXTRA_BODY` | `external_api`, `8192`, omitted, omitted, omitted, empty object | Select `external_api` or `reasoner` templates; set the output-token limit, sampling temperature, nucleus sampling, and top-k (when the provider accepts it); merge additional JSON into the external request |

Endpoint, model, and key are required when the feature is enabled. A request-time
external failure logs a warning and continues with the original prompt.

## Reasoner configuration

On DGX Spark and Jetson AGX Thor, `NIM_GPU_MEMORY_UTILIZATION` is required;
change other controls one at a time and measure the effect.

### Speculative decoding

| Name | Default | Use |
| --- | --- | --- |
| `NIM_USE_DFLASH` | `true` for Reasoner | Use the bundled Nano or Super DFlash draft; set `false` to run the target model without speculative decoding |
| `NIM_DFLASH_MODEL_PATH` | Empty | Nano or Super Reasoner: use an independent absolute local DFlash directory containing `config.json` and `model.safetensors`; see [Local directory](bring-your-own-checkpoint.md#local-directory) for the mount |

DFlash is on by default for Reasoner and does not change the request API; set
`NIM_USE_DFLASH=0` to disable it (Generator rejects an explicit enable).
`NIM_DFLASH_MODEL_PATH` accepts only an absolute local path, not `hf://`, and
`NIM_DFLASH_CONFIG` requires DFlash enabled and cannot set the `method` or
`model` keys.

### Context and scheduling

| Name | Default | Use |
| --- | --- | --- |
| `NIM_MAX_MODEL_LEN` | `-1` (auto) | Let the runtime choose a context length, bounded by the checkpoint limit of 262,144 tokens |
| `NIM_GPU_MEMORY_UTILIZATION` | `0.93` | Set the Reasoner GPU-memory target in `(0,1]`; startup warns when it exceeds current free memory but does not reduce it. On DGX Spark/GB10 and Jetson AGX Thor, set `0.80` for image-only or `0.70` for video or mixed-media workloads from the first preflight on; see [Set the Reasoner memory share](deployment.md#set-the-reasoner-memory-share-on-unified-memory-systems) |

### Caching and multimodal processing

| Name | Default | Use |
| --- | --- | --- |
| `NIM_MAX_IMAGES_PER_PROMPT` | Unset | Optionally set a nonnegative image limit; when unset, do not override the runtime limit |
| `NIM_MAX_VIDEOS_PER_PROMPT` | Unset | Optionally set a nonnegative video limit; when unset, do not override the runtime limit |

Prefer request-level `media_io_kwargs` over the operator-wide
`NIM_MEDIA_IO_KWARGS`. Unchunked multimodal input, the default, can raise peak
GPU memory; set `NIM_DISABLE_CHUNKED_MM_INPUT=0` only after measuring memory,
latency, and throughput.

### API behavior

| Name | Default | Use |
| --- | --- | --- |
| `NIM_DISABLE_LOG_REQUESTS` | `true` | Keep Reasoner request bodies out of logs |
| `NIM_DISABLE_RESPONSES_ROUTE` | `false` | Remove Responses routes when set to `true` |

## Advanced settings

Leave these at their defaults unless you have measured a need.

| Name | Runtime | Default | Use |
| --- | --- | --- | --- |
| `NIM_OFFLOAD_MODE` | Generator | Automatic | Request `none`, `model`, or `layer` when the model provides it |
| `NIM_UNIFIED_MEMORY_HOST_RESERVE_GIB` | Both | `16` | Binary GiB reserved for the host before profile selection on integrated GPUs; no effect on discrete GPUs |
| `NIM_GPU_MEMORY_HEADROOM_GIB` | Reasoner | `2` | Free binary GiB reserved per discrete GPU for selection and sizing; lowering it raises OOM risk |
| `NIM_MODEL_PROFILE` | Both | Empty | Pin an exact profile ID from the current image |
| `NIM_VIDEO_SAVE_QUALITY` | Generator | `7` | VP9 output quality 1-9; affects encoding, not diffusion |
| `NIM_ENABLE_TORCH_COMPILE` | Generator | `true` | Enable the compilation path |
| `NIM_LINEAR_BACKEND` | Generator | Quantized: `cutlass`; BF16: unset | `auto`, `cutlass`, `flashinfer_cutlass`, `flashinfer_cutedsl`, or `torch` for quantized DiT linear layers; an incompatible backend fails at model load |
| `NIM_TRITON_LOG_VERBOSE` | Generator | `0` | Backend log verbosity for diagnosis |
| `NIM_MAX_SEQUENCE_LENGTH` | Generator | `5120` | Startup prompt-token sequence length |
| `NIM_OFFLOAD_TEXT_GUARDRAIL` | Generator | Profile policy | Override whether the text guard sleeps on CPU during diffusion |
| `NIM_OFFLOAD_VIDEO_GUARDRAIL` | Generator | Profile policy | Override whether output guardrails sleep during diffusion |
| `NIM_ALLOW_UNSAFE_TRANSFER` | Generator | `false` | Bypass Transfer's VRAM admission check for an approved diagnostic; can fail with OOM |
| `NIM_DFLASH_BF16_KV_CACHE` | Reasoner | `false` on Hopper; `true` otherwise | BF16 DFlash KV cache; can improve acceptance but uses more memory |
| `NIM_DFLASH_CONFIG` | Reasoner | Empty object | Extra vLLM DFlash speculative-config fields as JSON |
| `NIM_MAX_NUM_BATCHED_TOKENS` | Reasoner | `16384` with unchunked multimodal input; otherwise `8192` | Scheduler token budget; unchunked multimodal input enforces at least `16384` |
| `NIM_MAX_NUM_SEQS` | Reasoner | `256` | Maximum scheduled sequences |
| `NIM_ENABLE_KV_CACHE_REUSE` | Reasoner | `true` | Prefix/KV-cache reuse |
| `NIM_ENABLE_CHUNKED_PREFILL` | Reasoner | `true` | Chunked prefill |
| `NIM_DISABLE_CHUNKED_MM_INPUT` | Reasoner | `true` | Require one whole multimodal item per scheduler iteration |
| `NIM_DISABLE_MM_PREPROCESSOR_CACHE` | Reasoner | `false` | Disable the multimodal preprocessor cache |
| `NIM_MEDIA_IO_KWARGS` | Reasoner | Video backend `pynvvc` | Replace the whole operator-level media I/O JSON object |
| `NIM_MM_PROCESSOR_KWARGS` | Reasoner | Unset | Operator-level multimodal processor options as a JSON object |
| `NIM_VIDEO_PRUNING_RATE` | Reasoner | `0` (disabled) | Video-token pruning from 0 through 1; greater than 0 enables it |
| `NIM_VIDEO_PRUNING_METHOD` | Reasoner | `vidcom2` | `vidcom2` or `evs` when pruning is enabled |
| `NIM_GUIDED_DECODING_BACKEND` | Reasoner | `xgrammar` | Structured-output backend |

## Secret handling

Inject `NGC_API_KEY`, `HF_TOKEN`, and `NIM_PROMPT_UPSAMPLING_API_KEY` as
separate secrets through the deployment environment, never through source
control, model-source URIs, image layers, requests, notebooks, or logs.
