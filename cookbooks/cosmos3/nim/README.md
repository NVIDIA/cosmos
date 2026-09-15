<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM

Deploy Cosmos3 models for world generation and multimodal reasoning. One image
contains two runtimes, but each container runs only one:

- **Generator** serves image/video generation, Action, and Transfer through
  `POST /v1/infer`.
- **Reasoner** serves OpenAI-compatible image/video understanding through Chat
  Completions and, when enabled, the Responses API.

> Deployment commands pin the current release image,
> `nvcr.io/nim/nvidia/cosmos3:2.0.0`. The [Support matrix](support-matrix.md)
> publishes the release profile-selection floors;
> confirm an available row in the running image's manifest. A matching profile
> establishes a candidate configuration, while cold start and representative
> requests establish host and workload compatibility.

## Choose what to run

Normal deployment does not require choosing a profile ID. Make a few
user-facing choices and let the NIM select the best compatible profile for the
visible GPUs.

1. Choose **Generator** or **Reasoner**.
2. Choose the model variant.
3. Optionally pin a precision. Generator prefers FP8 when compatible;
   Reasoner prefers BF16 on compute capability 8.0 through 8.8, FP8 on 8.9
   through 9.x, and NVFP4 on 10.0 or newer when the matching profile fits.
4. For Generator, choose **latency** or **throughput**.
5. Start the container. The NIM selects a compatible profile for the model,
   precision preference, performance objective, and host.
6. For each Generator request, set the explicit top-level `model_mode` for the
   task.

`NIM_MODEL_VARIANT` chooses the checkpoint at startup; request `model_mode`
chooses T2I, T2V, I2V, V2V, Transfer, or an Action operation.

### Generator variants

| Variant | Supported use |
| --- | --- |
| `nano` | General-purpose generation and compatible Generator tasks |
| `nano-droid` | DROID policy with an action-only response; BF16 only |
| `super` | General-purpose generation and compatible Generator tasks |
| `super-t2i` | Text-to-image only |
| `super-t2i-4step` | Four-step text-to-image only |
| `super-i2v` | Image-to-video only |
| `super-i2v-4step` | Four-step image-to-video only |

Select a Generator model with `NIM_MODEL_VARIANT`. For Generator, the variant
also determines Nano versus Super. Select `NIM_PERF_PROFILE=latency` to
prioritize individual request latency or `throughput` to prioritize aggregate
request rate. Latency is the software default, but make the choice explicit in
deployment automation.

Reasoner provides `nano` and `super`, also selected with
`NIM_MODEL_VARIANT`. Reasoner does not use `NIM_PERF_PROFILE`.

If `NIM_PRECISION` is omitted, Generator prefers FP8 when compatible.
Reasoner derives its preference from GPU compute capability: BF16 for 8.0
through 8.8, FP8 for 8.9 through 9.x, and NVFP4 for 10.0 or newer. Selection
uses another compatible precision when the preferred row is unavailable. Pin
precision only when the workload requires it.

Exact profile IDs, low-level tags, and offload overrides are advanced controls.
See [Deploy the NIM](deployment.md#advanced-profile-controls).

## Choose a task

| Task | Runtime | Input | Output | Guide |
| --- | --- | --- | --- | --- |
| Text-to-image | Generator | Prompt | JPEG image | [Generation](generation.md#text-to-image) |
| Text-to-video | Generator | Prompt | MP4 video | [Generation](generation.md#text-to-video) |
| Image-to-video | Generator | Prompt + image | MP4 video | [Generation](generation.md#image-to-video) |
| Video-to-video | Generator | Prompt + video | MP4 video | [Generation](generation.md#video-to-video) |
| Forward dynamics | Generator | Image + action trajectory | Rollout video | [Action](action.md#forward-dynamics) |
| Policy | Generator | Image + task/state | Video + predicted action | [Action](action.md#policy) |
| Nano-DROID policy | Generator | Image + task/current state | Predicted action | [Action](action.md#nano-droid-policy) |
| Inverse dynamics | Generator | Video | Video + predicted action | [Action](action.md#inverse-dynamics) |
| Video Transfer | Generator | Prompt + spatial control | Controlled MP4 video | [Transfer](transfer.md) |
| Image/video reasoning | Reasoner | Media + text | Text or structured result | [Reasoning](reasoning.md) |
| Responses API | Reasoner | Responses input | Response object/text | [Reasoning](reasoning.md#use-the-responses-api) |

The public Generator request model does not expose image-to-image or sound
generation. Local examples publish either selected runtime at
`http://localhost:8000`; remove the active example container before launching the other
runtime on the same host port.

## Get started

### Use an existing NIM endpoint

If an administrator already deployed the NIM, install and initialize the
[client tooling](prerequisites.md#client-tooling), then set the service URL:

```bash
cd cookbooks/cosmos3/nim
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -fsS "$NIM_URL/v1/health/ready"
curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
```

Confirm that metadata identifies the intended runtime and primary endpoint:
Generator uses `/v1/infer`; Reasoner uses `/v1/chat/completions`.

Run an example that matches the active runtime and model. For a general-purpose
Generator, start with an image request:

```bash
uv run python examples/t2i.py
```

For a Reasoner:

```bash
uv run python examples/reasoner.py --case image_caption
```

### Deploy the NIM yourself

1. Verify the GPU host against [Prerequisites](prerequisites.md).
2. Choose a compatible model and hardware configuration from the
   [Support matrix](support-matrix.md).
3. For a Reasoner on DGX Spark or Jetson AGX Thor, choose the documented
   [unified-memory share](deployment.md#set-the-reasoner-memory-share-on-unified-memory-systems)
   before the first preflight command.
4. Follow [Deployment](deployment.md) to authenticate, pull the image, set any
   required unified-memory share, run the pre-download profile preflight,
   prepare the cache, and launch either Generator or Reasoner.
5. Return to [Use an existing NIM endpoint](#use-an-existing-nim-endpoint) to
   verify the service and run the first request.

Specialist models accept only their documented task. Use
[`t2i_4step.py`](examples/t2i_4step.py),
[`i2v_4step.py`](examples/i2v_4step.py), or the
[Nano-DROID request](action.md#nano-droid-policy) only after launching the
matching variant.

Generator examples save decoded media and action JSON under
`examples/outputs/`. The Reasoner task runner prints its final answer and saves
request metadata, the raw response, final text, and validated JSON for
structured cases under the same ignored output directory.

## AI-assisted usage

Open this directory in a compatible coding assistant to get guided help with
host preparation, deployment, endpoint verification, request selection, and
troubleshooting. The bundled customer instructions and skill use the public
pages and examples as their source of truth and default to read-only assistance.

## Documentation

### Deploy and configure

- [Prerequisites](prerequisites.md)
- [Deployment](deployment.md)
- [Configuration](configuration.md)
- [Support matrix](support-matrix.md)
- [Helm deployment status](helm.md)
- [Bring your own checkpoint](bring-your-own-checkpoint.md)

### Use the APIs

- [API reference](api-reference.md)
- [Generation](generation.md)
- [Reasoning](reasoning.md)
- [Action](action.md)
- [Transfer](transfer.md)
- [Python examples](examples/)

### Operate

- [Operations and troubleshooting](operations.md)
- [Release notes](release-notes.md)
- [Acknowledgements](acknowledgements.md)

## Safety, license, and notices

Generator guardrails are enabled by default. Disabling them can remove
content-policy and face-privacy protections; see
[Guardrails](operations.md#guardrails).

This cookbook is licensed under the repository
[LICENSE](../../../LICENSE). The running NIM exposes bundled product license
information at `/v1/license`. The image also contains product terms, notices,
component licenses, and the included package-modification source described
under [Acknowledgements](acknowledgements.md).
