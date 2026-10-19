<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 NIM

Deploy Cosmos3 models for world generation and multimodal reasoning. One image
contains two runtimes, but each container runs only one:

- **Generator** serves image/video generation, Action, and Transfer through
  `POST /v1/infer`.
- **Reasoner** serves OpenAI-compatible image/video understanding through Chat
  Completions and, when enabled, the Responses API.

> Deployment commands pin the current release image,
> `nvcr.io/nim/nvidia/cosmos3:2.0.0`; compatible configurations are in the
> [Support matrix](support-matrix.md).

## Architecture

Cosmos3 is a Mixture-of-Transformers (MoT) world foundation model with two
jointly trained towers that share latent representations: a **Reasoner**
(vision-language) tower that grounds the prompt, and a **Generator** (world
simulator) tower that produces the output, such as video. The Generator runtime
uses both towers; the Reasoner runtime serves understanding. For details, see
the
[Cosmos3 technical report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf).

![Cosmos3 Mixture-of-Transformers architecture with Reasoner and Generator towers](../cosmos3-model-architecture.png)

## Choose what to run

Choose a runtime, a model variant (`NIM_MODEL_VARIANT`), optionally a
precision, and for Generator a latency or throughput objective; the NIM selects
a compatible profile, as described in
[How selection works](deployment.md#how-selection-works).

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

Local examples publish either runtime at `http://localhost:8000`; remove the
active example container before launching the other runtime on the same port.

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
3. Follow [Deployment](deployment.md) to authenticate, pull the image, set the
   [unified-memory share](deployment.md#set-the-reasoner-memory-share-on-unified-memory-systems)
   for a Reasoner on DGX Spark or Jetson AGX Thor, run the preflight, and
   launch Generator or Reasoner.
4. Return to [Use an existing NIM endpoint](#use-an-existing-nim-endpoint) to
   verify the service and run the first request.

Specialist variants accept only their task: run
[`t2i_4step.py`](examples/t2i_4step.py),
[`i2v_4step.py`](examples/i2v_4step.py), or the
[Nano-DROID request](action.md#nano-droid-policy) only against the matching
variant. Examples save outputs under the ignored `examples/outputs/` directory.

Validate the selected configuration with representative requests before
production use.

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
- [Helm deployment](helm.md)
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
- [Monitoring with Prometheus and Grafana](monitoring.md)
- [Release notes](release-notes.md)
- [License and notices](license.md)
- [Acknowledgements](acknowledgements/README.md)

## Safety, license, and notices

Generator guardrails are enabled by default. Disabling them can remove
content-policy and face-privacy protections; see
[Guardrails](operations.md#guardrails).

See [License and notices](license.md) for the terms. This cookbook is licensed
under the repository [LICENSE](../../../LICENSE). The running NIM serves its
license at `/v1/license`; the image also contains product terms, notices,
component licenses, and the package-modification source described in
[Acknowledgements](acknowledgements/README.md).

The NIM is distributed as a container image through the NVIDIA NGC Catalog,
which publishes a security scan report for each image: a security rating, CVE
severity by package, and links to CVE details.

## NVIDIA Developer Program

Want to learn more about NIMs? Join the
[NVIDIA Developer Program](https://developer.nvidia.com/join-nvidia-developer-program)
to get free access to self-hosting NVIDIA NIMs and microservices on up to 16
GPUs on any infrastructure: cloud, data center, or personal workstation.

Once you join the free NVIDIA Developer Program, you can access NIMs through the
[NVIDIA API Catalog](https://build.nvidia.com/explore/discover) at any time. For
enterprise-grade security, support, and API stability, select the option to
access NIM through our free 90-day NVIDIA AI Enterprise Trial with a business
email address.

Refer to the
[NVIDIA NIM FAQ](https://forums.developer.nvidia.com/t/nvidia-nim-faq/300317)
for additional information.
