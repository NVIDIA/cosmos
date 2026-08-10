<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM support matrix

Use this page to match a model to precision, GPU compute capability, GPU count,
per-device VRAM, and system-memory requirements.

> **Pre-release status:** These configurations are available for evaluation and
> may change before the public release. Confirm the selected configuration in
> `/v1/manifest` for the image you run. Final supported configurations will be
> published with the release.

## Choose a model

Choose the model first. The NIM then selects a compatible profile for the
visible host.

| Runtime | Model | Supported use | Automatic precision behavior |
| --- | --- | --- | --- |
| Generator | `nano` | General-purpose Generator tasks included by the image | Prefer FP8 when compatible; otherwise fall back |
| Generator | `nano-droid` | DROID policy with action-only output | BF16 only |
| Generator | `super` | General-purpose Generator tasks included by the image | Prefer FP8 when compatible; otherwise fall back |
| Generator | `super-t2i` | Full-step T2I only | Prefer FP8 when compatible; otherwise fall back |
| Generator | `super-t2i-4step` | Four-step T2I only | Prefer FP8 when compatible; otherwise fall back |
| Generator | `super-i2v` | Full-step I2V only | Prefer FP8 when compatible; otherwise fall back |
| Generator | `super-i2v-4step` | Four-step I2V only | Prefer FP8 when compatible; otherwise fall back |
| Reasoner | `nano` | Image/video reasoning | Prefer FP8 when compatible; BF16 and explicit NVFP4 rows also exist |
| Reasoner | `super` | Image/video reasoning | Prefer FP8 when compatible; BF16 and explicit NVFP4 rows also exist |

A Generator specialist rejects requests for other tasks. Confirm that
the selected image includes the model before deployment.

Nano Reasoner can optionally use DFlash speculative decoding. The selected
profile must include the draft artifact; Generator and Super Reasoner do not
support `NIM_USE_DFLASH=1`.

## GPU architecture and topology

Compatibility uses CUDA compute capability rather than a GPU SKU allowlist:

| Runtime | Precision | Minimum compute capability |
| --- | --- | ---: |
| Generator | BF16 | 8.7 |
| Generator | FP8 | 8.9 |
| Reasoner | BF16 | 8.0 |
| Reasoner | FP8 | 8.9 |
| Reasoner | NVFP4 | 10.0 |

Active Generator profiles use BF16 or FP8; there is no active Generator NVFP4
row.

Reasoner requires all visible GPUs to have the same compute capability.
Generator eligibility is calculated from the least-capable GPU and the smallest
per-device memory total. Use homogeneous GPUs for either runtime; mixed-GPU
configurations are not supported for pre-release evaluation.

All memory values below are binary GiB per device. Do not add VRAM across
devices to satisfy a per-device floor.

For an integrated GPU with unified host/device memory, automatic selection
subtracts a 16-GiB host reserve from the reported shared-memory total before
applying these floors. Generator selection keeps the model and both guardrails
resident, so only rows with **Model offload = None** and **Guardrails =
Resident** are eligible. Reasoner floors are also compared with the remaining
shared-memory total.

## Generator configurations

The **Super family** in this table means `super`, `super-t2i`,
`super-t2i-4step`, `super-i2v`, and `super-i2v-4step`. Transfer thresholds apply
only to the general-purpose `nano` and `super` variants.

| Variant group | Precision | Model offload | Guardrails during diffusion | GPUs | Generation minimum VRAM/device | Minimum effective system RAM | Transfer minimum VRAM/device |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `nano` | BF16 | None | Resident | 1, 2, 4, 8 | 58 GiB | No additional floor | 64 GiB |
| `nano` | BF16 | Layer | Automatic offload | 1 | 31 GiB | No additional floor | 35 GiB |
| `nano` | FP8 | None | Resident | 1, 2, 4, 8 | 44 GiB | No additional floor | 50 GiB |
| `nano` | FP8 | Model | Resident | 1 | 38 GiB | No additional floor | 44 GiB |
| `nano` | FP8 | Layer | Resident | 1 | 32 GiB | No additional floor | 38 GiB |
| `nano` | FP8 | Layer | Automatic offload | 1 | 31 GiB | No additional floor | 35 GiB |
| `nano-droid` | BF16 | None | Resident | 1, 2, 4, 8 | 58 GiB | No additional floor | N/A |
| Super family | BF16 | None | Resident | 1, 2, 4, 8 | 150 GiB | No additional floor | Base `super`: 160 GiB |
| Super family | BF16 | Model | Resident | 1 | 94 GiB | 150 GiB | Base `super`: 100 GiB |
| Super family | BF16 | Layer | Resident | 1 | 42 GiB | 150 GiB | Base `super`: 50 GiB |
| Super family | FP8 | None | Resident | 1, 2, 4, 8 | 94 GiB | No additional floor | Base `super`: 104 GiB |
| Super family | FP8 | Model | Resident | 1 | 64 GiB | No additional floor | Base `super`: 76 GiB |
| Super family | FP8 | Layer | Resident | 1 | 38 GiB | No additional floor | Base `super`: 50 GiB |

The 31-GiB Nano layer-offload configurations are intended to support ordinary
generation on 32-GB client GPUs such as the NVIDIA GeForce RTX 5090. This is not
a GPU SKU allowlist: the device must expose at least 31 binary GiB to the
runtime and meet the precision's compute-capability requirement. The RTX 5090
does not meet the 35-GiB Transfer minimum for these configurations.

A configuration without a listed system-memory floor still needs enough RAM
for the container, runtime, artifact materialization, and offloaded weights; a
general minimum is not yet available. The 150-GiB requirement is an explicit
startup gate for every Super-family BF16 model- and layer-offload configuration.
The NIM checks a container memory limit before host physical memory.

Leave offload and guardrail residency on automatic selection unless a specific
configuration has been validated for the deployment.

## Reasoner configurations

Reasoner does not use Generator latency/throughput or model-offload selectors:

| Model | Precision | GPUs | Tensor parallelism | Minimum VRAM/device | Minimum compute capability |
| --- | --- | ---: | ---: | ---: | ---: |
| `nano` | BF16 | 1 | 1 | 23.1 GiB | 8.0 |
| `nano` | FP8 | 1 | 1 | 23.1 GiB | 8.9 |
| `nano` | NVFP4 | 1 | 1 | 23.1 GiB | 10.0 |
| `super` | BF16 | 1 | 1 | 135 GiB | 8.0 |
| `super` | BF16 | 2 | 2 | 46 GiB | 8.0 |
| `super` | FP8 | 1 | 1 | 67 GiB | 8.9 |
| `super` | NVFP4 | 1 | 1 | 73 GiB | 10.0 |

When at least two compatible GPUs are visible for Super BF16, profile ranking
prefers the two-GPU TP2 layout. Nano Reasoner profiles include the DFlash draft
artifact, but DFlash remains opt-in.

## Automatic profile selection

Users normally set:

- runtime;
- model variant;
- Generator latency or throughput; and
- precision only when it must be pinned.

The NIM finds a compatible profile for those choices and the visible GPUs.
Automatic selection prefers FP8 when available, avoids offload when the model
fits normally, and prefers the largest compatible GPU layout. If no profile
fits, startup fails rather than selecting an incompatible combination.

Exact profile IDs and low-level manifest tags are advanced image-specific
controls. Do not copy them between images or hosts.

## Transfer headroom

Transfer can require more peak memory than ordinary generation on the same
model. Startup compares the selected profile's per-device floor and Transfer
overhead with the detected GPU memory. A deployment can therefore serve
ordinary generation while rejecting Transfer.

Use a larger GPU or a lower-memory configuration when Transfer does not fit.
`NIM_ALLOW_UNSAFE_TRANSFER=1` bypasses the check for diagnosis but can cause an
out-of-memory failure and does not make the deployment supported.

## Media and codecs

| Direction | Media | Pre-release formats/codecs and limits |
| --- | --- | --- |
| Input | Images | Final inventory not yet available; use the included fixtures |
| Input | Videos | Final inventory not yet available; use the included fixtures |
| Output | Generator image | JPEG in the pre-release version |
| Output | Generator video | VP9 in MP4 in the pre-release version |

The request schemas accept base64 and MIME-aware data URLs. HTTP(S) input is
available only when enabled and reachable from the container. Schema acceptance
does not guarantee every image format, video container, codec, chroma format,
frame rate, or remote source.

## Verify a running deployment

```bash
curl -fsS http://localhost:8000/v1/metadata | python3 -m json.tool
curl -fsS http://localhost:8000/v1/manifest | python3 -m json.tool
```

The selected image's manifest identifies the configurations it contains.
Metadata confirms the configuration chosen for the running container.
