<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM support matrix

Use this page to match a model to precision, GPU compute capability, GPU count,
per-device VRAM, and system-memory requirements.

> **Published release floors:** The tables below define the release
> profile-selection floors for GPU compute capability, GPU count, per-device
> VRAM, effective system memory, and Transfer headroom. Confirm that a matching
> configuration exists in `/v1/manifest` for the exact image you run. Meeting a
> floor establishes a candidate profile, not full host compatibility; see
> [Prerequisites](prerequisites.md#hardware-requirements).

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

Nano and Super Reasoner profiles include variant-specific DFlash drafts and use
them by default. Set `NIM_USE_DFLASH=0` to run either target model without
speculative decoding. Generator does not support DFlash.

## GPU architecture and topology

Compatibility uses CUDA compute capability, per-device VRAM, and effective
system RAM rather than a GPU SKU allowlist:

| Runtime | Precision | Minimum compute capability |
| --- | --- | ---: |
| Generator | BF16 | 8.0 |
| Generator | FP8 | 8.9 |
| Reasoner | BF16 | 8.0 |
| Reasoner | FP8 | 8.9 |
| Reasoner | NVFP4 | 10.0 |

Active Generator profiles use BF16 or FP8; there is no active Generator NVFP4
row.

Reasoner requires all visible GPUs to have the same compute capability.
Generator eligibility is calculated from the least-capable GPU and the smallest
per-device memory total. Use homogeneous GPUs for either runtime; mixed-GPU
configurations are not supported.

All memory values below are binary GiB per device. Do not add VRAM across
devices to satisfy a per-device floor. Static compatibility first checks total
VRAM. At startup, the NIM captures free memory once and requires each GPU used
by the selected layout to meet the same floor. For Reasoner, usable free memory
is measured after a runtime reserve: 2 GiB by default on a discrete GPU or the
configured host reserve on an integrated GPU.

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
| `nano` | BF16 | None | Resident | 1, 2, 4, 8 | 58 GiB | 16 GiB | 64 GiB |
| `nano` | BF16 | Layer | Offloaded | 1 | 31 GiB | 64 GiB | 35 GiB |
| `nano` | FP8 | None | Resident | 1, 2, 4, 8 | 44 GiB | 16 GiB | 50 GiB |
| `nano` | FP8 | Model | Resident | 1 | 38 GiB | 64 GiB | 44 GiB |
| `nano` | FP8 | Layer | Resident | 1 | 32 GiB | 64 GiB | 38 GiB |
| `nano` | FP8 | Layer | Offloaded | 1 | 31 GiB | 64 GiB | 35 GiB |
| `nano-droid` | BF16 | None | Resident | 1, 2, 4, 8 | 58 GiB | 16 GiB | N/A |
| Super family | BF16 | None | Resident | 1, 2, 4, 8 | 150 GiB | 16 GiB | Base `super`: 160 GiB |
| Super family | BF16 | Model | Resident | 1 | 93 GiB | 150 GiB | Base `super`: 99 GiB |
| Super family | BF16 | Layer | Resident | 1 | 42 GiB | 150 GiB | Base `super`: 50 GiB |
| Super family | FP8 | None | Resident | 1, 2, 4, 8 | 93 GiB | 16 GiB | Base `super`: 103 GiB |
| Super family | FP8 | Model | Resident | 1 | 64 GiB | 150 GiB | Base `super`: 76 GiB |
| Super family | FP8 | Layer | Resident | 1 | 38 GiB | 150 GiB | Base `super`: 50 GiB |

The 31-GiB Nano layer-offload configurations are intended to support ordinary
generation on 32-GB client GPUs such as the NVIDIA GeForce RTX 5090. This is not
a GPU SKU allowlist: the device must expose at least 31 binary GiB to the
runtime and meet the precision's compute-capability requirement. The RTX 5090
does not meet the 35-GiB Transfer minimum for these configurations.

The NIM applies a system-memory floor to every profile and filters incompatible
profiles before final selection. Generator startup also requires the current
free memory on each participating GPU to meet the generation floor in the
table. Resident Generator profiles use a 16-GiB selection floor, Nano offload
profiles use 64 GiB, and all Super offload profiles use 150 GiB. These profile floors do not define a general host-RAM
minimum; the release-wide CPU and RAM requirements remain unresolved. The
NIM checks a container memory limit before host physical memory.

Leave offload and guardrail residency on automatic selection unless a specific
configuration has been validated for the deployment.

## Reasoner configurations

Reasoner does not use Generator latency/throughput or model-offload selectors:

| Model | Precision | GPUs | Tensor parallelism | Minimum VRAM/device | Minimum effective system RAM | Minimum compute capability |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `nano` | BF16 | 1 | 1 | 23.1 GiB | 16 GiB | 8.0 |
| `nano` | FP8 | 1 | 1 | 23.1 GiB | 16 GiB | 8.9 |
| `nano` | NVFP4 | 1 | 1 | 23.1 GiB | 16 GiB | 10.0 |
| `super` | BF16 | 1 | 1 | 135 GiB | 16 GiB | 8.0 |
| `super` | BF16 | 2 | 2 | 73 GiB | 16 GiB | 8.0 |
| `super` | FP8 | 1 | 1 | 67 GiB | 16 GiB | 8.9 |
| `super` | NVFP4 | 1 | 1 | 73 GiB | 16 GiB | 10.0 |

When both Super BF16 layouts fit, profile ranking prefers the one-GPU TP1
layout. If TP1 does not have enough current free memory but TP2 fits on both
participating GPUs, automatic selection can fall back to TP2. To require TP2,
set `NIM_TAGS_SELECTOR='n_gpus=2,nim_tp=2'`. All Reasoner rows include the
variant-specific DFlash draft, and DFlash is enabled by default.

## Automatic profile selection

Users normally set:

- runtime;
- model variant;
- Generator latency or throughput; and
- precision only when it must be pinned.

The NIM first finds statically compatible profiles for those choices, visible
GPU totals, and effective system memory. It then uses one current-free-memory
snapshot without changing the requested model, precision, or Generator
performance objective. Automatic selection prefers FP8 when available and a
resident Generator profile when it fits. If the preferred layout is currently
too large, it can fall back across equivalent layouts, preferring Generator
layer offload before model offload and allowing a fitting Reasoner TP layout.
If none fits, startup fails. An explicit `NIM_MODEL_PROFILE` never falls back.

Exact profile IDs and low-level manifest tags are advanced image-specific
controls. Do not copy them between images or hosts. Free unrelated GPU memory
before launch and restart the NIM if you want automatic selection to reconsider
a layout.

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
uv run python examples/inspect_profile.py
```

Metadata confirms the configuration chosen for the running container. The
profile helper decodes the selected image's YAML manifest and prints the active
profile without its full artifact inventory. See
[Operations](operations.md#inspect-the-running-service) for the raw
management-endpoint workflow.
