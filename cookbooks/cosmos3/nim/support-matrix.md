<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM support matrix

Use this page to match a model to precision, GPU compute capability, GPU count,
per-device VRAM, and system-memory requirements.

> **Published requirements:** The tables below define GPU compute capability,
> GPU count, per-device VRAM, Transfer headroom, practical host RAM for
> non-offloading discrete-GPU deployments, and system-memory profile floors for
> offload. Confirm that a matching configuration exists in
> `/v1/manifest` for the exact image you run. The NIM's system-memory admission
> tag can be lower than the practical RAM requirement, so a selected profile is
> still only a candidate until cold start and representative requests succeed;
> see [Prerequisites](prerequisites.md#hardware-requirements).

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
| Reasoner | `nano` | Image/video reasoning | Prefer BF16 on compute capability 8.0 through 8.8, FP8 on 8.9 through 9.x, and NVFP4 on 10.0 or newer when compatible |
| Reasoner | `super` | Image/video reasoning | Prefer BF16 on compute capability 8.0 through 8.8, FP8 on 8.9 through 9.x, and NVFP4 on 10.0 or newer when compatible |

A Generator specialist rejects requests for other tasks. Confirm that
the selected image includes the model before deployment.

Nano and Super Reasoner profiles include variant-specific DFlash drafts and use
them by default. Set `NIM_USE_DFLASH=0` to run either target model without
speculative decoding. Generator does not support DFlash.

### Practical model recommendation

Profile compatibility is a hard hardware requirement, not a performance
recommendation. Choose the model family first, then let preflight select an
exact image-specific profile:

- On H200- and B200-class discrete GPUs, use Super when the workload needs it.
  Start with `NIM_MODEL_TYPE=generator` or `reasoner`, set
  `NIM_MODEL_VARIANT=super`, and—for Generator—set
  `NIM_PERF_PROFILE=latency`; normally leave precision, offload, tags, and the
  profile ID unset.
- On H100, RTX PRO 6000 Blackwell, and lower-throughput discrete devices,
  default to Nano for image and especially video generation. Super can remain
  profile-compatible through a resident or offload layout, but use it only for
  an explicit model requirement after preflight and representative testing.
- On unified-memory systems such as DGX Spark and Jetson AGX Thor, default to
  Nano for both runtimes. Some Super precision rows can fit the shared-memory
  floors, but Super is not recommended on these systems for practical
  turnaround.

These recommendations do not remove compatible profiles or replace the tables
below. After startup, verify the selected profile through `/v1/metadata` and
`/v1/manifest`; do not copy a profile ID from another image or host.

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

Meeting a minimum compute capability does not mean that every GPU at or above
it can run a profile. The memory floors, GPU count, and the exact image
manifest apply as well; see [Tested GPU inventory](#tested-gpu-inventory) for
the devices verified in this release.

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
applies these floors to the effective shared memory defined in [Unified-memory
thresholds](#unified-memory-thresholds). Generator selection keeps the model
and both guardrails resident, so only rows with **Model offload = None** and
**Guardrails = Resident** are eligible. Reasoner floors use the same effective
total.

### Unified-memory thresholds

Unified-memory systems use one DRAM pool for the host and the device. The
**effective shared memory** in this table is the device-reported total minus
the default 16-GiB host reserve. Change that reserve with
`NIM_UNIFIED_MEMORY_HOST_RESERVE_GIB` only from validated host measurements;
see [Advanced profile controls](configuration.md#advanced-profile-controls).

| Runtime/profile | Effective shared-memory floor | Unified-memory policy |
| --- | ---: | --- |
| Generator `nano` BF16; `nano-droid` BF16 | 58 GiB | Resident model and guardrails |
| Generator `nano` FP8 | 44 GiB | Resident model and guardrails |
| Generator `super` BF16 | 150 GiB | Resident model and guardrails |
| Generator `super` FP8 | 93 GiB | Resident model and guardrails |
| Reasoner `nano` BF16, FP8, or NVFP4 | 23.1 GiB/device | No CPU/model offload |
| Reasoner `super` BF16, TP1 | 135 GiB/device | No CPU/model offload |
| Reasoner `super` BF16, TP2 | 73 GiB/device | No CPU/model offload |
| Reasoner `super` FP8, TP1 | 67 GiB/device | No CPU/model offload |
| Reasoner `super` NVFP4, TP1 | 73 GiB/device | No CPU/model offload |

The effective floor is compared with the value returned by the running
system, not with a product's decimal memory label. Current free shared memory
is checked separately at startup. A profile must also meet its compute-capability,
GPU-count, and exact-image manifest requirements.

## Tested GPU inventory

The following devices are included in the release hardware inventory. The
discrete GPU rows are configured for official NIM testing; the unified-memory
rows are verified hardware targets for the shared-memory profile policy.
Product names are shortened for readability, grouped by architecture, and
sorted by nominal memory within each group.

Inventory membership does not mean that every runtime, model, precision, GPU
count, offload mode, and task is tested or available on that device. Use the
configuration tables below to find a candidate, and confirm that the exact
image manifest contains a matching profile. A device outside this inventory
can still be compatible when it meets the compute-capability and memory floors.

### Unified-memory systems

These systems share host and device memory; the capacity shown here is not
separate VRAM. Subtract the host reserve from the binary-GiB total reported by
the running system before comparing it with the effective floors above.

| Device | Nominal shared memory |
| --- | ---: |
| Jetson AGX Thor T5000 | 123 GiB |
| DGX Spark (GB10) | 128 GB (about 119 GiB) |

At these capacities, memory floors permit the resident Generator Nano BF16
and FP8 rows and the resident Super FP8 rows. For Reasoner, the Nano rows and
Super FP8 and NVFP4 TP1 rows meet the memory floor. This is capacity
compatibility, not a recommendation to use Super on these systems; prefer Nano
for practical turnaround. Compute capability, GPU count, current free memory,
and the exact image manifest still determine the selected profile.

Reasoner Super BF16 TP1 requires a 135-GiB effective pool after the host
reserve. It therefore exceeds the total capacity of both listed unified-memory
systems; reclaimable cache cannot make that profile fit. This differs from a
profile that fits the effective total but is temporarily blocked by the current
memory state.

### Blackwell Ultra

| GPU | Nominal GPU memory |
| --- | ---: |
| GB300 Workstation | 252 GB HBM3e |
| B300 SXM6 | 288 GB HBM3e |
| GB300 | 288 GB HBM3e |

### Blackwell

| GPU | Nominal GPU memory |
| --- | ---: |
| GeForce RTX 5090 | 32 GB GDDR7 |
| RTX PRO 4500 Blackwell Server Edition | 32 GB GDDR7 |
| RTX PRO 6000 Blackwell Server Edition | 96 GB GDDR7 |
| RTX PRO 6000 Blackwell Workstation Edition | 96 GB GDDR7 |
| B200 | 192 GB HBM3e |
| GB200 | 192 GB HBM3e |

### Hopper and Grace Hopper

| GPU | Nominal GPU memory |
| --- | ---: |
| H100 80 GB HBM3 | 80 GB HBM3 |
| H100 PCIe | 80 GB HBM3 |
| H100 NVL | 94 GB HBM3 |
| GH200 480 GB | 96 GB HBM3 |
| H20 | 96 GB HBM3 |
| H200 | 141 GB HBM3e |
| H200 NVL | 141 GB HBM3e |
| GH200 144 GB HBM3e | 144 GB HBM3e |

### Ada Lovelace

| GPU | Nominal GPU memory |
| --- | ---: |
| L20 | 48 GB GDDR6 |
| L40S | 48 GB GDDR6 |
| RTX 6000 Ada Generation | 48 GB GDDR6 |

### Ampere

| GPU | Nominal GPU memory |
| --- | ---: |
| A100 SXM4 40 GB | 40 GB HBM2e |
| A100 SXM4 80 GB | 80 GB HBM2e |

Nominal memory is shown only to order this inventory. The Thor and DGX
Spark entries are verified unified-memory targets, not separate-VRAM GPUs.
Profile selection uses
the per-device binary-GiB floors below and the memory reported by the running
system. For Grace Hopper, the table shows HBM GPU memory. In the GH200 480 GB
entry, 480 GB identifies CPU memory and is not GPU VRAM.

## Generator configurations

The `super` rows in this table also apply to `super-t2i`,
`super-t2i-4step`, `super-i2v`, and `super-i2v-4step`. Transfer thresholds apply
only to the general-purpose `nano` and `super` variants.

| Variant group | Precision | Model offload | Guardrails during diffusion | GPUs | Generation minimum VRAM/device | Host RAM guidance | Transfer minimum VRAM/device |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `nano` | BF16 | None | Resident | 1, 2, 4, 8 | 58 GiB | 40 GiB | 64 GiB |
| `nano` | BF16 | Layer | Offloaded | 1 | 31 GiB | 64 GiB | 35 GiB |
| `nano` | FP8 | None | Resident | 1, 2, 4, 8 | 44 GiB | 40 GiB | 50 GiB |
| `nano` | FP8 | Model | Resident | 1 | 38 GiB | 64 GiB | 44 GiB |
| `nano` | FP8 | Layer | Resident | 1 | 32 GiB | 64 GiB | 38 GiB |
| `nano` | FP8 | Layer | Offloaded | 1 | 31 GiB | 64 GiB | 35 GiB |
| `super` | BF16 | None | Resident | 1, 2, 4, 8 | 150 GiB | 112 GiB | 160 GiB |
| `super` | BF16 | Model | Resident | 1 | 93 GiB | 150 GiB | 99 GiB |
| `super` | BF16 | Layer | Resident | 1 | 42 GiB | 150 GiB | 50 GiB |
| `super` | FP8 | None | Resident | 1, 2, 4, 8 | 93 GiB | 78 GiB | 103 GiB |
| `super` | FP8 | Model | Resident | 1 | 64 GiB | 150 GiB | 76 GiB |
| `super` | FP8 | Layer | Resident | 1 | 38 GiB | 150 GiB | 50 GiB |

The 31-GiB Nano layer-offload configurations are intended to support ordinary
generation on 32-GB client GPUs such as the GeForce RTX 5090. This is not
a GPU SKU allowlist: the device must expose at least 31 binary GiB to the
runtime and meet the precision's compute-capability requirement. The RTX 5090
does not meet the 35-GiB Transfer minimum for these configurations.

The practical host RAM values for non-offloading profiles are empirical
requirements for a discrete-GPU NIM deployment. To avoid blocking an attempted
startup, the embedded profile tags check only a 16-GiB system-memory admission
floor for these Generator profiles. Passing that admission check does not
replace the 40-, 78-, or 112-GiB practical requirement in the table. Nano
offload profiles retain a 64-GiB profile floor, and Super offload profiles
retain a 150-GiB profile floor; separate empirical practical minima have not
been established for those offload rows. The NIM checks a container memory
limit before host physical memory.

These host RAM values do not change unified-memory guidance. DGX Spark and
Jetson AGX Thor use one host/device pool; evaluate them with [Unified-memory
thresholds](#unified-memory-thresholds), current `MemAvailable`, and the
required Reasoner memory-utilization setting rather than adding a separate host
RAM requirement.

Leave offload and guardrail residency on automatic selection unless a specific
configuration has been validated for the deployment.

## Reasoner configurations

Reasoner does not use Generator latency/throughput or model-offload selectors:

| Model | Precision | GPUs | Tensor parallelism | Minimum VRAM/device | Practical host RAM minimum | Minimum compute capability |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `nano` | BF16 | 1 | 1 | 23.1 GiB | 24 GiB | 8.0 |
| `nano` | FP8 | 1 | 1 | 23.1 GiB | 18 GiB | 8.9 |
| `nano` | NVFP4 | 1 | 1 | 23.1 GiB | 18 GiB | 10.0 |
| `super` | BF16 | 1 | 1 | 135 GiB | 76 GiB | 8.0 |
| `super` | BF16 | 2 | 2 | 73 GiB | 76 GiB | 8.0 |
| `super` | FP8 | 1 | 1 | 67 GiB | 46 GiB | 8.9 |
| `super` | NVFP4 | 1 | 1 | 73 GiB | 36 GiB | 10.0 |

The Reasoner values are empirical practical host RAM requirements on a
discrete-GPU host. The embedded profile tags deliberately check only 16 GiB of
system memory so an operator can attempt startup; that admission floor is not
the practical requirement for a running NIM. Unified-memory systems continue
to use the shared-pool guidance above without a separate host RAM requirement.

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
performance objective. Generator prefers FP8 when compatible. Reasoner prefers
BF16 on compute capability 8.0 through 8.8, FP8 on 8.9 through 9.x, and NVFP4
on 10.0 or newer, using another compatible precision when the preferred row is
unavailable. Selection prefers a resident Generator profile when it fits. If
the preferred layout is currently too large, it can fall back across equivalent
layouts, preferring Generator layer offload before model offload and allowing a
fitting Reasoner TP layout. If none fits, startup fails. An explicit
`NIM_MODEL_PROFILE` never falls back.

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

| Direction | Media | Documented formats/codecs and limits |
| --- | --- | --- |
| Input | Images | Use the included JPEG, PNG, and WebP fixtures; verify other formats against the deployed image |
| Input | Videos | Use the included MP4 fixtures; verify other containers and codecs against the deployed image |
| Output | Generator image | JPEG |
| Output | Generator video | VP9 in MP4 |

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
