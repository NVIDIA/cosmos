<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 NIM support matrix

Match a model to precision, GPU compute capability, GPU count, per-device VRAM,
and system memory. All memory values are binary GiB per device.

## Choose a model

Choose the model; the NIM selects a compatible profile and precision (see
[Precision selection](deployment.md#precision-selection)).

| Runtime | Model | Supported use |
| --- | --- | --- |
| Generator | `nano` | General-purpose Generator tasks |
| Generator | `nano-droid` | DROID policy with action-only output; BF16 only |
| Generator | `super` | General-purpose Generator tasks |
| Generator | `super-t2i` | Full-step T2I only |
| Generator | `super-t2i-4step` | Four-step T2I only |
| Generator | `super-i2v` | Full-step I2V only |
| Generator | `super-i2v-4step` | Four-step I2V only |
| Reasoner | `nano` | Image/video reasoning |
| Reasoner | `super` | Image/video reasoning |

A Generator specialist rejects requests for other tasks. After startup,
[verify the selection](deployment.md#verify-the-selection).

### Practical model recommendation

- H200- and B200-class discrete GPUs: use Super when the workload needs it,
  with `NIM_PERF_PROFILE=latency` for Generator; leave precision, offload,
  tags, and profile ID unset.
- H100, RTX PRO 6000 Blackwell, and lower-throughput discrete GPUs: default to
  Nano, especially for video. Use Super only for an explicit model requirement.
- Unified-memory systems (DGX Spark, Jetson AGX Thor): default to Nano for both
  runtimes. Some Super rows fit, but turnaround is impractical.

## GPU architecture and topology

Compatibility uses compute capability, per-device VRAM, and system RAM, not a
GPU SKU allowlist:

| Runtime | Precision | Minimum compute capability |
| --- | --- | ---: |
| Generator | BF16 | 8.0 |
| Generator | FP8 | 8.9 |
| Reasoner | BF16 | 8.0 |
| Reasoner | FP8 | 8.9 |
| Reasoner | NVFP4 | 10.0 |

Use homogeneous GPUs; mixed-GPU configurations are not supported, and
Reasoner requires the same compute capability on every GPU. Do not add VRAM
across devices. Each GPU in the selected layout must meet the floor in total
and in free memory at startup; Reasoner measures free memory after a runtime
reserve (2 GiB by default on a discrete GPU, the host reserve on an integrated
GPU). On unified-memory systems, Generator uses only rows with
**Model offload = None** and **Guardrails = Resident**.

### Unified-memory thresholds

On unified-memory systems, **effective shared memory** is the device-reported
total minus the default 16-GiB host reserve. Change that reserve with
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

Floors are compared with the binary-GiB total reported by the running system;
current free shared memory is checked separately at startup.

## Tested GPU inventory

Devices in the release hardware inventory, grouped by architecture and sorted
by nominal memory. Inventory membership does not mean every configuration is
tested on that device, and devices outside it can be compatible when they meet
the floors.

### Unified-memory systems

Capacity is a shared host/device pool, not separate VRAM:

| Device | Nominal shared memory |
| --- | ---: |
| Jetson AGX Thor T5000 | 123 GiB |
| DGX Spark (GB10) | 128 GB (about 119 GiB) |

These capacities fit the resident Generator Nano BF16/FP8 and Super FP8 rows,
and the Reasoner Nano and Super FP8/NVFP4 TP1 rows. Reasoner Super BF16 TP1
(135 GiB) exceeds both systems' total capacity, so freeing cache cannot make
it fit.

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

Nominal memory only orders this inventory. For GH200 480 GB, 480 GB is CPU
memory; the table shows HBM GPU memory.

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

The 31-GiB Nano layer-offload rows target 32-GB client GPUs such as the
GeForce RTX 5090, which does not meet their 35-GiB Transfer minimum.

> **Host RAM:** Values for non-offloading rows (Generator and Reasoner) are
> empirical practical minimums on discrete-GPU hosts. The profiles' own
> admission check is only 16 GiB so startup is not blocked; passing it does not
> satisfy the practical value. Offload rows carry a profile floor (Nano 64 GiB,
> Super 150 GiB) with no separate practical minimum established. A container
> memory limit counts before host physical memory. Unified-memory systems have
> no separate host RAM requirement; size them with
> [Unified-memory thresholds](#unified-memory-thresholds).

Leave offload and guardrail residency on automatic selection unless a
configuration is validated for the deployment.

## Reasoner configurations

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

Host RAM follows the note under
[Generator configurations](#generator-configurations). Selection prefers Super
BF16 TP1 and can fall back to TP2 when TP1 lacks free memory; to require TP2,
set `NIM_TAGS_SELECTOR='n_gpus=2,nim_tp=2'`.

## Transfer headroom

Transfer needs more peak memory than ordinary generation, so startup checks it
separately; a deployment can serve generation while rejecting Transfer. Use a
larger GPU or a lower-memory configuration. `NIM_ALLOW_UNSAFE_TRANSFER=1`
bypasses the check for diagnosis only and can cause out-of-memory failures.

## Media and codecs

| Direction | Media | Documented formats/codecs and limits |
| --- | --- | --- |
| Input | Images | Use the included JPEG, PNG, and WebP fixtures; verify other formats against the deployed image |
| Input | Videos | H.264, H.265/HEVC, MPEG-4, VP8, VP9, and AV1; other codecs return HTTP 422; any container the decoder opens |
| Output | Generator image | JPEG |
| Output | Generator video | VP9 in MP4 |

Inputs are base64 or data URLs; HTTP(S) input works only when enabled and
reachable from the container. See [API reference](api-reference.md) for the
media rules.
