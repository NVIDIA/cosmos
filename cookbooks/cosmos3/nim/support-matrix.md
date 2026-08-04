<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM support matrix

Use this page to match a model to released precision, GPU, memory, and media
support. The final tested matrix is **TBD**. Development profiles do not create
a public support guarantee.

## Model contracts

Choose the model first. The NIM then selects a compatible profile for the
visible host.

| Runtime | Model | Task contract | Automatic precision behavior |
| --- | --- | --- | --- |
| Generator | `nano` | General-purpose Generator tasks included by the release | Prefer FP8 when compatible; otherwise fall back |
| Generator | `nano-droid` | DROID policy with action-only output | BF16 only in the current implementation |
| Generator | `super` | General-purpose Generator tasks included by the release | Prefer FP8 when compatible; otherwise fall back |
| Generator | `super-t2i` | Full-step T2I only | Prefer FP8 when compatible; otherwise fall back |
| Generator | `super-t2i-4step` | Four-step T2I only | Prefer FP8 when compatible; otherwise fall back |
| Generator | `super-i2v` | Full-step I2V only | Prefer FP8 when compatible; otherwise fall back |
| Generator | `super-i2v-4step` | Four-step I2V only | Prefer FP8 when compatible; otherwise fall back |
| Reasoner | `nano` | Image/video reasoning | Prefer FP8 when compatible; otherwise fall back |
| Reasoner | `super` | Image/video reasoning | Prefer FP8 when compatible; otherwise fall back |

A Generator specialist rejects requests outside its task contract. Model
presence in the implementation does not mean that every model is included in
every image release.

Nano Reasoner can optionally use DFlash speculative decoding. The released
profile must include the draft artifact; Generator and Super Reasoner do not
support `NIM_USE_DFLASH=1`.

## Released hardware configurations

A configuration is supported only when one released row matches the selected
model, precision, GPU architecture, GPU count, per-device memory, and any
system-memory requirement.

| Runtime/model | Precision | GPU architecture | GPUs | Per-device VRAM | System RAM | Performance/offload | Transfer | Status |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| Generator variants | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | Release validation pending |
| Reasoner Nano/Super | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | N/A | N/A | Release validation pending |

Populate this table from the released image manifest and approved test
inventory. If a combination is not listed, treat it as unsupported.

## Automatic profile selection

Users normally set:

- runtime;
- Generator variant or Reasoner model size;
- Generator latency or throughput; and
- precision only when it must be pinned.

The NIM finds a compatible profile for those choices and the visible GPUs.
Automatic selection prefers FP8 when available, avoids offload when the model
fits normally, and uses a compatible GPU layout. If no profile fits, startup
fails rather than selecting an unsupported combination.

Exact profile IDs and low-level manifest tags are advanced release-specific
controls. Do not copy them between images or hosts.

## Latency, throughput, and lower-memory profiles

Generator requires a workload choice:

- `latency` prioritizes an individual request;
- `throughput` prioritizes aggregate request rate.

The software default is latency, but production deployment should make the
choice explicit. Reasoner does not use this selector.

Some Generator releases can provide model or guardrail offload profiles. They
reduce resident GPU memory at a latency cost and can move weights into system
RAM. Current Super BF16 model- and layer-offload profiles require 150 GiB of
effective system memory, measured against the container limit first. The
released table must confirm availability, GPU and system-memory floors, and
performance expectations before use.

## Transfer headroom

Transfer can require more peak memory than generation without Transfer on the
same model. Startup checks whether the selected profile and GPU have validated
headroom. A deployment can therefore serve generation while rejecting
Transfer.

Use a larger GPU or a released lower-memory profile when Transfer does not fit.
`NIM_ALLOW_UNSAFE_TRANSFER=1` bypasses the check for diagnosis but can cause an
out-of-memory failure and does not make the deployment supported.

## Media and codecs

| Direction | Media | Released formats/codecs and limits |
| --- | --- | --- |
| Input | Images | **TBD** |
| Input | Videos | **TBD** |
| Output | Generator image | Current implementation emits JPEG; release validation pending |
| Output | Generator video | Current implementation emits VP9 in MP4; release validation pending |

The request schemas accept base64 and MIME-aware data URLs. HTTP(S) input is
available only when enabled and reachable from the container. Schema acceptance
does not guarantee every image format, video container, codec, chroma format,
frame rate, or remote source.

## Verify a running deployment

```bash
curl -fsS http://localhost:8000/v1/metadata | python -m json.tool
curl -fsS http://localhost:8000/v1/manifest | python -m json.tool
```

Metadata confirms the selected model/profile; it does not expand the released
support boundary.
