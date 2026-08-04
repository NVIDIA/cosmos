<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM support matrix

This page is the canonical location for released model, precision, GPU,
profile, offload, and media compatibility. Use
[prerequisites.md](prerequisites.md) for general host requirements and
[deployment.md](deployment.md#select-a-profile) to select a compatible profile.

> **Release status:** The final tested and supported matrix is **TBD
> (release-dependent)**. Current source profiles are useful validation inputs
> but do not establish a public support guarantee.

## How to read the matrix

A configuration is supported only when all dimensions match a released row:

- runtime family: Generator or Reasoner;
- model size: Nano or Super;
- Generator model variant and its allowed request mode;
- model precision;
- GPU architecture and minimum compute capability;
- GPU count and homogeneous per-device VRAM;
- parallelism layout;
- Generator latency/throughput objective;
- Generator model and guardrail offload policy; and
- Transfer VRAM headroom, when Transfer is enabled.

Distinguish:

- **tested and supported** configurations explicitly listed for the release;
- **compatible** hardware that satisfies documented gates but is not part of
  the tested SKU list; and
- **unsupported or unverified** combinations, including configurations inferred
  only from development manifests.

## Released model and hardware profiles

| Runtime | Model/variant | Precision | GPU architecture / compute capability | GPUs | Per-device VRAM | Layout/offload | Transfer | Status |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| Generator | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | Release validation pending |
| Reasoner | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | N/A | Release validation pending |

Populate this table from the exact released manifest and approved test
inventory. Do not merge historical Generator and Reasoner tables or publish
development-only rows as released support.

## Profile behavior

Generator latency profiles can shard one request across compatible parallelism
dimensions. Generator throughput profiles can use data-parallel workers to
increase aggregate request rate. Reasoner profiles select their own
tensor-parallel allocation and do not use the Generator
`latency`/`throughput` selector.

Low-level manifest dimensions can include `nim_dp`, `nim_gp`, `nim_up`, and
`nim_tp`. Generator profiles also carry `model_variant`, `text_guard_offload`,
and `video_guard_offload`. Select with high-level variables where possible;
pin low-level tags only after reviewing the target image's manifest.

The current development source permits Generator FP8 on compute capability 8.9
or newer; Reasoner precision gates are separate. This is pre-release evidence,
not a public hardware guarantee. Populate the released table from the exact
published manifest.

## Low-VRAM offload

The current design includes Generator profiles that can trade performance for
lower GPU-memory residency:

- `none` keeps the normal model layout;
- model-level or layer-level offload reduces model residency at a latency cost;
- text-guard offload sleeps the text classifier during diffusion; and
- video-guard offload sleeps output-safety sessions during diffusion.

These are distinct dimensions. The released rows, supported model/precision
combinations, minimum memory, and performance expectations remain **TBD**.
Confirm availability before setting an offload selector or override.

## Transfer headroom

Generator profile compatibility is based on its ordinary generation floor.
Transfer has an additional measured peak-memory overhead. At startup, the NIM
checks whether the visible GPU has enough headroom for the selected profile;
a deployment can therefore serve ordinary generation while rejecting Transfer.

The released matrix must identify which profile/GPU rows have validated
Transfer headroom. `NIM_ALLOW_UNSAFE_TRANSFER=1` bypasses this protection but
can cause an out-of-memory failure and does not make the configuration
supported.

## Supported media and codecs

| Direction | Media | Released formats/codecs and limits |
| --- | --- | --- |
| Input | Images | **TBD (release-dependent)** |
| Input | Videos | **TBD (release-dependent)** |
| Output | Generator image | Current source emits JPEG; release validation pending |
| Output | Generator video | Current source emits VP9 in MP4; release validation pending |

Request schemas recognize base64 and MIME-aware data URLs; optional HTTP(S)
input also depends on runtime configuration. Schema acceptance does not prove
that every container, codec, chroma format, frame rate, or remote source is
supported. Record the validated release matrix here after image smoke tests.

## Inspect the selected release

After launch, inspect the active selection rather than assuming which profile
won:

```bash
curl -fsS http://localhost:8000/v1/metadata | python -m json.tool
curl -fsS http://localhost:8000/v1/manifest | python -m json.tool
```

Runtime metadata confirms the selected image/profile, Generator
`model_variant`, and checkpoint label. It does not expand the published support
boundary.
