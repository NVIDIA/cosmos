<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Evidence and release gates

## Authority order

Use sources in this order when they disagree:

1. Current Cosmos3 Certified NIM implementation, request models,
   configuration, profile code, and focused tests.
2. Behavior and OpenAPI observed from the exact RC or released image.
3. Approved release notes, manifest, model card, chart, and notice artifacts.
4. Historical first-party NIM documentation for durable coverage or
   explanations only.
5. Nearby Cosmos cookbook material for terminology, style, and approved asset
   reuse.
6. Other backends and the Cosmos framework for concepts only, never Certified
   NIM request fields or support claims.

Public docs must not expose private checkout paths, internal commit IDs, or
unreleased profile IDs. Record evidence status in prose whenever it affects
whether a claim is merely source-compatible or actually release-tested.

## Current source areas

For contract updates, inspect the current Certified NIM equivalents of:

- Generator request, Action, Transfer, response, resolution, and media models;
- Generator and Reasoner runtime routing and request normalization;
- environment-variable parsing and model/checkpoint source resolution;
- profile selection, hardware checks, profile generation, and manifest data;
- prompt upsampling and guardrail configuration;
- focused tests for each changed behavior; and
- live `/openapi.json`, `/v1/models`, metadata, and manifest endpoints when the
  selected image can run.

Use cookbook assets for representative examples, but validate every NIM request
against the NIM contract rather than copying another backend's adapter.

## Active release maintenance

- Deployment currently uses a versioned Cosmos3 2.6 staging bugfix RC, not a
  final release identity. Record source-derived evidence in maintainer files
  only and require the exact image manifest or live behavior before presenting
  claims as validated in the evaluation image. Before public release, `deployment.md`
  owns the exact evaluation image reference; update it on every RC bump.
- Never replace an RC reference with `latest`.
- The current support-matrix compute-capability, GPU-count, per-device VRAM,
  Transfer, and effective system-memory thresholds are approved for publication
  as release profile-selection floors. They establish a candidate profile, not
  full host compatibility, and remain subject to the exact image manifest.
- The final public image, release version/date, catalog URL, and model-card URL
  remain release-owned until approved.
- The current evaluation chart reference is
  `nvstaging/nim/nim-wfm:1.3.0`. The public chart URL is TBD and must replace
  the staging reference when approved. Keep the public Helm page explicit about
  this boundary and omit pull/install commands and values until the public URL,
  schema, and workflow are approved.

## Current source-derived contracts

Track evidence provenance in this maintainer reference. Public pages state the
corresponding product behavior directly without mentioning source-code
provenance, and direct users to runtime interfaces where availability must be
confirmed:

- Generator BF16 compute capability 8.0, updated Super VRAM/Transfer floors,
  and Reasoner Super BF16 TP2 at 73 GiB/device;
- effective system-memory selection floors of 16 GiB for resident Generator and
  Reasoner profiles, 64 GiB for Nano offload, and 150 GiB for Super offload;
- current-free-VRAM startup selection before model materialization, the
  container preflight invocation, equivalent-layout fallback, explicit-pin
  failure, and the Reasoner runtime reserve/utilization clamp;
- Reasoner TP1 preference, TP2 availability fallback, guided-decoding
  enforcement, Responses create normalization, and VidCom2 pruning defaults;
- default-on Nano and Super DFlash drafts, independent local draft overrides,
  hardware-derived BF16 KV-cache selection, and advanced DFlash JSON
  configuration;
- quantized Generator linear-backend selection; and
- runtime-aware metadata, health responses, and wrong-runtime diagnostics.

Regenerate the source profile export before reconciling tables. Generated
artifacts can lag profile policy source and must not silently override current
implementation or be presented as an approved image manifest.

## Current RC-validated contracts

The pre-download selector was run from the exact evaluation image pinned in
`deployment.md` (manifest digest
`sha256:40fc1382a557fe22e60e4ddaae5c4be6b187431786e610208b5b3d5261dc5ce2`)
on one NVIDIA H100 PCIe (compute capability 9.0, 81,559 MiB
total and 81,081 MiB free):

- Nano Generator selected an FP8, one-GPU profile with `offload=none` at the
  44-GiB VRAM and 16-GiB effective-system-memory floors, with Transfer
  admission enabled.
- Nano Reasoner selected an FP8, one-GPU profile at the 23.1-GiB VRAM and
  16-GiB effective-system-memory floors.
- `NGC_API_KEY` and checkpoint overrides were absent inside the container, and
  `/opt/nim/.cache` was absent both before and after the two selector runs.

The same image manifest contains 115 Generator profiles. Normalizing away
profile IDs, performance scenario, and repeated GPU topology produces 37
variant-specific hardware rows: six each for the five Super-family variants,
six for Nano, and one for Nano-DROID. The public table collapses the five
identical Super-family sets into six shared rows, so its 13 rows cover every
active Generator profile without retaining a documentation-only row. Transfer
minimums also match each applicable installed generation floor plus Transfer
overhead.

The pre-download selector was launched through Slurm/Pyxis because the
documentation host could not access its Docker daemon. This validates the
selector invocation, embedded manifest and Transfer-policy reconciliation, and
no-model-download boundary inside the exact image, not the documented Docker
wrapper or profile fallback cases.

The exact image was also launched separately as Nano Reasoner with Docker on one
H100 PCIe and reached readiness. Its `/v1/manifest` response contained the full
YAML manifest in the JSON `manifest_file` string, and `/v1/metadata` identified
one matching selected profile. The committed profile-inspection helper was
validated against these live responses. This establishes the
management-endpoint join and helper output for this Nano Reasoner configuration,
not Generator, inference behavior, or performance.

## Open release gates

Review this list on every substantive documentation update and remove, add, or
refine entries when evidence changes:

- final public image identity and release URLs;
- general CPU architecture, host RAM beyond profile-selection floors, disk,
  shared-memory, driver, Docker, and Container Toolkit requirements;
- exact supported image formats, video containers/codecs, URL fetching, and
  VP9-in-MP4 playback observations;
- exact released support for specialist Generator, Action, Transfer, and V2V
  combinations;
- Reasoner Responses create normalization plus storage/background/retrieve
  behavior, and guided-output enforcement in the selected image;
- Reasoner public-URL, text-only, and request-level video sampling behavior;
- remaining live management endpoints, metrics, logs, errors, and chart probes;
- approved startup, latency, and throughput measurements for each published
  reference configuration;
- prompt-upsampling integration behavior in the selected image;
- current-free-VRAM fallback, Reasoner utilization clamping, and discrete-GPU
  NVML startup behavior in the selected image;
- default-on Nano/Super DFlash, draft override, KV-cache, and advanced
  configuration behavior in the selected image;
- quantized Generator linear-backend behavior in the selected image;
- public Helm chart URL to replace the staging reference, plus approved values,
  installation workflow, and monitoring integration;
- approved reasoning-trace wording; and
- approved acknowledgements and product license/model-card links for the exact
  release.

Keep unresolved values visible in prose or tables. Do not place them in
runnable fences or fill them from an older NIM release.

## Validation boundaries

Static validation can establish that:

- Markdown targets and documented local paths exist;
- JSON examples parse;
- Python examples compile and their dependencies resolve from `uv.lock`;
- request builders use current field names and valid local assets; and
- documented commands are internally consistent and contain no obvious
  placeholders or secrets.

Static validation cannot establish released profile availability, runtime
behavior, hardware compatibility, media/codec support, metrics, logs, or
performance. Claim those only after testing the exact image on appropriate
hardware or receiving an approved release artifact.

The container pre-download preflight sits between static review and cold-start
validation. It can establish that the exact image has a candidate manifest row
and that selectors, effective system-memory floor, compute capability, total
VRAM, and current free VRAM pass profile selection. It cannot establish general
host compatibility, model download/materialization, model load, readiness,
media handling, inference behavior, or performance.

For live validation, record the exact image, runtime, model variant, precision,
hardware, endpoint, and request case. Validate Generator and Reasoner
separately because one selected profile launches only one backend.

## Publication review

Before publication:

- reconcile affected claims with current source and approved release evidence;
- verify canonical ownership and deliberate duplicates;
- update the evaluation image in deployment;
- inspect all unresolved release statements and confirm none implies a usable
  value;
- validate links, JSON, examples, dependencies, paths, and ignored outputs;
- search for obsolete fields, legacy images/endpoints, realistic secrets,
  private paths, and unsupported backend syntax;
- report tests that could not run and do not present static checks as live NIM
  validation; and
- review maintainer `AGENTS.md`, `SKILL.md`, and both references, plus the
  customer-assistant instructions and skill when affected, so AI guidance
  remains synchronized with the public documentation.
