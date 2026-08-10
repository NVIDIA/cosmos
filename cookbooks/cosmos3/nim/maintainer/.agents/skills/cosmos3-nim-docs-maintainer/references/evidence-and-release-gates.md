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

- Deployment currently uses a versioned Cosmos3 2.3 staging RC, not a final
  release identity. Record source-derived evidence in maintainer files only and
  require the exact image manifest or live behavior before presenting claims as
  validated in the evaluation image. Before public release, `deployment.md`
  owns the exact evaluation image reference; update it on every RC bump.
- Never replace an RC reference with `latest`.
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
- Reasoner guided-decoding enforcement and Responses create normalization;
- independent local DFlash draft overrides, BF16 KV-cache selection, and
  advanced DFlash JSON configuration; and
- runtime-aware metadata, health responses, and wrong-runtime diagnostics.

Regenerate the source profile export before reconciling tables. Generated
artifacts can lag profile policy source and must not silently override current
implementation or be presented as an approved image manifest.

## Open release gates

Review this list on every substantive documentation update and remove, add, or
refine entries when evidence changes:

- final public image identity and release URLs;
- released profile rows, including current-source compute capability, VRAM,
  Transfer, effective system-memory boundaries, and driver/toolkit floors;
- general CPU architecture, RAM, disk, and shared-memory requirements;
- exact supported image formats, video containers/codecs, URL fetching, and
  VP9-in-MP4 playback observations;
- exact released support for specialist Generator, Action, Transfer, and V2V
  combinations;
- Reasoner Responses create normalization plus storage/background/retrieve
  behavior, and guided-output enforcement in the selected image;
- Reasoner public-URL, text-only, and request-level video sampling behavior;
- live management endpoints, metrics, logs, errors, and chart probes;
- approved startup, latency, and throughput measurements for each published
  reference configuration;
- prompt-upsampling integration behavior in the selected image;
- DFlash draft override, KV-cache, and advanced configuration behavior in the
  selected image;
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
