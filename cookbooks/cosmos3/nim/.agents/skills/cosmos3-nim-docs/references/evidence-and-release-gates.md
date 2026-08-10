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

- Deployment currently uses a versioned Cosmos3 2.2 staging RC, not a final
  release identity. The exact reference is intentionally canonical in both
  `deployment.md` and `release-notes.md`; update both on every RC bump.
- Never replace an RC reference with `latest`.
- The final public image, release version/date, catalog URL, and model-card URL
  remain release-owned until approved.
- No Cosmos3 Certified NIM Helm chart is published yet. Keep Helm guidance
  conceptual and omit chart commands until the exact repository, version, and
  schema are available.

## Open release gates

Review this list on every substantive documentation update and remove, add, or
refine entries when evidence changes:

- final public image identity and release URLs;
- released profile rows, tested GPU boundary, and driver/toolkit floors;
- general CPU architecture, RAM, disk, and shared-memory requirements;
- exact supported image formats, video containers/codecs, URL fetching, and
  VP9-in-MP4 playback observations;
- published Generator and Reasoner BYOC formats and source boundary;
- exact released support for specialist Generator, Action, Transfer, and V2V
  combinations;
- Reasoner Responses storage/background/retrieve behavior;
- Reasoner public-URL, text-only, legacy media, and request-level video sampling
  behavior;
- live management endpoints, metrics, logs, errors, and chart probes;
- approved startup, latency, and throughput measurements for each published
  reference configuration;
- prompt-upsampling integration behavior in the selected image;
- final Helm chart identity, values, and monitoring integration;
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
- update the RC image in deployment and release notes together;
- inspect all remaining TBDs and confirm none implies a usable value;
- validate links, JSON, examples, dependencies, paths, and ignored outputs;
- search for obsolete fields, legacy images/endpoints, realistic secrets,
  private paths, and unsupported backend syntax;
- report tests that could not run and do not present static checks as live NIM
  validation; and
- review `AGENTS.md`, `SKILL.md`, and both references so editor guidance remains
  synchronized with the public documentation.
