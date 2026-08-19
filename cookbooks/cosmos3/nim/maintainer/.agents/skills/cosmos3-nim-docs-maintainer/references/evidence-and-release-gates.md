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
- The published tested-GPU inventory contains 22 of the SKUs in the current
  NIMCraft NIM configuration; L4 is intentionally excluded from publication.
  The published inventory identifies official validation targets; it does not
  establish a passing result for every profile or task on every listed SKU and
  is not a compatibility allowlist.
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
- quantized Generator linear-backend selection;
- the NIM API adaptation for the 18-case Reasoner catalog, including byte-equal
  vLLM user-prompt strings, data-URL media, request-level 4-FPS video sampling,
  an unpruned-video baseline, disabled NIM-native parsed reasoning, preserved
  prompt-authored `<think>` text, standard JSON Schema output, and local
  structural validators; and
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

The exact image was launched separately on one NVIDIA B200 as Super FP8
Reasoner with its bundled DFlash draft explicitly enabled. The committed
`robot_planning` case used the shared `robot_planning.png` fixture, original
Reasoner-notebook prompt, and seed 0. It completed with a five-subtask response
covering movement to the flower, grasp, pickup, movement to the red bottle, and
placement. This validates that one request and configuration, not a stable JSON
format, other seeds, other tasks, general Reasoner quality, or performance.

The same image and one-B200 Super FP8 profile were then run target-only with
NIM-native parsed reasoning disabled and the default VidCom2 pruning rate of
0.6. All 18 catalog
cases completed API and format validation,
including JSON parsing, timestamp/box/point invariants, and spatial artifact
generation. Qualitative review passed 12 case-specific checklists. The video
caption, first temporal-localization case, robotics next action, assisted-task
next action, marked-subject description, and flower trajectory remained
incomplete or semantically inaccurate. This validates the target-only catalog
transport and format paths, not general task quality or performance.

The target-only, native-reasoning-disabled FP8 profile was then rerun with
`NIM_VIDEO_PRUNING_RATE=0`. All 18 cases again completed with
`finish_reason=stop`, and all seven structured cases passed local format
validation. Fourteen qualitative checklists passed. The video caption improved
materially but stopped at preparation to place the white box rather than
explicitly completing placement. Assisted-task next action, marked-subject
description, and flower trajectory remained inaccurate. Disabling pruning
restored the complete first temporal timeline and the expected smart-charger
next action; this result motivated keeping pruning disabled in the catalog
baseline.

The exact image was then launched on one NVIDIA B200 as Super BF16 target-only,
with NIM-native parsed reasoning and video pruning disabled. All 18 catalog
cases completed with
`finish_reason=stop`, and all seven structured cases passed format validation.
The BF16 run corrected the flower-trajectory result seen with FP8; video caption,
assisted-task next action, and marked-subject description remained incomplete.
The `robot_planning` case explicitly sent `temperature=0`, `top_p=0.8`,
`top_k=20`, `presence_penalty=0`, `repetition_penalty=1`, and `seed=0`. Its
five-subtask response was byte-for-byte identical to the controlled raw-vLLM
Super BF16 response, but both omitted an explicit flower release and arm
retreat. This validates deterministic parity for that request, not sampled
parity, complete task semantics, general model quality, or performance.

The exact image was subsequently launched on one NVIDIA H200 as Super BF16
without DFlash or video pruning. Before inference, all 18 catalog user prompts
were checked byte for byte against the runtime strings in
`cookbooks/cosmos3/reasoner/run_with_vllm.ipynb`. NIM-native parsed reasoning
remained disabled; six prompts still contained their original literal `<think>`
instructions. All 18 requests completed with `finish_reason=stop`, and all seven
structured cases passed JSON and local semantic format validation. Four text
cases returned visible `<think>` blocks in `message.content`; guided output kept
the two structured trajectory responses as schema-conforming JSON.

This exact-prompt run retained deterministic robot-planning parity, the physical
plausibility answer `A`, a near-identical situation-understanding answer, a
matching driving Action-CoT class, and a grounding box close to the recorded
vLLM answer. It did not establish catalog-wide output parity: video captioning
placed the white box back on the table instead of in the shipping box, and the
assisted-task case proposed removing the old cartridge instead of obtaining the
new one. The interval answer omitted one box carrier, and the timestamp range
started earlier than the recorded vLLM answer. The vLLM notebook contains no
embedded outputs and was not rerun concurrently, so comparisons other than the
separately controlled robot-planning request use the recorded prompt-guide
answers and are qualitative, not a backend equivalence test.

Two live correctness findings remain internal release gates:

- With native reasoning controls enabled, both DFlash and target-only returned
  `finish_reason=stop` with `message.content=null` and placed the complete text
  or structured answer in `message.reasoning`. This reproduced with 512- and
  2048-token reasoning budgets, so the catalog keeps NIM-native parsed
  reasoning disabled.
- For a fixed-seed 2D grounding request, DFlash produced invalid coordinates.
  In the follow-up control, DFlash passed 0/10 greedy requests at
  `temperature=0` and 4/20 sampled seeds; target-only passed 10/10 and 20/20,
  respectively. Failures included reversed corners, dropped digits,
  out-of-range values, and malformed JSON. The public catalog therefore uses
  the target-only baseline without describing this unresolved release finding.

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
  configuration behavior in the selected image, including resolution of the
  Super FP8 spatial-output correctness finding recorded above;
- quantized Generator linear-backend behavior in the selected image;
- public Helm chart URL to replace the staging reference, plus approved values,
  installation workflow, and monitoring integration;
- controlled, concurrent vLLM/NIM review of all exact-prompt catalog outputs,
  especially video captioning, assisted-task next action, interval/timestamp
  completeness, trajectory semantics, and robot-planning release/retreat;
  plus evaluation of default VidCom2 output quality on the video catalog;
- native reasoning-parser separation of reasoning from non-empty final content,
  plus approved reasoning-trace wording; and
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
