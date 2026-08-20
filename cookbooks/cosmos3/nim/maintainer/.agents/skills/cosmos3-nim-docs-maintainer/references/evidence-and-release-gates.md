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

- Deployment currently uses the versioned Cosmos3 1.0.0 experimental RC
  `1.0.0-rc.experimental.20260820211500`, not a final release identity. Record
  source-derived evidence in maintainer files only and require the exact image
  manifest or live behavior before presenting claims as validated in the
  evaluation image. Before public release, `deployment.md` owns the exact
  evaluation image reference; update it on every RC bump.
- Never replace an RC reference with `latest`.
- The current support-matrix compute-capability, GPU-count, per-device VRAM,
  Transfer, and effective system-memory thresholds are approved for publication
  as release profile-selection floors. They establish a candidate profile, not
  full host compatibility, and remain subject to the exact image manifest.
- The published tested-GPU inventory contains the 22 discrete SKUs in the
  current NIMCraft NIM configuration; L4 is intentionally excluded from
  publication. It also includes Jetson AGX Thor T5000 and DGX Spark (GB10) as
  verified unified-memory targets. Inventory membership identifies an official
  validation target; it does not establish a passing result for every profile
  or task on every listed device and is not a compatibility allowlist.
- Transfer remains profile-compatible wherever its separate per-device memory
  floor is met, but its practical evaluation recommendation starts with an RTX
  PRO 6000 Blackwell 96-GB, H100 80-GB, or higher-throughput compatible
  discrete GPU. DGX Spark is not recommended for Transfer turnaround even when
  a compatible profile fits. Keep this operational recommendation distinct
  from support-matrix eligibility and do not turn it into an expected runtime.
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
- unified-memory selection subtracts the default 16-GiB host reserve from the
  reported shared-memory total for static profile floors, uses resident model
  and guardrails for Generator, and applies the same reserve to Reasoner
  current-free-memory admission. The current implementation records the
  Jetson AGX Thor T5000 as a 123-GiB unified-memory measurement target with an
  approximately 6.8-GiB observed host working set, and identifies GB10/DGX
  Spark for unified-memory Reasoner runtime defaults;
- effective system-memory selection floors of 16 GiB for resident Generator and
  Reasoner profiles, 64 GiB for Nano offload, and 150 GiB for Super offload;
- current-free-VRAM startup selection before model materialization, the
  container preflight invocation, equivalent-layout fallback, explicit-pin
  failure, and the Reasoner runtime reserve/utilization clamp;
- exact arbitrary Generator video frame counts within the mode and resolution
  bounds, generation at the next native `1 + 4k` count, trimming to the exact
  requested output count, and rounded V2V latent-index validation;
- Reasoner TP1 preference, TP2 availability fallback, guided-decoding
  enforcement, Responses create normalization, disabled-by-default video-token
  pruning with VidCom2 as the selected method when enabled, and operator-level
  media I/O and multimodal processor JSON options;
- default-on Nano and Super DFlash drafts, independent local draft overrides,
  hardware-derived BF16 KV-cache selection, and advanced DFlash JSON
  configuration;
- quantized Generator linear-backend selection;
- the NIM API adaptation for the 18-case Reasoner catalog, including byte-equal
  vLLM user-prompt strings, data-URL media, request-level 4-FPS video sampling,
  disabled NIM-native thinking, preserved prompt-authored `<think>` text,
  explicit effective sampling controls, prompt-constrained JSON extraction,
  opt-in NIM JSON Schema guidance, and local structural validators; and
- runtime-aware metadata, health responses, and wrong-runtime diagnostics.

Regenerate the source profile export before reconciling tables. Generated
artifacts can lag profile policy source and must not silently override current
implementation or be presented as an approved image manifest.

## Current RC validation status

The supplied build-completion record establishes the exact evaluation image
pinned in `deployment.md`. The source delta from the superseded RC adds exact
arbitrary Generator video frame counts: it removes the public `1 + 4k` cadence
restriction, generates non-native requests at the next native count, trims the
decoded result to the requested count, and validates V2V latent indexes against
the rounded internal count. An authenticated registry lookup confirmed that the
tag resolves to multi-architecture manifest-list digest
`sha256:a3e67b87dc28303936ae6ca03fba6ce83b0d185a1435117991eb6814db233b2a`.
The embedded NIM manifest release and profile inventory, preflight behavior,
cold start, management endpoints, and inference have not been validated in
this documentation update. Do not carry the superseded RC observations below
forward as current-image validation.

## Historical validation from the superseded 20260820 180843 RC

The superseded evaluation image had manifest digest
`sha256:c0a6b8a14c05bee46609a87798876dca62cea7f703d6ae552a26559d2298ad51`.
Its embedded manifest reported release
`1.0.0-rc.experimental.20260820180843`, with 115 Generator profiles and 7
Reasoner profiles. Direct option parsing in the image confirmed that unset
video pruning resolves to disabled and that `NIM_MM_PROCESSOR_KWARGS` accepts
a JSON object. The Reasoner
preflight exported `pynvvc` as the operator-level video backend.

On NVIDIA H100 NVL (compute capability 9.0, 95,830 MiB total and 95,322 MiB
free per visible device), the pre-download selector passed with one-GPU
profiles for both runtimes:

- Nano Generator selected FP8, `offload=none`, latency profile
  `845653ddaf5445077909499d031b8e57a249052dced3c4644ef9dc2f71898c8c`, with
  44-GiB VRAM and 16-GiB effective-system-memory floors; Transfer admission
  passed.
- Nano Reasoner selected FP8 profile
  `0ef8dad974a6e18226d70838ead8161670fbdd871ce4bc1efcd3f707a2bce612`, with
  23.1-GiB VRAM and 16-GiB effective-system-memory floors.

These preflight results establish candidate-profile compatibility only.

The exact image was also cold-started on one NVIDIA RTX PRO 6000 Blackwell
Server Edition (compute capability 12.0, 97,887 MiB total and 97,252 MiB free)
as the one-GPU Nano FP8 latency configuration. It reached readiness and
reported `model_type=generator`, `inference_endpoint=/v1/infer`, and
`model_variant=nano`. Paired AV policy requests used the same `av_0.jpg`, seed
0, and inference controls; only explicit left-turn and right-turn task prompts
differed. Both returned structurally valid `[60,9]` actions and 61-frame,
832-by-480 rollouts. Manual review confirmed that the left prompt produced a
left turn and the right prompt produced a right turn. This validates the two
included language-conditioned Nano AV policy cases for one run each, not Super,
BF16, other fixtures or seeds, safe-driving behavior, or performance.

The exact image was also launched on one NVIDIA H200 as Super FP8 profile
`c70e25d00f876b14b07441fc7920b8a7001487aecf3f258739bfbcc9a208e4a9`, with
the bundled DFlash draft enabled and video pruning unset. All 18 catalog cases
completed API and local format validation on the prompt-constrained default
path, and the opt-in guided-output grounding request also passed. One-time
comparison with `maintainer/reasoner-semantic-fixtures.yaml` gave grounding IoU
0.926 against the recorded vLLM box, above the 0.75 review threshold. Temporal
localization returned one whole-video event instead of the fixture's minimum
three events and therefore failed semantic-quality review despite passing local
JSON and timestamp validation. The same Super FP8 profile was then run
target-only with `NIM_USE_DFLASH=0`: 17 of 18 cases passed local format
validation, while `describe_anything` returned `description` instead of the
requested `caption` field.

One-run manual review against the catalog criteria rated direct vLLM Super at
14 pass and 4 fail, NIM with DFlash at 13 pass, 2 partial, and 3 fail, and NIM
target-only at 14 pass and 4 fail. All three missed video-caption completion and
the driving Action-CoT pedestrian. vLLM and target-only segmented temporal
localization while DFlash collapsed it into one event; only target-only passed
the assisted-task next action, while only target-only failed
`describe_anything` format. These sampled results validate request transport,
extraction, and local format handling and identify review targets; they do not
establish stable catalog-wide semantic quality or performance.

The tutorial image `vllm/vllm-openai:cosmos3` at digest
`sha256:db0bb920b0b54e82ea96a98659bbd21921f87d0dcfc86feffdafa2db3f08be55`
was run directly on the same H200. Super BF16 ran at TP1 rather than the
tutorial's TP4 because only one GPU was allocated. Its grounding result passed
the local validator and had IoU 0.931 against the fixture. The tutorial's video
combination of server-level `num_frames=-1` and request-level
`mm_processor_kwargs` at 4 FPS failed with HTTP 400 for both Super and Edge:
the loader supplied 37 frames before the processor attempted to sample again.
Using request-level `media_io_kwargs` at 4 FPS, as the NIM adaptation does,
completed successfully. With the catalog's explicit effective sampling and
thinking controls, all 18 Super cases passed API and local format validation.
Super temporal localization returned seven ordered segments covering pickup,
dispensing, placement, and retraction; Edge returned one whole-video segment
and failed the three-event semantic floor. Edge grounding passed under the same
explicit controls. This direct comparison validates the NIM-specific media
adaptation and the one-run fixture thresholds, not stable quality or TP4 output
parity.

## Open release gates

Review this list on every substantive documentation update and remove, add, or
refine entries when evidence changes:

- final public image identity and release URLs;
- general CPU architecture, host RAM beyond profile-selection floors, disk,
  shared-memory, driver, Docker, and Container Toolkit requirements;
- exact supported image formats, video containers/codecs, URL fetching, and
  VP9-in-MP4 playback observations;
- exact released support for specialist Generator, Action combinations beyond
  the validated Nano FP8 AV policy cases, Transfer, and V2V combinations;
- Reasoner Responses create normalization plus storage/background/retrieve
  behavior, and guided-output enforcement in the selected image;
- Reasoner public-URL, text-only, request-level video sampling, and
  operator-level media I/O and multimodal processor behavior;
- remaining live management endpoints, metrics, logs, errors, and chart probes;
- approved startup, latency, and throughput measurements for each published
  reference configuration;
- prompt-upsampling integration behavior in the selected image;
- current-free-VRAM fallback, Reasoner utilization clamping, and discrete-GPU
  NVML startup behavior in the selected image;
- default-on Nano/Super DFlash, draft override, KV-cache, and advanced
  configuration behavior in the selected image, including repeated-seed Super
  FP8 spatial-output correctness;
- quantized Generator linear-backend behavior in the selected image;
- public Helm chart URL to replace the staging reference, plus approved values,
  installation workflow, and monitoring integration;
- repeated-seed vLLM/NIM review of the current catalog quality gaps, especially
  temporal segmentation, video-caption completion, assisted-task next action,
  trajectory semantics, and driving Action-CoT pedestrian identification; plus
  evaluation of explicitly enabled VidCom2 output quality on the video catalog;
- approved native-thinking output and reasoning-trace wording for the current
  image; and
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
