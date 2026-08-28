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

Customer-facing docs must not expose private checkout paths, internal commit
IDs, or private image-specific profile IDs. Exact source commit hashes may be
retained in this maintainer reference as internal provenance for image-to-image
comparisons.
Record evidence status in prose whenever it affects whether a claim is merely
source-compatible or actually release-tested.

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

- Deployment uses the exact Cosmos3 image
  `nvcr.io/nvstaging/nim/cosmos3:2.0.0-8bd2d17cdc1091ba`.
  `deployment.md` owns this reference; update it for each approved image and
  never replace it with `latest`.
- The source revision corresponding to this image is
  `8b4798a08154b854475b1b239ff8b33d55240666`. The previous recorded baseline,
  `4b505c903687efa80ffff65dd5d66753227ab8d4`, is its ancestor, and the complete
  range was reviewed for contract changes. Retain the current revision as the
  comparison baseline for the next image.
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
- Model-family compatibility remains floor-based, but practical guidance
  recommends Super on H200- and B200-class discrete GPUs when needed and Nano
  for generation on H100, RTX PRO 6000 Blackwell, lower-throughput discrete
  GPUs, and unified-memory systems. A fitting Super row on those hosts remains
  available but is not the default turnaround recommendation. Do not turn this
  guidance into a fixed latency claim.
- Transfer remains profile-compatible wherever its separate per-device memory
  floor is met, but its practical evaluation recommendation starts with an RTX
  PRO 6000 Blackwell 96-GB, H100 80-GB, or higher-throughput compatible
  discrete GPU. DGX Spark is not recommended for Transfer turnaround even when
  a compatible profile fits. Keep this operational recommendation distinct
  from support-matrix eligibility and do not turn it into an expected runtime.
- Cosmos3 Helm deployment guidance is not included. Keep `helm.md` limited to
  that status and do not adapt another NIM's chart or values.

## Current source-derived contracts

Track evidence provenance in this maintainer reference. Public pages state the
corresponding product behavior directly without mentioning source-code
provenance, and direct users to runtime interfaces where availability must be
confirmed:

- Generator BF16 compute capability 8.0, updated Super VRAM/Transfer floors,
  Reasoner Super BF16 TP2 at 73 GiB/device, and architecture-derived Reasoner
  precision preference: BF16 for compute capability 8.0 through 8.8, FP8 for
  8.9 through 9.x, and NVFP4 for 10.0 or newer;
- unified-memory selection subtracts the default 16-GiB host reserve from the
  reported shared-memory total for static profile floors, uses resident model
  and guardrails for Generator, and applies the same reserve to Reasoner
  current-free-memory admission. Current-state selection uses Linux
  `MemAvailable`, which includes the kernel's reclaimable-cache estimate,
  rather than CUDA's `MemFree` view. Startup also attempts a scoped
  `POSIX_FADV_DONTNEED` release for large files under the NIM cache on unified
  memory; that product behavior does not authorize an assistant to clear host
  caches. The current implementation records the Jetson AGX Thor T5000 as a
  123-GiB unified-memory measurement target with an approximately 6.8-GiB
  observed host working set, and identifies GB10/DGX Spark for unified-memory
  Reasoner runtime defaults;
- effective system-memory selection floors of 16 GiB for resident Generator and
  Reasoner profiles, 64 GiB for Nano offload, and 150 GiB for Super offload;
- current-free-VRAM startup selection before model materialization, the
  container preflight invocation, equivalent-layout fallback, explicit-pin
  failure, the Reasoner runtime reserve, and the explicit unified-memory
  utilization target, which is not reduced automatically;
- exact arbitrary Generator video frame counts within the mode and resolution
  bounds, generation at the next native `1 + 4k` count, trimming to the exact
  requested output count, and rounded V2V latent-index validation;
- Reasoner TP1 preference, TP2 availability fallback, guided-decoding
  enforcement, Responses create normalization, disabled-by-default video-token
  pruning with VidCom2 as the selected method when enabled, operator-level
  media I/O and multimodal processor JSON options, default-disabled multimodal
  input chunking, and the resulting minimum 16,384 batched-token budget;
- default-on Nano and Super DFlash drafts, independent local draft overrides,
  hardware-derived BF16 KV-cache selection, and advanced DFlash JSON
  configuration;
- quantized Generator linear-backend selection;
- the NIM API adaptation for the 18-case Reasoner catalog, including byte-equal
  vLLM user-prompt strings, data-URL media, request-level 4-FPS video sampling,
  disabled NIM-native thinking, preserved prompt-authored `<think>` text,
  explicit effective sampling controls, prompt-constrained JSON extraction,
  parse-status recording without structural validation, opt-in NIM JSON Schema
  guidance, and best-effort spatial annotation; and
- runtime-aware metadata, health responses, and wrong-runtime diagnostics.

Regenerate the source profile export before reconciling tables. Generated
artifacts can lag profile policy source and must not silently override current
implementation or be presented as an approved image manifest.

## Current image validation status

The exact image resolves to OCI manifest-list digest
`sha256:9fea8c0c0df52914e43d7c7e663fc9c7c7ca902fdeacb8b7e7b45f0ecfc9c2ee`,
with `amd64` manifest digest
`sha256:dad096ce3552363e7ebf2f1805f8b0da099679af396a6581e6b2f2dfea7a9950`
and `arm64` manifest digest
`sha256:3a4af0a82426c9b6b31703357ca5433e07c324f9d1275c7ca402668df35767dc`.
Its labels and embedded manifest report version and release `2.0.0`. The
manifest contains 115 Generator and 7 Reasoner profiles. Regenerating all 122
profiles from the recorded source revision produced the same normalized tag
inventory as the embedded manifest. All 19,937 embedded artifact references
use the public `ngc://nim/nvidia/cosmos3` model location; none uses an
`nvstaging` artifact URI.

On one NVIDIA H100 PCIe GPU with compute capability 9.0, 81,559 MiB total, and
81,081 MiB free, exact-image pre-download profile selection passed for Nano
Generator FP8 latency and Nano Reasoner FP8. These checks establish candidate
profile compatibility only. Direct image checks confirmed the Reasoner
precision preferences for `sm_80`, `sm_89`, `sm_90`, `sm_100`, and `sm_120`.
Option parsing confirmed that `NIM_DISABLE_CHUNKED_MM_INPUT` defaults to `true`,
the resulting effective `NIM_MAX_NUM_BATCHED_TOKENS` is `16384`, and
`NIM_GPU_MEMORY_UTILIZATION` defaults to `0.93`.

The image contains nonempty `/opt/nim/LICENSE`, `/opt/nim/MODEL_LICENSE`, and
`/opt/nim/NOTICE` files plus nine files under `/opt/nim/licenses`. The NIM terms
reference the NVIDIA Software License and Product-Specific Terms for NVIDIA AI
Products; the model terms identify OpenMDW-1.1. Checksums pass for the
package-modification source bundle under `/usr/share/cosmos3/oss-source`. Cold
start, management endpoints including `/v1/license`, and inference were not run
for this validation.

## Open validation and documentation gaps

Review this list on every substantive documentation update and remove, add, or
refine entries when evidence changes:

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
- current-free-VRAM fallback, Reasoner utilization handling on discrete GPUs,
  and discrete-GPU NVML startup behavior in the selected image;
- default-on Nano/Super DFlash, draft override, KV-cache, and advanced
  configuration behavior in the selected image, including repeated-seed Super
  FP8 spatial-output correctness;
- quantized Generator linear-backend behavior in the selected image;
- Cosmos3 Helm chart, approved values, installation workflow, and monitoring
  integration;
- repeated-seed vLLM/NIM review of the current catalog quality gaps, especially
  temporal segmentation, video-caption completion, assisted-task next action,
  trajectory semantics, and driving Action-CoT pedestrian identification; plus
  evaluation of explicitly enabled VidCom2 output quality on the video catalog;
- approved native-thinking output and reasoning-trace wording for the current
  image; and
- external product and model-card links associated with the exact image.

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
- update the exact image in deployment;
- inspect all unresolved requirements and confirm none implies an unsupported
  fixed value;
- validate links, JSON, examples, dependencies, paths, and ignored outputs;
- search for obsolete fields, legacy images/endpoints, realistic secrets,
  private paths, and unsupported backend syntax;
- report tests that could not run and do not present static checks as live NIM
  validation; and
- review maintainer `AGENTS.md`, `SKILL.md`, and both references, plus the
  customer-assistant instructions and skill when affected, so AI guidance
  remains synchronized with the public documentation.
