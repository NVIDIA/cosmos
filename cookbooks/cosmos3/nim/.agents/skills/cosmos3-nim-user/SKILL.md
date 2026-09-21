---
name: cosmos3-nim-user
description: Guide customers through Cosmos3 Certified NIM deployment, endpoint verification, API example selection, hardware compatibility, and troubleshooting. Use for operating or calling the NIM from the public cookbooks/cosmos3/nim directory; do not use for maintaining the documentation itself.
license: OpenMDW-1.1
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM customer workflow

Work from `cookbooks/cosmos3/nim` and read `AGENTS.md`. Use the public pages in
this directory as the source of truth.

## 1. Identify the customer's path

Ask only for information needed to route the workflow:

1. Do they already have a reachable NIM endpoint, or must they deploy one?
2. Do they need Generator or Reasoner?
3. Which task do they want to perform?
4. For a new deployment, what GPU count, compute capability, per-device total
   and currently free VRAM, and effective host/container RAM are available? Is
   the device discrete or unified memory? For unified memory, also collect
   `MemFree`, `MemAvailable`, `Cached`, `SReclaimable`, and `Shmem`.

Do not ask for the value of `NGC_API_KEY`, another token, private input media, or
unredacted logs. It is sufficient to confirm that required secrets are set.

## 2. Use an existing endpoint

Help the customer establish `NIM_URL`, then use the documented client tooling
and perform the runtime preflight:

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -fsS "$NIM_URL/v1/health/ready"
curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
```

Require metadata to identify the intended runtime and endpoint before choosing
an example:

- Generator: `model_type` is `generator`; primary endpoint is `/v1/infer`.
- Reasoner: `model_type` is `reasoner`; primary endpoint is
  `/v1/chat/completions`.

Use `/v1/models` only after this runtime check. If metadata identifies the wrong
runtime, do not rewrite the request for that runtime; correct the deployment or
`NIM_URL`.

## 3. Prepare a new deployment

Guide the customer through the canonical pages in this order:

1. `prerequisites.md` for host and client preparation.
2. `support-matrix.md` for model, precision, GPU, VRAM, system-memory, and
   Transfer eligibility.
3. `deployment.md` for the exact image, authentication, pre-download profile
   preflight, cache, runtime selector, launch, readiness, logs, and cleanup.
4. `configuration.md` only when a documented non-default setting is needed.
5. The existing-endpoint preflight above before the first inference request.

Use only the exact image and commands in `deployment.md`; never substitute
`latest`. After the image pull, run the documented pre-download profile
preflight with the intended selectors and GPUs. For a Reasoner on DGX Spark/GB10
or Jetson AGX Thor, every preflight and launch command shown to the customer,
starting with the first one, must include
`-e NIM_GPU_MEMORY_UTILIZATION=0.80` for an image-only workload or
`-e NIM_GPU_MEMORY_UTILIZATION=0.70` for video or mixed-media workloads. Do not
show a generic Reasoner command first and add the setting later. Treat preflight
success as evidence for a candidate profile only, not full host compatibility;
cold start and representative requests remain required. One service container
starts one runtime. The preflight container is temporary and `--rm` removes
only that container. Preserve a failed service container long enough to inspect
its logs, and ask before removing containers or cached data.
Before NGC login, explain the documented Docker credential-storage behavior and
external-helper option without presenting a helper as a prerequisite;
`--password-stdin` protects shell input, not Docker's stored credential. Include
`docker logout nvcr.io` when guiding evaluation cleanup, and explain that logout
does not revoke the NGC personal key.

For hardware guidance:

- detect DGX Spark/GB10 and Jetson AGX Thor as unified-memory systems; for an
  unfamiliar device, use the exact-image preflight's unified-memory report
  rather than inferring from a marketing memory label;
- evaluate every participating device's total and currently free memory against
  the per-device floor, including the documented Reasoner runtime reserve; on a
  discrete-GPU host also require the practical host RAM minimum for the selected
  non-offloading model/precision row rather than treating the 16-GiB profile
  admission tag as sufficient;
- on unified memory, use `MemAvailable` for current state because it includes
  reclaimable cache; report `MemFree` and the cache components separately,
  never add cache to `MemAvailable`, and do not add the discrete-host practical
  RAM requirement to the single shared pool;
- never add VRAM across devices;
- use the Transfer minimum when Transfer must be served, and distinguish
  profile compatibility from the task's practical hardware recommendation:
  prefer an RTX PRO 6000 Blackwell 96-GB, H100 80-GB, or higher-throughput
  compatible discrete GPU for Transfer rather than DGX Spark;
- use the tested-GPU inventory as the official validation scope, not as a
  compatibility allowlist for every profile or task;
- recommend Super on H200- and B200-class discrete GPUs when the workload needs
  it; default to Nano for generation on H100, RTX PRO 6000 Blackwell, lower-
  throughput discrete devices, and all unified-memory devices; treat a fitting
  Super row on those systems as compatibility, not a performance
  recommendation;
- choose runtime, model variant, and Generator latency/throughput selectors
  first, normally leaving precision, offload, tags, and profile ID unset;
  Generator then prefers FP8 when compatible, while Reasoner prefers BF16 on
  compute capability 8.0 through 8.8, FP8 on 8.9 through 9.x, and NVFP4 on 10.0
  or newer; let preflight choose the exact image-specific profile;
- after cold start, require `examples/inspect_profile.py` to match the selected
  profile from `/v1/metadata` to the YAML embedded in `/v1/manifest`, and use
  the support matrix for documented requirements;
- use the current RTX 5090 guidance and thresholds in `support-matrix.md` to
  distinguish ordinary generation from Transfer eligibility;
- explain that preflight/startup takes one free-memory snapshot and can fall
  back only to an equivalent lower-memory layout; an explicit profile pin never
  falls back;
- use the published floors to choose a candidate without claiming that they
  define fixed general CPU, RAM, disk, shared-memory, driver, Docker, or
  Container Toolkit requirements; and
- leave requirements without a documented fixed minimum unresolved and require
  cold-start validation rather than inventing a value.

### Unified-memory decision procedure

1. Run the read-only memory report in
   `prerequisites.md#inspect-unified-memory-capacity-and-current-state` and
   present `MemFree`, `MemAvailable`, `Cached`, `SReclaimable`, `Shmem`, and the
   approximate reclaimable cache. Explain that `MemAvailable` already includes
   reclaimable memory.
2. If the customer needs Reasoner, classify the workload before presenting any
   Docker command. Use `NIM_GPU_MEMORY_UTILIZATION=0.80` for image-only or
   `0.70` for video or mixed media. Include the value explicitly in the first
   preflight command and every later Reasoner launch command; the `0.93` default
   is not reduced automatically on the unified pool.
3. Compare the requested profile family's floor with effective total capacity
   after the host reserve. If the floor exceeds effective total, report
   **exceeds total capacity**; cache reclamation cannot change that result. For
   example, Reasoner Super BF16 TP1 requires 135 GiB effective memory and cannot
   fit the DGX Spark shared pool.
4. If the floor fits effective total but preflight rejects current free memory,
   report **fits hardware but not the current memory state**. Identify active
   workloads and rerun preflight after the operator stops only those workloads.
5. Recommend model-level selectors, not an image-specific profile ID. On
   H200/B200-class hardware, a Generator Super request begins with
   `NIM_MODEL_TYPE=generator`, `NIM_MODEL_VARIANT=super`, and
   `NIM_PERF_PROFILE=latency`. On DGX Spark, Thor, H100, or RTX PRO 6000
   Blackwell, recommend `NIM_MODEL_VARIANT=nano` for practical generation
   turnaround unless the customer has an explicit Super requirement.
6. Run preflight with those selectors and the intended GPU visibility. Present
   a pass only as candidate-profile compatibility.
7. After startup, check readiness and `/v1/metadata`, then run
   `uv run python examples/inspect_profile.py`; report the model selector and
   the actual profile selected from that image.

Never clear host page cache, delete model cache, stop unidentified processes,
or reboot automatically. Present host page-cache reclamation only as an
approved diagnostic with its host-wide impact, require explicit operator
confirmation, and collect before-and-after memory and preflight evidence.

## 4. Route the task

Use the canonical task page and its committed example instead of constructing a
new transport or translating another backend's request:

| Customer intent | Runtime | Page | Primary example |
| --- | --- | --- | --- |
| Text-to-image | Generator | `generation.md` | `examples/t2i.py` |
| Text-to-video | Generator | `generation.md` | `examples/t2v.py` |
| Image-to-video | Generator | `generation.md` | `examples/i2v.py` |
| Video-to-video | Generator | `generation.md` | `examples/v2v.py` |
| Forward, policy, or inverse dynamics | Generator | `action.md` | `examples/action.py` |
| Controlled video generation | Generator | `transfer.md` | `examples/transfer.py` |
| Image or video understanding | Reasoner | `reasoning.md` | `examples/reasoner.py` |
| Responses API | Reasoner | `reasoning.md` | `examples/reasoner_responses.py` |
| Custom checkpoint | Either | `bring-your-own-checkpoint.md` | Follow the runtime-specific launch flow |

Initialize the pinned client environment through `prerequisites.md` before
running `uv run python examples/...`. Do not add ad hoc dependencies. The
Generator examples reject unknown CLI arguments before endpoint or media work,
then require metadata to identify the Generator runtime, `/v1/infer`, a
selected profile, and a model variant compatible with the example before they
submit inference. Do not bypass this preflight. For Reasoner, use
`examples/reasoner.py --list-cases` or the JSON `--describe`
mode to select from the complete catalog without contacting an endpoint. Then
follow `reasoning.md`, preserve the catalog's exact vLLM user-prompt text and
explicit effective sampling controls, keep its NIM-specific media transport,
and use the default prompt-constrained output path unless the customer asks to
exercise NIM guided output. Confirm the selected profile before treating output
as representative of that deployment. Literal `<think>` instructions in a
parity prompt are visible response text. Keep API/format validation separate
from the catalog's qualitative review criteria.

## 5. Troubleshoot safely

Use `operations.md` as the owner for health, logs, generic errors, resource
failures, and diagnostics. Use the task page for task-specific validation.

Collect the minimum sanitized evidence needed:

- the exact image identifier, without credentials;
- intended runtime and model variant;
- redacted `/v1/metadata` and, when configuration is relevant,
  `/v1/manifest`;
- GPU count, compute capability, and per-device total and free memory;
- HTTP status and response error envelope; and
- the relevant container log excerpt with tokens, private URLs, and media
  removed.

Before retrying a synchronous Generator or Transfer request, check whether it is
still active and inspect service health and logs. Treat the documented
30-minute general Generator and 60-minute Transfer client values as timeout
ceilings, not expected latency or an SLO. The Generator backend retains a
separate 30-minute default unless the operator changes it at container launch.

Do not use unsafe overrides as fixes. If an approved diagnostic requires one,
state its risk, limit it to that diagnostic, and restore the default afterward.

## 6. State the result

Tell the customer:

- what was checked;
- which runtime, page, and example apply;
- which commands are safe to run next;
- which assumptions or unresolved release requirements remain; and
- whether the conclusion comes from documentation or from observed endpoint
  output.
