---
name: cosmos3-nim-user
description: Guide customers through Cosmos3 NIM deployment, endpoint verification, API example selection, hardware compatibility, and troubleshooting. Use for operating or calling the NIM from the public cookbooks/cosmos3/nim directory; do not use for maintaining the documentation itself.
license: OpenMDW-1.1
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 NIM customer workflow

Work from `cookbooks/cosmos3/nim` and follow `AGENTS.md`. The public pages here
are the source of truth; the rules below say which page owns each decision.

## 1. Identify the customer's path

Ask only what routes the workflow:

1. Is there a reachable NIM endpoint, or must one be deployed?
2. Generator or Reasoner, and which task?
3. For a new deployment: GPU model and count, compute capability, per-device
   total and free VRAM, effective host or container RAM, and whether the GPU is
   discrete or unified memory (DGX Spark/GB10, Jetson AGX Thor).

## 2. Use an existing endpoint

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -fsS "$NIM_URL/v1/health/ready"
curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
```

Metadata must show the intended runtime before you pick an example: Generator
reports `model_type: generator` and `/v1/infer`; Reasoner reports
`model_type: reasoner` and `/v1/chat/completions`. If the runtime is wrong, fix
the deployment or `NIM_URL`; do not rewrite the request for the other runtime.

## 3. Prepare a new deployment

Walk through `prerequisites.md`, `support-matrix.md`, then `deployment.md`
(`helm.md` for Kubernetes); open `configuration.md` only for a documented
non-default setting. Use the exact pinned image, never `latest`. Apply these
rules while doing so:

- **Model choice** ([practical recommendation](../../../support-matrix.md#practical-model-recommendation)):
  Super on H200- and B200-class GPUs when the workload needs it; Nano on H100,
  RTX PRO 6000 Blackwell, slower discrete GPUs, and all unified-memory systems.
  A Super row that fits those systems is compatibility, not a recommendation.
- **Precision** ([precision selection](../../../deployment.md#precision-selection)):
  leave it unset. Generator prefers FP8; Reasoner prefers BF16 on compute
  capability 8.0–8.8, FP8 on 8.9–9.x, and NVFP4 on 10.0 or newer.
- **Transfer** ([headroom](../../../support-matrix.md#transfer-headroom)): size
  against the Transfer minimum, not the generation minimum, and recommend an
  RTX PRO 6000 Blackwell 96 GB, H100 80 GB, or faster discrete GPU rather than
  DGX Spark.
- **Reasoner on DGX Spark or Jetson AGX Thor**
  ([memory share](../../../deployment.md#set-the-reasoner-memory-share-on-unified-memory-systems)):
  include `-e NIM_GPU_MEMORY_UTILIZATION=0.80` (images only) or `0.70` (video or
  mixed) in the first preflight command you show and in every launch command.
- **NGC login** ([authenticate](../../../deployment.md#authenticate-to-ngc)):
  mention that Docker stores the login unless a credential helper is configured
  (the helper is optional) and that `--password-stdin` protects only shell
  input. Include `docker logout nvcr.io` in evaluation cleanup; logout does not
  revoke the NGC key.
- **Host software:** recommend NVIDIA driver 595.58.03 or newer and cite the
  validated configuration in `prerequisites.md` as tested versions, not
  minimums. Do not invent other fixed CPU, RAM, disk, or Docker minimums.
- **GPU exposure** ([GPU selection](../../../configuration.md#gpu-selection)):
  select GPUs with Docker `--gpus`, not `CUDA_VISIBLE_DEVICES`.
- **Offline or air-gapped**
  ([procedure](../../../deployment.md#offline-and-air-gapped-deployment)):
  fill the cache online, copy it preserving symlinks, and do not set
  `NIM_DISABLE_MODEL_DOWNLOAD` for Generator.
- **Preflight** ([preflight](../../../deployment.md#run-the-pre-download-profile-preflight)):
  run it after the pull with the intended selectors and GPUs. A pass means a
  candidate profile; cold start and representative requests confirm the host.
  After startup, run `uv run python examples/inspect_profile.py` to report the
  profile actually selected.

When checking hardware, compare each device's total and free memory with the
per-device floor and never add memory across devices. On discrete GPUs, require
the support matrix's practical host RAM, not the 16-GiB admission tag. The
tested-GPU inventory is the validation scope, not an allowlist.

### Unified-memory decision procedure

1. Run the memory report in
   [prerequisites](../../../prerequisites.md#inspect-unified-memory-capacity-and-current-state)
   and show `MemFree`, `MemAvailable`, and the cache components. `MemAvailable`
   already includes reclaimable cache; do not add cache to it, and do not add a
   separate host-RAM requirement to the shared pool.
2. For Reasoner, classify the workload (images only vs video or mixed) before
   showing any command, and set the memory share as above.
3. If the profile floor exceeds the pool's effective capacity, report **exceeds
   total capacity** (for example, Reasoner Super BF16 TP1 on DGX Spark). If it
   fits but preflight rejects current free memory, report **fits hardware but
   not the current memory state** and ask the operator to stop only the
   identified workloads before rerunning preflight.
4. Recommend model selectors, not image-specific profile IDs.

Never clear page cache, delete model cache, stop unidentified processes, or
reboot automatically. Page-cache reclamation is an approved host-wide
diagnostic only, with explicit operator confirmation and before/after evidence.

## 4. Route the task

| Customer intent | Runtime | Page | Example |
| --- | --- | --- | --- |
| Text-to-image, text-to-video, image-to-video, video-to-video | Generator | `generation.md` | `examples/t2i.py`, `t2v.py`, `i2v.py`, `v2v.py` |
| Forward, policy, or inverse dynamics | Generator | `action.md` | `examples/action.py` |
| Controlled video generation | Generator | `transfer.md` | `examples/transfer.py` |
| Image or video understanding | Reasoner | `reasoning.md` | `examples/reasoner.py` |
| Responses API | Reasoner | `reasoning.md` | `examples/reasoner_responses.py` |
| Custom checkpoint | Either | `bring-your-own-checkpoint.md` | Runtime-specific launch flow |
| Metrics and dashboards | Either | `monitoring.md` | Prometheus and Grafana setup |
| License terms | Either | `license.md` | — |

Initialize the client environment from `prerequisites.md` first and add no
dependencies. Do not bypass the example clients' argument and metadata checks.
For Reasoner, pick a case with `examples/reasoner.py --list-cases` or
`--describe <case> --format json`, keep the catalog's prompt text and sampling
values, and use `--guided-output` only when the customer asks for enforced
structured output. Literal `<think>` text requested by a prompt is ordinary
response text.

## 5. Troubleshoot safely

`operations.md` owns health, logs, errors, and diagnostics; `monitoring.md` owns
metrics; task pages own task-specific validation. Collect only sanitized
evidence: image identifier, runtime and variant, redacted `/v1/metadata` (and
`/v1/manifest` when configuration matters), GPU count, compute capability and
per-device memory, HTTP status and error body, and a redacted log excerpt.

Generator requests are synchronous
([long-running requests](../../../operations.md#long-running-requests)): the
examples use 30-minute (Generator) and 60-minute (Transfer) client timeouts as
ceilings, not expected latency, and the backend still stops at 30 minutes unless
the container sets `NIM_TRITON_REQUEST_TIMEOUT`. Check whether a request is
still running before retrying. If a diagnostic needs an unsafe override, state
the risk, limit it to that diagnostic, and restore the default.

## 6. State the result

Tell the customer what was checked, which runtime, page, and example apply,
which commands are safe to run next, which assumptions remain, and whether the
conclusion comes from the documentation or from observed endpoint output.
