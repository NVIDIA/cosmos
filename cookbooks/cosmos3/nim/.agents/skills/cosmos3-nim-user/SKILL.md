---
name: cosmos3-nim-user
description: Guide customers through Cosmos3 Certified NIM deployment, endpoint verification, API example selection, hardware compatibility, and troubleshooting. Use for operating or calling the NIM from the public cookbooks/cosmos3/nim directory; do not use for maintaining the documentation itself.
license: OpenMDW-1.1
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM customer workflow

Work from `cookbooks/cosmos3/nim` and read `AGENTS.md`. Use the public pages in
this directory as the source of truth. If the user wants to edit these pages,
direct them to open `maintainer/` instead of applying this skill.

## 1. Identify the customer's path

Ask only for information needed to route the workflow:

1. Do they already have a reachable NIM endpoint, or must they deploy one?
2. Do they need Generator or Reasoner?
3. Which task do they want to perform?
4. For a new deployment, what GPU count, compute capability, per-device total
   and currently free VRAM, and effective host/container RAM are available?

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
preflight with the intended selectors and GPUs. Treat success as evidence for a
candidate profile only, not full host compatibility; cold start and
representative requests remain required. One service container starts one
runtime. The preflight container is temporary and `--rm` removes only that
container. Preserve a failed service container long enough to inspect its logs,
and ask before removing containers or cached data.
Before NGC login, explain the documented Docker credential-storage behavior and
external-helper option without presenting a helper as a prerequisite;
`--password-stdin` protects shell input, not Docker's stored credential. Include
`docker logout nvcr.io` when guiding evaluation cleanup, and explain that logout
does not revoke the NGC personal key.

For hardware guidance:

- evaluate every participating device's total and currently free memory against
  the per-device floor, including the documented Reasoner runtime reserve;
- never add VRAM across devices;
- use the Transfer minimum when Transfer must be served, and distinguish
  profile compatibility from the task's practical hardware recommendation:
  prefer an RTX PRO 6000 Blackwell 96-GB, H100 80-GB, or higher-throughput
  compatible discrete GPU for Transfer rather than DGX Spark;
- use the tested-GPU inventory as the official validation scope, not as a
  compatibility allowlist for every profile or task;
- use `uv run python examples/inspect_profile.py` to match the active profile
  from `/v1/metadata` to the YAML embedded in `/v1/manifest`, and use the
  support matrix for documented requirements;
- use the current RTX 5090 guidance and thresholds in `support-matrix.md` to
  distinguish ordinary generation from Transfer eligibility;
- explain that preflight/startup takes one free-memory snapshot and can fall
  back only to an equivalent lower-memory layout; an explicit profile pin never
  falls back;
- use the published floors to choose a candidate without claiming that they
  resolve unavailable general CPU, RAM, disk, shared-memory, driver, Docker, or
  Container Toolkit requirements; and
- leave requirements described as not yet available unresolved.

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
