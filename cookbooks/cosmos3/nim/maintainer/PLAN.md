<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM documentation creation plan

> Authoring artifact maintained on the `egor/nim_docs_update` branch. This is a
> project execution plan, not an end-user guide.
>
> This plan defines the execution order, page ownership, validation gates, and
> completion criteria for the public documentation. See [SOURCES.md](SOURCES.md)
> for source provenance, resolved contracts, discrepancies, and the detailed
> coverage matrices.

## Current status and authorization boundary

- Source discovery and information-architecture planning are complete.
- Public documentation has been refreshed against Cosmos3 NIM source commit
  `22e36fd6d8a5c2eb709b1ec937d4bb5ad1a36480`.
- The user journey now starts with runtime, model, optional precision, and the
  Generator latency/throughput choice; profile IDs are advanced controls.
- Maintainer planning and provenance live under `maintainer/`, outside the
  public guide navigation.
- Static validation for this refresh is tracked below; released-image
  validation remains pending.
- Release-owned facts remain `TBD (release-dependent)` until authoritative
  release evidence is available.

## Objective

Create standalone, human- and AI-readable documentation for the unified Cosmos3
Certified NIM under `cookbooks/cosmos3/nim` on branch
`egor/nim_docs_update`.

The final guide set must:

- cover no less than the durable user-facing topics in the previous official
  Cosmos3 Generator documentation;
- preserve the applicable historical Reasoner/VLM coverage;
- retain the deleted historical first-party `documentation.md` as a coverage
  floor while deriving current behavior from implementation, tests, generated
  profiles, and live release OpenAPI;
- describe only API behavior supported by the current Certified NIM source or
  validated released image;
- use the current cookbook's structure, terminology, links, code-fence, table,
  asset, and SPDX conventions;
- distinguish source-derived claims, release-validated claims, historical
  context, and release-dependent TBDs; and
- provide complete, editable examples without exposing credentials or private
  source-tree details.

## Planned public artifacts

```text
cookbooks/cosmos3/nim/
├── README.md
├── release-notes.md
├── prerequisites.md
├── deployment.md
├── configuration.md
├── support-matrix.md
├── helm.md
├── bring-your-own-checkpoint.md
├── api-reference.md
├── generation.md
├── reasoning.md
├── action.md
├── transfer.md
├── operations.md
├── acknowledgements.md
└── examples/
    ├── common.py
    ├── t2i.py
    ├── t2i_4step.py
    ├── t2v.py
    ├── i2v.py
    ├── i2v_4step.py
    ├── v2v.py
    ├── reasoner.py
    ├── reasoner_stream.py
    ├── reasoner_responses.py
    ├── action.py
    └── transfer.py
```

Use a hub-and-spoke structure. `README.md` is the concise entry point; focused
pages own distinct fact classes; plain-Python scripts own the complete runnable
examples. Do not replace this with one large document, one file per endpoint,
or a hand-maintained machine-readable manifest.

## Canonical page ownership

| Artifact | Canonical responsibility |
| --- | --- |
| `README.md` | Product scope, runtime/model-first selection model, capability index, first requests, and guide navigation |
| `release-notes.md` | Released versions, image tags, compatibility changes, limitations, and upgrade guidance |
| `prerequisites.md` | Host hardware/software, storage, shared memory, NGC access, and setup verification |
| `deployment.md` | `NGC_API_KEY`, Docker login, cache, launch flags, ports, selectors, readiness, and shutdown |
| `configuration.md` | Shared, Generator, Reasoner, selection, and prompt-upsampling environment variables |
| `support-matrix.md` | Released model, precision, GPU, VRAM, profile, offload, and codec compatibility |
| `helm.md` | Kubernetes prerequisites, secrets, values, GPUs, storage, probes, rollout, and verification |
| `bring-your-own-checkpoint.md` | Generator and Reasoner checkpoint sources, layouts, mounts/downloads, profile validation, launch, and verification |
| `api-reference.md` | Runtime routing, common Generator top-level fields, strict JSON behavior, common response, and live schema |
| `generation.md` | T2I, T2V, I2V, V2V, conditioning media, frame/resolution rules, prompt upsampling, output decoding, reproducibility, generation failures |
| `reasoning.md` | Chat Completions, Responses, streaming, image/video media, sampling, structured outputs, prompt/task guidance |
| `action.md` | Complete `action_params` contract, forward dynamics, policy, inverse dynamics, domains, action shapes, and response |
| `transfer.md` | Complete `transfer` contract, controls, defaults, derived/precomputed forms, combinations, and chunking |
| `operations.md` | Health/readiness, management endpoints, generic errors, metrics, logs, guardrails, diagnostics, and troubleshooting |
| `acknowledgements.md` | Approved third-party notices for the exact released image only |
| `examples/` | Minimal editable requests with API calls and primary response handling visible in each script; only strict local-media encoding and image/video decoding are shared |

When a workflow needs a fact owned by another page, summarize only what is
needed and link to the canonical section. Do not duplicate full field,
environment-variable, profile, or troubleshooting tables.

## Execution plan

### Phase 0: Refresh evidence

1. Record branch, commit, and tracked worktree state for the cookbook, NIM,
   product-documentation, and framework repositories.
2. Diff the current NIM against the snapshots in `SOURCES.md`, focusing on:
   request models, Generator specialist contracts, Reasoner routing/model
   sources, environment variables, prompt upsampling, profile inputs/generation,
   and tests. Treat deleted source documentation as historical evidence.
3. Update the resolved contracts, coverage matrices, discrepancies, and TBD
   ledger before public drafting.
4. If a release image is available, capture Generator and Reasoner evidence
   separately: readiness, models, OpenAPI, requests, errors, metrics, and logs.

Gate: every changed source fact is reconciled or explicitly deferred.

### Phase 1: Create the public scaffold

1. Create only the agreed pages and `examples/` directory.
2. Add the current OpenMDW-1.1 SPDX notice to every new Markdown and Python
   artifact.
3. Add stable page titles, one-paragraph scope statements, and relative
   navigation links.
4. Add semantic placeholders only for ledger-backed release facts, for example
   `<NIM_IMAGE:TBD>`.

Gate: all navigation targets exist; no placeholder implies a usable value.

### Phase 2: Establish shared and task API contracts

Draft the compact `api-reference.md` and task-owned contracts from current
request models, routing code, tests, and live OpenAPI when available.

It must define:

- runtime routing and common Generator `/v1/infer` fields in
  `api-reference.md`;
- frame/resolution/media rules in `generation.md`;
- complete nested Action and Transfer contracts in their task pages;
- complete Reasoner API behavior in `reasoning.md`; and
- management endpoints and generic errors in `operations.md`.

Gate: every field and endpoint has one canonical owner; the compact API page
does not repeat task-specific tables.

### Phase 3: Create runnable cookbook examples

1. Keep `common.py` limited to strict MIME-aware local-media encoding and
   strict image/video decoding.
2. Maintain one local script for each planned task surface, showing its URL/client,
   request call, status handling, and primary output directly.
3. Reuse existing cookbook assets.
4. Use `http://localhost:8000` as the documented default through `NIM_URL`,
   unless the final release convention changes.
5. Use dynamic `/v1/models` discovery for Reasoner examples where appropriate.
6. Keep representative prompts, request fields, and case meaning synchronized
   with reviewed NIM fixtures while retaining the cookbook's direct-call
   teaching structure. Validate the local scripts independently against
   current API models, runtime routing, tests, and live OpenAPI when available.
7. Exclude internal commands, moving `main` URLs presented as pinned, local
   `file://` media, vLLM-Omni multipart endpoints, and unsupported fields.

Gate: every script parses, uses complete editable request dictionaries, names
the runtime/endpoint, handles its output, and contains no credential value.

### Phase 4: Write task guides

Draft in this order:

1. `generation.md`
2. `reasoning.md`
3. `action.md`
4. `transfer.md`

Each guide must include scope, prerequisites/link to deployment, active runtime
and endpoint, one minimal complete request, response/artifact handling,
task-specific parameters, common failures, and links to the API reference and
complete scripts.

Prompt upsampling belongs in generation, configuration, and operations; it is
not a separate generation endpoint. Document it as optional, Generator-only,
limited to T2I/T2V/I2V, using a separate external-service secret, and falling
back to the original prompt on request-time failures.

Gate: every documented workflow has a local cookbook example or an explicit
reason why a runnable example is not currently provided.

### Phase 5: Write release, prerequisites, deployment, and platform pages

Draft:

- `release-notes.md` with release-owned identity and compatibility placeholders;
- `prerequisites.md` with host requirements and verification;
- `support-matrix.md` with release-gated hardware/profile/media tables;
- `configuration.md` with environment-variable contracts;
- `helm.md` with Kubernetes setup and chart-owned values left TBD;
- `bring-your-own-checkpoint.md` with separate Generator local-path and
  Reasoner local/Hugging Face `NIM_MODEL_PATH` contracts; and
- `deployment.md` with:
  - creation and safe handling of an NGC personal API key;
  - runtime variable `NGC_API_KEY` and Docker username literal `$oauthtoken`;
  - image pull, cache permissions, GPU exposure, shared memory, ulimits, ports,
    cold-start behavior, readiness, cleanup, and profile selectors;
  - Generator versus Reasoner selection; and
  - links to the focused requirement, configuration, support, Helm, and BYOC
    pages.

Do not use `NGC_TOKEN` as a runtime variable. Do not invent the final image URL,
tag, driver floor, support matrix, or Helm identity.

Gate: a user can follow the minimal Docker flow with only explicitly identified
release TBDs, and every non-obvious flag is explained.

### Phase 6: Write operations

Draft `operations.md` from current runtime controls and release observations:

- liveness versus readiness and cold-start interpretation;
- health, model, metadata, version, manifest, license, metrics, and OpenAPI
  inspection as supported by each active runtime;
- log levels, distributed diagnostics, profile confirmation, and performance
  implications;
- Generator guardrails and the risks of disabling them;
- prompt-upsampling warnings, timeouts, response failures, and fallback;
- Prometheus/Grafana only with release-confirmed endpoints/metrics; and
- symptom/cause/fix tables covering deployment and every capability family.

Gate: no metric, endpoint, error string, or log sample is presented as current
without source or released-image evidence.

### Phase 7: Assemble the overview last

Draft `README.md` only after the focused pages and examples exist. Include:

- the unified selected-runtime mental model;
- capability, input/output, and endpoint matrix;
- the shortest NGC login, launch, and readiness path;
- one minimal Generator and one minimal Reasoner request;
- choose-your-task navigation; and
- release/model-card, license, and acknowledgements links when approved.

Gate: the overview summarizes existing canonical content and does not become a
second deployment or API reference.

### Phase 8: Add acknowledgements when available

Keep `acknowledgements.md` artifact-dependent. Populate it only from the
approved notices inventory for the exact released Certified NIM image. Do not
infer it from dependencies or copy the historical Generator acknowledgement
artifact.

Gate: notice provenance and publication approval are known. Until then, retain
an explicit TBD rather than fabricated content.

### Phase 9: Reconcile inbound cookbook links

If edits outside `cookbooks/cosmos3/nim` are authorized, update the root and
nearby Cosmos3 README surfaces that still advertise separate legacy images,
narrower Generator support, or unsupported Reasoner fields. Replace duplicated
setup with concise summaries and links to the new canonical guides.

If scope remains path-only, leave those files untouched and record the known
contradictions in the handoff or pull-request description.

Gate: no edited integration surface contradicts the new Certified NIM guide.

### Phase 10: Validate and review

Run all applicable gates:

1. Check Markdown links and generated anchors.
2. Parse every embedded JSON request.
3. Syntax-check every Python example.
4. Validate Generator bodies against current models or live OpenAPI.
5. Validate Reasoner request construction and live routes when available.
6. Smoke-test every documented capability/profile mode on supported hardware,
   or label the missing runtime validation explicitly.
7. Compare all defaults, ranges, routes, profiles, and media claims with current
   code and released-image evidence.
8. Audit all 49 previous Generator topic groups, all 36 Reasoner/VLM groups,
   all 21 current source-guide sections, and every local cookbook example.
9. Search for legacy images, vLLM-only endpoints/fields, `NGC_TOKEN`, local
   paths, internal commands, credentials, and unledgered placeholders.
10. Confirm acknowledgements and legal links use approved release artifacts.
11. Confirm generated outputs and credentials are untracked and ignored.
12. Review the complete guide set for canonical ownership, terminology,
    navigation, and human/AI readability.

Gate: every coverage-matrix row has an evidence-backed destination, an explicit
correction/exclusion, or a visible TBD; all runnable/static checks pass or have
a documented environment/runtime limitation.

## Release-dependent TBD policy

The following do not block source-based drafting, but they block publishing the
specific affected claim or command:

- final NGC image repository and tag;
- approved public product/release/model-card URLs and naming;
- released profile/support matrix and driver/toolkit floors;
- final Helm chart identity and values;
- published BYOC boundary;
- live Generator/Reasoner OpenAPI, metrics, logs, and media behavior;
- released prompt-upsampling integration contract;
- approved Reasoner reasoning-trace wording; and
- exact-image acknowledgements and license/EULA links.

Use `TBD (release-dependent)` and map every placeholder to the `SOURCES.md` TBD
ledger. Historical values must never fill a missing release fact implicitly.

## Completion criteria

The documentation project is complete only when:

- every planned non-deferred artifact exists and carries the repository SPDX
  notice;
- canonical facts are consistent across pages and examples;
- every required historical/current coverage row is resolved;
- every example is statically valid and runtime-validated where infrastructure
  permits, with missing live validation clearly identified;
- credentials and private/internal details are absent;
- all links, anchors, requests, outputs, and navigation are verified;
- every remaining TBD is genuinely release-owned and visibly labeled; and
- stale inbound documentation is updated when authorized or explicitly called
  out when outside scope.

## Progress checklist

- [x] Inventory source repositories and authority order.
- [x] Audit previous Generator and Reasoner/VLM documentation floors.
- [x] Audit the current source guide and API implementation.
- [x] Map capabilities and facts to canonical pages.
- [x] Record discrepancies, reusable assets, and release-dependent TBDs.
- [x] Agree on the hub-and-spoke information architecture.
- [x] Receive explicit authorization to draft public documentation.
- [x] Refresh the original source snapshots.
- [x] Reconcile the 2026-08-03 NIM drift: model variants, shared BYOC,
  Nano-DROID/action-only output, Transfer VRAM admission, profile-backed
  guardrail residency, and Reasoner video pruning.
- [x] Reconcile the 2026-08-04 drift: explicit Generator `model_mode`, renamed
  request fields, Nano Reasoner DFlash, and system-memory admission for Super
  BF16 offload profiles.
- [x] Create the public scaffold, API reference, and twelve Python files,
  including the shared helper and specialist four-step T2I/I2V requests.
- [x] Write the generation, reasoning, action, and transfer guides.
- [x] Write deployment, operations, acknowledgements, and the overview.
- [x] Split release notes, prerequisites, configuration, support matrix, Helm,
  and BYOC into focused canonical pages.
- [x] Reduce `api-reference.md` to shared routing and Generator envelope
  material; move mode-specific contracts to task and operations pages.
- [x] Keep provisional profile details structural and release-owned values TBD.
- [x] Simplify the user journey around runtime, model, optional precision, and
  Generator latency/throughput; leave exact profiles and tags as advanced.
- [x] Move planning and source provenance into the maintainer-only directory.
- [x] Validate Markdown links/anchors, fence labels, embedded JSON, SPDX headers,
  Python syntax, offline asset resolution, and request construction.
- [x] Run a 70-topic public coverage probe spanning the previous Generator,
  Reasoner/VLM, and current first-party guide requirements; every probe passed.
- [x] Statically check 14 Generator payloads against field names extracted from
  the authoritative request classes plus frame/action-shape invariants.
- [x] Add source-backed T2I request, JPEG response, prompt-upsampling, and
  guardrail documentation from the merged NIM `cosmos3` branch.
- [x] Establish independently maintained cookbook teaching examples with direct
  API calls, shallow media helpers, and one selected action/transfer request per
  invocation.
- [ ] Validate both runtimes against the final released image and resolve or
  retain the release-dependent TBD ledger.
- [ ] Update stale inbound cookbook/root NIM summaries if that broader scope is
  authorized; otherwise report them as known follow-up work.

## Authoring execution record

Public drafting was authorized and completed on 2026-07-27. Static validation
passed for the source available in this checkout.

The repository-prescribed locked NIM environment still cannot start because
building `nim-sdk` needs the unavailable private `sw-nemollm-rust` registry.
Static validation compiled every local cookbook example, checked public payload
fields and invariants, and exercised request builders that do not require a
live NIM.

The focused-page restructuring was completed in the same authoring pass.
`deployment.md` was reduced to Docker deployment and profile selection, while
`api-reference.md` was reduced to routing and the shared Generator envelope.
The split added release notes, prerequisites, configuration, support matrix,
Helm, and BYOC pages. Recursive links/anchors, SPDX headers, JSON, Python,
shell, YAML, example compilation, credential terminology, and
`git diff --check` were revalidated after redistribution.

The incremental T2I refresh was completed on 2026-07-29 from the merged NIM
`cosmos3` state at
`74064b2318222018af446b03701f8a8cbeaa28c3`. It added the one-frame request
contract, JPEG response and example, T2I prompt upsampling and
visual-guardrail coverage, and removed four obsolete Generator execution
variables.

The full update refresh was first performed from NIM source commit
`243e05f8eecb44766d90f2843adb46356ae77a17`, then advanced to
`22e36fd6d8a5c2eb709b1ec937d4bb5ad1a36480`. The latest refresh adds the
breaking explicit-mode Generator request contract, Nano Reasoner DFlash, and
system-memory admission for Super BF16 offload profiles. It supersedes the
older current contracts for profile inventory, BYOC, Generator request/response
modality, Action, Transfer admission, and advanced environment variables. The
source profile generator produced 122 development rows (115 Generator and 7
Reasoner) in a temporary output; those rows remain pre-release evidence.
Offline validation compiled all twelve examples, validated 16 cookbook
Generator payloads and eight documented JSON payloads against the latest
`Cosmos3Request`, parsed every JSON fence, checked local Markdown links/anchors
and SPDX headers, verified profile parallelism, DFlash artifacts, system-memory
tags, and all seven variants, asserted the refreshed source contracts, and
passed `git diff --check`.

A later editorial pass removed duplicated launch/configuration material,
reordered basic tasks before specialist behavior, reduced repeated status
language, made configuration progressive, removed speculative Helm values, and
moved this plan and source ledger under `maintainer/`. Recursive links, anchors,
JSON, SPDX, Python syntax, model/precision claims, and whitespace were checked
again.

Live API, profile, media/codec, metrics, log, Helm, BYOC, and acknowledgements
validation remains release-dependent and is labeled as such in the public
guides. No value from an older separate Generator or Reasoner NIM was used to
silently fill those gaps.
