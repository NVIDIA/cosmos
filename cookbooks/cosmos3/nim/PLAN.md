# Cosmos3 Certified NIM documentation creation plan

> Local authoring artifact. Do not commit this file.
>
> This plan defines the execution order, page ownership, validation gates, and
> completion criteria for the public documentation. See [SOURCES.md](SOURCES.md)
> for source provenance, resolved contracts, discrepancies, and the detailed
> coverage matrices.

## Current status and authorization boundary

- Source discovery and information-architecture planning are complete.
- Public documentation drafting has not started.
- `README.md` must remain `# TBD` until the user explicitly authorizes drafting.
- Release-owned facts may remain `TBD (release-dependent)` while other sections
  are drafted from current implementation evidence.

## Objective

Create standalone, human- and AI-readable documentation for the unified Cosmos3
Certified NIM under `cookbooks/cosmos3/nim` on branch
`egor/cosmos3_nim_docs`.

The final guide set must:

- cover no less than the durable user-facing topics in the previous official
  Cosmos3 Generator documentation;
- preserve the applicable historical Reasoner/VLM coverage;
- cover every current first-party `documentation.md` section and example by
  documenting, correcting, intentionally excluding, or visibly deferring it;
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
├── deployment.md
├── api-reference.md
├── generation.md
├── reasoning.md
├── action.md
├── transfer.md
├── operations.md
├── acknowledgements.md
└── examples/
    ├── common.py
    ├── t2v.py
    ├── i2v.py
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
| `README.md` | Product scope, selected Generator/Reasoner runtime model, capability/endpoint index, minimum launch and first requests, guide navigation |
| `deployment.md` | Prerequisites, `NGC_API_KEY`, Docker login, cache, launch flags, ports, selectors, profiles/hardware, prompt-upsampling configuration, readiness, BYOC, Helm |
| `api-reference.md` | Endpoint inventory, request/response fields, defaults, ranges, media forms, validation, HTTP semantics, model and OpenAPI discovery |
| `generation.md` | T2V, I2V, V2V, conditioning media, prompt upsampling, output decoding, reproducibility, generation failures |
| `reasoning.md` | Chat Completions, Responses, streaming, image/video media, sampling, structured outputs, prompt/task guidance |
| `action.md` | Forward dynamics, policy, inverse dynamics, domains, action shapes, chunk/frame constraints |
| `transfer.md` | Edge, blur, depth, segmentation, WSM, derived/precomputed controls, combinations, chunking |
| `operations.md` | Health/readiness, inspection endpoints, metrics, logs, guardrails, prompt-upsampling fallback, diagnostics, troubleshooting |
| `acknowledgements.md` | Approved third-party notices for the exact released image only |
| `examples/` | Complete editable requests, shared media encoding, request transport, response decoding, safe artifact output |

When a workflow needs a fact owned by another page, summarize only what is
needed and link to the canonical section. Do not duplicate full field,
environment-variable, profile, or troubleshooting tables.

## Execution plan

### Phase 0: Refresh evidence

1. Record branch, commit, and tracked worktree state for the cookbook, NIM,
   product-documentation, and framework repositories.
2. Diff the current NIM against the snapshots in `SOURCES.md`, focusing on:
   request models, Reasoner routing, environment variables, prompt upsampling,
   profiles, tests, `documentation.md`, and `examples/`.
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

### Phase 2: Establish the API contract

Draft `api-reference.md` first from current request models, routing code, tests,
and live OpenAPI when available.

It must define:

- shared management endpoints versus the active Generator or Reasoner API;
- Generator `/v1/infer`, all current request modes, nested action/transfer
  structures, response shapes, defaults, ranges, conflicts, and errors;
- Reasoner Chat Completions, streaming, Responses routes, sampling/media
  extensions, model discovery, and validation boundaries; and
- accepted media representations while leaving unverified formats/codecs and
  public-URL behavior visibly gated.

Gate: task pages can reference one canonical field contract without inventing
or duplicating schema facts.

### Phase 3: Adapt runnable examples

1. Adapt `common.py` for readiness, MIME-aware data URLs, JSON/OpenAI requests,
   base64 MP4 decoding, action/text output, and safe artifact paths.
2. Adapt one script for each planned task surface.
3. Reuse existing cookbook assets; do not copy files from the proprietary NIM
   source tree.
4. Use `http://localhost:8000` as the documented default through `NIM_URL`,
   unless the final release convention changes.
5. Use dynamic `/v1/models` discovery for Reasoner examples where appropriate.
6. Exclude internal commands, moving `main` URLs presented as pinned, local
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

Prompt upsampling belongs in generation, deployment, and operations; it is not
a separate generation endpoint. Document it as optional, Generator-only,
limited to T2V/I2V, using a separate external-service secret, and falling back
to the original prompt on request-time failures.

Gate: every first-party example has been adapted, merged deliberately, excluded
with a reason, or left behind an explicit release-validation gate.

### Phase 5: Write deployment

Draft `deployment.md` with:

- prerequisites and release-dependent version placeholders;
- creation and safe handling of an NGC personal API key;
- runtime variable `NGC_API_KEY` and Docker username literal `$oauthtoken`;
- image pull, cache permissions, GPU exposure, shared memory, ulimits, ports,
  cold-start behavior, readiness, cleanup, and profile selectors;
- Generator versus Reasoner selection and released hardware/profile tables;
- prompt-upsampling endpoint/model/template/secret configuration;
- Generator-scoped BYOC unless the release proves a wider boundary; and
- Helm/Kubernetes setup with chart-owned values left TBD until confirmed.

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
   all 21 current source-guide sections, and every first-party example.
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
- [x] Audit the current source guide and all first-party examples.
- [x] Map capabilities and facts to canonical pages.
- [x] Record discrepancies, reusable assets, and release-dependent TBDs.
- [x] Agree on the hub-and-spoke information architecture.
- [ ] Receive explicit authorization to draft public documentation.
- [ ] Refresh sources and complete phases 1-10 above.
