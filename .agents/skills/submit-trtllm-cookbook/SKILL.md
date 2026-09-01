---
name: submit-trtllm-cookbook
description: Audit and submit TensorRT-LLM cookbook PRs by checking the exact upstream serving and offline contracts, building the source container through gpu-run, and running every added or changed example. Use only for TensorRT-LLM cookbook or recipe work, not other inference frameworks.
---

# Submit a TensorRT-LLM Cookbook

Finish with a source-backed, runtime-backed cookbook PR. Do not treat syntax
checks, mocked responses, or a few representative requests as validation of a
larger example set.

## Establish the exact revisions

Record these before changing the cookbook:

- Cookbook repository, PR URL, base branch and SHA, head branch and SHA.
- TensorRT-LLM repository, selected ref and resolved commit SHA.
- Container tag and immutable image ID or digest after the build.
- Checkpoint identifiers and revisions when they are pinned.

Use current TensorRT-LLM `main` when the required implementation has merged.
Use a TensorRT-LLM PR head only when the cookbook intentionally targets an
unmerged change. Never keep testing an old PR head merely because the cookbook
originally depended on it. Re-resolve the selected ref immediately before the
build.

Reconcile the cookbook branch with its current base before expensive testing.
Inspect conflicts semantically; a mechanically clean merge does not prove that
the examples still describe the current API.

## Keep a complete execution ledger

Create or refresh `.plans/submit-trtllm-cookbook.md` in the cookbook worktree.
Keep it uncommitted unless the repository or user wants plans checked in. Start
it with the revision fingerprints above and a test matrix containing:

| ID | Cookbook/example | Interface | Server/config | Model/variant | Input/control | Expected artifact | Status | Evidence |
|---|---|---|---|---|---|---|---|---|

Use `TODO`, `PASS`, `FAIL`, `BLOCKED`, or `EXCLUDED`. Expand loops, parameter
lists, model families, distilled variants, control types, and advertised
capabilities into separate rows. Include commands in Markdown cells as well as
code cells. `EXCLUDED` requires a concrete reason and upstream source evidence;
unsupported material should normally be removed from the cookbook. An example
that is expensive or needs more GPUs is `BLOCKED`, not implicitly covered by a
smaller representative run.

Discover the matrix from the full PR diff against the current base. Enumerate
every added or changed notebook, script, recipe, launch command, serving flow,
offline flow, and README claim. Do not rely on a hand-maintained list from an
earlier review.

For every execution, record the exact command, `gpu-run` run ID, node and GPU
count, timestamps, result, relevant server log location, and output artifact
metadata. This ledger is the source for the final PR report.

## Audit TensorRT-LLM from source

Inspect the selected TensorRT-LLM checkout before trusting its documentation.
Use `rg` to trace each cookbook input end to end.

For serving examples, trace:

1. Route and HTTP method.
2. Request schema and multipart or JSON parsing.
3. Preprocessing and model-specific dispatch.
4. Worker/model invocation and supported modes.
5. Synchronous or asynchronous response handling, content retrieval, and
   error behavior.

Resolve the exact content type, field names, JSON-encoded subfields, uploaded
media fields, defaults, constraints, lifecycle endpoints, and returned artifact
format. Read implementation tests as additional evidence, but treat current
runtime code as authoritative.

For offline examples, trace the CLI or Python entry point through config
loading, preprocessing, model invocation, and serialization. Resolve required
files, accepted shapes and lengths, defaults, output names and formats, and GPU
topology from the implementation and checked-in configs.

Compare those findings with the actual notebook/script request construction and
launch commands. A matching endpoint name alone is not enough.

## Build through gpu-run

Before any GPU work, read
`~/code/gpu_run/manuals/agent-manual.md` completely and follow its current
instructions. Do not use direct `ssh`, manual synchronization, `scancel`, or
`gpu-run cancel`.

Build TensorRT-LLM's current source container from the selected checkout via
`gpu-run`; do not silently substitute a released wheel or stale prebuilt image.
Use the checkout's current container build entry point rather than preserving a
command from an older PR. Record the build command, run ID, source SHA, image
tag, and image ID or digest.

Use the smallest GPU allocation that the exact server config or offline flow
requires. Ask before requesting more than one GPU, a different GPU family, or a
long lease unless the user already authorized that exact cost. If the user
names an existing allocation, adopt that exact node and do not submit another
job. Before loading a model, perform read-only ownership and GPU-activity checks;
if another worker is using it, stop and report that instead of competing for
the device. Obtain approval before an unrequested image pull, checkpoint
download, container launch, or new allocation.

Use asynchronous gpu-run mode for long builds or servers and attach to the
recorded run instead of polling or starting a duplicate. Treat infrastructure
loss separately from a program or test failure, as the gpu-run manual requires.

## Prove and then encode the contract

When the request contract is uncertain, first make the smallest real request
succeed against the exact built revision. Use its observed request, response,
logs, and artifact to correct the cookbook. Then execute the corrected
checked-in example; an ad hoc curl or reconstructed Python snippet is not a pass
for notebook code that was never run.

Run cheap repository checks before GPU inference, but never use them as a
replacement for it. Start the documented server or offline command, wait for a
real readiness signal, and execute every row in the ledger. Reuse a compatible
server across rows when that does not change the documented behavior.

For each row, verify more than process exit status:

- The request reached the expected route and implementation without a server
  traceback.
- The response status, content type, and lifecycle match the source contract.
- The artifact can be decoded by the appropriate tool.
- Image/video dimensions, frame count, frame rate, audio streams, tensor keys,
  shapes, or other mode-specific invariants match the example.
- Server logs confirm the intended model, mode, and input path.

Record whether guardrails or optional safety components were enabled. A run
with a component disabled or unavailable can validate generation, but it is not
certification of that omitted component.

If a row fails, inspect the captured client and server logs. Determine whether
the cookbook, the selected TensorRT-LLM revision, model access, or infrastructure
is responsible. Patch only from source and runtime evidence, then rerun that row
and every row affected by the shared code. Preserve failed attempts in the
ledger.

After runtime fixes, run the repository's notebook JSON, output-clearing, link,
formatting, lint, and other applicable cookbook checks. If the cookbook head,
base integration, TensorRT-LLM SHA, container, or shared request helper changes
materially, invalidate and rerun the affected rows.

## Ready-for-review gate

Do not mark the PR ready while any required row is `TODO`, `FAIL`, or `BLOCKED`.
All advertised examples must be `PASS`, or explicitly `EXCLUDED` with evidence
and the corresponding unsupported material removed or clearly scoped out. In
particular, do not say "representative GPU inference passed" when untested
variants remain in the PR.

Before handoff, verify:

- The PR has no unresolved base conflicts.
- The tested cookbook head is the pushed head.
- The tested TensorRT-LLM SHA and container fingerprint are recorded.
- Every matrix row has direct evidence.
- Repository checks pass at the pushed head.
- The PR description no longer contains stale dependency, draft, or untested
  claims.

Inspect `git status`, every intended diff, and the staged diff. Stage explicit
paths only, commit, push, and include clickable commit links in the proposed PR
response.

The final report should contain the two source SHAs, container fingerprint,
GPU/node summary, a compact all-row result table, artifact checks, repository
checks, known limitations, and commit links. Drafting that report does not grant
permission to post it. Post or resolve GitHub content only when the task or user
authorizes that exact destination, and always obtain separate explicit
permission before posting to Slack.
