---
name: submit-trtllm-cookbook
description: Audit and submit TensorRT-LLM cookbook PRs by surveying what sibling backends already ship, checking the exact upstream serving and offline contracts, building through gpu-run, running every added or changed example, comparing against the reference implementation side by side, and delivering artifacts for human review. Use only for TensorRT-LLM cookbook or recipe work, not other inference frameworks.
---

# Submit a TensorRT-LLM Cookbook

Finish with a source-backed, runtime-backed cookbook PR, plus artifacts a human
can eyeball. Do not treat syntax checks, mocked responses, or a few
representative requests as validation of a larger example set.

## Know what you are shipping

A cookbook change is almost never one file. Enumerate the deliverables before
writing any of them:

| Artifact | Ship it when |
|---|---|
| Notebook `run_<scenario>_with_<backend>.ipynb` | the runnable walkthrough — usually the primary deliverable |
| Modality README (the cookbook folder's own) | quickstart snippet plus the "Notebook walkthrough" paragraph |
| Shared environment/setup guide | launch command for a new checkpoint, mode, or topology |
| Repository root index table | one row per notebook, describing scope honestly |
| Assets (prompts, negative prompts, reference media, specs) | new inputs the examples read |
| Benchmark page | only when you actually produced numbers |

If a change advertises a capability, every place that lists capabilities must
agree. Find the sibling entries for an adjacent backend and match their shape
and altitude rather than inventing a new format.

### Notebook hygiene without a Jupyter session

Assume notebooks are edited as JSON, by an agent or from an editor that cannot
execute them. These checks stand in for "I opened it and it looked fine", and
are mandatory before pushing:

- the file parses as JSON, and cell ids are unique
- every code cell has `outputs: []` and `execution_count: null`
- every code cell compiles: `compile("".join(cell["source"]), cell["id"], "exec")`
- the file round-trips through the repo's own serialization (match the existing
  indent and `ensure_ascii` settings) so the diff shows only intended lines
- `git diff --check` is clean
- whatever notebook CI the repository defines passes at the pushed head

Prefer patching notebooks with a script that asserts on its own match counts
over hand-editing JSON; a silent zero-match replacement is the common failure.

To execute a notebook headlessly, drive it with `nbconvert --to notebook
--execute` from a checkout, so repo-root discovery and relative asset paths
resolve exactly as they do for a reader. Executing a hand-written script that
merely resembles the notebook is not evidence the notebook runs.

## Survey before you design

Do not invent a scenario list. Derive it.

1. Identify every sibling backend the cookbook already supports for this
   modality (reference implementation, other serving stacks, in-process APIs).
2. For each, enumerate the scenarios it actually **runs**. A manifest or config
   entry with no call site is declared, not shipped — check for the call site.
3. Take the union. That is the candidate set a reader will expect.
4. Intersect it with what the target revision genuinely supports, established
   from source (see the audit section). For every candidate you drop, record
   the concrete reason and the source evidence.

State the resulting coverage explicitly in the PR: which scenarios ship, which
do not, and why. "Matches what backend X already ships" is a much stronger
justification than a list you chose.

### Report what is already stale

While surveying, audit the current cookbook and report findings to the user
rather than silently fixing them:

- capability claims that no longer match the implementation
- launch commands referencing moved, renamed, or deleted configs
- checked-in assets that have drifted from the upstream defaults they mirror
- index rows pointing at renamed or removed notebooks
- pinned versions, tags, or revisions that have been superseded

Fix only what is in scope for the change at hand; surface the rest as a list so
the user can decide. Drive-by fixes that a change genuinely depends on are fine,
but call them out separately and explain the dependency.

## Establish the exact revisions

Record these before changing the cookbook:

- Cookbook repository, PR URL, base branch and SHA, head branch and SHA.
- TensorRT-LLM repository, selected ref and resolved commit SHA.
- Build fingerprint: container tag and image ID/digest, or the native build's
  reported version string.
- Reference implementation repository and resolved SHA.
- Checkpoint identifiers and revisions when they are pinned.

Use current TensorRT-LLM `main` when the required implementation has merged. Use
a TensorRT-LLM PR head only when the cookbook intentionally targets an unmerged
change. Never keep testing an old PR head merely because the cookbook originally
depended on it. Re-resolve the selected ref immediately before the build.

Before building, check whether a released tag already contains the commits the
cookbook needs (`git tag --contains <sha>`). If one does, and the cookbook tells
readers to use released artifacts, validating against that tagged artifact is
more honest than validating a source build they will never run. Say which you
chose and why.

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

**A change to a shared asset or helper expands the matrix.** Editing a prompt
file, payload builder, or request helper that other examples also read puts
those examples back in scope. Either run them or mark them `BLOCKED` with the
cost; never assume an untouched example is unaffected by a file it reads.

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
5. Synchronous or asynchronous response handling, content retrieval, and error
   behavior.

Resolve the exact content type, field names, JSON-encoded subfields, uploaded
media fields, defaults, constraints, lifecycle endpoints, and returned artifact
format. Read implementation tests as additional evidence, but treat current
runtime code as authoritative.

Pay particular attention to values the request does **not** send. Server-side
defaults are selected by other fields, and a parameter you omit is still a
decision someone made on your behalf. Trace which branch each omitted parameter
lands in and what it implies, rather than assuming omission is neutral.

For offline examples, trace the CLI or Python entry point through config
loading, preprocessing, model invocation, and serialization. Resolve required
files, accepted shapes and lengths, defaults, output names and formats, and GPU
topology from the implementation and checked-in configs.

Compare those findings with the actual notebook/script request construction and
launch commands. A matching endpoint name alone is not enough.

## Build and run through gpu-run

Before any GPU work, read `~/code/gpu_run/manuals/agent-manual.md` completely and
follow its current instructions. Do not use direct `ssh`, manual synchronization,
`scancel`, or `gpu-run cancel`.

Build the selected TensorRT-LLM revision from source through `gpu-run`, by
whichever path the environment documents (source container, or a native build in
a persistent workspace). Do not silently substitute a released wheel or stale
prebuilt image for a source build you claimed to run. Use the checkout's current
build entry point rather than preserving a command from an older PR. Record the
build command, run ID, source SHA, and build fingerprint.

Use the smallest GPU allocation the exact server config or offline flow requires.
Follow the user's standing policy on allocation; where none is established, ask
before requesting more than one GPU, a different GPU family, or a long lease. If
the user names an existing allocation, adopt that exact node and do not submit
another job. Before loading a model, perform read-only ownership and GPU-activity
checks; if another worker is using the device, stop and report rather than
competing for it. Obtain approval before an unrequested image pull, checkpoint
download, container launch, or new allocation.

Use asynchronous gpu-run mode for long builds or servers and attach to the
recorded run instead of polling or starting a duplicate. Treat infrastructure
loss separately from a program or test failure, as the gpu-run manual requires;
node reallocation mid-run is normal and the workspace survives it.

Release allocations when the work is finished, and say that you did.

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
  traceback. Confirm the route from the server's own access log, not from the
  client's intent.
- The response status, content type, and lifecycle match the source contract.
- The artifact can be decoded by the appropriate tool.
- Image/video dimensions, frame count, frame rate, audio streams, tensor keys,
  shapes, or other mode-specific invariants match the example.
- Server logs confirm the intended model, mode, and input path.

Record whether guardrails or optional safety components were enabled. A run with
a component disabled or unavailable can validate generation, but it is not
certification of that omitted component.

If a row fails, inspect the captured client and server logs. Determine whether
the cookbook, the selected TensorRT-LLM revision, model access, or infrastructure
is responsible. Patch only from source and runtime evidence, then rerun that row
and every row affected by the shared code. Preserve failed attempts in the
ledger.

After runtime fixes, run the repository's notebook JSON, output-clearing, link,
formatting, lint, and other applicable cookbook checks. If the cookbook head,
base integration, TensorRT-LLM SHA, build fingerprint, or a shared request helper
changes materially, invalidate and rerun the affected rows.

## Compare against the reference implementation

A TensorRT-LLM example is only as trustworthy as its agreement with the
reference implementation the checkpoint was published against. For every
scenario in the matrix, run the reference path as well as the TensorRT-LLM path.

Match everything that can be matched: same inputs, same seed, same
sampler/steps/guidance, same resolution and length. Where the two stacks'
documented defaults disagree, prefer the model card or reference defaults and say
so — otherwise a visible difference only means you compared two configurations.

Then build one combined artifact per scenario, with both sides labelled and on
equal footing:

- **video** — horizontal stack, equal frame counts, re-encoded from decoded
  frames so source bitrate differences do not confound the comparison
- **image** — side by side with a label bar
- **numeric output** (actions, trajectories, embeddings) — paired or overlaid
  plots on shared axes, plus the summary statistics that matter
- **text** — a diff

Interpretation rules:

- Cross-backend sampling is not bit-reproducible. Compare composition and
  behavior, not pixels. Where the project defines perceptual or numeric gates,
  those gates are the authority, not your impression.
- One sample per scenario is an observation, not a quality judgement. Say which
  it is.
- A large, structured divergence is a finding worth chasing. It usually means the
  two paths are not running the same configuration, and the cause is more often
  a parameter nobody sent than a genuine backend difference.
- If the reference cannot run a scenario, record that; do not drop the row.

## Deliver artifacts for review

Human review needs files, not adjectives. Save everything worth looking at to:

```
~/sim/<project>/<feature-or-checkpoint-name>/
```

Put in it, with backend-prefixed names so the pairing is obvious:

- raw outputs from each backend, per scenario
- the combined side-by-side artifact per scenario
- the executed notebook
- the server log
- the exact request payloads sent

Then **tell the user the directory path** and list the contents with shapes,
sizes, and what each file is. Do not make them hunt. State plainly which
artifacts you inspected yourself and which you only verified mechanically, so
they know what still needs their eyes.

## Ready-for-review gate

Do not mark the PR ready while any required row is `TODO`, `FAIL`, or `BLOCKED`.
All advertised examples must be `PASS`, or explicitly `EXCLUDED` with evidence
and the corresponding unsupported material removed or clearly scoped out. In
particular, do not say "representative GPU inference passed" when untested
variants remain in the PR.

Before handoff, verify:

- The PR has no unresolved base conflicts.
- The tested cookbook head is the pushed head.
- The tested TensorRT-LLM SHA and build fingerprint are recorded.
- Every matrix row has direct evidence, including rows pulled in by shared-asset
  or shared-helper changes.
- The scenario survey and its exclusions are stated in the PR.
- Repository checks pass at the pushed head.
- Comparison artifacts exist and the review directory has been reported.
- Stale-cookbook findings have been surfaced to the user.
- The PR description no longer contains stale dependency, draft, or untested
  claims.

Inspect `git status`, every intended diff, and the staged diff. Stage explicit
paths only, commit, push, and include clickable commit links in the proposed PR
response.

The final report should contain the two source SHAs, build fingerprint, GPU/node
summary, a compact all-row result table, artifact checks, the review directory
path, repository checks, known limitations, and commit links. Drafting that
report does not grant permission to post it. Post or resolve GitHub content only
when the task or user authorizes that exact destination, and always obtain
separate explicit permission before posting to Slack.
