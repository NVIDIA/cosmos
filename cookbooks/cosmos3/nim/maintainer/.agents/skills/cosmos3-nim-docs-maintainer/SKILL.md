---
name: cosmos3-nim-docs-maintainer
description: Edit and review the Cosmos3 Certified NIM public documentation, examples, client environment, and release-specific guidance. Use from cookbooks/cosmos3/nim/maintainer for documentation changes, including RC bumps, API updates, page ownership changes, and consistency reviews.
license: OpenMDW-1.1
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 NIM documentation maintainer workflow

Work from `cookbooks/cosmos3/nim/maintainer`. Read `AGENTS.md` before editing.
The public documentation root is `..`; run path-sensitive documentation and
validation commands from that directory.

Use these references on demand:

- [Page ownership](references/page-ownership.md) identifies the canonical home
  for each fact and the nearby integration surfaces.
- [Evidence and release gates](references/evidence-and-release-gates.md) defines
  source authority, unresolved release facts, and validation boundaries.

## 1. Classify the change

Identify whether the request changes:

- public wording or navigation;
- a runnable command or Python example;
- an API, configuration, profile, or task contract;
- client or host prerequisites;
- dependencies or the uv environment;
- an RC/release-owned value; or
- page ownership or the editing workflow itself.

Read every affected public page and example completely before editing. For a
contract change, inspect the authoritative current NIM source/tests or supplied
release evidence rather than relying on the existing prose alone. For profile
or support-matrix changes, regenerate the current source profile export and
compare its tags with profile policy code before editing; generated artifacts
can lag source and are not image-validation evidence.

## 2. Choose the canonical owner

Use `references/page-ownership.md`. Update the owning page first. Elsewhere,
repeat only the minimum needed to complete a workflow and link to the owner.
Do not create competing field, configuration, profile, or troubleshooting
references.

If the change affects a nearby Cosmos3 README outside this directory, edit it
only when that scope is authorized; otherwise report the stale integration
surface in the handoff.

## 3. Label the evidence correctly

Classify each changed claim for maintainer records. Keep the classification
internal: customer-facing pages state product behavior directly and never say
that the documentation or a behavior claim is based on source code.

Classify each changed claim as:

- **Source-derived:** established by current implementation, models, profiles,
  or tests but not yet validated in the selected image.
- **RC/release-validated:** observed from the exact image, manifest, OpenAPI, or
  approved release artifact.
- **Historical:** retained only for rationale or coverage.
- **Unresolved:** release-owned and clearly identified as not yet available in
  prose or a table.

Never upgrade source evidence to a tested support claim. Never use historical
values to fill an unresolved release fact.

## 4. Edit runnable material

- State the required runtime and endpoint near each request.
- Establish the working directory before relative commands.
- Keep the one-time pinned environment initialization in `prerequisites.md`;
  link to it from task pages instead of repeating it.
- Use `uv run python examples/...` for project examples and `python3` only for
  direct host-side standard-library snippets.
- Keep commands copyable: no unresolved values, unsupported metavariables, or
  fake credentials in runnable fences.
- Keep NGC credentials out of shell history and source control.
- Keep request dictionaries, API calls, status handling, and primary outputs
  visible in task scripts.
- Do not present a timeout, one observed run, or complete client-command wall
  time as expected service latency. Require release-specific configuration and
  measurement evidence before publishing performance expectations.
- When dependencies change, update `pyproject.toml`, regenerate `uv.lock`, and
  update all affected commands and prerequisite text in the same change.

For a pre-release image bump, update `NIM_IMAGE` in `deployment.md`. Do not use
`latest`. Keep the exact approved staging Helm reference in `helm.md`, but keep
pull/install guidance conceptual until the public chart URL, schema, and
workflow are approved. Replace the staging reference when that public release
artifact is available. Keep acknowledgement inventory visibly unavailable until
approved notices exist, and keep release notes limited to the initial unified
release rather than migrations between unreleased development contracts.

## 5. Synchronize editor guidance

Every public documentation or example change must review `AGENTS.md`, this
skill, and both reference files. Review the customer-assistant instructions and
skill when a change affects customer routing, commands, or safety. Update them
in the same change when needed:

| Change | Editor file to update |
| --- | --- |
| Page added, removed, renamed, or given new responsibility | `references/page-ownership.md` |
| Durable command, dependency, security, or style rule changed | `AGENTS.md` and, if procedural, `SKILL.md` |
| Source authority or editing/validation workflow changed | `SKILL.md` |
| RC/release status, validation boundary, or open gate changed | `references/evidence-and-release-gates.md` |
| Customer workflow, task routing, command, or operational-safety behavior changed | `../AGENTS.md` and `../.agents/skills/cosmos3-nim-user/SKILL.md` |

A wording-only correction normally requires review but no editor-file edit.
The final handoff should state whether editor guidance changed or remained
applicable.

## 6. Validate proportionally

Run checks appropriate to the change, without claiming unavailable live
coverage:

```bash
cd ..
uv lock --check
uv run --locked python -m compileall -q examples
git diff --check
```

Also:

- verify changed local Markdown targets and path-sensitive commands;
- parse changed JSON examples;
- search for stale command forms, obsolete fields, unresolved runnable values,
  realistic secrets, and duplicated RC image references;
- run CLI `--help` or import smoke tests when runner behavior changes; and
- run live requests only when the exact image and required hardware are
  available, recording the image reference and active runtime.

Before finishing, reread the complete diff for consistency between public docs,
examples, dependency metadata, `AGENTS.md`, and this skill. Report static and
live validation separately.
