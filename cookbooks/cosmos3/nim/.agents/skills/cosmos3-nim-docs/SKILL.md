---
name: cosmos3-nim-docs
description: Edit and review the Cosmos3 Certified NIM documentation, examples, client environment, and release-specific guidance. Use for documentation changes under cookbooks/cosmos3/nim, including RC bumps, API updates, page ownership changes, and consistency reviews.
license: OpenMDW-1.1
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 NIM documentation workflow

Work from `cookbooks/cosmos3/nim`. Read `AGENTS.md` before editing.

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
release evidence rather than relying on the existing prose alone.

## 2. Choose the canonical owner

Use `references/page-ownership.md`. Update the owning page first. Elsewhere,
repeat only the minimum needed to complete a workflow and link to the owner.
Do not create competing field, configuration, profile, or troubleshooting
references.

If the change affects a nearby Cosmos3 README outside this directory, edit it
only when that scope is authorized; otherwise report the stale integration
surface in the handoff.

## 3. Label the evidence correctly

Classify each changed claim as:

- **Source-derived:** established by current implementation, models, profiles,
  or tests but not yet validated in the selected image.
- **RC/release-validated:** observed from the exact image, manifest, OpenAPI, or
  approved release artifact.
- **Historical:** retained only for rationale or coverage.
- **Unresolved:** release-owned and kept visibly TBD in prose or a table.

Never upgrade source evidence to a tested support claim. Never use historical
values to fill an unresolved release fact.

## 4. Edit runnable material

- State the required runtime and endpoint near each request.
- Establish the working directory before relative commands.
- Use `uv run python examples/...` for project examples and `python3` only for
  direct host-side standard-library snippets.
- Keep commands copyable: no unresolved values, unsupported metavariables, or
  fake credentials in runnable fences.
- Keep NGC credentials out of shell history and source control.
- Keep request dictionaries, API calls, status handling, and primary outputs
  visible in task scripts.
- When dependencies change, update `pyproject.toml`, regenerate `uv.lock`, and
  update all affected commands and prerequisite text in the same change.

For an RC image bump, update `NIM_IMAGE` in `deployment.md` and the matching
entry in `release-notes.md`. Do not use `latest`. Keep Helm guidance descriptive
until an exact approved chart and version exist.

## 5. Synchronize editor guidance

Every public documentation or example change must review `AGENTS.md`, this
skill, and both reference files. Update them in the same change when needed:

| Change | Editor file to update |
| --- | --- |
| Page added, removed, renamed, or given new responsibility | `references/page-ownership.md` |
| Durable command, dependency, security, or style rule changed | `AGENTS.md` and, if procedural, `SKILL.md` |
| Source authority or editing/validation workflow changed | `SKILL.md` |
| RC/release status, validation boundary, or open gate changed | `references/evidence-and-release-gates.md` |

A wording-only correction normally requires review but no editor-file edit.
The final handoff should state whether editor guidance changed or remained
applicable.

## 6. Validate proportionally

Run checks appropriate to the change, without claiming unavailable live
coverage:

```bash
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
