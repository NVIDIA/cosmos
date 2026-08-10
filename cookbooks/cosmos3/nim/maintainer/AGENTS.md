<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM maintainer instructions

These instructions apply when maintaining the public documentation, Python
examples, and supporting metadata one directory above this folder. Open
`cookbooks/cosmos3/nim/maintainer` in the documentation editor so this file and
the maintainer skill are discovered. The public documentation root is `..`.

For substantive documentation work, load
`.agents/skills/cosmos3-nim-docs-maintainer/SKILL.md` and its references. These
maintainer instructions take precedence over the customer-assistant guidance in
`../AGENTS.md` when editing the documentation.

## Evidence and scope

- Prefer current Certified NIM implementation, request models, configuration,
  profiles, and tests. Use behavior observed from the selected RC or released
  image when available. Historical documentation is a coverage source, not an
  authority for current names, defaults, commands, or support claims.
- Distinguish source-derived, RC/release-validated, historical, and unresolved
  claims. Do not silently fill release-owned gaps from memory or old releases.
- Treat timeout ceilings, individual observations, and end-to-end client command
  times as distinct from validated service latency. Publish expected
  performance only with an approved release image, reference configuration, and
  measurement method.
- Keep private source paths, internal commit IDs, development profile IDs,
  credentials, and realistic secret values out of public documentation.
- Put each fact in its canonical page. Consult the skill's
  `references/page-ownership.md` before adding or duplicating reference
  material.

## Runnable examples and dependencies

- Establish the public documentation root (`..`) as the working directory
  before path-sensitive commands. Keep the one-time pinned environment setup in
  `prerequisites.md`; task pages link to that setup instead of repeating it. Run
  pinned examples as
  `uv run python examples/...`.
- Use `python3` for direct host-side standard-library commands and `python` only
  inside the uv project environment. Python Markdown fence labels remain
  `python`.
- Keep example dependencies in `pyproject.toml`, commit `uv.lock`, and update
  dependency metadata, documentation, and examples together. Do not document
  ad hoc `pip install` or `uv run --with` commands.
- Declare required client tools, minimum versions, and installation instructions
  before first use. Keep client tools separate from NIM host/container
  requirements.
- Every fenced command must use a usable value. State unresolved release values
  as not yet available in prose or tables, not as runnable placeholders.
- Never use `latest` for the NIM image. Before public release, keep the exact
  evaluation image reference in `deployment.md`.
- Do not invent Helm commands, chart names, versions, or values before an
  approved chart is published. Keep the Helm page explicit about availability.
- Keep release notes user-facing and concise. Until the first public release,
  document only the initial unified release and the current request and
  configuration contract, not development migration history.
- Keep acknowledgement status visible without inventing an inventory before the
  image-specific notices are approved.

## Editing and validation

- Preserve the OpenMDW-1.1 SPDX notice and existing Markdown style.
- Keep task scripts directly editable: show request construction, the API call,
  status handling, and primary output without hiding them behind a large helper
  abstraction.
- Validate affected links, paths, JSON, Python syntax, documented commands, and
  `uv.lock`. Report static checks separately from live NIM validation.
- Do not add CI workflows, repository automation, or unrelated tooling unless
  the user explicitly requests it.

## Keep editor guidance current

Every documentation change must review this `AGENTS.md` and
`.agents/skills/cosmos3-nim-docs-maintainer/` for affected instructions, page
ownership, source references, validation steps, and release gates. Update the
editor files in the same change whenever their guidance would otherwise become
stale:

- page additions, removals, or responsibility changes update
  `.agents/skills/cosmos3-nim-docs-maintainer/references/page-ownership.md`;
- command conventions or durable editing rules update this file and, when
  procedural, `SKILL.md`;
- source-contract and validation-workflow changes update `SKILL.md`;
- RC/release status and resolved or new release gates update
  `.agents/skills/cosmos3-nim-docs-maintainer/references/evidence-and-release-gates.md`;
- customer workflow, task routing, or operational-safety changes update
  `../AGENTS.md` and `../.agents/skills/cosmos3-nim-user/SKILL.md`; and
- dependency policy changes update this file, `../pyproject.toml`, `../uv.lock`,
  and the relevant public guides together.
