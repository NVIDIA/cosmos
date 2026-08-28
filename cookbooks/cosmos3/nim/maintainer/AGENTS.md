<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM maintainer instructions

These instructions apply to the public documentation, examples, and metadata in
`..`. Open this `maintainer` directory in the editor and load
`.agents/skills/cosmos3-nim-docs-maintainer/SKILL.md` plus its references.
These instructions take precedence over `../AGENTS.md` for documentation work.

## Evidence and scope

- Prefer the current Certified NIM implementation and exact selected image.
  Historical material is a coverage source, not authority for current behavior.
- Distinguish source-derived, image-validated, and unresolved claims internally.
  Public pages state product behavior directly and do not mention source-code
  provenance.
- Keep maintainer references current rather than chronological. Remove
  superseded validation narratives when no active gate depends on them; Git
  history is the archive.
- Keep private paths, private image-specific profile IDs, credentials, and
  realistic secrets out of customer-facing documentation. Exact source commit
  hashes may be stored in `maintainer/` as internal evidence for image-to-image
  source comparisons;
  do not copy them into customer-facing pages or examples.
- Put each fact in its canonical page. Use
  `.agents/skills/cosmos3-nim-docs-maintainer/references/page-ownership.md`.
- Do not publish performance from individual runs, timeout ceilings, or complete
  client-command times.

## Runnable material

- Run path-sensitive commands from `..`. Use the pinned environment from
  `prerequisites.md`: `uv run python examples/...`; use `python3` only for
  direct host-side standard-library commands.
- Update `pyproject.toml`, `uv.lock`, examples, and documentation together when
  dependencies change. Do not document ad hoc installs or `uv run --with`.
- Keep commands copyable, declare tools before use, and never put unresolved
  values or `latest` in runnable commands. Preserve the workload-specific
  `NIM_GPU_MEMORY_UTILIZATION` setting in the first Reasoner preflight and
  launch commands for DGX Spark/GB10 and Jetson AGX Thor.
- Preserve the documented credential-helper guidance and evaluation logout.
  `--password-stdin` does not protect Docker's stored credential.
- Keep task scripts directly editable. Parse arguments before endpoint or media
  work, reject unknown arguments, and verify runtime, endpoint, selected
  profile, and compatible model variant before inference.

## Editing and validation

- For cluster RC checks, run the public Docker command with only these local
  adaptations: canonicalize `NIM_CACHE`, mount it at `/opt/nim/.cache`, run as
  `--user "$(id -u):$(id -g)"`, set `HOME` and `XDG_CACHE_HOME` under that
  cache, skip the public `chmod`, and use `--gpus all` under Slurm.
- Keep Reasoner semantic fixtures under `maintainer/`; never expose them through
  public pages or runner output.
- Preserve the OpenMDW-1.1 SPDX notice and existing Markdown style.
- Validate affected links, paths, JSON, Python, documented commands, and
  `uv.lock`. Report static checks separately from live NIM validation.
- Do not add CI, repository automation, or unrelated tooling unless requested.

## Keep editor guidance current

Review this file, the maintainer skill, and both references for every public
change. Update only the affected guidance:

- page ownership changes update
  `.agents/skills/cosmos3-nim-docs-maintainer/references/page-ownership.md`;
- durable command or editing rules update this file and, when procedural, the
  maintainer `SKILL.md`;
- source-contract or validation-workflow changes update the maintainer
  `SKILL.md`;
- current image evidence or release gates update
  `.agents/skills/cosmos3-nim-docs-maintainer/references/evidence-and-release-gates.md`;
- customer workflow or safety changes update `../AGENTS.md` and
  `../.agents/skills/cosmos3-nim-user/SKILL.md`; and
- dependency policy changes update this file, `../pyproject.toml`, `../uv.lock`,
  and the affected public guides.
