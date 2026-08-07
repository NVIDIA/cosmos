<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Documentation rules

- Declare required client executables, their minimum versions, and installation
  instructions before the first command that uses them. Keep client tooling
  separate from NIM host and container prerequisites.
- Establish the command working directory once. Run Python examples from this
  cookbook directory with `uv run python examples/...` so `uv` discovers the
  checked-in `pyproject.toml` and `uv.lock`.
- Keep example dependencies in `pyproject.toml` and commit the resolved
  `uv.lock`. Do not use ad hoc `pip install` or `uv run --with` commands in
  user-facing documentation.
- Use `python3` for host shell commands. Use `python` only inside the `uv`
  project environment; Python Markdown fence labels are unaffected.
- Update prerequisite text, dependency metadata, the lockfile, and every
  affected command together. Validate that the lockfile is current and search
  the complete documentation set for stale command forms before publishing.
- On every Cosmos3 2.2 release-candidate bump, update the exact `NIM_IMAGE`
  reference in both `deployment.md` and `release-notes.md`. Never replace the
  versioned RC tag with `latest`; the documentation check requires both files
  to agree.
- Keep unresolved values out of runnable fenced blocks. Run
  `python3 maintainer/check_code_fences.py` from this cookbook directory before
  publishing; CI runs the same check.
