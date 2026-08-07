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
