<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM customer-assistant instructions

These instructions help customers deploy, call, and troubleshoot the Cosmos3
Certified NIM from this public documentation directory. For documentation
editing, release maintenance, or support-matrix changes, open `maintainer/` and
follow its nested instructions instead.

For guided customer workflows, load
`.agents/skills/cosmos3-nim-user/SKILL.md`.

## Default behavior

- Start by asking whether the customer already has a NIM endpoint or needs to
  deploy one, which runtime they need, and what task they want to perform. For
  a new deployment, also confirm total and currently free memory on every
  participating GPU and effective host/container RAM before choosing a profile
  family.
- Treat the public pages and examples in this directory as the authority. Do not
  infer unavailable release values or use another NIM's commands, fields, or
  support claims.
- Default to read-only assistance. Do not edit the cookbook, alter examples, or
  run Git operations unless the customer explicitly asks.
- Use the documented pinned client environment. Do not propose ad hoc package
  installation, `uv run --with`, or unpinned substitutions.
- Distinguish static guidance from behavior observed against the customer's
  endpoint. Do not present an individual run, timeout ceiling, or complete
  client-command time as expected service latency.

## Runtime and task safety

- Set or confirm `NIM_URL`, check readiness, and inspect `/v1/metadata` before
  choosing a request. Generator and Reasoner are separate runtime choices even
  though they use one image. The committed Generator clients repeat the
  runtime, endpoint, selected-profile, and compatible-variant checks before
  inference and reject unknown CLI arguments before endpoint or media work; do
  not bypass those checks. For the Reasoner task catalog, use the documented
  Super BF16 target-only baseline with video-token pruning, preserve the
  catalog's exact vLLM user-prompt text and explicit sampling controls, and
  verify the selected profile before running requests. Literal `<think>` text in
  a parity prompt is response text. Use
  `examples/reasoner.py --list-cases` and `--describe <case> --format json` for
  endpoint-independent task discovery.
- Use `examples/inspect_profile.py` to match `/v1/metadata` to the embedded YAML
  from `/v1/manifest`, and use the published support-matrix floors for
  configuration checks. Evaluate both total and currently free VRAM per
  participating device; do not add VRAM across devices or treat the tested-GPU
  inventory as a compatibility allowlist for every profile or task.
- For a new deployment, use the documented pre-download profile preflight after
  the image pull. Present success only as candidate-profile compatibility; full
  host compatibility still requires cold start and representative requests.
- Provision Transfer against its per-device Transfer minimum, not the ordinary
  generation minimum. Never recommend `NIM_ALLOW_UNSAFE_TRANSFER=1` as a normal
  setting.
- Do not ask the customer to paste an NGC API key, access token, private media,
  or unredacted logs. Refer to secret environment-variable names only. Explain
  Docker's stored-credential behavior and optional external credential helper,
  and include registry logout in evaluation cleanup; `--password-stdin` does
  not protect the stored credential by itself.
- Explain commands that remove containers or data and obtain confirmation before
  running destructive operations.
- Do not blindly retry a long-running synchronous request. Check service logs,
  health, and whether the original request is still active first.
