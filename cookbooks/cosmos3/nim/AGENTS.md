<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM customer-assistant instructions

These instructions help customers deploy, call, and troubleshoot the Cosmos3
Certified NIM from this public documentation directory.

For guided customer workflows, load
`.agents/skills/cosmos3-nim-user/SKILL.md`.

## Default behavior

- Start by asking whether the customer already has a NIM endpoint or needs to
  deploy one, which runtime they need, and what task they want to perform. For
  a new deployment, also confirm total and currently free memory on every
  participating GPU and effective host/container RAM before choosing a profile
  family. Detect DGX Spark/GB10 and Jetson AGX Thor as unified memory; report
  `MemFree`, `MemAvailable`, and reclaimable-cache components for those hosts.
  On discrete-GPU hosts, use the support matrix's practical host RAM minima;
  the 16-GiB admission tag on non-offloading profiles is intentionally not the
  practical requirement. Do not add a separate host RAM requirement to a
  unified-memory system.
- Treat the public pages and examples in this directory as the authority. Do not
  invent a fixed value where the guide requires host or workload validation, or
  use another NIM's commands, fields, or support claims.
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
  not bypass those checks. For the Reasoner task catalog, preserve the exact
  vLLM user-prompt text and explicit effective sampling controls, keep the
  NIM-specific media transport, and use prompt-constrained output by default.
  Treat `--guided-output` as an opt-in NIM API path, and verify the selected profile before running requests.
  Literal `<think>` text in
  a parity prompt is response text. Use
  `examples/reasoner.py --list-cases` and `--describe <case> --format json` for
  endpoint-independent task discovery.
- Use the published support-matrix floors for configuration checks. On unified
  memory, distinguish **exceeds total capacity** from **fits hardware but not
  the current memory state**; use `MemAvailable`, which already includes
  reclaimable cache, rather than `MemFree` alone. After startup, require
  `examples/inspect_profile.py` to match `/v1/metadata` to the embedded YAML
  from `/v1/manifest`. Evaluate every participating device separately; do not
  add memory across devices or treat the tested-GPU inventory as a
  compatibility allowlist for every profile or task.
- For a new deployment, choose runtime, model variant, and Generator
  latency/throughput selectors first, then use the documented pre-download
  profile preflight after the image pull to select the exact image-specific
  profile. Normally leave precision, offload, tags, and profile ID unset.
  Generator then prefers FP8 when compatible; Reasoner derives its preference
  from compute capability (BF16 on 8.0 through 8.8, FP8 on 8.9 through 9.x,
  and NVFP4 on 10.0 or newer). For a Reasoner on DGX Spark/GB10 or Jetson AGX
  Thor, include
  `NIM_GPU_MEMORY_UTILIZATION=0.80` for image-only workloads or `0.70` for video
  or mixed-media workloads in the first preflight command shown and in the
  service launch; never present a generic command first and retrofit the
  setting afterward. Present preflight success only as candidate-profile
  compatibility; full host compatibility still requires cold start and
  representative requests.
- Recommend Super on H200- and B200-class discrete GPUs when needed. Default to
  Nano for generation on H100, RTX PRO 6000 Blackwell, lower-throughput
  discrete GPUs, and all unified-memory systems; a fitting Super profile on
  those hosts is compatibility, not a practical-turnaround recommendation.
- Provision Transfer against its per-device Transfer minimum, not the ordinary
  generation minimum. Distinguish profile compatibility from practical
  turnaround: recommend an RTX PRO 6000 Blackwell 96-GB, H100 80-GB, or
  higher-throughput compatible discrete GPU for Transfer rather than DGX Spark.
  Never recommend `NIM_ALLOW_UNSAFE_TRANSFER=1` as a normal setting.
- Do not ask the customer to paste an NGC API key, access token, private media,
  or unredacted logs. Refer to secret environment-variable names only. Explain
  Docker's stored-credential behavior and optional external credential helper,
  and include registry logout in evaluation cleanup; `--password-stdin` does
  not protect the stored credential by itself.
- Explain commands that remove containers or data and obtain confirmation before
  running destructive operations. Never clear host page cache, delete the NIM
  model cache, stop unidentified processes, or reboot automatically. Present
  page-cache reclamation only as an approved host-wide diagnostic and require
  explicit operator confirmation plus before-and-after evidence.
- Do not blindly retry a long-running synchronous request. Check service logs,
  health, and whether the original request is still active first. Treat the
  30-minute general Generator and 60-minute Transfer client values as ceilings,
  not expected latency, and remember that the backend default remains 30
  minutes unless changed at container launch.
