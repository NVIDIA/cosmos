<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Canonical page ownership

Use one canonical owner for each fact class. Paths in this table are relative
to the public documentation root (`..` from the maintainer folder). Other pages
may include the minimum needed for a complete workflow and link to the owner.

| Artifact | Canonical responsibility |
| --- | --- |
| `README.md` | Product scope, selected Generator/Reasoner runtime model, capability index, quick starts, AI-assistance discovery, and guide navigation |
| `AGENTS.md` and `.agents/skills/cosmos3-nim-user/SKILL.md` | Customer-assistant routing, safety boundaries, canonical-page selection, and endpoint-guided workflow without duplicating reference facts |
| `release-notes.md` | Concise description of the initial unified release; versioned history begins only after public release |
| `prerequisites.md` | Host hardware/software, client tooling and pinned environment initialization, storage, shared memory, NGC/network access, and setup verification |
| `deployment.md` | NGC login, exact image, cache, Docker launch flags, selectors, ports, readiness, and shutdown |
| `configuration.md` | Shared, Generator, Reasoner, profile-selection, and prompt-upsampling environment variables |
| `support-matrix.md` | Model, precision, GPU, VRAM, profile, offload, and media/codec compatibility with evidence status |
| `helm.md` | Kubernetes prerequisites and concepts; exact chart commands only after chart publication |
| `bring-your-own-checkpoint.md` | Generator and Reasoner checkpoint sources, layouts, mounts/downloads, validation, launch, and failures |
| `api-reference.md` | Runtime routing, common Generator envelope and response, strict JSON behavior, and live schema inspection |
| `generation.md` | T2I, T2V, I2V, V2V, frame/resolution/media rules, prompt upsampling, decoding, and generation failures |
| `reasoning.md` | Reasoner Chat Completions and Responses, media, sampling, structured output, and task-runner behavior |
| `action.md` | Forward dynamics, policy, inverse dynamics, Nano-DROID, action shapes/domains, and responses |
| `transfer.md` | Transfer controls, precomputed/derived forms, defaults, combinations, chunking, and admission behavior |
| `operations.md` | Health, management endpoints, generic errors, metrics, logs, guardrails, diagnostics, and troubleshooting |
| `acknowledgements.md` | Availability and approved third-party notices for the exact released image |
| `examples/` | Complete editable client requests, primary response handling, media helpers, and generated outputs |
| `pyproject.toml` and `uv.lock` | Reproducible Python client environment for all examples |
| `maintainer/` | Documentation editing instructions, evidence and release gates, page ownership, and maintainer validation workflow |

## Duplication rules

- `README.md` may show a minimal deployment and one request per runtime, but
  detailed launch behavior belongs in `deployment.md`.
- Task pages own mode-specific fields. `api-reference.md` owns only routing and
  the common Generator envelope.
- `configuration.md` owns environment-variable tables; workflow pages may show
  only variables needed for that workflow.
- `support-matrix.md` owns compatibility tables. Other pages should not copy
  profile matrices.
- `operations.md` owns generic diagnosis. Task guides own task-specific failure
  rows.
- Before public release, `deployment.md` owns the exact evaluation image.
  Release notes describe only the initial unified release without duplicating a
  mutable staging tag.
- The Helm page states chart availability without inventing commands. The
  acknowledgements page states notice availability without inventing inventory.

## Integration surfaces

When broader edits are authorized, check these nearby pages for stale Certified
NIM claims and link them to the canonical guide instead of duplicating setup:

- repository `README.md`;
- `cookbooks/cosmos3/README.md`;
- `cookbooks/cosmos3/generator/audiovisual/README.md`;
- `cookbooks/cosmos3/reasoner/README.md`;
- `cookbooks/cosmos3/generator/action/README.md`; and
- `cookbooks/cosmos3/generator/transfer/README.md`.

If those files are outside the authorized scope, do not edit them silently;
identify any contradiction in the handoff or pull-request description.
