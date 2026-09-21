<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM release notes

## Initial unified Cosmos3 NIM release

This is the first release of the unified Cosmos3 Certified NIM. One container
image includes both runtime choices:

- **Generator** serves image and video generation, Action, and Transfer through
  `POST /v1/infer`.
- **Reasoner** serves OpenAI-compatible image and video understanding through
  Chat Completions and the Responses API.

Each container starts one selected runtime. See [Deployment](deployment.md) for
the current image and launch instructions.
