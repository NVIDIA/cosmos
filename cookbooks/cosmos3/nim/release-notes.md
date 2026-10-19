<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 NIM release notes

## 2.0.0: Initial unified Cosmos3 NIM release

This is the first release of the unified Cosmos3 NIM, published as
`nvcr.io/nim/nvidia/cosmos3:2.0.0`. One container image includes both runtime
choices:

- **Generator** serves image and video generation, Action, and Transfer through
  `POST /v1/infer`.
- **Reasoner** serves OpenAI-compatible image and video understanding through
  Chat Completions and the Responses API.

See [Deployment](deployment.md) for launch instructions.
