<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM release notes

This page records user-visible changes, compatibility requirements, known
limitations, and upgrade guidance for released Cosmos3 Certified NIM images.
Use an explicit image tag from the corresponding release entry.

## Initial unified release — TBD

> **Current release candidate:** Deployment examples use
> `nvcr.io/nvstaging/nim/cosmos3:2.2.0-rc.20260805164511-12ca3dacb921e392`.
> This staging reference is not a final release identity and changes with each
> approved RC bump.

| Item | Status |
| --- | --- |
| Release version and date | **TBD** |
| NGC image repository and tag | Current RC: `nvcr.io/nvstaging/nim/cosmos3:2.2.0-rc.20260805164511-12ca3dacb921e392`; final release **TBD** |
| NGC Catalog and model-card URLs | **TBD** |
| Helm chart and version | **TBD** |
| Supported hardware and model matrix | Semi-final source-profile requirements documented; released-row approval pending |
| Released Generator model variants | **TBD** |
| Released Generator and Reasoner BYOC boundary | **TBD** |
| Transfer-enabled model/GPU rows | **TBD** |
| Known limitations and upgrade notes | **TBD** |

### Development API migration

The current Generator API requires top-level `model_mode`, renames
`num_output_frames` to `num_frames`, `steps` to `num_inference_steps`, and
`image`/`video` to `input_reference`, and moves `action_params.mode` to the top
level. Older request fields return HTTP 422. Generator responses are unchanged.

Current Reasoner development behavior adds Qwen3 parsed-reasoning controls,
OpenAI tool calling, `developer`-message normalization, and strict
`include_reasoning` and `top_logprobs` types. Media errors return HTTP 422;
other request-shape and sampling errors return HTTP 400.

Current shared runtime diagnostics normalize successful Generator and Reasoner
health responses, report `model_type` and `inference_endpoint` through
`/v1/metadata`, and return a runtime-aware NIM error envelope for missing and
wrong-runtime routes. These source-derived behaviors still require validation
against the selected release image.

The current documentation describes the source-backed unified runtime model:
one image contains Generator and Reasoner profiles, and one selected profile
starts one backend. This statement is not a substitute for a published release
entry or compatibility notice.

Do not infer a release version, image tag, support guarantee, or upgrade path
from the development branch. Before publication, replace this section with the
approved release inventory and retain older entries for users operating pinned
images.
