<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Bring your own Cosmos3 checkpoint

Use this page to replace the selected Generator or Reasoner checkpoint while
preserving the Certified NIM server, profile selection, and runtime contract.

> `NIM_MODEL_PATH` provides the implementation described below. Confirm the
> accepted checkpoint inventory in the released [support matrix](support-matrix.md).

## Supported boundary

`NIM_MODEL_PATH` is the shared checkpoint-source variable:

| Runtime | Accepted source | What remains profile-owned |
| --- | --- | --- |
| Generator | Absolute local directory | Server, profile, and Generator guardrail artifacts |
| Reasoner | Absolute local directory or `hf://owner/repository[:revision]` | Server, selected runtime layout, and profile compatibility policy |

The former Generator variable `NIM_FT_CHECKPOINT` is not part of the current
contract. Do not reuse historical Transfer variables such as
`NIM_EDGE_CHECKPOINT`, `NIM_VIS_CHECKPOINT`, `NIM_DEPTH_CHECKPOINT`, or
`NIM_SEG_CHECKPOINT`.

## Generator checkpoint

### Expected layout

The current Generator path expects this structural shape:

```text
/byoc/cosmos3/
├── transformer/
│   ├── config.json
│   └── <weight shards>
├── vae/
├── scheduler/
└── model_index.json
```

The runtime reads `transformer/config.json` to infer model size and precision,
then cross-checks both against the selected profile. Directory names alone do
not prove compatibility with the released image.

### Launch

Start with the standard [Generator launch](deployment.md#launch-generator).
Mount the checkpoint read-only and add these Docker options:

```bash
export BYOC_CHECKPOINT='/host/path/to/generator-checkpoint'
```

Add these options to the `docker run` command:

```text
-e NIM_MODEL_PATH=/byoc/cosmos3 \
-v "$BYOC_CHECKPOINT:/byoc/cosmos3:ro"
```

Choose a Generator variant, precision, and latency/throughput objective that
match the checkpoint. Generator guardrails remain profile-owned artifacts. The cache must be writable,
NGC artifact access may still be required, and
`NIM_DISABLE_MODEL_DOWNLOAD=1` is rejected for Generator profiles.

## Reasoner checkpoint

### Local directory

A local Reasoner checkpoint must use an absolute in-container path and contain,
at minimum:

- a supported Cosmos3 Reasoner or Cosmos3 Omni `config.json`;
- safetensors weights or a valid safetensors index and all referenced shards;
- tokenizer files; and
- processor or preprocessor configuration.

The runtime infers Nano versus Super, BF16/FP8/NVFP4 precision, and Reasoner
versus Omni layout. It then selects a compatible Reasoner profile. An explicit
`NIM_MODEL_SIZE`, `NIM_PRECISION`, or `NIM_MODEL_PROFILE` must agree with the
checkpoint.

Use the same read-only mount pattern as the Generator, but select the Reasoner:

```bash
-e NIM_MODEL_TYPE=reasoner \
-e NIM_MODEL_PATH=/byoc/cosmos3-reasoner \
-v "$BYOC_CHECKPOINT:/byoc/cosmos3-reasoner:ro"
```

For a completely local checkpoint, `NIM_DISABLE_MODEL_DOWNLOAD=1` prevents
profile artifact download after source resolution.

### Hugging Face source

The Reasoner also accepts:

```bash
-e NIM_MODEL_TYPE=reasoner \
-e NIM_MODEL_PATH='hf://owner/repository:revision' \
-e HF_TOKEN
```

Omit `:revision` to use `main`. Inject `HF_TOKEN` only when the repository
requires it; never place the token in the URI, image layer, or documentation.
The downloaded snapshot is stored under the writable NIM cache.

An `hf://` source requires network materialization, so it cannot be combined
with `NIM_DISABLE_MODEL_DOWNLOAD=1`. For offline operation, pre-download the
checkpoint and use an absolute local path instead.

## Discovery and validation

At startup, the NIM:

1. parses `NIM_MODEL_PATH` and rejects unsupported or unsafe source forms;
2. validates the checkpoint layout and infers its model properties;
3. checks explicit selectors and the selected profile against those properties;
4. materializes any runtime-owned artifacts still required; and
5. fails before inference when the source, layout, or profile is incompatible.

Adjust the selectors or checkpoint rather than bypassing compatibility checks.
The exact accepted release inventory belongs in the
[Support matrix](support-matrix.md).

## Verify the active checkpoint

Wait for readiness and inspect metadata:

```bash
until curl -fsS http://localhost:8000/v1/health/ready >/dev/null; do
  sleep 10
done

curl -fsS http://localhost:8000/v1/metadata | python -m json.tool
```

The `checkpoint` field reports `default` for bundled artifacts or identifies the
configured source. Generator metadata also reports `model_variant`. Confirm the
selected profile, then run a representative request and compare its result with
the checkpoint's validation baseline.

## Common failures

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Local path does not exist | Host path was not mounted at the exact `NIM_MODEL_PATH` value | Align the read-only bind destination and environment path |
| Relative path rejected | Local checkpoint sources must be absolute | Use an absolute in-container path |
| Permission denied | Container cannot traverse or read the mount | Fix host ownership/ACLs while retaining a read-only checkpoint mount |
| Required file missing | Checkpoint layout, tokenizer, processor, index, or weight shards are incomplete | Compare with the released BYOC contract and export again |
| Model size or precision mismatch | Explicit selector/profile disagrees with inferred checkpoint properties | Select a compatible released profile or use a matching checkpoint |
| Generator rejects disabled downloads | Generator still needs profile-owned guardrails | Remove `NIM_DISABLE_MODEL_DOWNLOAD=1` and provide NGC/cache access |
| Hugging Face source rejected offline | `hf://` requires download but downloads are disabled | Use an absolute pre-downloaded local path |
| Hugging Face authorization fails | Token, repository ID, revision, network, or cache is invalid | Check `HF_TOKEN`, URI, connectivity, and writable cache without logging the token |
| Metadata shows `default` | Override was omitted, rejected, or applied to another container | Inspect launch environment, mounts, startup logs, and `/v1/metadata` |

For broader startup, cache, GPU, and readiness diagnosis, see
[Operations](operations.md#byoc).
