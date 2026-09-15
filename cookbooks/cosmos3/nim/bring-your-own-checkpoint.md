<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Bring your own Cosmos3 checkpoint

Use this page to replace the selected Generator or Reasoner checkpoint while
preserving the Certified NIM server, profile selection, and runtime contract.

> Set `NIM_MODEL_PATH` to replace the bundled Generator or Reasoner checkpoint.
> At startup, the NIM validates the checkpoint layout, inferred model properties,
> and compatibility with the selected configuration before serving requests.

## Supported boundary

`NIM_MODEL_PATH` is the shared checkpoint-source variable:

| Runtime | Accepted source | What the NIM still provides |
| --- | --- | --- |
| Generator | Absolute local directory | Server, profile, and Generator guardrail artifacts |
| Reasoner | Absolute local directory or `hf://owner/repository[:revision]` | Server, selected runtime layout, and profile compatibility policy |
| Nano or Super Reasoner DFlash draft | Absolute local directory only | Primary Reasoner checkpoint, server, and profile policy |

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

The runtime reads `transformer/config.json` to infer the Nano or Super base
variant and precision, then cross-checks both against the selected profile.
Transformer dimensions do not identify a specialist Generator contract, so
select the exact checkpoint contract with `NIM_MODEL_VARIANT`. Directory names
alone do not prove compatibility with the selected image.

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
match the checkpoint. The NIM still provides Generator guardrail artifacts.
The cache must be writable, NGC artifact access may still be required, and
`NIM_DISABLE_MODEL_DOWNLOAD=1` is rejected for Generator profiles.

## Reasoner checkpoint

### Local directory

A local Reasoner checkpoint must use an absolute in-container path and contain,
at minimum:

- a supported Cosmos3 Reasoner or Cosmos3 Omni `config.json`;
- safetensors weights or a valid safetensors index and all referenced shards;
- tokenizer files; and
- processor or preprocessor configuration.

The runtime infers the `nano` or `super` variant, BF16/FP8/NVFP4 precision,
and Reasoner versus Omni layout. It then selects a compatible Reasoner profile.
An explicit `NIM_MODEL_VARIANT`, `NIM_PRECISION`, or `NIM_MODEL_PROFILE` must
agree with the checkpoint.

Use the same read-only mount pattern as the Generator, but select the Reasoner:

```bash
-e NIM_MODEL_TYPE=reasoner \
-e NIM_MODEL_PATH=/byoc/cosmos3-reasoner \
-v "$BYOC_CHECKPOINT:/byoc/cosmos3-reasoner:ro"
```

For a completely local checkpoint, `NIM_DISABLE_MODEL_DOWNLOAD=1` prevents
profile artifact download after source resolution. Because Reasoner enables
DFlash by default, also provide a local draft or set `NIM_USE_DFLASH=0` for
completely local target-only operation.

When DFlash is enabled, Nano and Super Reasoner each need their corresponding
draft artifact containing `config.json` and `model.safetensors`. The selected
profile can provide it under the primary model workspace. A separate read-only
local draft can be supplied through `NIM_DFLASH_MODEL_PATH`; the draft does not
need the target model's tokenizer or processor files:

```text
/byoc/cosmos3-dflash/
├── config.json
└── model.safetensors
```

To use an independent draft, set a host directory and add the following options
to the standard Reasoner `docker run` command:

```bash
export DFLASH_CHECKPOINT="$HOME/models/cosmos3-dflash"
```

```text
-e NIM_USE_DFLASH=1 \
-e NIM_DFLASH_MODEL_PATH=/byoc/cosmos3-dflash \
-v "$DFLASH_CHECKPOINT:/byoc/cosmos3-dflash:ro"
```

To train your own matching Cosmos3-Nano DFlash draft on a custom dataset, or
after fine-tuning the Nano Reasoner target, follow the [Model Optimizer DFlash
training recipe](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/speculative_decoding/recipes/train_dflash_cosmos3_nano.ipynb).

`NIM_DFLASH_MODEL_PATH` must be an absolute in-container local path and does not
accept `hf://`. If neither the selected image nor an independent mount supplies
the required draft, set `NIM_USE_DFLASH=0`.

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
checkpoint and use an absolute local path instead. If DFlash is enabled, use the
draft supplied by the selected profile or mount a separate local draft with
`NIM_DFLASH_MODEL_PATH`; a Hugging Face URI is not accepted for the draft.

## Discovery and validation

At startup, the NIM:

1. parses `NIM_MODEL_PATH` and rejects unsupported or unsafe source forms;
2. validates the checkpoint layout and infers its model properties;
3. checks explicit selectors and the selected profile against those properties;
4. materializes any runtime-owned artifacts still required; and
5. fails before inference when the source, layout, or profile is incompatible.

Adjust the selectors or checkpoint rather than bypassing compatibility checks.
Use only checkpoint types confirmed for the selected image.

## Verify the active checkpoint

Wait for readiness and inspect metadata:

```bash
until curl -fsS http://localhost:8000/v1/health/ready >/dev/null; do
  sleep 10
done

curl -fsS http://localhost:8000/v1/metadata | python3 -m json.tool
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
| Required file missing | Checkpoint layout, tokenizer, processor, index, or weight shards are incomplete | Compare with the required layout and export again |
| Model variant or precision mismatch | Explicit selector/profile disagrees with inferred checkpoint properties | Select a compatible configuration or use a matching checkpoint |
| Generator rejects disabled downloads | Generator still needs NIM-provided guardrails | Remove `NIM_DISABLE_MODEL_DOWNLOAD=1` and provide NGC/cache access |
| Hugging Face source rejected offline | `hf://` requires download but downloads are disabled | Use an absolute pre-downloaded local path |
| Hugging Face authorization fails | Token, repository ID, revision, network, or cache is invalid | Check `HF_TOKEN`, URI, connectivity, and writable cache without logging the token |
| DFlash path is rejected | The draft path is relative, uses `hf://`, is not mounted, or lacks one of its two required files | Use an absolute local mount containing `config.json` and `model.safetensors` |
| DFlash configuration is rejected | DFlash is disabled, the draft does not match the Nano/Super target, or `NIM_DFLASH_CONFIG` sets reserved `method`/`model` keys | Use the selected Reasoner's matching draft, keep DFlash enabled, and remove reserved keys; see [Speculative decoding](configuration.md#speculative-decoding) |
| Metadata shows `default` | Override was omitted, rejected, or applied to another container | Inspect launch environment, mounts, startup logs, and `/v1/metadata` |

For broader startup, cache, GPU, and readiness diagnosis, see
[Operations](operations.md#byoc).
