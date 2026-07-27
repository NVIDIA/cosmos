<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Bring your own Cosmos3 checkpoint

Use this page to replace the Generator diffusion checkpoint while preserving
the selected NIM profile's runtime, guardrails, and supporting artifacts.

> **Release status:** Current source supports `NIM_FT_CHECKPOINT` for a
> Generator diffusion checkpoint. It does not establish Reasoner BYOC. The
> final published runtime boundary and accepted checkpoint format are **TBD
> (release-dependent)**.

## Supported boundary

BYOC replaces the Generator model weights identified by
`NIM_FT_CHECKPOINT`. It does not replace the NIM server, profile manifest,
guardrail bundle, or other profile-owned runtime artifacts.

Do not reuse historical Transfer BYOC variables such as
`NIM_EDGE_CHECKPOINT`, `NIM_VIS_CHECKPOINT`, `NIM_DEPTH_CHECKPOINT`, or
`NIM_SEG_CHECKPOINT`. They are not part of the current Certified NIM's
source-backed configuration contract.

## Expected checkpoint layout

The current source expects this structural shape:

```text
/byoc/cosmos3/
├── transformer/
│   ├── config.json
│   └── <weight shards>
├── vae/
├── scheduler/
└── model_index.json
```

The released image must define the exact accepted serialization, configuration
fields, model sizes, and precisions. Do not infer compatibility solely because
a directory has these names.

## Prepare the launch

Use the normal released image and writable model cache. Mount the checkpoint
read-only at an absolute in-container path:

```bash
export NIM_IMAGE='<NIM_IMAGE:TBD>'
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim/cosmos3}"
export BYOC_CHECKPOINT='/host/path/to/checkpoint'

docker run --rm --name cosmos3-generator-byoc \
  --gpus '"device=0"' \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=generator \
  -e NIM_MODEL_SIZE=nano \
  -e NIM_PRECISION=bf16 \
  -e NIM_PERF_PROFILE=latency \
  -e NIM_FT_CHECKPOINT=/byoc/cosmos3 \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -v "$BYOC_CHECKPOINT:/byoc/cosmos3:ro" \
  "$NIM_IMAGE"
```

The selector values are illustrative. Set model size, precision, GPU count, and
other selectors to a profile compatible with the actual checkpoint and the
released [support matrix](support-matrix.md).

The normal cache must remain writable because derived engines and intermediate
artifacts are not written into the read-only checkpoint mount.

## Discovery and validation

At startup, the current NIM:

1. resolves the absolute `NIM_FT_CHECKPOINT` path;
2. inspects checkpoint configuration to infer model size and precision;
3. cross-checks those values against the selected profile; and
4. fails before inference when the path, layout, or profile is incompatible.

This prevents a checkpoint/profile mismatch from failing later inside model
load or inference. Adjust the selectors; do not bypass the cross-check.

## Verify the active checkpoint

Wait for readiness, then inspect metadata:

```bash
until curl -fsS http://localhost:8000/v1/health/ready >/dev/null; do
  sleep 10
done

curl -fsS http://localhost:8000/v1/metadata | python -m json.tool
```

Confirm that the reported checkpoint is the in-container override path and
that the selected profile matches the intended model size and precision. Then
run a representative Generator request and compare output quality with the
checkpoint's validation baseline.

## Common failures

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Checkpoint path does not exist | Host path was not mounted at the value of `NIM_FT_CHECKPOINT` | Compare the bind mount destination with the absolute in-container variable |
| Permission denied | Container cannot traverse/read the mounted directory | Fix host ownership/ACLs while keeping the checkpoint mount read-only |
| Required file or directory missing | Checkpoint layout or serialization is incomplete/incompatible | Compare against the released BYOC contract and export the checkpoint again |
| Model size or precision mismatch | Selected profile does not match checkpoint metadata | Set compatible selectors from the released support matrix |
| Engine/materialization failure | Writable cache, disk, RAM, shared memory, or profile resources are insufficient | Check cache permissions/resources and retain startup logs |
| Metadata shows bundled checkpoint | Override was omitted, rejected, or applied to the wrong container | Inspect environment, mount, startup logs, and `/v1/metadata` |

For broader startup, cache, GPU, and readiness diagnosis, see
[Operations](operations.md#byoc).
