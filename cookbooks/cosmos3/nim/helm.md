<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Deploy the Cosmos3 Certified NIM with Helm

The current pre-release evaluation chart is pinned to:

```text
nvstaging/nim/nim-wfm:1.3.0
```

This is a staging chart reference, not a final public chart identity. Pin the
exact chart and NIM image versions together. Approved Cosmos3 values, resource
names, installation commands, and monitoring integration are not yet available
in this cookbook.

## Prerequisites

Prepare:

- a Kubernetes cluster with supported NVIDIA GPU nodes;
- NVIDIA GPU Operator or another supported device-plugin stack;
- `kubectl` and Helm access to the target namespace;
- a storage class suitable for the model cache;
- outbound NGC access or an approved pre-populated-cache workflow; and
- an `NGC_API_KEY` with NGC Catalog access.

Choose a model and hardware configuration from the
[Support matrix](support-matrix.md), then confirm that the selected image
contains that configuration before setting GPU resources.

## Inspect the evaluation chart

Use the exact `nvstaging/nim/nim-wfm:1.3.0` reference through the approved NGC
Helm workflow and inspect that version's README and values schema before
installation. This cookbook does not yet publish a pull or install command for
the staging chart. Do not copy values from another NIM or chart version.

## Create NGC secrets

The deployment needs one secret for the image pull and one for runtime artifact
download:

```bash
kubectl create secret docker-registry ngc-image-pull \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY"

kubectl create secret generic cosmos3-ngc \
  --from-literal=NGC_API_KEY="$NGC_API_KEY"
```

The runtime secret key must be named `NGC_API_KEY`. In production, use the
cluster's approved secret manager or CI injection instead of commands that can
expose credentials through shell history or process inspection.

## Required chart concepts

Map the following concepts to names from the pinned chart schema:

- explicit image repository and tag;
- image-pull and runtime NGC secrets;
- `NIM_MODEL_TYPE`;
- `NIM_MODEL_VARIANT` for Generator or Reasoner;
- `NIM_PERF_PROFILE` for Generator;
- optional `NIM_PRECISION` pin;
- GPU and system-memory limits matching a released configuration, including
  any additional RAM required by model offload;
- a writable model cache mounted at `/opt/nim/.cache`;
- an adequately sized in-memory `/dev/shm` volume;
- service port `8000`, unless deliberately changed;
- liveness `/v1/health/live` and readiness `/v1/health/ready`;
- a startup budget for cold download, materialization, load, and warmup; and
- a security context that can write to the cache.

No values file is included until the approved Cosmos3 key names and settings
for this chart version are available.

## Storage and scaling

A persistent cache avoids repeated downloads and materialization. Select a
pattern supported by the chart and storage system:

- one PVC per replica avoids concurrent writers but duplicates artifacts;
- a shared `ReadWriteMany` PVC can reduce downloads but needs validated locking,
  ownership, and throughput; and
- `hostPath` ties a workload to a node and adds security and scheduling risks.

Each replica must receive a complete GPU allocation compatible with its model.
Validate cold start, rolling updates, cache ownership, scale-up, and probe
budgets before production use.

## Install and verify

Use only approved install values, resource names, and port-forward instructions
for the pinned chart version. After rollout, require readiness before sending
traffic:

```bash
curl -f http://localhost:8000/v1/health/ready
```

For logs, metrics, and probes, see [Operations](operations.md). For Pending Pods,
mount failures, or startup failures, see
[Startup and deployment troubleshooting](operations.md#startup-and-deployment).
