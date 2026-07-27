<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Deploy the Cosmos3 Certified NIM with Helm

Use this page to plan a Kubernetes deployment, configure NGC secrets, storage,
GPUs, probes, and services, and verify a Helm release. Use
[prerequisites.md](prerequisites.md) and the
[support matrix](support-matrix.md) to choose compatible GPU nodes.

> **Release status:** The final chart repository, chart version, values schema,
> resource names, and supported monitoring integration are **TBD
> (release-dependent)**. Do not copy an older Cosmos NIM chart name or values
> schema without confirmation.

## Prerequisites

Prepare:

- a Kubernetes cluster with compatible NVIDIA GPU nodes;
- NVIDIA GPU Operator or an equivalent supported device-plugin stack;
- `kubectl` access to the target namespace;
- Helm compatible with the released chart;
- a default or explicitly selected storage class for the model cache;
- outbound NGC access, or an approved pre-populated-cache workflow; and
- an `NGC_API_KEY` with the required NGC Catalog access.

## Select and inspect the chart

After the released chart identity is known, pin its version and inspect its own
documentation rather than copying a generic values catalogue:

```bash
export NIM_HELM_CHART='<NIM_HELM_CHART:TBD>'
export NIM_HELM_VERSION='<NIM_HELM_VERSION:TBD>'

helm show readme "$NIM_HELM_CHART" --version "$NIM_HELM_VERSION"
helm show values "$NIM_HELM_CHART" --version "$NIM_HELM_VERSION"
```

Use the value names from that exact schema. The examples below describe
required concepts; placeholder keys are not an approved release contract.

## Create NGC secrets

Create separate secrets for pulling the image and downloading runtime model
artifacts:

```bash
kubectl create secret docker-registry ngc-image-pull \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY"

kubectl create secret generic cosmos3-ngc \
  --from-literal=NGC_API_KEY="$NGC_API_KEY"
```

The runtime secret must contain a key literally named `NGC_API_KEY`. The
image-pull secret uses the Docker registry format. These commands can expose
values through shell history or local process inspection; use the cluster's
approved secret manager or CI injection in production.

## Configure values

The released values schema must provide or map these concepts:

- explicit NIM image repository and tag;
- image-pull and runtime NGC secret references;
- `NIM_MODEL_TYPE` and compatible profile selectors;
- GPU resource limits matching one released profile;
- a writable model cache mounted at `/opt/nim/.cache`;
- an in-memory `/dev/shm` volume sized for the released configuration;
- HTTP service port `8000`, unless deliberately changed;
- liveness `/v1/health/live` and readiness `/v1/health/ready`;
- a startup budget long enough for cold download, materialization, load, and
  warmup;
- pod/container security context and cache ownership; and
- optional metrics or OpenTelemetry settings validated for the chart release.

A conceptual values file follows. Replace every placeholder key with the exact
released chart schema:

```yaml
image:
  repository: <NIM_IMAGE_REPOSITORY:TBD>
  tag: <NIM_IMAGE_TAG:TBD>

imagePullSecrets:
  - name: ngc-image-pull

model:
  ngcAPISecret: cosmos3-ngc

env:
  - name: NIM_MODEL_TYPE
    value: generator
  - name: NIM_MODEL_SIZE
    value: nano

resources:
  limits:
    nvidia.com/gpu: 1

persistence:
  enabled: true
```

This file is illustrative until the final chart values are published.

## Configure storage

A persistent cache avoids repeated model downloads and materialization.
Available patterns depend on the chart and cluster:

- one PVC per StatefulSet replica avoids concurrent writers but duplicates
  artifacts;
- a `ReadWriteMany` PVC can share artifacts, subject to the released cache
  locking, ownership, and storage-performance guidance;
- direct NFS may need cluster-specific mount configuration; and
- `hostPath` binds a workload to a node and has significant security and
  scheduling implications.

An `emptyDir` cache is ephemeral and forces cold startup after rescheduling.
Ensure any persistent mount is writable by the container security context.

## Install and verify

After preparing a release-valid values file:

```bash
helm upgrade --install cosmos3-nim "$NIM_HELM_CHART" \
  --version "$NIM_HELM_VERSION" \
  --values values.yaml

kubectl rollout status deployment/cosmos3-nim --timeout=30m
kubectl port-forward service/cosmos3-nim 8000:8000
curl -f http://localhost:8000/v1/health/ready
```

Deployment and service names are illustrative until the chart contract is
published. If the chart uses a StatefulSet, monitor that resource instead.
Send inference only after readiness returns HTTP 200.

## Scale and monitor

Each replica must receive a GPU allocation compatible with its selected
profile. CPU-based autoscaling signals usually do not represent GPU inference
capacity; use release-validated workload or request metrics.

Persistent RWX sharing can reduce download time while creating concurrency and
storage-throughput bottlenecks. Validate cold start, rolling upgrades, scale-up,
cache ownership, and probe budgets before production use.

For metrics and logs, see [Operations](operations.md). For Pending Pods, mount
failures, probe failures, or rejected values, see
[Startup and deployment troubleshooting](operations.md#startup-and-deployment).
