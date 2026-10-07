<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Deploy the Cosmos3 Certified NIM with Helm

Use this page to deploy the Cosmos3 Certified NIM on Kubernetes with the shared
`nim-wfm` Helm chart. It follows the NVIDIA
[Deploy with Helm](https://docs.nvidia.com/nim/cosmos/latest/helm.html) guide
with Cosmos3 image and runtime settings. Model selection, hardware floors, and
verification are the same as in [Deployment](deployment.md).

## Prerequisites

- A Kubernetes cluster with GPU nodes that match a row in the
  [Support matrix](support-matrix.md), with the
  [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/)
  installed.
- Helm and `kubectl` access to the target namespace. The commands below use the
  default namespace.
- A storage class for the persistent model cache.
- `NGC_API_KEY` exported as described in
  [Network and NGC access](prerequisites.md#network-and-ngc-access).

## Create secrets

The chart needs an image pull secret and a generic secret with a key named
`NGC_API_KEY`:

```bash
kubectl create secret docker-registry ngc-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY"
kubectl create secret generic ngc-api \
  --from-literal=NGC_API_KEY="$NGC_API_KEY"
```

## Fetch the chart

Choose a `nim-wfm` chart version in the
[NGC Catalog](https://catalog.ngc.nvidia.com/) and download it:

```bash
export NIM_WFM_VERSION='<chart-version>'
helm fetch "https://helm.ngc.nvidia.com/nim/charts/nim-wfm-${NIM_WFM_VERSION}.tgz" \
  --username='$oauthtoken' \
  --password="$NGC_API_KEY"
```

The chart README lists every value supported by that version:

```bash
helm show readme "nim-wfm-${NIM_WFM_VERSION}.tgz"
helm show values "nim-wfm-${NIM_WFM_VERSION}.tgz"
```

## Configure values

Save the following as `cosmos3-values.yaml`. It deploys the Nano Generator,
optimized for latency, on one GPU:

```yaml
image:
  repository: nvcr.io/nim/nvidia/cosmos3
  tag: "2.0.0"
imagePullSecrets:
  - name: ngc-secret
model:
  ngcAPISecret: ngc-api
  nimCache: /opt/nim/.cache
persistence:
  enabled: true
  size: <cache-size>
resources:
  limits:
    nvidia.com/gpu: 1
env:
  - name: NIM_MODEL_TYPE
    value: generator
  - name: NIM_MODEL_VARIANT
    value: nano
  - name: NIM_PERF_PROFILE
    value: latency
```

Adjust it for the chosen configuration:

- **Model and GPUs:** Change `NIM_MODEL_VARIANT`, `NIM_PERF_PROFILE`, and
  `nvidia.com/gpu` together using the
  [Generator configurations](support-matrix.md#generator-configurations). The
  GPU count does not select a GPU type; use `nodeSelector`, `affinity`, or
  `tolerations` to target eligible nodes.
- **Cache size:** Replace `<cache-size>` with a volume size for the selected
  model artifacts and materialization. No single floor is published.
- **Memory:** A pod memory limit counts as system memory during profile
  selection. Keep any limit at or above the host RAM value of the selected row.
- **Shared memory:** The Docker launch allocates 16 GiB of `/dev/shm`. If
  `helm template` output shows no `/dev/shm` mount, add a memory-backed
  `emptyDir` through `extraVolumes` and `extraVolumeMounts`.
- **Chart-owned settings:** Use `model.nimCache`, `model.ngcAPISecret`,
  `model.apiPort`, `model.jsonLogging`, and `model.logLevel` instead of
  `NIM_CACHE_PATH`, `NGC_API_KEY`, `NIM_HTTP_API_PORT`, `NIM_LOGGING_JSONL`,
  and `NIM_LOG_LEVEL`. Add other [Configuration](configuration.md) variables to
  `env`.

To check profile compatibility before a cold download, run the Docker
[pre-download profile preflight](deployment.md#run-the-pre-download-profile-preflight)
on a host with the same GPUs and selectors.

### Reasoner

For Reasoner, replace `env` with the Reasoner selectors and choose the GPU
count from the [Reasoner configurations](support-matrix.md#reasoner-configurations).
Reasoner does not use `NIM_PERF_PROFILE`:

```yaml
env:
  - name: NIM_MODEL_TYPE
    value: reasoner
  - name: NIM_MODEL_VARIANT
    value: nano
```

On DGX Spark/GB10 or Jetson AGX Thor nodes, also set
`NIM_GPU_MEMORY_UTILIZATION` to `"0.80"` for image-only workloads or `"0.70"`
for video or mixed-media workloads; see
[Set the Reasoner memory share](deployment.md#set-the-reasoner-memory-share-on-unified-memory-systems).

## Install

Review the rendered manifests, then install the release:

```bash
helm template cosmos3 "nim-wfm-${NIM_WFM_VERSION}.tgz" -f cosmos3-values.yaml
helm install cosmos3 "nim-wfm-${NIM_WFM_VERSION}.tgz" -f cosmos3-values.yaml
kubectl get pods -w
```

Cold download, materialization, and model load can take much longer than pod
startup. Send requests only after the pod reports `Ready`.

## Verify

The chart creates a `ClusterIP` Service with no ingress, and the NIM does not
handle authentication. Port-forward the Service for testing:

```bash
kubectl port-forward service/cosmos3-nim-wfm 8000:http-api
```

In another terminal:

```bash
export NIM_URL=http://localhost:8000
curl -fsS "$NIM_URL/v1/health/ready"
curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
```

Confirm that metadata reports `generator` and `/v1/infer`, or `reasoner` and
`/v1/chat/completions`. Then run a first request from
[Use an existing NIM endpoint](README.md#use-an-existing-nim-endpoint).

## Storage and scaling

With `persistence.enabled` and the default `statefulSet.enabled: true`, each
replica gets its own persistent volume claim and downloads the model before it
becomes ready. To share one cache across replicas, use a `ReadWriteMany`
storage class and set `persistence.accessMode: ReadWriteMany`. With
`statefulSet.enabled: false` and a `ReadWriteOnce` volume, scaling beyond one
pod is likely to fail. The chart also supports `nfs` and `hostPath`;
`hostPath` ties pods to one node and has security implications.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Pod stays `Pending` | Run `kubectl describe pod <pod>` and check `Events` for insufficient GPUs, untolerated taints, or an unbound volume claim |
| Pod restarts while preparing the model workspace | The startup probe ran out before the download finished; increase `startupProbe.failureThreshold` |
| Profile selection fails | Compare GPU type, count, free memory, and pod memory limit with the [Support matrix](support-matrix.md); see [Troubleshooting](operations.md#troubleshooting) |

## Remove the deployment

```bash
helm uninstall cosmos3
kubectl get pvc
```

StatefulSet volume claims are retained by default and keep the model cache.
Delete them and the `ngc-secret` and `ngc-api` secrets only when they are no
longer needed.
