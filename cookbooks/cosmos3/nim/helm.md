<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Deploy the Cosmos3 Certified NIM with Helm

Use this page to deploy the Cosmos3 Certified NIM on Kubernetes with the shared
`nim-wfm` Helm chart. Model selection, hardware floors, and verification are
the same as in [Deployment](deployment.md).

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

This page uses `nim-wfm` chart version 1.3.0:

```bash
export NIM_WFM_VERSION='1.3.0'
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
  size: 200Gi
sharedMemory:
  sizeLimit: 16Gi
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
- **Cache size:** The Nano Generator cache uses about 22 GiB after download
  and materialization; `200Gi` leaves room for other variants and profile
  changes. No single floor is published, so size the volume for the selected
  model with headroom.
- **Memory:** A pod memory limit counts as system memory during profile
  selection. Keep any limit at or above the host RAM value of the selected row.
- **Shared memory:** The chart mounts a memory-backed `emptyDir` at `/dev/shm`
  by default. `sharedMemory.sizeLimit: 16Gi` matches the 16 GiB that the Docker
  launch allocates.
- **Pod user:** The chart runs the pod as UID and GID 1000 with
  `fsGroup: 1000` (`podSecurityContext`). The cache volume must be writable by
  that user; storage that ignores `fsGroup`, such as some NFS exports and
  `hostPath` directories, needs matching ownership.
- **Chart-owned settings:** Use `model.nimCache`, `model.ngcAPISecret`,
  `model.apiPort`, `model.grpcPort`, `model.inferenceProtocol`,
  `model.jsonLogging`, and `model.logLevel` instead of `NIM_CACHE_PATH`,
  `NGC_API_KEY`, `NIM_HTTP_API_PORT`, `NIM_GRPC_API_PORT`,
  `NIM_INFERENCE_PROTOCOL`, `NIM_LOGGING_JSONL`, and `NIM_LOG_LEVEL`. Add other
  [Configuration](configuration.md) variables to `env`.

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

### OpenTelemetry

To export traces and metrics over OTLP, add these entries to `env`. The
endpoint assumes an OpenTelemetry Collector DaemonSet that listens on host
port 4318; change `OTEL_EXPORTER_OTLP_ENDPOINT` for other collector layouts:

```yaml
  - name: NIM_ENABLE_OTEL
    value: "1"
  - name: OTEL_SERVICE_NAME
    value: cosmos3
  - name: OTEL_TRACES_EXPORTER
    value: otlp
  - name: OTEL_METRICS_EXPORTER
    value: otlp
  - name: HOST_IP
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  - name: OTEL_EXPORTER_OTLP_ENDPOINT
    value: "http://$(HOST_IP):4318"
```

`HOST_IP` must come before `OTEL_EXPORTER_OTLP_ENDPOINT` so that Kubernetes
can expand it.

### Other chart features

The chart README documents these optional features; all are off by default:

- **Prometheus metrics:** Set `metrics.serviceMonitor.enabled: true` to create
  a Prometheus Operator `ServiceMonitor` that scrapes `/v1/metrics` on the
  `http-api` port. See [Metrics](operations.md#metrics).
- **Ingress:** `ingress.enabled`, `ingress.className`, `ingress.hosts`, and
  `ingress.tls`. The NIM does not authenticate requests, so protect any
  ingress outside the cluster.
- **Autoscaling:** `autoscaling.enabled` with `minReplicas`, `maxReplicas`, and
  `metrics`. CPU and memory metrics are of limited use for scaling a NIM; use
  custom metrics, for example through `prometheus-adapter`. Each new replica
  must download or mount the model before it becomes ready; see
  [Storage and scaling](#storage-and-scaling).

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

The chart supports these cache options; use only one:

| Option | Values | Notes |
| --- | --- | --- |
| StatefulSet volume claims | `persistence.enabled: true` with the default `statefulSet.enabled: true` | Default. Each replica gets its own claim and downloads the model before it becomes ready |
| Shared volume claim | `persistence.enabled: true`, `statefulSet.enabled: false`, `persistence.accessMode: ReadWriteMany` | One cache shared by all replicas; needs a `ReadWriteMany` storage class |
| Existing claim | `persistence.existingClaim: <claim-name>` | Reuse a pre-filled cache. Run one replica unless the claim is `ReadWriteMany` |
| Direct NFS | `nfs.enabled: true`, `nfs.server`, `nfs.path` | Mount options cannot be set per pod; an NFS-backed claim is usually a better choice |
| `hostPath` | `hostPath.enabled: true`, `hostPath.path` | Ties pods to one node and has security implications |

With `statefulSet.enabled: false` and a `ReadWriteOnce` volume, scaling or
rolling upgrades beyond one pod are likely to fail. Use a `ReadWriteMany`
storage class, manually cloned `ReadOnlyMany` volumes of a pre-filled cache,
or direct NFS. Without persistence, every pod start downloads the model again.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Pod stays `Pending` | Run `kubectl describe pod <pod>` and check `Events` for insufficient GPUs, untolerated taints, or an unbound volume claim |
| Pod restarts while preparing the model workspace | The startup probe ran out before the download finished; increase `startupProbe.failureThreshold` |
| Scaling or upgrade fails with `statefulSet.enabled: false` | The cache volume is `ReadWriteOnce`; see [Storage and scaling](#storage-and-scaling) |
| Profile selection fails | Compare GPU type, count, free memory, and pod memory limit with the [Support matrix](support-matrix.md); see [Troubleshooting](operations.md#troubleshooting) |

## Remove the deployment

```bash
helm uninstall cosmos3
kubectl get pvc
```

StatefulSet volume claims are retained by default and keep the model cache.
Delete them and the `ngc-secret` and `ngc-api` secrets only when they are no
longer needed.
