<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Deploy the Cosmos3 NIM with Helm

Use this page to deploy the Cosmos3 NIM on Kubernetes with the shared
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
  size: 220Gi
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

The chart also accepts these selectors as `model.selection.type`,
`model.selection.variant`, `model.selection.precision`, and
`model.selection.performanceProfile` values. Setting a selector both there and
in `env` fails the render.

Adjust it for the chosen configuration:

- **Model and GPUs:** Change `NIM_MODEL_VARIANT`, `NIM_PERF_PROFILE`, and
  `nvidia.com/gpu` together using the
  [Generator configurations](support-matrix.md#generator-configurations). The
  GPU count does not select a GPU type; use `nodeSelector` or `affinity` to
  target eligible GPU nodes, and `tolerations` if those nodes are tainted. On
  nodes with MIG enabled, request the MIG resource instead of
  `nvidia.com/gpu`, for example `nvidia.com/mig-4g.24gb: 2`.
- **Cache size:** `220Gi` is a starting size for large model files. Set `persistence.storageClass` if the
  cluster has no default storage class.
- **Memory:** A pod memory limit counts as system memory during profile
  selection. Keep any limit at or above the host RAM value of the selected row.
- **Shared memory:** The chart mounts a memory-backed `emptyDir` at `/dev/shm`
  by default. `sharedMemory.sizeLimit: 16Gi` matches the 16 GiB that the Docker
  launch allocates.
- **Pod user:** The chart runs the pod as UID and GID 1000 with
  `fsGroup: 1000` (`podSecurityContext`). The cache volume must be writable by
  that user; storage that ignores `fsGroup`, such as some NFS exports and
  `hostPath` directories, needs matching ownership.

Set these through chart values rather than `env`:

| Chart value | Replaces |
| --- | --- |
| `model.nimCache` | `NIM_CACHE_PATH` |
| `model.ngcAPISecret` | `NGC_API_KEY` |
| `model.apiPort` | `NIM_HTTP_API_PORT` |
| `model.jsonLogging` | `NIM_LOGGING_JSONL` |
| `model.logLevel` | `NIM_LOG_LEVEL` |

Add other [Configuration](configuration.md) variables to `env`.

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
startup. Send requests only after the pod reports `Ready`. Kubernetes restarts
the container if it is not ready within `startupProbe.initialDelaySeconds` plus
`startupProbe.failureThreshold` × `startupProbe.periodSeconds`; raise
`failureThreshold` for slow first downloads.

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
| Existing claim | `persistence.enabled: true`, `persistence.existingClaim: <claim-name>` | Reuse a pre-filled cache. Run one replica unless the claim is `ReadWriteMany` |
| Direct NFS | `nfs.enabled: true`, `nfs.server`, `nfs.path` | Mount options cannot be set per pod; an NFS-backed claim is usually a better choice |
| `hostPath` | `hostPath.enabled: true`, `hostPath.path` | Ties pods to one node and has security implications |

Set `replicaCount` for a fixed number of replicas. With the default
StatefulSet, you can pre-populate caches by scaling up to the maximum replica
count once and then scaling down; the retained claims let later scale-ups
start without a new download.

With `statefulSet.enabled: false` and a `ReadWriteOnce` volume, scaling or
rolling upgrades beyond one pod are likely to fail. Use a `ReadWriteMany`
storage class, manually cloned `ReadOnlyMany` volumes of a pre-filled cache,
or direct NFS. Without persistence, the cache is an `emptyDir` and every pod
start downloads the model again.

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

## Chart parameters

These are the `nim-wfm` chart values used with the chart version pinned on this
page (1.3.0); run `helm show values "nim-wfm-${NIM_WFM_VERSION}.tgz"` for the
authoritative list.

### Deployment parameters

| Name | Description | Value |
| --- | --- | --- |
| `affinity` | Affinity settings for deployment. | `{}` |
| `containerSecurityContext` | Sets privilege and access control settings for container (Only affects the main container, not pod-level). | `{}` |
| `customCommand` | Overrides command line options sent to the NIM with the array listed here. | `[]` |
| `customArgs` | Overrides command line arguments of the NIM container with the array listed here. | `[]` |
| `env` | Adds arbitrary environment variables to the main container. | `[]` |
| `extraEnvFrom` | Adds arbitrary ConfigMap or Secret sources to the main container environment. | `[]` |
| `extraVolumes` | Adds arbitrary additional volumes to the deployment set definition. | `{}` |
| `extraVolumeMounts` | Adds volume mounts to the main container from `extraVolumes`. | `{}` |
| `sharedMemory.enabled` | Mounts a RAM-backed `emptyDir` at `/dev/shm`. | `true` |
| `sharedMemory.sizeLimit` | Optional Kubernetes quantity limiting the shared-memory volume. | `""` |
| `image.repository` | Specifies the NIM-WFM image to deploy. | `nvcr.io/nvidia/nemo-microservices/nemoretriever-parse` |
| `image.tag` | Specifies the image tag or version. | `""` |
| `image.pullPolicy` | Sets the image pull policy. | `IfNotPresent` |
| `imagePullSecrets` | Specifies a list of secret names that are needed for the main container and any init containers. | |
| `initContainers` | Specifies model init containers, if needed. | |
| `initContainers.extraInit` | Fully specify any additional init containers your use case requires. | `[]` |
| `nodeSelector` | Sets node selectors for the NIM -- for example `nvidia.com/gpu.present: "true"`. | `{}` |
| `runtimeClassName` | Selects the container runtime class for workload pods. | `""` |
| `priorityClassName` | Selects the Kubernetes priority class for workload pods. | `""` |
| `topologySpreadConstraints` | Configures pod topology spreading. | `[]` |
| `podAnnotations` | Sets additional annotations on the main deployment pods. | `{}` |
| `podSecurityContext` | Specifies security context settings for pod. | |
| `podSecurityContext.runAsUser` | Specify user UID for pod. | `1000` |
| `podSecurityContext.runAsGroup` | Specify group ID for pod. | `1000` |
| `podSecurityContext.fsGroup` | Specify file system owner group id. | `1000` |
| `proxyCA` | Specify a certificate for a custom proxy. When `proxyCA` is set native TLS is used for downloading from NGC. | |
| `proxyCA.enabled` | Specify true if NIM is run behind a proxy. | `false` |
| `proxyCA.secretName` | Specify a name of the Kubernetes secret containing the certificate. The secret is created before the deployment. Must be used together with `proxyCA.keyName` | `""` |
| `proxyCA.keyName` | Specify a name of the key inside the secret which contains the certificate. Must be used together with `proxyCA.secretName` | `""` |
| `terminationGracePeriodSeconds` | Seconds Kubernetes allows the NIM to terminate gracefully. | `60` |
| `replicaCount` | Specify static replica count for deployment. | `1` |
| `resources` | Specify resources limits and requests for the running service. | |
| `resources.limits.nvidia.com/gpu` | Specify number of GPUs to present to the running service. | `1` |
| `serviceAccount` | Options to specify service account for the deployment. | |
| `serviceAccount.create` | Specifies whether a service account should be created. | `false` |
| `serviceAccount.annotations` | Sets annotations to be added to the service account. | `{}` |
| `serviceAccount.name` | Specifies the name of the service account to use. If it is not set and create is `true`, a name is generated using a `fullname` template. | `""` |
| `statefulSet.enabled` | Enables `statefulset` deployment. Enabling `statefulSet` allows PVC templates for scaling. If using central PVC with RWX `accessMode`, this isn't needed. | `true` |
| `tolerations` | Specify tolerations for pod assignment. Allows the scheduler to schedule pods with matching taints. | |

### Autoscaling parameters

Values used for creating a `Horizontal Pod Autoscaler`. If autoscaling is not
enabled, the rest are ignored. NVIDIA recommends usage of the custom metrics
API, commonly implemented with the Prometheus Adapter. Standard metrics of CPU
and memory are of limited use in scaling NIM.

| Name | Description | Value |
| --- | --- | --- |
| `autoscaling.enabled` | Enables horizontal pod autoscaler. | `false` |
| `autoscaling.minReplicas` | Specify minimum replicas for autoscaling. | `1` |
| `autoscaling.maxReplicas` | Specify maximum replicas for autoscaling. | `10` |
| `autoscaling.metrics` | Array of metrics for autoscaling. | `[]` |

### Ingress parameters

| Name | Description | Value |
| --- | --- | --- |
| `ingress.enabled` | Enables ingress. | `false` |
| `ingress.className` | Specify class name for Ingress. | `""` |
| `ingress.annotations` | Specify additional annotations for ingress. | `{}` |
| `ingress.hosts` | Specify list of hosts each containing lists of paths. | |
| `ingress.hosts[0].host` | Specify name of host. | `chart-example.local` |
| `ingress.hosts[0].paths[0].path` | Specify ingress path. | `/` |
| `ingress.hosts[0].paths[0].pathType` | Specify path type. | `ImplementationSpecific` |
| `ingress.tls` | Specify list of pairs of TLS `secretName` and hosts. | `[]` |

### Probe parameters

| Name | Description | Value |
| --- | --- | --- |
| `livenessProbe.enabled` | Enables `livenessProbe`. | `true` |
| `livenessProbe.method` | `LivenessProbe` `http` or `script`, but no script is currently provided. | `http` |
| `livenessProbe.command` | `LivenessProbe` script command to use (unsupported at this time). | `["myscript.sh"]` |
| `livenessProbe.path` | `LivenessProbe` endpoint path. | `/v1/health/live` |
| `livenessProbe.initialDelaySeconds` | Initial delay seconds for `livenessProbe`. | `15` |
| `livenessProbe.timeoutSeconds` | Timeout seconds for `livenessProbe`. | `1` |
| `livenessProbe.periodSeconds` | Period seconds for `livenessProbe`. | `10` |
| `livenessProbe.successThreshold` | Success threshold for `livenessProbe`. | `1` |
| `livenessProbe.failureThreshold` | Failure threshold for `livenessProbe`. | `3` |
| `readinessProbe.enabled` | Enables `readinessProbe`. | `true` |
| `readinessProbe.path` | Readiness Endpoint Path. | `/v1/health/ready` |
| `readinessProbe.initialDelaySeconds` | Initial delay seconds for `readinessProbe`. | `15` |
| `readinessProbe.timeoutSeconds` | Timeout seconds for `readinessProbe`. | `1` |
| `readinessProbe.periodSeconds` | Period seconds for `readinessProbe`. | `10` |
| `readinessProbe.successThreshold` | Success threshold for `readinessProbe`. | `1` |
| `readinessProbe.failureThreshold` | Failure threshold for `readinessProbe`. | `3` |
| `startupProbe.enabled` | Enables `startupProbe`. | `true` |
| `startupProbe.path` | `StartupProbe` Endpoint Path. | `/v1/health/ready` |
| `startupProbe.initialDelaySeconds` | Initial delay seconds for `startupProbe`. | `40` |
| `startupProbe.timeoutSeconds` | Timeout seconds for `startupProbe`. | `1` |
| `startupProbe.periodSeconds` | Period seconds for `startupProbe`. | `10` |
| `startupProbe.successThreshold` | Success threshold for `startupProbe`. | `1` |
| `startupProbe.failureThreshold` | Failure threshold for `startupProbe`. | `180` |

### Helm test parameters

The readiness test always uses the HTTP management endpoint, including when
gRPC inference is selected. The inference test is opt-in because request
payloads are model-specific, and it is only rendered for
`model.inferenceProtocol: http`.

| Name | Description | Value |
| --- | --- | --- |
| `tests.image.repository` | Container image repository used by Helm test pods. | `curlimages/curl` |
| `tests.image.tag` | Container image tag used by Helm test pods. | `8.6.0` |
| `tests.image.pullPolicy` | Container image pull policy used by Helm test pods. | `IfNotPresent` |
| `tests.readiness.enabled` | Enables the HTTP readiness Helm test. | `true` |
| `tests.readiness.path` | HTTP path used by the readiness Helm test. | `/v1/health/ready` |
| `tests.readiness.timeoutSeconds` | Timeout in seconds for the readiness request. | `30` |
| `tests.metadata.enabled` | Enables the runtime metadata Helm test. | `false` |
| `tests.metadata.path` | HTTP path used by the metadata Helm test. | `/v1/metadata` |
| `tests.metadata.timeoutSeconds` | Timeout in seconds for the metadata request. | `30` |
| `tests.metadata.expectedModelType` | Optional expected `model_type` value. | `""` |
| `tests.metadata.expectedInferenceEndpoint` | Optional expected `inference_endpoint` value. | `""` |
| `tests.inference.enabled` | Enables the HTTP inference Helm test. This test is only rendered when `model.inferenceProtocol` is `http`. | `false` |
| `tests.inference.path` | HTTP path used by the inference Helm test. | `/v1/infer` |
| `tests.inference.timeoutSeconds` | Timeout in seconds for the inference request. | `300` |
| `tests.inference.expectedStatus` | Expected HTTP status code from the inference request. | `200` |
| `tests.inference.payload` | Request body sent by the inference Helm test. Required when the inference test is enabled. | `""` |
| `tests.inference.expectedResponseField` | Optional response field required in the inference response. | `""` |

### Metrics parameters

| Name | Description | Value |
| --- | --- | --- |
| `metrics` | Configures Prometheus integration for the NIM HTTP metrics endpoint. | |
| `metrics.serviceMonitor` | Options for `serviceMonitor` to use the Prometheus Operator and the primary service object. | |
| `metrics.serviceMonitor.enabled` | Enables `serviceMonitor` creation. | `false` |
| `metrics.serviceMonitor.additionalLabels` | Specify additional labels for `serviceMonitor`. | `{}` |
| `metrics.serviceMonitor.port` | Named Service port scraped by ServiceMonitor. | `http-api` |
| `metrics.serviceMonitor.path` | HTTP metrics path scraped by ServiceMonitor. | `/v1/metrics` |
| `metrics.serviceMonitor.interval` | Optional Prometheus scrape interval. | `""` |
| `metrics.serviceMonitor.scrapeTimeout` | Optional Prometheus scrape timeout. | `""` |

### Models parameters

| Name | Description | Value |
| --- | --- | --- |
| `model.nimCache` | Path to mount writable storage or a pre-filled model cache for the NIM. | `/model-store` |
| `model.name` | Optional model name retained for compatibility with NIM-specific values files. | `nvidia/Cosmos3-Nano` |
| `model.ngcAPISecret` | Name of a secret containing `NGC_API_KEY` for model downloads. When `model.ngcAPIKey` is set, the chart generates this secret; leave the name empty to generate only the image pull secret. | `ngc-api` |
| `model.ngcAPIKey` | NGC API key literal used to generate the image pull secret and, when `model.ngcAPISecret` is nonempty, the model-download secret. | `""` |
| `model.hfTokenSecret` | Name of a pre-existing secret with a key named `HF_TOKEN` for Hugging Face model downloads. | `""` |
| `model.apiPort` | Specifies the HTTP API and management port. | `8000` |
| `model.grpcPort` | Specifies the gRPC inference port. | `50051` |
| `model.inferenceProtocol` | Selects `http` or `grpc` inference. The HTTP port remains available for health and management endpoints in gRPC mode. | `http` |
| `model.selection.type` | Optional runtime selector mapped to `NIM_MODEL_TYPE`. | `""` |
| `model.selection.variant` | Optional model variant mapped to `NIM_MODEL_VARIANT`. | `""` |
| `model.selection.precision` | Optional precision mapped to `NIM_PRECISION`; empty preserves runtime auto-selection. | `""` |
| `model.selection.performanceProfile` | Optional Generator objective mapped to `NIM_PERF_PROFILE`. | `""` |
| `model.labels` | Specifies extra labels to be added on deployed pods. | `{}` |
| `model.jsonLogging` | Turns JSON lines logging on or off. | `true` |
| `model.logLevel` | Log level of the NIM service. Possible values are TRACE, DEBUG, INFO, WARNING, ERROR, and CRITICAL. | `INFO` |

### Storage parameters

| Name | Description | Value |
| --- | --- | --- |
| `persistence` | Specify settings to modify the behavior and use of persistent volumes for model weights. | |
| `persistence.enabled` | Enables the use of persistent volumes. | `false` |
| `persistence.existingClaim` | Specifies an existing persistent volume claim. If using `existingClaim`, run only one replica or use a `ReadWriteMany` storage setup. | `""` |
| `persistence.storageClass` | Specifies the persistent volume storage class. If set to `"-"`, this disables dynamic provisioning. If left undefined or set to null, the cluster default storage provisioner is used. | `""` |
| `persistence.accessMode` | Specify `accessMode`. If using an NFS or similar setup, you can use `ReadWriteMany`. | `ReadWriteOnce` |
| `persistence.stsPersistentVolumeClaimRetentionPolicy.whenDeleted` | Specifies persistent volume claim retention policy when deleted. Only used with Stateful Set volume templates. | `Retain` |
| `persistence.stsPersistentVolumeClaimRetentionPolicy.whenScaled` | Specifies persistent volume claim retention policy when scaled. Only used with Stateful Set volume templates. | `Retain` |
| `persistence.size` | Specifies the size of the persistent volume claim (for example 40Gi). | `50Gi` |
| `persistence.annotations` | Adds annotations to the persistent volume claim. | `{}` |
| `hostPath` | Configures model cache on local disk on the nodes using `hostPath` -- for special cases. You should understand the security implications before using this option. | |
| `hostPath.enabled` | Enable `hostPath`. | `false` |
| `hostPath.path` | Specifies path on the node used as a `hostPath` volume. | `/model-store` |
| `nfs` | Configures the model cache to sit on shared direct-mounted NFS. NOTE: you cannot set mount options using direct NFS mount to pods without a node-intalled nfsmount.conf. An NFS-based `PersistentVolumeClaim` is likely better in most cases. | |
| `nfs.enabled` | Enables direct pod NFS mount. | `false` |
| `nfs.path` | Specify path on NFS server to mount. | `/exports` |
| `nfs.server` | Specify NFS server address. | `nfs-server.example.com` |
| `nfs.readOnly` | Set to true to mount as read-only. | `false` |

### Service parameters

| Name | Description | Value |
| --- | --- | --- |
| `service.type` | Specifies the service type for the deployment. | `ClusterIP` |
| `service.name` | Overrides the default service name. | `""` |
| `service.apiPort` | Specifies the HTTP API and management port for the service. | `8000` |
| `service.grpcPort` | Specifies the gRPC inference port for the service. It is exposed when `model.inferenceProtocol=grpc`. | `50051` |
| `service.annotations` | Specifies additional annotations to be added to the service. | `{}` |
| `service.labels` | Specifies additional labels to be added to the service. | `{}` |
