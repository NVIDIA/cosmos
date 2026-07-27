<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Deploy the Cosmos3 Certified NIM

Use this page to authenticate to NGC, launch the Cosmos3 Certified NIM, select a
Generator or Reasoner profile, manage the model cache, and verify readiness.

> **Release-dependent values:** The final NGC image repository/tag, published
> support matrix, and release URLs are **TBD**. Commands use `<NIM_IMAGE:TBD>`
> until the release owner supplies the authoritative value. Do not replace it
> with an older separate Generator or Reasoner image.

## Deployment model

One container image contains profiles for both runtime families. One selected
profile starts one API/backend:

| `NIM_MODEL_TYPE` | Backend/API | Use |
| --- | --- | --- |
| `generator` | Generator `POST /v1/infer` | Video generation, action, and transfer |
| `reasoner` | OpenAI-compatible completion/Responses APIs | Image/video understanding and reasoning |

Run separate containers when both APIs must be available simultaneously. Give
each container a distinct host port and cache strategy.

## Before you deploy

Verify the host against [Prerequisites](prerequisites.md), then choose a
released configuration from the [Support matrix](support-matrix.md). For
Kubernetes, continue with [Deploy with Helm](helm.md) after completing NGC
authentication below.

## Create and protect an NGC API key

Create a personal API key in the
[NGC account setup](https://org.ngc.nvidia.com/setup) flow and include access to
the NGC Catalog service. Export it only in the shell that launches the NIM:

```bash
export NGC_API_KEY='<your-ngc-api-key>'
```

The runtime variable is `NGC_API_KEY`. Do not use `NGC_TOKEN` in launch
commands. Never place the real key in source control, notebooks, command output,
saved request JSON, or documentation.

Authenticate Docker using the literal special username `$oauthtoken`:

```bash
echo "$NGC_API_KEY" \
  | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

Docker authentication pulls the container. Passing `NGC_API_KEY` into the
container authorizes cold-start model artifact download.

## Set the released image

Replace this placeholder after the final NGC repository and tag are approved:

```bash
export NIM_IMAGE='<NIM_IMAGE:TBD>'
```

Pin an explicit release tag. Do not use `latest` in reproducible deployments.
Discover available tags from the final NGC Catalog page or NGC CLI once the
public repository is known.

## Prepare the model cache

A persistent cache avoids downloading/materializing model artifacts on every
restart:

```bash
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim/cosmos3}"
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R a+rwX "$LOCAL_NIM_CACHE"
```

Mount it at `/opt/nim/.cache`. In a production environment, prefer a
group/ACL/UID-specific permission policy over world-writable permissions. The
container must be able to create files in the mount.

Keep the cache between runs. A cold cache can add substantial download,
engine-build, load, and warmup time. Remove it only to reclaim space or when a
release explicitly requires a clean artifact layout.

## Launch a Generator profile

This is the minimal structural launch. Replace `<NIM_IMAGE:TBD>` and choose
selectors supported by the final release:

```bash
docker run --rm --name cosmos3-generator \
  --gpus '"device=0"' \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=generator \
  -e NIM_MODEL_SIZE=nano \
  -e NIM_PERF_PROFILE=latency \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

The current source defaults to Generator when `NIM_MODEL_TYPE` is omitted, but
set it explicitly in documentation, automation, and multi-container setups.

## Launch a Reasoner profile

Use a different host port when keeping the Generator running:

```bash
docker run --rm --name cosmos3-reasoner \
  --gpus '"device=1"' \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -p 8001:8000 \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=reasoner \
  -e NIM_MODEL_SIZE=nano \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

The Reasoner has no `latency`/`throughput` profile axis, so do not set
`NIM_PERF_PROFILE` or `profile=...` for it. Point Reasoner clients at
`http://localhost:8001` in this two-container example.

## Understand the Docker flags

| Flag | Purpose |
| --- | --- |
| `--rm --name` | Remove the stopped container and give it a stable operational name |
| `--gpus` | Expose the GPU set from which profile selection can choose |
| `--shm-size` | Expand `/dev/shm` for media/intermediate buffers; final recommended size is release-dependent |
| `--ulimit memlock=-1` | Allow engine/model processes to pin required memory |
| `--ulimit stack=67108864` | Avoid small default stacks during model/engine setup |
| `--ulimit nofile=65536:65536` | Allow many checkpoint/artifact files to open concurrently |
| `-p HOST:8000` | Publish the container's HTTP API on the selected host port |
| `-e NGC_API_KEY` | Pass the existing shell variable without writing its value into the command |
| `-v ...:/opt/nim/.cache` | Persist downloaded and materialized model artifacts |

If multiple NIM containers publish additional internal/backend ports in a final
release recipe, every container needs unique host-side port numbers even when
container-side ports are identical.

## Wait for readiness

Liveness means the HTTP process exists; readiness means the selected model can
serve requests:

```bash
curl -f http://localhost:8000/v1/health/live

until curl -fsS http://localhost:8000/v1/health/ready >/dev/null; do
  sleep 10
done
```

Cold materialization, compilation, loading, and warmup can take much longer
than HTTP startup. Send inference only after readiness returns HTTP 200. Do not
use liveness as a substitute.

## Select a profile

Profiles describe compatible model artifacts and runtime layouts. The exact
released profile grid is intentionally not reproduced here while it is still
being finalized. The stable selection concepts are:

- runtime: Generator or Reasoner;
- model size: Nano or Super;
- precision: for example BF16, FP8, or NVFP4 when supported by the released
  artifact and GPU architecture;
- Generator objective: latency or aggregate throughput;
- GPU count and parallelism layout; and
- Generator offload mode for lower-VRAM deployments.

User-facing selectors:

| Variable | Meaning |
| --- | --- |
| `NIM_MODEL_TYPE` | `generator` or `reasoner` |
| `NIM_MODEL_SIZE` | `nano` or `super` |
| `NIM_PRECISION` | Requested precision when a compatible released profile exists |
| `NIM_PERF_PROFILE` | Generator-only `latency` or `throughput` |
| `NIM_OFFLOAD_MODE` | Generator offload preference, such as `none`, `model`, or `layer`, when released |
| `NIM_TAGS_SELECTOR` | Comma-separated exact tags such as `model_size=nano,n_gpus=2` |
| `NIM_MODEL_PROFILE` | Exact manifest profile ID; use only when intentionally pinning a reviewed release |

Shorthands and the same keys in `NIM_TAGS_SELECTOR` must not conflict.

At a high level, selection:

1. removes profiles incompatible with visible GPU count, VRAM, compute
   capability, and runtime-specific constraints;
2. applies explicit selector tags and an exact profile pin, if supplied;
3. applies soft defaults only when they keep at least one compatible candidate;
4. prefers layouts that use the requested/available resources; and
5. applies runtime-specific layout ranking.

Inspect `/v1/metadata` and `/v1/manifest` after startup rather than assuming
which row won.

### Latency, throughput, and parallelism

- Generator latency profiles shard one request across compatible parallelism
  dimensions to reduce per-request latency.
- Generator throughput profiles use data-parallel workers/replicas where the
  released layout permits, increasing aggregate request rate rather than
  necessarily reducing one request's latency.
- Reasoner profiles select their tensor-parallel GPU allocation without the
  Generator performance-scenario tag. Scale aggregate Reasoner throughput with
  external replicas and load balancing unless a release documents otherwise.

`nim_dp`, `nim_gp`, `nim_up`, and `nim_tp` are low-level manifest tags. Prefer
the high-level shorthands unless debugging or pinning a layout validated for the
exact release.

### Low-VRAM offload profiles

The current design includes Generator profiles that can offload model state to
reduce resident GPU-memory requirements. Conceptually:

- `none` keeps the normal resident layout and is preferred when it fits;
- model-level offload reduces GPU residency with added transfer/startup cost;
  and
- layer-level offload can reduce GPU residency further with a larger latency
  tradeoff.

Offload profiles are intended for access on lower-VRAM GPUs, not peak
performance. Current source profiles are provisional, may exist only for
specific model/precision/GPU-count combinations, and can change before release.
Confirm released offload rows and memory requirements in the
[Support matrix](support-matrix.md) before setting `NIM_OFFLOAD_MODE`.

## Continue configuring the deployment

- [Configuration](configuration.md) lists launch-time environment variables,
  defaults, conflicts, and prompt-upsampling settings.
- [Bring your own checkpoint](bring-your-own-checkpoint.md) covers the
  Generator checkpoint override and validation.
- [Deploy with Helm](helm.md) covers Kubernetes secrets, storage, GPU
  resources, probes, and rollout.

## Stop the NIM

```bash
docker stop cosmos3-generator
docker stop cosmos3-reasoner
```

The examples use `--rm`, so stopped containers are removed automatically. Keep
the cache unless intentionally reclaiming disk space.

For health, logs, metrics, guardrails, and troubleshooting, continue with
[operations.md](operations.md).
