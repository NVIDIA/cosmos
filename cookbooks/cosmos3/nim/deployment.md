<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Deploy the Cosmos3 Certified NIM

Use this page to authenticate to NGC, choose a model, launch Generator or
Reasoner, and verify the selected service. The commands pin the current 2.0.0
release image. Use the [Support matrix](support-matrix.md) to choose a
compatible configuration. For an active container, `/v1/manifest` is the
authority for available profiles.

## How selection works

Users normally select a runtime and model, not a profile ID:

1. Choose Generator or Reasoner.
2. Choose a model variant.
3. Optionally pin precision. Generator prefers FP8 when compatible; Reasoner
   derives its preference from GPU compute capability.
4. For Generator, choose latency or throughput.
5. The NIM selects the best compatible profile for those choices and the
   visible GPUs.

A profile is the resolved deployment configuration: model artifacts,
precision, GPU layout, and any required GPU/system-memory residency policy.
Automatic selection filters on effective system memory and stable total VRAM,
then takes one snapshot of current free memory on every visible GPU. It prefers
a compatible profile that avoids offload and makes effective use of the
available GPUs, but can choose an equivalent lower-memory offload or
tensor-parallel layout when the preferred layout does not currently fit. On an
integrated GPU with unified host/device memory, selection reserves host memory
and uses resident Generator model and guardrail profiles. Startup fails
if the chosen model cannot run on the host. See [Support
matrix](support-matrix.md#gpu-architecture-and-topology) for the memory and
shared-memory rules, then confirm the selected profile in the exact image
manifest.

### Select a Generator model

| `NIM_MODEL_VARIANT` | Supported use |
| --- | --- |
| `nano` | General-purpose Nano Generator |
| `nano-droid` | Nano-DROID policy; BF16 only |
| `super` | General-purpose Super Generator |
| `super-t2i` | Full-step T2I specialist |
| `super-t2i-4step` | Four-step T2I specialist |
| `super-i2v` | Full-step I2V specialist |
| `super-i2v-4step` | Four-step I2V specialist |

For Generator, `NIM_MODEL_VARIANT` determines Nano versus Super and selects
an exact general-purpose or specialist model.

Choose the workload objective explicitly:

| `NIM_PERF_PROFILE` | Optimize for |
| --- | --- |
| `latency` | Lower latency for an individual request |
| `throughput` | Higher aggregate request rate |

The software defaults to `latency` when the selector is omitted.

### Select a Reasoner model

Set `NIM_MODEL_TYPE=reasoner` and choose `NIM_MODEL_VARIANT=nano` or `super`.
Reasoner does not use `NIM_PERF_PROFILE`. Nano and Super Reasoner enable their
bundled DFlash speculative-decoding drafts by default. Set `NIM_USE_DFLASH=0`
to run the selected Reasoner target model without DFlash; see
[Configuration](configuration.md#speculative-decoding).

### Precision selection

Omit `NIM_PRECISION` for normal automatic selection. Generator prefers FP8
when compatible. Reasoner prefers BF16 on compute capability 8.0 through 8.8,
FP8 on 8.9 through 9.x, and NVFP4 on 10.0 or newer; selection uses another
compatible precision when the preferred row is unavailable. Set
`NIM_PRECISION=bf16`, `fp8`, or another available value only when the workload
requires an explicit precision. Nano-DROID currently has BF16 profiles only.

## Before you deploy

Verify the host against [Prerequisites](prerequisites.md) and choose a
compatible configuration from the [Support matrix](support-matrix.md). For the
current availability of Helm guidance, see [Helm deployment](helm.md).

## Authenticate to NGC

Create and export `NGC_API_KEY` as described under
[Network and NGC access](prerequisites.md#network-and-ngc-access). Docker saves
registry credentials after login. Docker Desktop configures a native credential
store automatically. Without an available helper, standalone Docker Engine
stores the credential in a base64-encoded form in `$HOME/.docker/config.json`
and prints a warning.

For stronger at-rest protection, optionally install and initialize a
[supported credential helper](https://docs.docker.com/reference/cli/docker/login/#credential-stores),
ensure its `docker-credential-*` program is on `PATH`, and merge a
registry-specific entry into `$HOME/.docker/config.json`. For example, an
initialized `pass` store uses:

```json
{
  "credHelpers": {
    "nvcr.io": "pass"
  }
}
```

This example requires `docker-credential-pass` on `PATH`. Use the suffix of
the helper installed for your platform. If you previously logged in to
`nvcr.io` without a helper and want to migrate the saved credential, run
`docker logout nvcr.io` before configuring the helper. Then authenticate in the
same shell:

```bash
echo "$NGC_API_KEY" \
  | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

The literal Docker username is `$oauthtoken`. `NGC_API_KEY` authorizes model
artifact download inside the container; do not substitute `NGC_TOKEN` or place
the key in source control. `--password-stdin` keeps the key out of shell history
and command logs, but it does not encrypt the credential Docker stores after
login. A credential helper protects that saved credential; without one, use the
evaluation logout step when you finish.

Pin and pull the current release image:

```bash
export NIM_IMAGE='nvcr.io/nim/nvidia/cosmos3:2.0.0'
docker pull "$NIM_IMAGE"
```

Do not replace the versioned tag with `latest` or another unvalidated image.

### Set the Reasoner memory share on unified-memory systems

Before the first Reasoner preflight or launch on DGX Spark/GB10 or Jetson AGX
Thor, set a workload-specific share of the unified host/device memory pool. For
an image-only Reasoner workload, use:

```bash
export NIM_GPU_MEMORY_UTILIZATION=0.80
```

For any Reasoner workload that includes video, or a mix of images and video,
use:

```bash
export NIM_GPU_MEMORY_UTILIZATION=0.70
```

The default `0.93` is not reduced automatically and can leave too little memory
for the host and media processing. Pass the exported value with
`-e NIM_GPU_MEMORY_UTILIZATION` in every Reasoner preflight and service-launch
command on these systems, starting with the first command. Do not set this
Reasoner-only variable for Generator or as a routine override on a discrete
GPU. See [Reasoner context and scheduling](configuration.md#context-and-scheduling)
for the variable contract.

## Run the pre-download profile preflight

After the image is present locally, run profile selection without starting the
server or downloading model artifacts. This example checks the same Nano
Generator selectors used by the launch example:

```bash
docker run --rm \
  --gpus '"device=0"' \
  -e NIM_MODEL_TYPE=generator \
  -e NIM_MODEL_VARIANT=nano \
  -e NIM_PERF_PROFILE=latency \
  --entrypoint /bin/bash \
  "$NIM_IMAGE" \
  -lc '
    set -e
    output=/tmp/cosmos3-preflight.env
    /opt/nim/.venv/bin/python -m profile_selection.startup --output "$output"
    printf "Profile preflight passed.\n"
  '
```

Use the runtime, model, precision, performance, offload, profile-selection
environment, GPU exposure, and container memory limit intended for the real
deployment. Expose every GPU that deployment will use. For Reasoner, set
`NIM_MODEL_TYPE=reasoner`, select `nano` or `super`, and omit
`NIM_PERF_PROFILE`.

On DGX Spark or Jetson AGX Thor, use the value exported in [Set the Reasoner
memory share on unified-memory systems](#set-the-reasoner-memory-share-on-unified-memory-systems)
in the first preflight command. This complete example checks Nano Reasoner:

```bash
docker run --rm \
  --gpus '"device=0"' \
  -e NIM_MODEL_TYPE=reasoner \
  -e NIM_MODEL_VARIANT=nano \
  -e NIM_GPU_MEMORY_UTILIZATION="$NIM_GPU_MEMORY_UTILIZATION" \
  --entrypoint /bin/bash \
  "$NIM_IMAGE" \
  -lc '
    set -e
    output=/tmp/cosmos3-preflight.env
    /opt/nim/.venv/bin/python -m profile_selection.startup --output "$output"
    printf "Profile preflight passed.\n"
  '
```

Do not pass `NGC_API_KEY`, `NIM_MODEL_PATH`, or `NIM_DFLASH_MODEL_PATH` to this
bundled-profile check. Checkpoint-source validation is a separate BYOC step,
and an `hf://` source can download files.

A successful preflight proves that the image manifest contains a candidate
profile and that the selectors, effective system-memory floor, GPU compute
capability, total VRAM, and current free VRAM pass profile selection. For
Generator, it also reports Transfer VRAM admission in the logs. It does not
load the model or establish general CPU, host-RAM, disk, shared-memory, driver,
Docker, Container Toolkit, media, or inference compatibility. Only a successful
cold start and representative requests establish those later stages.

The preflight writes only inside its temporary container. `--rm` removes that
container when the command exits; it does not remove the image or model cache.
If preflight fails, resolve the reported selector or resource issue before
starting a cold download.

## Prepare the model cache

```bash
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim/cosmos3}"
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R a+rwX "$LOCAL_NIM_CACHE"
```

The examples use permissive local-development permissions. In production, use
an appropriate UID, group, or ACL policy. Keep the cache between runs to avoid
repeated downloads and materialization.

## Launch Generator

The local examples publish the selected runtime at `http://localhost:8000`.
Run one runtime at a time when using this default host port.

This example explicitly chooses the general-purpose Nano model and latency:

```bash
docker run -d --name cosmos3-generator \
  --gpus '"device=0"' \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=generator \
  -e NIM_MODEL_VARIANT=nano \
  -e NIM_PERF_PROFILE=latency \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

The command starts the container in the background. Follow startup logs with
`docker logs -f cosmos3-generator`; press Ctrl+C to stop following logs without
stopping the container. On a discrete-GPU host, provision at least 40 GiB of
host RAM if automatic selection chooses a non-offloading Nano profile; the
16-GiB profile admission check is intentionally lower than that practical
requirement. Use the separate RAM floor in the support matrix if selection
chooses an offload profile.

Replace the model and performance objective with values supported by the image.
Expose the GPU count required by the selected configuration instead of keeping
`--gpus '"device=0"'` when a multi-GPU configuration is required. Add
`-e NIM_PRECISION=fp8` only when precision must be pinned.

## Launch Reasoner

The Reasoner uses the same default host URL as the Generator. If you launched
Generator above, remove it before reusing host port `8000`:

```bash
docker rm -f cosmos3-generator
```

### DGX Spark and Jetson AGX Thor

Use Nano and pass the unified-memory share exported before preflight. This
command is ready for an image-only workload when the exported value is `0.80`,
or a video or mixed-media workload when it is `0.70`:

```bash
docker run -d --name cosmos3-reasoner \
  --gpus '"device=0"' \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=reasoner \
  -e NIM_MODEL_VARIANT=nano \
  -e NIM_GPU_MEMORY_UTILIZATION="$NIM_GPU_MEMORY_UTILIZATION" \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

Do not omit `-e NIM_GPU_MEMORY_UTILIZATION` and rely on the `0.93` default on
these unified-memory systems. Follow startup logs with
`docker logs -f cosmos3-reasoner`; press Ctrl+C to stop following logs without
stopping the container.

### Discrete GPU example

This example launches the Super FP8 Reasoner on a compatible discrete GPU. The
one-GPU configuration requires compute capability 8.9 or newer, at least 67
GiB of total and currently usable VRAM after the Reasoner reserve, and 46 GiB
of practical host RAM; see the [Reasoner
configurations](support-matrix.md#reasoner-configurations):

```bash
docker run -d --name cosmos3-reasoner \
  --gpus '"device=0"' \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=reasoner \
  -e NIM_MODEL_VARIANT=super \
  -e NIM_PRECISION=fp8 \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

The command pins Super FP8 and uses the DFlash and video-token pruning
defaults. It starts the Reasoner in the background. Follow startup logs with
`docker logs -f cosmos3-reasoner`; press Ctrl+C to stop following logs without
stopping the container. Expose all GPUs required by the selected Reasoner
configuration. To use another compatible Reasoner configuration, change the
model and precision selectors together and revalidate representative requests.

Both runtimes listen on container HTTP port `8000`; the Docker mapping chooses
the host port. To run both containers concurrently, publish one on another
unused host port and set that client's `NIM_URL` accordingly.

## Wait for readiness

Liveness means the HTTP process exists. Readiness means the selected model can
serve requests:

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -f "$NIM_URL/v1/health/live"

until curl -fsS "$NIM_URL/v1/health/ready" >/dev/null; do
  sleep 10
done
```

Cold download, materialization, compilation, model load, and warmup can take
much longer than HTTP startup. Send inference only after readiness succeeds.

## Verify the selection

```bash
curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
curl -fsS "$NIM_URL/v1/manifest" | python3 -m json.tool
```

Metadata confirms the selected model and profile. It also reports `model_type`
and `inference_endpoint`; verify that they identify `generator`
and `/v1/infer`, or `reasoner` and `/v1/chat/completions`, before sending an
inference request. This is verification, not a normal profile-selection step.

## Advanced profile controls

Most deployments should stop at model, optional precision, and Generator
latency/throughput selection. Use these controls only for a validated need:

| Variable | Use |
| --- | --- |
| `NIM_OFFLOAD_MODE` | Request an available lower-memory model offload mode |
| `NIM_TAGS_SELECTOR` | Filter profiles by exact manifest tags |
| `NIM_MODEL_PROFILE` | Pin one reviewed profile ID from the exact image |

Exact profile pins and low-level tags reduce portability across hosts and
releases. If automatic selection fails, first choose a smaller model or
compatible precision rather than copying a profile ID from another system.
See [Configuration](configuration.md#advanced-profile-controls) for details.

## Docker flag summary

| Flag | Purpose |
| --- | --- |
| `--gpus` | Expose the GPUs available to automatic profile selection |
| `--shm-size` | Provide shared memory for media and intermediate buffers |
| `--ulimit` | Raise memory-lock, stack, and open-file limits |
| `-p HOST:8000` | Publish the selected host port |
| `-e NGC_API_KEY` | Pass the exported NGC credential |
| `-e NIM_GPU_MEMORY_UTILIZATION=...` | Set the required Reasoner memory share on DGX Spark or Jetson AGX Thor |
| `-v ...:/opt/nim/.cache` | Persist model artifacts |

GPU counts, compute-capability gates, VRAM floors, and practical host RAM are
summarized in the [configuration matrix](support-matrix.md). Each participating
GPU must have enough current free memory at startup, not only enough total
capacity. On discrete-GPU hosts, non-offloading profiles require 40 GiB for
Generator Nano, 78 GiB for Generator Super FP8, 112 GiB for Generator Super
BF16, 24 GiB for Reasoner Nano BF16, 18 GiB for Reasoner Nano FP8 or NVFP4,
36 GiB for Reasoner Super NVFP4, 46 GiB for Reasoner Super FP8, or 76 GiB for
Reasoner Super BF16. Their embedded profile
tags check only 16 GiB so an attempted startup is not blocked; passing that
admission check does not satisfy the practical requirement. Nano offload
profiles retain a 64-GiB profile floor, and Super offload profiles retain 150
GiB. Docker or Kubernetes memory limits count as available system memory.
Unified-memory systems continue to use the shared-pool table and do not add a
separate host RAM requirement. Confirm that the selected row and its tags are
present in the target image before deployment.

## Next steps

- [Configuration](configuration.md) lists user-facing and advanced variables.
- [Bring your own checkpoint](bring-your-own-checkpoint.md) covers local and
  Hugging Face model sources.
- [API reference](api-reference.md) routes requests to the right task guide.
- [Operations](operations.md) covers health, logs, metrics, and failures.

Remove the example containers when finished:

```bash
docker rm -f cosmos3-generator
docker rm -f cosmos3-reasoner
docker logout nvcr.io
```

Docker logout removes the saved `nvcr.io` login from the configured credential
store; it does not revoke the NGC personal key. Deactivate or delete a dedicated
evaluation key in the NGC user interface when it is no longer needed. The
persistent model cache remains.
