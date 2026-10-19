<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Deploy the Cosmos3 NIM

Use this page to authenticate to NGC, choose a model, launch Generator or
Reasoner, and verify the selected service. The commands pin the current 2.0.0
release image. Use the [Support matrix](support-matrix.md) to choose a
compatible configuration.

## How selection works

Users normally select a runtime and model, not a profile ID:

1. Choose Generator or Reasoner.
2. Choose a model variant.
3. Optionally pin precision. Generator prefers FP8 when compatible; Reasoner
   derives its preference from GPU compute capability.
4. For Generator, choose latency or throughput.
5. The NIM selects the best compatible profile for those choices and the
   visible GPUs.

Selection prefers a resident profile that uses the visible GPUs well and can
fall back to an equivalent lower-memory offload or tensor-parallel layout when
current free GPU memory is short. If no profile fits, startup fails; free GPU
memory and restart. See
[GPU architecture and topology](support-matrix.md#gpu-architecture-and-topology)
for the memory rules.

### Select a Generator model

| `NIM_MODEL_VARIANT` | Supported use |
| --- | --- |
| `nano` | General-purpose Nano Generator |
| `nano-droid` | Nano-DROID policy with an action-only response; BF16 only |
| `super` | General-purpose Super Generator |
| `super-t2i` | Full-step T2I specialist |
| `super-t2i-4step` | Four-step T2I specialist |
| `super-i2v` | Full-step I2V specialist |
| `super-i2v-4step` | Four-step I2V specialist |

Specialist variants accept only their task. Set the workload objective
explicitly:

| `NIM_PERF_PROFILE` | Optimize for |
| --- | --- |
| `latency` | Lower latency for an individual request |
| `throughput` | Higher aggregate request rate |

The default is `latency`.

### Select a Reasoner model

Set `NIM_MODEL_TYPE=reasoner` and choose `NIM_MODEL_VARIANT=nano` or `super`.
Reasoner rejects `NIM_PERF_PROFILE` at startup. DFlash speculative decoding is
on by default; set `NIM_USE_DFLASH=0` to disable it (see
[Configuration](configuration.md#speculative-decoding)).

### Precision selection

Omit `NIM_PRECISION` normally: Generator prefers FP8 when compatible, and
Reasoner prefers BF16 on compute capability 8.0 through 8.8, FP8 on 8.9 through
9.x, and NVFP4 on 10.0 or newer, falling back to another compatible precision.
Set `NIM_PRECISION=bf16`, `fp8`, or another available value only when the
workload requires it; Nano-DROID is BF16 only.

## Before you deploy

Verify the host against [Prerequisites](prerequisites.md) and choose a
compatible configuration from the [Support matrix](support-matrix.md). For a
Kubernetes deployment, follow [Deploy with Helm](helm.md).

## Authenticate to NGC

Export `NGC_API_KEY` (not `NGC_TOKEN`) as described under
[Network and NGC access](prerequisites.md#network-and-ngc-access); never put the
key in source control. Docker saves the registry login after `docker login`:
Docker Desktop uses a native credential store, while standalone Docker Engine
without a helper stores it base64-encoded in `$HOME/.docker/config.json` and
prints a warning.

For stronger at-rest protection, configure a
[supported credential helper](https://docs.docker.com/reference/cli/docker/login/#credential-stores)
whose `docker-credential-*` program is on `PATH`, for example
`docker-credential-pass` with this entry in `$HOME/.docker/config.json`:

```json
{
  "credHelpers": {
    "nvcr.io": "pass"
  }
}
```

To migrate an existing saved login, run `docker logout nvcr.io` first. Then log
in with the literal username `$oauthtoken`:

```bash
echo "$NGC_API_KEY" \
  | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

`--password-stdin` keeps the key out of shell history but does not encrypt the
saved credential; without a helper, log out when you finish evaluating.

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

The `0.93` default is not reduced automatically and can starve the host and
media processing. Pass `-e NIM_GPU_MEMORY_UTILIZATION` in every Reasoner
preflight and launch command on these systems. Do not set it for Generator or
routinely on a discrete GPU; see
[Reasoner context and scheduling](configuration.md#context-and-scheduling).

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

Use the selectors, GPU exposure, and container memory limit of the real
deployment. For Reasoner, set `NIM_MODEL_TYPE=reasoner`, select `nano` or
`super`, and omit `NIM_PERF_PROFILE`.

On DGX Spark or Jetson AGX Thor, pass the
[exported memory share](#set-the-reasoner-memory-share-on-unified-memory-systems).
This example checks Nano Reasoner:

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

Preflight success is necessary, not sufficient: it confirms that the image
has a profile matching the selectors, compute capability, system-memory floor,
and total and free VRAM (and logs Generator Transfer admission), but it does
not load the model. Only a cold start and representative requests confirm host
RAM, disk, shared memory, the driver stack, media, and inference. Preflight
writes nothing outside its removed container. Resolve any reported failure
before a cold download.

## Prepare the model cache

```bash
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim/cosmos3}"
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R a+rwX "$LOCAL_NIM_CACHE"
```

These permissions suit local development. The container runs as UID/GID 1000
(`nvs`); in production grant that UID/GID write access through a group or ACL.
Keep the cache between runs to avoid repeated downloads.

## Launch Generator

This example runs general-purpose Nano with latency on host port 8000:

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

The container starts in the background. Follow startup logs with
`docker logs -f <container>` (here `cosmos3-generator`); Ctrl+C stops following
without stopping the container.

Change the model and objective as needed, expose every GPU a multi-GPU
configuration requires, and add `-e NIM_PRECISION=fp8` only to pin precision.

## Launch Reasoner

Reasoner uses the same host port; remove a running Generator first:

```bash
docker rm -f cosmos3-generator
```

### DGX Spark and Jetson AGX Thor

Use Nano and pass the exported unified-memory share (`0.80` or `0.70`):

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

### Discrete GPU example

This example launches Super FP8 Reasoner, which needs compute capability 8.9+,
67 GiB of usable VRAM, and 46 GiB of host RAM (see
[Reasoner configurations](support-matrix.md#reasoner-configurations)):

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

It uses the DFlash and video-token pruning defaults. To use another
configuration, change the model and precision selectors together and expose
the GPUs it needs.

To run both runtimes concurrently, publish one on another host port and set
that client's `NIM_URL` accordingly.

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

A cold start (download, materialization, model load, warmup) takes much longer
than HTTP startup. Send inference only after readiness succeeds.

## Verify the selection

```bash
curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
curl -fsS "$NIM_URL/v1/manifest" | python3 -m json.tool
```

Metadata reports the selected model and profile. Confirm that `model_type` and
`inference_endpoint` are `generator` and `/v1/infer`, or `reasoner` and
`/v1/chat/completions`, before sending requests.

## Offline and air-gapped deployment

The NIM can start offline from a cache filled by an earlier online start. The
cache holds only the selected profile's artifacts, so fill it for every
configuration you will run offline.

1. On a host with NGC access, launch each configuration as shown in
   [Launch Generator](#launch-generator) or [Launch Reasoner](#launch-reasoner),
   with the offline host's selectors and GPU exposure, and wait for readiness.
   Several configurations can share one cache.
2. Copy the cache with a tool that preserves symbolic links:

   ```bash
   rsync -a "$LOCAL_NIM_CACHE/" /path/to/offline-cache/
   ```

   Snapshot files are relative symbolic links into `blobs/`, so tools that
   dereference links break the cache. On the offline host,
   `find /path/to/offline-cache -xtype l` must print nothing. Apply the
   ownership from [Prepare the model cache](#prepare-the-model-cache).
3. On the offline host, set `LOCAL_NIM_CACHE` to the copy and run the same
   launch command. Generator can omit `-e NGC_API_KEY`; keep it for Reasoner,
   whose keyless offline start has not been validated.
4. For Generator, do not set `NIM_DISABLE_MODEL_DOWNLOAD=1`; startup rejects it
   because guardrail artifacts must be materialized from the cached files.
5. [Wait for readiness](#wait-for-readiness). With a complete cache, every
   artifact logs `Skipping download, using cached copy of file`; the
   `Downloading manifest profile` line does not indicate network access.

To test a cache on a connected host before moving it, replace `-p 8000:8000`
with `--network none` and check readiness inside the container:

```bash
docker exec cosmos3-generator python3 -c \
  'import urllib.request as u; print(u.urlopen("http://localhost:8000/v1/health/ready").status)'
```

## Advanced profile controls

Offload modes, tag selectors, and profile-ID pins are for validated needs only;
see [Configuration](configuration.md#advanced-profile-controls).

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

GPU counts, VRAM floors, and host RAM are in the
[support matrix](support-matrix.md).

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

Logout does not revoke the NGC key; deactivate or delete an evaluation key in
the NGC user interface. The model cache remains.
