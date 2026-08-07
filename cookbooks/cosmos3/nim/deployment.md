<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Deploy the Cosmos3 Certified NIM

Use this page to authenticate to NGC, choose a model, launch Generator or
Reasoner, and verify the selected service. The final NGC image remains **TBD**;
semi-final source-profile hardware requirements are documented in the
[support matrix](support-matrix.md) pending release approval.

## How selection works

Users normally select a runtime and model, not a profile ID:

1. Choose Generator or Reasoner.
2. Choose a model variant.
3. Optionally pin precision; otherwise FP8 is preferred when compatible.
4. For Generator, choose latency or throughput.
5. The NIM selects the best compatible profile for those choices and the
   visible GPUs.

A profile is the resolved deployment configuration: model artifacts,
precision, GPU layout, and any required GPU/system-memory residency policy.
Automatic selection prefers a compatible profile that avoids offload and makes
effective use of the available GPUs. On an integrated GPU with unified
host/device memory, selection reserves host memory and uses resident Generator
model and guardrail profiles. Startup fails if the chosen model cannot run on
the host. See [Support matrix](support-matrix.md#gpu-architecture-and-topology)
for the shared-memory rule.

### Select a Generator model

| `NIM_MODEL_VARIANT` | Contract |
| --- | --- |
| `nano` | General-purpose Nano Generator |
| `nano-droid` | Nano-DROID policy; current implementation is BF16 only |
| `super` | General-purpose Super Generator |
| `super-t2i` | Full-step T2I specialist |
| `super-t2i-4step` | Four-step T2I specialist |
| `super-i2v` | Full-step I2V specialist |
| `super-i2v-4step` | Four-step I2V specialist |

For Generator, `NIM_MODEL_VARIANT` determines Nano versus Super and selects
an exact general-purpose or specialist checkpoint contract.

Choose the workload objective explicitly:

| `NIM_PERF_PROFILE` | Optimize for |
| --- | --- |
| `latency` | Lower latency for an individual request |
| `throughput` | Higher aggregate request rate |

The software defaults to `latency` when the selector is omitted.

### Select a Reasoner model

Set `NIM_MODEL_TYPE=reasoner` and choose `NIM_MODEL_VARIANT=nano` or `super`.
Reasoner does not use `NIM_PERF_PROFILE`. Nano Reasoner can optionally enable
DFlash speculative decoding with
`NIM_USE_DFLASH=1`; see [Configuration](configuration.md#speculative-decoding).

### Precision selection

Omit `NIM_PRECISION` for normal automatic selection. FP8 is preferred when the
chosen model, released profiles, and GPU support it; otherwise selection falls
back to another compatible precision. Set `NIM_PRECISION=bf16`, `fp8`, or
another released value only when the workload requires an explicit precision.
Nano-DROID currently has BF16 profiles only.

## Before you deploy

Verify the host against [Prerequisites](prerequisites.md) and the released
[Support matrix](support-matrix.md). For Kubernetes, see
[Deploy with Helm](helm.md).

## Authenticate to NGC

Create an API key with NGC Catalog access, then export it in the shell that
launches the NIM:

```bash
export NGC_API_KEY='<your-ngc-api-key>'
echo "$NGC_API_KEY" \
  | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

The literal Docker username is `$oauthtoken`. `NGC_API_KEY` authorizes model
artifact download inside the container; do not substitute `NGC_TOKEN` or place
the key in source control.

Set the released image with an explicit tag:

```bash
export NIM_IMAGE='<NIM_IMAGE:TBD>'
```

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

This example explicitly chooses the general-purpose Nano model and latency:

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
  -e NIM_MODEL_VARIANT=nano \
  -e NIM_PERF_PROFILE=latency \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

Replace the model and performance objective with values supported by the
released image. Add `-e NIM_PRECISION=fp8` only when precision must be pinned.

## Launch Reasoner

Use another GPU and host port when Generator remains active:

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
  -e NIM_MODEL_VARIANT=nano \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

Point Reasoner clients at `http://localhost:8001` in this two-container setup.

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

Metadata confirms the selected model, profile, and Generator variant. This is
verification, not a normal profile-selection step.

## Advanced profile controls

Most deployments should stop at model, optional precision, and Generator
latency/throughput selection. Use these controls only for a validated need:

| Variable | Use |
| --- | --- |
| `NIM_OFFLOAD_MODE` | Request a released lower-memory model offload mode |
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
| `-v ...:/opt/nim/.cache` | Persist model artifacts |

GPU counts, compute-capability gates, and VRAM floors are summarized in the
[semi-final profile matrix](support-matrix.md). Current Super-family BF16 model-
and layer-offload profiles require 150 GiB of effective system memory. Docker
or Kubernetes memory limits count as the available system memory. Confirm that
the selected row is present in the target image before deployment.

## Next steps

- [Configuration](configuration.md) lists user-facing and advanced variables.
- [Bring your own checkpoint](bring-your-own-checkpoint.md) covers local and
  Hugging Face model sources.
- [API reference](api-reference.md) routes requests to the right task guide.
- [Operations](operations.md) covers health, logs, metrics, and failures.

Stop the examples with:

```bash
docker stop cosmos3-generator
docker stop cosmos3-reasoner
```

The containers use `--rm`; the persistent model cache remains.
