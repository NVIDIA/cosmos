<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM

Deploy and use the Cosmos3 Certified NIM for world generation and multimodal
reasoning. The image contains profiles for both runtime families, but one
selected profile starts one backend/API at a time:

- **Generator** — image and video generation, action, and transfer through
  `POST /v1/infer`.
- **Reasoner** — OpenAI-compatible image/video understanding through Chat
  Completions and, when enabled, the Responses API.

> **Release status:** The final NGC image repository/tag, model-card and release
> URLs, exact supported hardware/profile matrix, Helm chart, and released-image
> acknowledgements are **TBD**. The guides intentionally preserve those as
> release-dependent placeholders while documenting the current source-backed
> API and deployment structure.

## Choose a task

| Task | Runtime | Input | Output | Guide |
| --- | --- | --- | --- | --- |
| Text-to-image | Generator | Prompt | JPEG image | [Generation](generation.md#text-to-image) |
| Text-to-video | Generator | Prompt | MP4 video | [Generation](generation.md#text-to-video) |
| Image-to-video | Generator | Prompt + image | MP4 video | [Generation](generation.md#image-to-video) |
| Video-to-video | Generator | Prompt + video | MP4 video | [Generation](generation.md#video-to-video) |
| Forward dynamics | Generator | Image + action trajectory | Rollout video | [Action](action.md#forward-dynamics) |
| Policy | Generator | Image + task/state | Video + predicted action | [Action](action.md#policy) |
| Nano-DROID policy | Generator specialist | Image + task/current state | Predicted action | [Action](action.md#nano-droid-policy) |
| Inverse dynamics | Generator | Video | Video + predicted action | [Action](action.md#inverse-dynamics) |
| Video transfer | Generator | Prompt + spatial control | Controlled MP4 video | [Transfer](transfer.md) |
| Image/video reasoning | Reasoner | Messages with media + text | Text or structured result | [Reasoning](reasoning.md) |
| Streaming reasoning | Reasoner | Chat Completions request | Text deltas | [Reasoning](reasoning.md#stream-chat-completions) |
| Responses API | Reasoner | Responses input | Response object/text | [Reasoning](reasoning.md#use-the-responses-api) |

The current Certified NIM source does not expose image-to-image or sound
generation through its public request model. Other Cosmos3 backends may have
broader modality support; do not copy their endpoints or fields into NIM
requests without translation.

## Documentation

| Page | Use it for |
| --- | --- |
| [Release notes](release-notes.md) | Released versions, compatibility changes, limitations, and upgrade guidance |
| [Prerequisites](prerequisites.md) | Host hardware, software, storage, shared memory, NGC access, and setup verification |
| [Deployment](deployment.md) | NGC authentication, Docker, cache, runtime/profile selection, readiness, and shutdown |
| [Configuration](configuration.md) | Shared, Generator, Reasoner, selection, BYOC, guardrail-memory, and prompt-upsampling variables |
| [Support matrix](support-matrix.md) | Released model variant, precision, GPU, VRAM, profile, offload, Transfer, and codec compatibility |
| [Deploy with Helm](helm.md) | Kubernetes secrets, values, GPUs, storage, probes, rollout, and verification |
| [Bring your own checkpoint](bring-your-own-checkpoint.md) | Generator and Reasoner checkpoint sources, mounts, selectors, validation, and verification |
| [API reference](api-reference.md) | Runtime routing, common Generator fields and response, task-contract links, and live schema |
| [Generation](generation.md) | T2I, T2V, I2V, V2V, specialist four-step variants, output decoding, and prompt upsampling |
| [Reasoning](reasoning.md) | Chat Completions, streaming, Responses, media, sampling, and prompting |
| [Action](action.md) | Forward dynamics, policy, inverse dynamics, Nano-DROID, domains, and action shapes |
| [Transfer](transfer.md) | Edge, blur, depth, segmentation, WSM, and transfer tuning |
| [Operations](operations.md) | Health, inspection, logs, metrics, guardrails, and troubleshooting |
| [Acknowledgements](acknowledgements.md) | Third-party notices for the exact released image; currently TBD |
| [Examples](examples/) | Complete editable Python requests and artifact handling |

## Quickstart

Review [Prerequisites](prerequisites.md) and choose a released configuration
from the [Support matrix](support-matrix.md) before launching the image.

### Authenticate to NGC

Create an NGC personal API key with NGC Catalog access, then:

```bash
export NGC_API_KEY='<your-ngc-api-key>'
echo "$NGC_API_KEY" \
  | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

The runtime variable is `NGC_API_KEY`, not `NGC_TOKEN`. Never commit, print, or
save the real key in requests or notebooks.

### Launch the Generator

Replace the image placeholder with the final released repository and tag:

```bash
export NIM_IMAGE='<NIM_IMAGE:TBD>'
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim/cosmos3}"
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R a+rwX "$LOCAL_NIM_CACHE"

docker run --rm --name cosmos3-generator \
  --gpus '"device=0"' \
  --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=generator \
  -e NIM_MODEL_SIZE=nano \
  -e NIM_PERF_PROFILE=latency \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

Exact requirements and profile availability are release-dependent. Do not use
the example selector values as a support statement.

Wait for readiness:

```bash
until curl -fsS http://localhost:8000/v1/health/ready >/dev/null; do
  sleep 10
done
```

Generate and decode a short deterministic video:

```bash
curl -fsS -X POST http://localhost:8000/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A storm trooper vacuuming the beach.",
    "resolution": "480_16_9",
    "num_output_frames": 49,
    "fps": 24.0,
    "seed": 0
  }' \
  -o /tmp/cosmos3-response.json

python - <<'PY'
import base64
import json
from pathlib import Path

result = json.loads(Path("/tmp/cosmos3-response.json").read_text())
Path("cosmos3-output.mp4").write_bytes(base64.b64decode(result["b64_video"]))
PY
```

### Launch the Reasoner

Stop the Generator or use another GPU/host port. The Reasoner does not accept a
Generator performance-profile selector:

```bash
docker run --rm --name cosmos3-reasoner \
  --gpus '"device=0"' \
  --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=reasoner \
  -e NIM_MODEL_SIZE=nano \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

Discover the model and run the image example:

```bash
python -m pip install openai
curl -sS http://localhost:8000/v1/models | python -m json.tool
python cookbooks/cosmos3/nim/examples/reasoner.py --case image
```

## Profiles and low-VRAM operation

Profile selection considers runtime, model size, precision, GPU count/hardware,
Generator latency versus throughput, checkpoint variant, and optional model or
guardrail offload. General-purpose `nano` and `super` Generator variants accept
the ordinary capability set; specialist variants can restrict requests to T2I,
I2V, or Nano-DROID policy. Select a specialist with `NIM_MODEL_VARIANT` only
when it appears in the released manifest.

The current generated profile grid is provisional, so this cookbook does not
publish it as a final support table. Use the released image's manifest and model
card for exact availability and inspect `/v1/metadata` after startup. Generator
metadata reports the selected `model_variant`. See
[Select a profile](deployment.md#select-a-profile) for the stable selection
concepts.

## Run the Python examples

Generator examples use `requests`; Reasoner examples use the OpenAI Python
client. Install both once:

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
python -m pip install requests openai

python cookbooks/cosmos3/nim/examples/t2i.py
# With NIM_MODEL_VARIANT=super-t2i-4step:
python cookbooks/cosmos3/nim/examples/t2i_4step.py
python cookbooks/cosmos3/nim/examples/t2v.py
python cookbooks/cosmos3/nim/examples/i2v.py
# With NIM_MODEL_VARIANT=super-i2v-4step:
python cookbooks/cosmos3/nim/examples/i2v_4step.py
python cookbooks/cosmos3/nim/examples/v2v.py
python cookbooks/cosmos3/nim/examples/action.py --case forward_dynamics
python cookbooks/cosmos3/nim/examples/transfer.py --case precomputed_edge

python cookbooks/cosmos3/nim/examples/reasoner.py --case image
python cookbooks/cosmos3/nim/examples/reasoner_stream.py
python cookbooks/cosmos3/nim/examples/reasoner_responses.py
```

Generator examples write a decoded JPEG or MP4—and predicted action JSON when
applicable—under `cookbooks/cosmos3/nim/examples/outputs/`. A specialist policy
can return action JSON without visual media. Reasoner examples print text
directly. The output directory is ignored by the repository.

## Safety, license, and notices

Generator guardrails are enabled by default in the current source. Disabling
them can remove content-policy and face-privacy protections; see
[Guardrails](operations.md#guardrails).

This cookbook is licensed under the repository
[LICENSE](../../../LICENSE). The running NIM exposes bundled product license
information at `/v1/license`; use the final NGC model card for authoritative
product terms and intended-use information.

Third-party notices for the exact released image are tracked in
[acknowledgements.md](acknowledgements.md) and remain TBD until the approved
release/build inventory is available.
