# Cosmos3 Cookbooks: Environment Setup

Shared environment setup for every Cosmos3 cookbook (Reasoner and Generator).
Each cookbook README links back here for the backend(s) it supports — pick the
backend you want to run and follow that one section.

| Backend | Use it for | Used by |
| --- | --- | --- |
| [Cosmos Framework](#cosmos-framework) | Native PyTorch inference, launched with `torchrun` | Reasoner, Generator (Audiovisual, Action, **Transfer**) |
| [Diffusers](#diffusers) | Direct generation with `Cosmos3OmniPipeline` | Generator (Audiovisual) |
| [Transformers](#transformers) | Hugging Face Transformers inference | Reasoner |
| [vLLM](#vllm) | OpenAI-compatible reasoning server (image/video understanding) | Reasoner |
| [vLLM-Omni](#vllm-omni) | OpenAI-compatible generation server (image/video/audio/action/transfer) | Generator (Audiovisual, Action, **Transfer**) |
| [Reasoner NIM](#reasoner-nim) | Prebuilt OpenAI-compatible reasoning server (image/video understanding); no venv | Reasoner |
| [Generator NIM](#generator-nim) | Prebuilt NGC container serving the Cosmos3 Generator for Text-to-Video and Image-to-Video inference | Generator (Audiovisual) |

## Prerequisites

- Linux with NVIDIA GPU access.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/), `git`, and `git-lfs` installed.
- Hugging Face access to the gated Cosmos3 model repos. Generator also requires
  access to the gated
  [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail)
  HF repository. Authenticate once before the first run:

  ```bash
  uvx hf@latest auth login
  # or:
  export HF_TOKEN=<your_token>
  ```

  To disable the guardrail, set `enable_safety_checker=False` (Diffusers), `guardrails: false`
  (vLLM-Omni `extra_params`/`extra_args`), or
  `--no-guardrails` (Cosmos Framework). For Generator NIM set environment variables `NIM_ENABLE_TEXT_GUARDRAILS=0 NIM_ENABLE_VIDEO_GUARDRAILS=0`.
- NIMs don't need Hugging Face access; instead, an NGC API key is required
  (used as `NGC_API_KEY`). You can generate one on [build.nvidia.com](https://build.nvidia.com/) or [NGC](https://catalog.ngc.nvidia.com/), then run `docker login nvcr.io` once (username `$oauthtoken`, password = your key). This repository uses the Reasoner NIM image `nvcr.io/nim/nvidia/cosmos3-reasoner` and the Generator NIM image `nvcr.io/nim/nvidia/cosmos3-generator`.
- For the Cosmos Framework backend: access to `git@github.com:NVIDIA/cosmos-framework.git`.
- Enough local disk for the venv/image, the uv cache, and the model cache. Nano
  downloads plus CUDA dependencies can take tens of GiB.

### CUDA driver and the `cuXXX` backend

Several backends pin a CUDA build of `torch`/`vllm` that **must match your NVIDIA
driver**. Pick the tag that matches the CUDA version your driver supports:

| Driver CUDA | Backend tag | Notes |
| --- | --- | --- |
| 13.x | `cu130` | Default in the notebooks. |
| 12.x | `cu128` | Use when a compatible wheel is available for the selected package version. |

vLLM does not publish a wheel for every CUDA minor version, so
`--torch-backend=auto` is not reliable here — choose the pair that matches your
driver.

## Cosmos Framework

Native PyTorch inference through the Cosmos Framework checkout. Used by the
`run_*_with_cosmos_framework.ipynb` notebooks and the Cosmos Framework
quickstarts.

From the `cosmos` repo root, clone (or reuse) the framework checkout:

```bash
mkdir -p packages
git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos3
cd packages/cosmos3
```

Install the framework dependencies into its venv. The inference path currently
imports modules from the training extras, so use the `*-train` dependency group
that matches your driver (see [CUDA driver and the `cuXXX` backend](#cuda-driver-and-the-cuxxx-backend)):

```bash
# lerobot tracks test artifacts with git-LFS that this cookbook does not need;
# skipping smudge avoids failures from missing LFS blobs in uv's git mirror.
export GIT_LFS_SKIP_SMUDGE=1

# CUDA 13 driver (default):
uv sync --all-extras --group=cu130-train

# CUDA 12.x driver:
# uv sync --all-extras --group=cu128-train
```

The notebooks honor `COSMOS3_UV_GROUP` (default `cu130-train`); set
`export COSMOS3_UV_GROUP=cu128-train` before launching them on CUDA 12.x systems.

This produces a venv at `packages/cosmos3/.venv`. Run framework commands either
by activating it (`source .venv/bin/activate`) or via its absolute interpreter
(`.venv/bin/python`, `.venv/bin/torchrun`).

### Recommended base image (optional)

For CUDA 13, NVIDIA documents the [NGC PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
`nvcr.io/nvidia/pytorch:25.09-py3` as the recommended starting point; for CUDA 12 use
`nvcr.io/nvidia/pytorch:25.06-py3`. See the repo root
[Which base container should I use?](../../README.md#which-base-container-should-i-use)
and [Cosmos Framework setup](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/setup.md#recommended-base-image).

Inside that image (or any minimal GPU host), install the system packages below **before**
your first `torchrun` inference — `uv sync --all-extras` alone is not enough for
guardrails.

### System packages (required for Framework guardrails)

Framework inference enables **guardrails by default**. The video guardrail path imports
OpenCV (via RetinaFace), which needs graphics libraries that are often missing on
headless servers and minimal containers.

From `packages/cosmos3` (or the framework repo root), with `apt-get` available.
NGC and many training containers run as **root** — use `apt-get` directly (no `sudo`).
On a normal host where you are not root, prefix with `sudo`.

```bash
apt-get update
apt-get install -y --no-install-recommends \
  curl ffmpeg git-lfs libgl1 libglib2.0-0 libx11-dev libxcb1 tree wget
```

Verify OpenCV imports after `source .venv/bin/activate`:

```bash
python -c "import cv2; print(cv2.__version__)"
```

If you see `libxcb.so.1: cannot open shared object file`, the `libxcb1` / `libgl1`
packages above were not installed. The same fix is documented in the repo root
[troubleshooting guide](../../README.md#import-fails-with-libxcbso1-cannot-open-shared-object-file).

When using the **NGC PyTorch base image**, clear `LD_LIBRARY_PATH` after activating the
venv so the container’s bundled libtorch does not shadow the venv (see
[Cosmos Framework FAQ — PyTorch import inside NGC](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/faq.md)):

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH=
```

Guardrails also require Hugging Face access to the gated safety models (accept the
license and set `HF_TOKEN` as in [Prerequisites](#prerequisites)). To disable guardrails
for a one-off run, pass `--no-guardrails` to `cosmos_framework.scripts.inference`.

## Diffusers

Direct generation with `Cosmos3OmniPipeline` (Generator · Audiovisual). Create a
venv and install the backend, choosing `--torch-backend` to match your driver
(see [CUDA driver and the `cuXXX` backend](#cuda-driver-and-the-cuxxx-backend)):

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate

uv pip install --torch-backend=cu130 \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate \
  av \
  cosmos_guardrail \
  huggingface_hub \
  imageio \
  imageio-ffmpeg \
  torch \
  torchvision \
  transformers
```

## Transformers

Local Python inference for the Cosmos3 Reasoner. This backend uses the
Transformers Cosmos3 integration and loads only the Reasoner tower from the
unified Cosmos3 checkpoint.

Cosmos3 support first appears in the Transformers `v5.11.0` release tag. Create
a venv and install Transformers `5.11.0` or newer:

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate

uv pip install --torch-backend=auto \
  accelerate \
  av \
  pillow \
  "safetensors>=0.8.0" \
  torch \
  "torchvision==0.25.0" \
  "transformers>=5.11.0"
```

`--torch-backend=auto` lets uv detect the CUDA build of `torch`/`torchvision`
that matches your NVIDIA driver. Pin a backend such as `cu128` or `cu130` if
your environment needs an explicit CUDA wheel.

Use `Cosmos3OmniForConditionalGeneration` with `AutoProcessor` and either
`nvidia/Cosmos3-Nano` or `nvidia/Cosmos3-Super`. See the
[Reasoner Transformers quickstart](reasoner/README.md#run-with-transformers)
for a runnable image example and video input notes.

## vLLM

OpenAI-compatible **reasoning** server for the Reasoner cookbook (image/video
understanding). Native Cosmos3 Reasoner support first appears in the vLLM
`v0.23.0` stable release:

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate

# CUDA 13 driver:
uv pip install --torch-backend=cu130 "vllm>=0.23.0"

# CUDA 12.x driver:
# uv pip install --torch-backend=cu128 "vllm>=0.23.0"
```

The vLLM wheel and the torch backend must be compatible — see
[CUDA driver and the `cuXXX` backend](#cuda-driver-and-the-cuxxx-backend).

If your vLLM build reports that DeepGEMM is unavailable, disable it before
starting the server:

```bash
export VLLM_USE_DEEP_GEMM=0
```

> When launching with `.venv/bin/vllm` instead of activating the venv, make sure
> `.venv/bin` is on `PATH` (e.g. `source .venv/bin/activate`). FlashInfer's
> just-in-time kernel build shells out to `ninja`, which lives in the venv.

### Start the server

All Reasoner cookbooks talk to an OpenAI-compatible chat-completions API. After
[installing vLLM](#vllm), run the commands below from
`cookbooks/cosmos3/reasoner` (same working directory as
[`run_with_vllm.ipynb`](reasoner/run_with_vllm.ipynb)). That sets
`$(dirname "$(pwd)")` to `<cosmos>/cookbooks/cosmos3`, which matches the
notebook's `COSMOS3_MEDIA_ROOT`.

**Cosmos3-Nano** (single GPU, port 8000):

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve nvidia/Cosmos3-Nano \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path "$(dirname "$(pwd)")" \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --port 8000
```

**Cosmos3-Super** (four GPUs; default in [`run_with_vllm.ipynb`](reasoner/run_with_vllm.ipynb), port 8001):

```bash
export COSMOS3_MEDIA_ROOT="$(dirname "$(pwd)")"
export VLLM_PORT="${VLLM_PORT:-8001}"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve nvidia/Cosmos3-Super \
  --tensor-parallel-size 4 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path "$COSMOS3_MEDIA_ROOT" \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --port "$VLLM_PORT"
```

**Cosmos3-Edge** (single GPU, port 8000):

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve nvidia/Cosmos3-Edge \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path "$(dirname "$(pwd)")" \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --port 8000
```

The Super notebook polls `/health` for up to 1800 seconds on first start while CUDA
graphs compile.

| Option | Use |
| --- | --- |
| `--tensor-parallel-size` | Number of GPUs for tensor-parallel inference |
| `--mm-encoder-tp-mode data` | Data parallelism for the visual encoder |
| `--media-io-kwargs '{"video": {"num_frames": -1}}'` | Lets the processor see all frames before downstream sampling |
| `--allowed-local-media-path` | Must cover local `file://` media paths; defaults to `<cosmos>/cookbooks/cosmos3` when run from `cookbooks/cosmos3/reasoner` |

## vLLM-Omni

OpenAI-compatible **generation** server (image/video/audio/action/transfer) for the
Generator cookbooks.

Cosmos3 checkpoints can exceed the default server init timeout — always pass
`--init-timeout 1800` on every `vllm serve` command below.

### Guardrails (gated dependency)

The vLLM-Omni server loads gated
[nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail)
at startup by default. Without Hugging Face access to that repo, the server
exits before serving requests. Per-request `guardrails: false` in `extra_params`
(see [Prerequisites](#prerequisites)) does not fix this — the guardrail models
must load at startup.

To disable guardrails server-wide (you are responsible for
[license compliance](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license)),
add `--no-guardrails` to any `vllm serve` command below:

```bash
vllm serve nvidia/Cosmos3-Nano \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --no-guardrails \
  --allowed-local-media-path / \
  --port 8000 \
  --init-timeout 1800
```

Alternatively, pass a
[`--deploy-config`](../../README.md#generator-with-vllm-omni) as documented in
the repository root README. See also the
[vLLM-Omni Cosmos3-Nano recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-Nano.md).

### Option 1: Docker (recommended)

The released image `vllm/vllm-omni:cosmos3` supports the Generator modalities,
including transfer controls. Pull once:

```bash
docker pull vllm/vllm-omni:cosmos3
```

Set paths once; adjust for your checkout and cache location:

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export COSMOS3_WORKDIR="${COSMOS3_WORKDIR:-$(pwd)}"
export COSMOS3_HOST_PORT="${COSMOS3_HOST_PORT:-8000}"
```

The container listens on port 8000; `-p "${COSMOS3_HOST_PORT}:8000"` publishes it
on the host. Generator notebooks often use `COSMOS3_HOST_PORT=8001` so port 8000
stays free for a Reasoner server. The Docker commands run from `/workspace`, so
repo-local paths such as `cookbooks/...` resolve inside the container.

**Cosmos3-Nano** (single GPU):

```bash
docker run --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${COSMOS3_WORKDIR}:/workspace" \
  -p "${COSMOS3_HOST_PORT}:8000" --ipc=host \
  -w /workspace \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --port 8000 \
    --init-timeout 1800
```

**Cosmos3-Super** (all GPUs; add tensor parallelism and layerwise offload):

```bash
docker run --runtime nvidia --gpus all \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${COSMOS3_WORKDIR}:/workspace" \
  -p "${COSMOS3_HOST_PORT}:8000" --ipc=host \
  -w /workspace \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Super \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --tensor-parallel-size 4 \
    --enable-layerwise-offload \
    --port 8000 \
    --init-timeout 1800
```

**Cosmos3-Edge** (single GPU):

```bash
docker run --runtime nvidia --gpus all \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${COSMOS3_WORKDIR}:/workspace" \
  -p "${COSMOS3_HOST_PORT}:8000" --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Edge \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --port 8000 \
    --init-timeout 1800
```

**Cosmos3-Super-Text2Image-4Step** (all GPUs; add tensor parallelism and layerwise offload):

```bash
docker run --runtime nvidia --gpus all \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${COSMOS3_WORKDIR}:/workspace" \
  -p "${COSMOS3_HOST_PORT}:8000" --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Super-Text2Image-4Step \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --tensor-parallel-size 4 \
    --enable-layerwise-offload \
    --port 8000 \
    --init-timeout 1800
```

**Cosmos3-Super-Image2Video-4Step** (all GPUs; add tensor parallelism and layerwise offload):

```bash
docker run --runtime nvidia --gpus all \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${COSMOS3_WORKDIR}:/workspace" \
  -p "${COSMOS3_HOST_PORT}:8000" --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Super-Image2Video-4Step \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --tensor-parallel-size 4 \
    --enable-layerwise-offload \
    --port 8000 \
    --init-timeout 1800
```

Mount any directory that holds local media or action JSON files referenced in
requests. Set `--allowed-local-media-path /` (as above) when the whole container
filesystem should be readable.

vLLM-Omni prints `Application startup complete.` when the API is ready.

### Option 2: Native venv

To install from `main` instead of Docker, create a venv and pick the CUDA build
that matches your driver (see
[CUDA driver and the `cuXXX` backend](#cuda-driver-and-the-cuxxx-backend)):

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate

# CUDA 13 driver:
uv pip install --torch-backend=cu130 \
  "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@main"

# CUDA 12.x driver:
# uv pip install --torch-backend=cu128 \
#   "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@main"
```

Run the same `vllm serve` arguments as in the Docker commands above, directly on
the host (no `docker run` wrapper):

```bash
vllm serve nvidia/Cosmos3-Nano \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / \
  --port 8000 \
  --init-timeout 1800
```

For Super, add `--tensor-parallel-size 4 --enable-layerwise-offload`.

Additional parallelism options (Docker or native):

| Option | Use |
| --- | --- |
| `--cfg-parallel-size 2` | Runs positive and negative CFG branches on two GPUs |
| `--ulysses-degree 2` | Ulysses sequence parallelism across GPUs |

Ensure the server has enough GPUs for the product of enabled degrees
(`tensor_parallel_size` × `cfg_parallel_size` × `ulysses_degree`).

## NIM

Prebuilt NGC containers for Cosmos3. Like vLLM-Omni, NIM runs from Docker, so
there is no venv or `--torch-backend` to manage. Unlike the Hugging Face based
backends, NIM authenticates with an NGC API key instead of a Hugging Face token
(see [Prerequisites](#prerequisites)).

Authenticate Docker to NGC once:

```bash
export NGC_API_KEY=<your_key>
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

Both NIMs expose readiness at `GET /v1/health/ready` after model download,
engine initialization, and warmup complete.

### Reasoner NIM

A prebuilt container that serves the Reasoner over an OpenAI-compatible API for
image and video understanding.

Start a Nano Reasoner server (publishes the API on port 8000; the first run
downloads the model into `~/.cache/nim`):

```bash
export NGC_API_KEY=<your_key>

docker run --runtime=nvidia --gpus all \
  --shm-size=32GB \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -e NIM_MODEL_SIZE=nano \
  -v ~/.cache/nim:/opt/nim/.cache \
  -u $(id -u) \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0
```

For **Cosmos3-Super-Reasoner** (the larger model), set `-e NIM_MODEL_SIZE=super`.
The container serves `nvidia/cosmos3-nano-reasoner` (or
`nvidia/cosmos3-super-reasoner`); pass that exact name as the request `model`, or
resolve it dynamically with `client.models.list()`.

### Generator NIM

A prebuilt container that serves **Cosmos3-Generator Text-to-Video and Image-to-Video
only** through `POST /v1/infer`. The NIM infers mode from the request fields:
non-empty `prompt` with no `image` means Text-to-Video; `image` provided means Image-to-Video. The
response is JSON with a base64-encoded MP4 in `b64_video`.

It does **not** expose text-to-image, video-to-video, sound/audio generation,
action modes, or transfer controls. Use vLLM-Omni or Cosmos Framework for those
broader Generator workflows.

Start a Nano Generator server (default `NIM_MODEL_SIZE=nano`, `NIM_PRECISION=fp8`,
`NIM_PERF_PROFILE=latency`):

```bash
export NGC_API_KEY=<your_key>
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R 777 "$LOCAL_NIM_CACHE" 2>/dev/null || true

docker run --runtime=nvidia --gpus all \
  --shm-size=32GB \
  --ulimit nofile=65536:65536 \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/cosmos3-generator:1.0.0
```

For **Cosmos3-Super Generator**, add `-e NIM_MODEL_SIZE=super`. Other selection
knobs:

| Env var | Values | Default | Use |
| --- | --- | --- | --- |
| `NIM_MODEL_SIZE` | `nano`, `super` | `nano` | Selects 8B Nano or 32B Super |
| `NIM_PRECISION` | `bf16`, `fp8`, `nvfp4` | `fp8` | Selects precision; `nvfp4` requires Blackwell |
| `NIM_PERF_PROFILE` | `latency`, `throughput` | `latency` | Optimizes profile selection objective |
| `NIM_TAGS_SELECTOR` | comma-separated `key=value` filters | unset | Advanced profile pinning, e.g. `model_size=super,nim_tp=2` |

A quick T2V smoke test:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/infer \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A humanoid robot walks through a futuristic warehouse, inspecting shelves of mechanical components.",
    "seed": 42,
    "guidance_scale": 6.0,
    "steps": 35,
    "resolution": "256",
    "num_output_frames": 25,
    "fps": 24.0
  }' | jq -r '.b64_video' | base64 -d > /tmp/cosmos3_generator_nim_t2v.mp4
```

## Verify the environment

For the Cosmos Framework / Diffusers / vLLM venvs, check that PyTorch sees the GPU:

```bash
.venv/bin/python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

For a vLLM / vLLM-Omni / NIM server, confirm it is serving the model (use the host
port you set with `COSMOS3_HOST_PORT` or `VLLM_PORT`):

```bash
curl http://localhost:8000/v1/models
```

A NIM server also exposes a readiness endpoint that returns `200` once the model
is loaded:

```bash
curl http://localhost:8000/v1/health/ready
```
