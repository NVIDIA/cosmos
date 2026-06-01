# Cosmos3 Cookbooks: Environment Setup

Shared environment setup for every Cosmos3 cookbook (Reasoner and Generator).
Each cookbook README links back here for the backend(s) it supports — pick the
backend you want to run and follow that one section.

| Backend | Use it for | Used by |
| --- | --- | --- |
| [Cosmos Framework](#cosmos-framework) | Native PyTorch inference, launched with `torchrun` | Reasoner, Generator (Audiovisual, Action) |
| [Diffusers](#diffusers) | Direct generation with `Cosmos3OmniPipeline` | Generator (Audiovisual) |
| [Transformers](#transformers-coming-soon) | Hugging Face Transformers inference | Reasoner |
| [vLLM](#vllm) | OpenAI-compatible reasoning server (image/video understanding) | Reasoner |
| [vLLM-Omni](#vllm-omni) | OpenAI-compatible generation server (image/video/audio/action) | Generator (Audiovisual, Action) |
| [NIM](#nim) | Prebuilt OpenAI-compatible reasoning server (image/video understanding); no venv | Reasoner |

## Prerequisites

- Linux with NVIDIA GPU access.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/), `git`, and `git-lfs` installed.
- Hugging Face access to the gated Cosmos3 model repos. Authenticate once before the first run:

  ```bash
  uvx hf@latest auth login
  # or:
  export HF_TOKEN=<your_token>
  ```

- For the Cosmos Framework and vLLM backends: access to `git@github.com:NVIDIA/cosmos-framework.git`.
- For the NIM backend: an NGC API key (used as `NGC_API_KEY`), which you can generate on [build.nvidia.com](https://build.nvidia.com/nvidia/cosmos3-nano-reasoner) or [NGC](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/cosmos3-reasoner), plus a one-time `docker login nvcr.io` (username `$oauthtoken`, password = your key). The HF login above is not needed for NIM.
- Enough local disk for the venv/image, the uv cache, and the model cache. Nano
  downloads plus CUDA dependencies can take tens of GiB.

### CUDA driver and the `cuXXX` backend

Several backends pin a CUDA build of `torch`/`vllm` that **must match your NVIDIA
driver**. Pick the tag that matches the CUDA version your driver supports:

| Driver CUDA | Backend tag | Notes |
| --- | --- | --- |
| 13.x | `cu130` | Default in the notebooks (e.g. `vllm==0.21.0`). |
| 12.x | `cu128` | Use the `cu128` pair instead (e.g. `vllm==0.19.1`). |

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

## Transformers (coming soon)

Support for Transformers-based Reasoner inference is coming soon.

## vLLM

OpenAI-compatible **reasoning** server for the Reasoner cookbook (image/video
understanding). Create a venv and install vLLM plus the `vllm-cosmos3` plugin,
which registers the `Cosmos3ReasonerForConditionalGeneration` architecture:

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate

# CUDA 13 driver:
uv pip install --torch-backend=cu130 "vllm==0.21.0" \
  "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3"

# CUDA 12.x driver:
# uv pip install --torch-backend=cu128 "vllm==0.19.1" \
#   "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3"
```

The vLLM version and the torch backend are paired — see
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
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
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
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
  --tensor-parallel-size 4 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path "$COSMOS3_MEDIA_ROOT" \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --port "$VLLM_PORT"
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

OpenAI-compatible **generation** server (image/video/audio/action) for the
Generator cookbooks.

Cosmos3 checkpoints can exceed the default server init timeout — always pass
`--init-timeout 1800` on every `vllm serve` command below.

### Option 1: Docker (recommended)

The prebuilt image `vllm/vllm-omni:cosmos3` supports every Generator modality
(including action). Pull once:

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
stays free for a Reasoner server.

**Cosmos3-Nano** (single GPU):

```bash
docker run --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${COSMOS3_WORKDIR}:/workspace" \
  -p "${COSMOS3_HOST_PORT}:8000" --ipc=host \
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

Mount any directory that holds local media or action JSON files referenced in
requests. Set `--allowed-local-media-path /` (as above) when the whole container
filesystem should be readable.

vLLM-Omni prints `Application startup complete.` when the API is ready.

### Option 2: Native venv (limited modalities)

To install from the upstreaming PR branch instead of Docker (text-to-image,
text-to-video, and image-to-video only — not action or sound yet), create a venv
and pick the CUDA build that matches your driver (see
[CUDA driver and the `cuXXX` backend](#cuda-driver-and-the-cuxxx-backend)):

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate

# CUDA 13 driver:
uv pip install --torch-backend=cu130 \
  "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@refs/pull/3454/head"

# CUDA 12.x driver:
# uv pip install --torch-backend=cu128 \
#   "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@refs/pull/3454/head"
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

A prebuilt container that serves the Reasoner over an OpenAI-compatible API for
image and video understanding. Like vLLM-Omni this is a Docker image, so there is
no venv or `--torch-backend` to manage; unlike the other backends it
authenticates with an NGC API key instead of Hugging Face (see
[Prerequisites](#prerequisites)).

Start a Nano server (publishes the OpenAI-compatible API on port 8000; the first
run downloads the model into `~/.cache/nim`):

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
