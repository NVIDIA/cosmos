# Setup Guide

> **Skill:** `.agents/skills/cosmos3-setup/SKILL.md`

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Virtual Environment](#virtual-environment)
    - [CUDA Variants](#cuda-variants)
  - [Docker Container](#docker-container)
- [Downloading Checkpoints](#downloading-checkpoints)
- [Troubleshooting](#troubleshooting)
  - [PyTorch Import Issue](#pytorch-import-issue)
  - [Dependency Issue](#dependency-issue)
  - [Python Issue](#python-issue)
  - [CUDA Issue](#cuda-issue)

______________________________________________________________________

<!--TOC-->

## System Requirements

- NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
- NVIDIA driver compatible with [CUDA version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [NVIDIA CUDA >=12.8](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu)
- Linux x86-64/aarch64
- glibc >=2.35 (e.g Ubuntu >=22.04)
- Python >=3.10

## Installation

If you encounter issues, see [Troubleshooting](#troubleshooting).

Clone the repository:

```bash
git clone git@github.com:nvidia-cosmos/cosmos3.git
cd cosmos3
```

Install one of the following environments:

<details open><summary><b>Virtual Environment</b></summary>

### Virtual Environment

Install system dependencies:

```shell
sudo apt-get install -y --no-install-recommends curl ffmpeg libx11-dev tree wget
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Install the package using one of the following methods:

<details open><summary><b>UV Sync: fully reproducible environment</b></summary>

```shell
uv sync --all-extras --group=cu130-train
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

</details>

<details><summary><b>UV Pip: virtual environment</b></summary>

```shell
# Create virtual environment (skip if using an existing environment)
uv venv --clear && source .venv/bin/activate && export LD_LIBRARY_PATH=

uv pip install -r pyproject.toml --all-extras --group=cu130-train
uv pip install -e .
```

</details>

<details><summary><b>UV Pip: system environment</b></summary>

```shell
uv pip install --system --break-system-packages -r pyproject.toml --all-extras --group=cu130-train
```

</details>

<details><summary><b>Advanced: custom torch/cuda versions</b></summary>

```shell
cuda_name=cu130
torch_name=torch210

# 1. Create and activate the virtual environment
uv venv --clear && source .venv/bin/activate

# 2. Install the desired torch/cuda versions
uv pip install "torch==2.10.0" "torchvision" --torch-backend=$cuda_name

# 3. Install the package with desired extras
uv pip install -r pyproject.toml --all-extras

# 4. Install one of the following attention backends:
# * Blackwell
uv pip install "natten==0.21.6.dev6+$cuda_name.$torch_name" -f https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0/natten
# * Hopper
uv pip install "flash-attn-3-nv==1.0.3+$cuda_name.$torch_name" -f https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0/flash-attn-3-nv
# * Ada/Ampere
uv pip install "flash-attn==2.7.4.post1+$cuda_name.$torch_name" -f https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0/flash-attn
```

If there is no attention backend wheel for your torch/cuda versions, you can build one using [cosmos-dependencies](https://github.com/nvidia-cosmos/cosmos-dependencies).

</details>

Optional package extras:

- `guardrail`: Guardrails
- `serve`: Online inference (ray, gradio).
- `train`: Training

#### CUDA Variants

| CUDA Version                | Inference       | Training              | Notes                                                                                                                                    |
| --------------------------- | --------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **CUDA 13.0 (recommended)** | `--group=cu130` | `--group=cu130-train` | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |
| CUDA 12.8                   | `--group=cu128` | `--group=cu128-train` | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |

</details>

<details><summary><b>Docker Container</b></summary>

### Docker Container

Please make sure you have access to Docker on your machine and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

Build the containers:

```bash
image_tag=$(docker build -f Dockerfile -q .)
```

Run the container:

```bash
docker run -it --runtime=nvidia --ipc=host --rm -v .:/workspace -v /workspace/.venv -v /root/.cache:/root/.cache -e HF_TOKEN="$HF_TOKEN" $image_tag
```

Optional arguments:

- `--ipc=host`: Use host system's shared memory, since parallel torchrun consumes a large amount of shared memory. If not allowed by security policy, increase `--shm-size` ([documentation](https://docs.docker.com/engine/containers/run/#runtime-constraints-on-resources)).
- `-v /root/.cache:/root/.cache`: Mount host cache to avoid re-downloading cache entries.
- `-e HF_TOKEN="$HF_TOKEN"`: Set Hugging Face token to avoid re-authenticating.

If you get `docker: Error response from daemon: unknown or invalid runtime name: nvidia`, you need to [configure docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker):

```shell
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

</details>

## Downloading Checkpoints

1. Get a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with `Read` permission
2. Login to [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli): `uvx hf auth login`
3. Accept the [NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/Cosmos-Guardrail1).
4. Test: `uvx hf@latest download --repo-type model nvidia/Cosmos-Guardrail1 --revision d6d4bfa899a71454a700907664f3e88f503950cf --include "README.md"`

If you encounter issues:

1. Check that you don't have conflicting [environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables) (e.g. `HF_TOKEN`): `printenv | grep HF_`
2. Check that your [token](https://huggingface.co/settings/tokens) has sufficient permissions.

Checkpoints are automatically downloaded during inference and post-training. To modify the checkpoint cache location, set the [`HF_HOME`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) environment variable.

## Troubleshooting

### PyTorch Import Issue

Errors:

- `ImportError: cannot import name '_functionalization' from 'torch._C'`

Clear the library path in your current shell:

```shell
export LD_LIBRARY_PATH=
```

This applies to the current session only. To persist, add the line to your `Dockerfile` or `~/.bashrc`.

If this doesn't fix the issue, try [reinstalling venv](#dependency-issue).

### Dependency Issue

Errors:

- `ModuleNotFoundError: No module named <module_name>`

Reinstall venv:

```shell
uv sync --all-extras --group=cu130-train --reinstall
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

If this doesn't fix the issue, try [reinstalling uv](#python-issue).

### Python Issue

Errors:

- `fatal error: Python.h: No such file or directory`

Reinstall uv and venv:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install --reinstall
rm -rf .venv
uv sync --all-extras --group=cu130-train --reinstall
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

### CUDA Issue

- `OSError: <lib_name>: cannot open shared object file: No such file or directory`

Ensure you have [CUDA installed](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu). The major version must match between the system and virtual environment cuda versions.

```shell
sudo apt-get install -y --no-install-recommends cuda-toolkit-<cuda_major_version>
```

Alternatively, use the [Docker container](#docker-container).
