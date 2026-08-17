<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Prerequisites for the Cosmos3 Certified NIM

Use this page to prepare and verify a host before pulling or launching the
Cosmos3 Certified NIM. Profile-specific GPU, precision, and VRAM compatibility
belongs to the [support matrix](support-matrix.md).

> **Published release floors:** GPU compute capability, GPU count, per-device
> VRAM, Transfer headroom, and configuration-specific system-memory floors are
> published in the [Support matrix](support-matrix.md). Full host compatibility
> cannot be established before cold start: general CPU architecture, host RAM,
> disk, shared-memory, driver, Docker, and Container Toolkit requirements are
> not yet available.

## Hardware requirements

Plan for:

- the CPU architecture supported by the selected image;
- homogeneous NVIDIA GPUs compatible with at least one available
  model/precision/configuration combination;
- enough total and currently free GPU memory on every participating device for
  the selected profile;
- enough host RAM and free disk for the container, downloaded artifacts,
  materialization, and temporary files; and
- enough shared memory for staged image/video buffers and multi-process work.

| Requirement | Current requirement |
| --- | --- |
| CPU architecture | Not yet available |
| GPU compute capability | Generator: BF16 `>=8.0`, FP8 `>=8.9`; Reasoner: BF16 `>=8.0`, FP8 `>=8.9`, NVFP4 `>=10.0` |
| GPU count and per-device VRAM | See the [Generator](support-matrix.md#generator-configurations) and [Reasoner](support-matrix.md#reasoner-configurations) tables |
| Host RAM | General minimum not yet available; profile-selection floors are 16 GiB for resident/Reasoner profiles, 64 GiB for Nano offload, and 150 GiB for Super offload |
| Free disk | Not yet available |
| Container shared memory | Not yet available |

Do not add together memory from multiple GPUs to satisfy a per-device floor.
If the deployment must serve Transfer, provision each GPU against the
**Transfer minimum VRAM/device** column in the
[Generator configurations](support-matrix.md#generator-configurations), not the
**Generation minimum VRAM/device** value.

Reasoner requires GPUs with the same compute capability. Use homogeneous GPUs
for either runtime because mixed-GPU support is not established. Static profile
compatibility evaluates the smallest per-device memory total and effective
system RAM exposed to the container. At startup, the selector also captures the
current free memory of each visible GPU once; existing allocations can force an
equivalent lower-memory layout or make selection fail. Confirm the selected
configuration in the exact image's manifest before deployment.

On an integrated GPU where device and host share one memory pool, the selector
withholds 16 GiB for the host by default before comparing the remaining shared
memory with profile floors. Generator selection uses resident model and
resident guardrail profiles on these systems; CPU-offload profiles do not
reduce shared-memory use.

Lower-VRAM profiles can keep model weights in system memory. Resident Generator
and Reasoner profiles carry a 16-GiB selection floor, Nano model/layer-offload
profiles carry 64 GiB, and Super model/layer-offload profiles carry 150 GiB. The
NIM checks a container memory limit before host physical memory, so a lower
Docker or Kubernetes limit can make an otherwise capable host incompatible.
These profile-selection floors do not resolve the release-wide host-RAM
requirement for the container, runtime, materialized artifacts, and workload.
They are sufficient to choose a candidate profile before model artifacts are
downloaded. After pulling the image, run the documented
[pre-download profile preflight](deployment.md#run-the-pre-download-profile-preflight)
before cold start.

## Software requirements

The host requires:

- a Linux distribution supported by NVIDIA Container Toolkit;
- a compatible `glibc`;
- an NVIDIA driver compatible with the selected image;
- Docker Engine; and
- NVIDIA Container Toolkit configured for Docker; discrete-GPU deployments
  must expose the NVML utility capability used to measure free memory without
  creating CUDA contexts.

| Software | Required version |
| --- | --- |
| Linux and `glibc` | Not yet available |
| NVIDIA driver | Not yet available |
| Docker Engine | Not yet available |
| NVIDIA Container Toolkit | Not yet available |

The CUDA user-space libraries required by the NIM are provided inside the
container. Follow the driver and Container Toolkit installation instructions
for the selected image instead of installing an unrelated host CUDA toolkit
solely for the NIM.

## Client tooling

Run the documented API and Python example commands from a client machine. The
client can be the NIM host or another Linux system that can reach the service.
These tools are separate from the software inside the NIM container:

| Tool | Minimum version | Used for |
| --- | --- | --- |
| `curl` | 7.61 | Health checks and direct HTTP API requests |
| `uv` | 0.11.0 | Creating the pinned Python environment and running examples |
| `python3` | 3.10 | Standard-library JSON formatting and inline decoding snippets |
| `ffmpeg` and `ffplay` | 6.1 | Optional playback and conversion of generated VP9-in-MP4 video |

On Ubuntu 24.04, install the distribution-provided client tools, then install
`uv` with its official standalone installer:

```bash
sudo apt-get update
sudo apt-get install -y curl python3 ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

The `uv` installer installs the current release. Verify that every client tool
meets the table before continuing:

```bash
curl --version | head -n 1
uv --version
python3 --version
ffmpeg -version | head -n 1
ffplay -version | head -n 1
```

For another supported client distribution, use its package manager for
`curl`, Python 3, and FFmpeg; use the same official installer for `uv`.

### Initialize the example environment

After checking out the repository, create the pinned client environment once:

```bash
cd cookbooks/cosmos3/nim
uv sync --locked
```

This command creates or updates the project `.venv` from `pyproject.toml` and
`uv.lock`. Run it again after either file changes. Subsequent `uv run python
examples/...` commands reuse that environment instead of installing ad hoc
request dependencies as part of an example invocation.

## Network and NGC access

For a normal cold start, the host must reach:

- `nvcr.io` to pull the container; and
- the NGC model storage used to download and materialize profile artifacts.

You also need an
[NGC personal API key](https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html)
with NGC Catalog access. For an evaluation, create a dedicated key with only
the required services and the shortest practical expiration. NGC personal keys
can be rotated, deactivated, or deleted from the NGC user interface. Read the
key into the shell that you will use for deployment without placing the secret
in shell history:

```bash
read -rsp "Enter your NGC API key: " NGC_API_KEY
export NGC_API_KEY
echo
```

The runtime variable is `NGC_API_KEY`, not `NGC_TOKEN`. Keep this shell open,
do not save the key in source control, and continue with
[Authenticate to NGC](deployment.md#authenticate-to-ngc) to log Docker in.

An air-gapped deployment requires the selected image and a correctly
pre-populated cache prepared through an approved workflow. Merely disabling
model download does not create the required artifacts.

## Verify the host

Check the operating system, CPU architecture, `glibc`, driver, Docker, and
Container Toolkit before pulling the NIM:

```bash
uname -m
ldd --version | head -n 1
nvidia-smi
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.free --format=csv
docker version
nvidia-ctk --version
docker info | sed -n '/Runtimes/,$p' | head
```

Then verify that Docker can expose the intended GPUs:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

The earlier detailed query reports compute capability and total and free memory
in MiB; divide MiB by 1024 when comparing with the binary-GiB profile floors.
Free memory can change after this check, so stop unrelated GPU workloads before
launch. The Docker command may pull the `ubuntu` image. It verifies GPU
container access and, on discrete systems, NVML access; it does not verify
Cosmos3 profile compatibility. Compare the reported
devices and memory with the [support matrix](support-matrix.md) before launch.

For installation and verification failures, see
[Troubleshooting](operations.md#troubleshooting).
