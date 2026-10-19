<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Prerequisites for the Cosmos3 NIM

Prepare and verify a host before pulling or launching the Cosmos3 NIM;
per-configuration GPU, VRAM, and host RAM requirements are in the
[support matrix](support-matrix.md).

## Hardware requirements

| Requirement | Current requirement |
| --- | --- |
| CPU architecture | The release image provides `amd64` and `arm64` variants; use the variant selected by the container runtime |
| GPU compute capability | Generator: BF16 `>=8.0`, FP8 `>=8.9`; Reasoner: BF16 `>=8.0`, FP8 `>=8.9`, NVFP4 `>=10.0` |
| GPU count and per-device VRAM | See the [Generator](support-matrix.md#generator-configurations) and [Reasoner](support-matrix.md#reasoner-configurations) tables |
| Host RAM | Practical minimum per configuration in the [Generator](support-matrix.md#generator-configurations) and [Reasoner](support-matrix.md#reasoner-configurations) tables |
| Free disk | Provision for the image, selected model artifacts, materialization, and outputs; no single workload-independent floor is documented |
| Container shared memory | The Docker reference launch allocates 16 GiB; validate the requirement for the selected media workload and concurrency |

Do not add together memory from multiple GPUs to satisfy a per-device floor.
If the deployment must serve Transfer, provision each GPU against the
**Transfer minimum VRAM/device** column in the
[Generator configurations](support-matrix.md#generator-configurations), not the
**Generation minimum VRAM/device** value.

Use homogeneous GPUs; Reasoner requires the same compute capability on every
GPU. Each GPU must also have enough free memory at startup, so stop unrelated
GPU workloads before launch.

Lower-VRAM offload profiles keep model weights in system memory and carry
higher RAM floors; see the
[Generator configurations](support-matrix.md#generator-configurations).

### Inspect unified-memory capacity and current state

DGX Spark/GB10 and Jetson AGX Thor share one physical memory pool between CPU
and GPU. Summarize the host memory state with this read-only command:

```bash
python3 - <<'PY'
keys = ("MemTotal", "MemFree", "MemAvailable", "Cached", "SReclaimable", "Shmem")
values = {}
with open("/proc/meminfo", encoding="ascii") as meminfo:
    for line in meminfo:
        name, value, *_ = line.replace(":", "").split()
        if name in keys:
            values[name] = int(value) * 1024
for name in keys:
    print(f"{name}: {values.get(name, 0) / 1024**3:.2f} GiB")
reclaimable = max(
    values.get("Cached", 0)
    + values.get("SReclaimable", 0)
    - values.get("Shmem", 0),
    0,
)
print(f"Approximate reclaimable cache: {reclaimable / 1024**3:.2f} GiB")
PY
```

`MemFree` is immediately unused memory. `MemAvailable` already includes
reclaimable cache; do not add the cache value to it, and do not assume every
cached byte can be reclaimed.

A memory-related preflight failure has two materially different causes:

- A floor above the device-reported total after the unified-memory host reserve
  **exceeds total capacity**. Stopping processes or reclaiming cache cannot make
  that profile fit.
- A floor that fits the effective total but fails the current-memory check
  **fits the hardware but not the current memory state**. Inspect `MemFree`,
  `MemAvailable`, and the cache components, stop identified competing
  workloads, and rerun preflight.

Before the first Reasoner preflight on these systems, set
[`NIM_GPU_MEMORY_UTILIZATION`](deployment.md#set-the-reasoner-memory-share-on-unified-memory-systems)
and keep it through launch. Clearing page cache or deleting the model cache is
not part of normal deployment; see
[Unified-memory diagnostics](operations.md#unified-memory-diagnostics).

## Software requirements

| Software | Required version |
| --- | --- |
| Linux and `glibc` | Use a Linux host supported by the selected NVIDIA driver and container stack |
| NVIDIA driver | Recommended: NVIDIA driver 595.58.03 or newer. The image is built on CUDA 13.0 user-space libraries; confirm the host with the [verification steps](#verify-the-host) and a container start |
| Docker Engine | Use a maintained Docker Engine release supported by NVIDIA Container Toolkit |
| NVIDIA Container Toolkit | Use a maintained release configured for the host driver and Docker Engine; on discrete GPUs, expose the NVML utility capability used to measure free memory |

This release was validated with the following host configuration (tested
versions, not minimums):

| Component | Validated version |
| --- | --- |
| OS, architecture, and GPU | Ubuntu 24.04.5 LTS on `aarch64` with NVIDIA GH200 480GB (96 GB GPU memory) |
| NVIDIA driver | 595.58.03 |
| Docker Engine | 27.0.3 |
| NVIDIA Container Toolkit | 1.20.1 |

The image provides the CUDA user-space libraries; do not install a host CUDA
toolkit solely for the NIM. Do not narrow the image's
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`; Reasoner needs the `video`
capability for NVDEC decoding.

## Client tooling

Run the API and Python examples from the NIM host or another Linux system that
can reach the service:

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

Verify that every client tool meets the table:

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

This creates the project `.venv` from `pyproject.toml` and `uv.lock`; rerun it
after either file changes. `uv run python examples/...` reuses it.

## Network and NGC access

For a normal cold start, the host must reach:

- `nvcr.io` to pull the container; and
- the NGC model storage used to download and materialize profile artifacts.

You also need an
[NGC personal API key](https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html)
with NGC Catalog access. For an evaluation, create a dedicated key with only
the required services and a short expiration. Read it into the deployment shell
without placing it in shell history:

```bash
read -rsp "Enter your NGC API key: " NGC_API_KEY
export NGC_API_KEY
echo
```

The variable is `NGC_API_KEY`, not `NGC_TOKEN`. Do not save the key in source
control; continue with [Authenticate to NGC](deployment.md#authenticate-to-ngc).

An air-gapped host needs a pre-populated cache; see
[Offline and air-gapped deployment](deployment.md#offline-and-air-gapped-deployment).

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

The GPU query reports memory in MiB; divide by 1024 to compare with the
binary-GiB floors in the [support matrix](support-matrix.md). The Docker check
verifies GPU container access, not Cosmos3 profile compatibility.

For installation and verification failures, see
[Troubleshooting](operations.md#troubleshooting).
