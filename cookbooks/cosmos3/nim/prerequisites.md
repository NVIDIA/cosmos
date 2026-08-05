<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Prerequisites for the Cosmos3 Certified NIM

Use this page to prepare and verify a host before pulling or launching the
Cosmos3 Certified NIM. Profile-specific GPU, precision, and VRAM compatibility
belongs to the [support matrix](support-matrix.md).

> **Release status:** Semi-final GPU compute, count, VRAM, and profile-specific
> system-memory requirements are available in the current source and summarized
> in the [support matrix](support-matrix.md). Exact CPU architecture, general
> host RAM, disk, shared-memory, driver, and container-toolkit requirements
> remain **TBD (release-dependent)**. Historical requirements from separate
> Cosmos NIMs are not substitutes for the unified image's release requirements.

## Hardware requirements

Plan for:

- the CPU architecture supported by the released image;
- homogeneous NVIDIA GPUs compatible with at least one released
  model/precision/profile combination;
- enough GPU memory on every participating device for the selected profile;
- enough host RAM and free disk for the container, downloaded artifacts,
  materialization, and temporary files; and
- enough shared memory for staged image/video buffers and multi-process work.

| Requirement | Current requirement |
| --- | --- |
| CPU architecture | **TBD (release-dependent)** |
| GPU compute capability | Generator: BF16 `>=8.7`, FP8 `>=8.9`; Reasoner: BF16 `>=8.0`, FP8 `>=8.9`, NVFP4 `>=10.0` |
| GPU count and per-device VRAM | See the semi-final [Generator](support-matrix.md#semi-final-generator-profiles) and [Reasoner](support-matrix.md#semi-final-reasoner-profiles) tables |
| Host RAM | General minimum **TBD**; Super-family BF16 model/layer offload requires 150 GiB of effective system memory |
| Free disk | **TBD (release-dependent)** |
| Container shared memory | **TBD (release-dependent)** |

Do not add together memory from multiple GPUs to satisfy a per-device floor.
Reasoner requires GPUs with the same compute capability; use homogeneous GPUs
for either runtime because mixed-GPU support is not established. The profile
selector evaluates the smallest per-device memory total exposed to the
container.

On an integrated GPU where device and host share one memory pool, the selector
withholds 16 GiB for the host by default before comparing the remaining shared
memory with profile floors. Generator selection uses resident model and
resident guardrail profiles on these systems; CPU-offload profiles do not
reduce shared-memory use.

Lower-VRAM profiles can keep model weights in system memory. Every current
Super-family BF16 model- and layer-offload row requires 150 GiB of effective
system memory. The NIM checks a container memory limit before host physical
memory, so a lower Docker or Kubernetes limit can make an otherwise capable
host incompatible. A profile without an explicit RAM floor still requires
memory for the container, runtime, materialized artifacts, and offloaded
weights.

## Software requirements

The host requires:

- a Linux distribution supported by NVIDIA Container Toolkit;
- a compatible `glibc`;
- an NVIDIA driver supported by the released image;
- Docker Engine; and
- NVIDIA Container Toolkit configured for Docker.

| Software | Minimum released version |
| --- | --- |
| Linux and `glibc` | **TBD** |
| NVIDIA driver | **TBD** |
| Docker Engine | **TBD** |
| NVIDIA Container Toolkit | **TBD** |

The CUDA user-space libraries required by the NIM are provided inside the
container. Follow the driver and Container Toolkit installation instructions
for the released image instead of installing an unrelated host CUDA toolkit
solely for the NIM.

## Network and NGC access

For a normal cold start, the host must reach:

- `nvcr.io` to pull the container; and
- the NGC model storage used to download and materialize profile artifacts.

You also need an NGC personal API key with NGC Catalog access. The runtime
variable is `NGC_API_KEY`, not `NGC_TOKEN`. See
[Authenticate to NGC](deployment.md#authenticate-to-ngc)
for export and Docker-login instructions.

An air-gapped deployment requires the released image and a correctly
pre-populated cache prepared through an approved workflow. Merely disabling
model download does not create the required artifacts.

## Verify the host

Check the operating system, CPU architecture, `glibc`, driver, Docker, and
Container Toolkit before pulling the NIM:

```bash
uname -m
ldd --version | head -n 1
nvidia-smi
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
docker version
nvidia-ctk --version
docker info | sed -n '/Runtimes/,$p' | head
```

Then verify that Docker can expose the intended GPUs:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

The earlier detailed query reports compute capability and memory in MiB; divide
MiB by 1024 when comparing with the binary-GiB profile floors. The Docker
command may pull the `ubuntu` image. It verifies GPU container access, not
Cosmos3 profile compatibility. Compare the reported devices and memory with the
[support matrix](support-matrix.md) before launch, then confirm that the target
image actually contains the selected row.

For installation and verification failures, see
[Troubleshooting](operations.md#troubleshooting).
