<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Prerequisites for the Cosmos3 Certified NIM

Use this page to prepare and verify a host before pulling or launching the
Cosmos3 Certified NIM. Profile-specific GPU, precision, and VRAM compatibility
belongs to the [support matrix](support-matrix.md).

> **Release status:** Exact CPU architecture, RAM, disk, shared-memory, driver,
> container-toolkit, GPU, and VRAM requirements are **TBD
> (release-dependent)**. Historical requirements from separate Cosmos NIMs are
> not substitutes for the unified image's released support matrix.

## Hardware requirements

Plan for:

- the CPU architecture supported by the released image;
- homogeneous NVIDIA GPUs compatible with at least one released
  model/precision/profile combination;
- enough GPU memory on every participating device for the selected profile;
- enough host RAM and free disk for the container, downloaded artifacts,
  materialization, and temporary files; and
- enough shared memory for staged image/video buffers and multi-process work.

| Requirement | Released value |
| --- | --- |
| CPU architecture | **TBD** |
| Supported GPU architectures and compute capability | **TBD; see [support matrix](support-matrix.md)** |
| Host RAM | **TBD** |
| Free disk | **TBD** |
| Container shared memory | **TBD** |

Do not add together the memory of heterogeneous GPUs to claim compatibility.
Use a released profile whose GPU count, per-device VRAM, precision, and
architecture requirements all match the visible devices.

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
[Create and protect an NGC API key](deployment.md#create-and-protect-an-ngc-api-key)
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
docker version
nvidia-ctk --version
docker info | sed -n '/Runtimes/,$p' | head
```

Then verify that Docker can expose the intended GPUs:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

This command may pull the `ubuntu` image. It verifies GPU container access, not
Cosmos3 profile compatibility. Compare the reported devices and memory with
the released [support matrix](support-matrix.md) before launch.

For installation and verification failures, see
[Troubleshooting](operations.md#troubleshooting).
