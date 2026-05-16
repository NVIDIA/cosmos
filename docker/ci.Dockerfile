# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This Dockerfile is used in the cosmos3.yml pipeline.
#
# Build:
#   docker build --build-arg CUDA_VERSION=12.8.1 -f packages/cosmos3/docker/ci.Dockerfile -t nvcr.io/nvidian/cosmos3-ci:v0.1.0 .
#   docker build --build-arg CUDA_VERSION=13.0.2 -f packages/cosmos3/docker/ci.Dockerfile -t nvcr.io/nvidian/cosmos3-ci:v0.1.0_arm .
#
# **Latest container tags:**
# * x86_64: nvcr.io/nvidian/cosmos3-ci:v0.1.0
# * arm64:  nvcr.io/nvidian/cosmos3-ci:v0.1.0_arm

ARG CUDA_VERSION=13.0.2
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04

# Keep in sync with .cosmos3-variables in ci_cd/cosmos3.yml
ARG UV_VERSION=0.10.8
ARG JUST_VERSION=1.46.0
ARG PYTHON_VERSION=3.13
ARG PRE_COMMIT_VERSION=4.5.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install packages
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libx11-dev \
    tree \
    wget

# Install uv: https://docs.astral.sh/uv/getting-started/installation/
# https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Install just: https://just.systems/man/en/pre-built-binaries.html
RUN command -v just || curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin --tag ${JUST_VERSION}

RUN uv python install ${PYTHON_VERSION} --default

RUN uv tool install "pre-commit>=${PRE_COMMIT_VERSION}"
