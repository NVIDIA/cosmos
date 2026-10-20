# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg git libgl1 libglib2.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/vlmevalkit
COPY requirements.txt setup.py setup.cfg README-vlmevalkit.md ./
# Evaluation is API-based; keep its PyTorch stack off the serving GPU.
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt einops accelerate 'setuptools<81' evaluate jupyterlab==4.4.10

COPY launchable/public-harness.patch /tmp/public-harness.patch
# Keep the public release and internal NVIDIA fork in separate Python environments.
RUN git init /opt/vantage-public \
    && git -C /opt/vantage-public remote add origin https://github.com/Clemson-Capstone/VANTAGE-Bench.git \
    && git -C /opt/vantage-public fetch --depth 1 origin ec1ebdebb955332a9d7fdd9e98f39627b2822ff5 \
    && git -C /opt/vantage-public checkout --detach FETCH_HEAD \
    && git -C /opt/vantage-public apply /tmp/public-harness.patch \
    && python -m venv /opt/vantage-public/.venv \
    && /opt/vantage-public/.venv/bin/pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && /opt/vantage-public/.venv/bin/pip install --no-cache-dir -r /opt/vantage-public/requirements.txt \
        'transformers==4.57.1' 'qwen-vl-utils==0.0.14' 'setuptools<81' evaluate gdown \
    && cd /opt/vantage-public \
    && .venv/bin/pip install --no-cache-dir --no-deps -e . \
    && .venv/bin/python -c 'import vlmeval' \
    && .venv/bin/pip freeze > /opt/public-evaluation-packages.txt
COPY . .
RUN pip install --no-cache-dir --no-deps -e . \
    && python -c 'import vlmeval' \
    && pip freeze > /opt/evaluation-packages.txt
ARG COSMOS_COMMIT=unknown
ENV COSMOS_COMMIT=${COSMOS_COMMIT} PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface
WORKDIR /opt/vlmevalkit/launchable
CMD ["python", "jupyter_config.py"]
