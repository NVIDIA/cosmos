#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${HF_TOKEN:?Supply HF_TOKEN as a secret Brev launch parameter}"
command -v docker >/dev/null
command -v python3 >/dev/null
docker compose version
docker info >/dev/null
export COSMOS_COMMIT
COSMOS_COMMIT=$(git rev-parse HEAD)
export LAUNCHABLE_STATE_DIR=${LAUNCHABLE_STATE_DIR:-"$PWD/state"}
mkdir -p "$LAUNCHABLE_STATE_DIR"/{cache,data,results,credentials/aws}
mkdir -p "$LAUNCHABLE_STATE_DIR/data/"{public,internal}/LMUData
# Persist only the generated notebook credential, never HF/model/judge credentials.
if [[ ! -f .env ]]; then
    python3 -c 'import secrets; print("JUPYTER_TOKEN=" + secrets.token_hex(32))' > .env
fi
chmod 600 .env
if [[ -z "${COSMOS_API_BASE:-}" ]]; then
    nvidia-smi
    docker compose --profile local pull cosmos-endpoint
    docker compose --profile local up -d --build
else
    # BYOM does not start or allocate a local model/GPU.
    docker compose up -d --build vantage-eval
fi
printf '%s\n' 'Jupyter is starting on 127.0.0.1:8888. Open its Brev secure link.' \
    'Retrieve the notebook login token with: docker compose exec vantage-eval jupyter server list' \
    'Run the notebook cells to prepare data, wait for the endpoint, and evaluate.'
