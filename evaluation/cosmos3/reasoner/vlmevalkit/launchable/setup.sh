#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command -v docker >/dev/null
command -v python3 >/dev/null
docker compose version
docker info >/dev/null
export COSMOS_COMMIT
COSMOS_COMMIT=$(git rev-parse HEAD)
export LAUNCHABLE_STATE_DIR=${LAUNCHABLE_STATE_DIR:-"$PWD/state"}
mkdir -p "$LAUNCHABLE_STATE_DIR"/{cache,data,results,credentials/aws}
mkdir -p "$LAUNCHABLE_STATE_DIR/data/"{public,internal}/LMUData
# Persist notebook authentication and its non-secret bind address, never model/judge credentials.
export JUPYTER_BIND_ADDRESS
JUPYTER_BIND_ADDRESS=$(python3 - "$@" <<'PY'
import ipaddress
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

if sys.argv[1:] not in ([], ["--brev"]):
    raise SystemExit("Usage: bash setup.sh [--brev]")
path = Path(".env")
lines = path.read_text().splitlines() if path.exists() else [
    "JUPYTER_TOKEN=" + secrets.token_hex(32)
]
stored = dict(line.split("=", 1) for line in lines if line.startswith("JUPYTER_BIND_ADDRESS="))
address = os.environ.get("JUPYTER_BIND_ADDRESS") or stored.get("JUPYTER_BIND_ADDRESS", "127.0.0.1")
if sys.argv[1:] == ["--brev"] and not os.environ.get("JUPYTER_BIND_ADDRESS"):
    try:
        interfaces = json.loads(subprocess.check_output(
            ["ip", "-j", "-4", "addr", "show", "dev", "wt0"], timeout=5))
        addresses = [item["local"] for interface in interfaces for item in interface["addr_info"]
                     if item["family"] == "inet" and item.get("scope") == "global"]
        if len(addresses) != 1:
            raise ValueError("expected exactly one global wt0 IPv4 address")
        address = addresses[0]
        if ipaddress.IPv4Address(address) not in ipaddress.IPv4Network("100.64.0.0/10"):
            raise ValueError("wt0 is not in the private Brev mesh range")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        raise SystemExit(f"Brev private notebook binding unavailable: {error}") from error
allowed = ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10")
if not any(ipaddress.IPv4Address(address) in ipaddress.IPv4Network(network) for network in allowed):
    raise SystemExit("JUPYTER_BIND_ADDRESS must be a loopback or private IPv4 address, never a public/wildcard address")
lines = [line for line in lines if not line.startswith("JUPYTER_BIND_ADDRESS=")]
path.write_text("\n".join(lines + ["JUPYTER_BIND_ADDRESS=" + address]) + "\n")
print(address)
PY
)
chmod 600 .env
if [[ -z "${COSMOS_API_BASE:-}" ]]; then
    nvidia-smi
    docker compose --profile local pull cosmos-endpoint
    docker compose --profile local up -d --build
else
    # BYOM does not start or allocate a local model/GPU.
    docker compose up -d --build vantage-eval
fi
printf 'Jupyter is starting on %s:8888. Open its authenticated secure link or SSH tunnel.\n' "${JUPYTER_BIND_ADDRESS:-127.0.0.1}"
printf '%s\n' \
    'Retrieve the notebook login token with: docker compose exec vantage-eval jupyter server list' \
    'Run the notebook cells to prepare data, wait for the endpoint, and evaluate.'
