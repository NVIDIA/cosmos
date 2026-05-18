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

"""Shared utilities for Action dataset export and evaluation workflows."""

import base64
import fcntl
import json
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger
from tqdm import tqdm

DEFAULT_GCS_CREDENTIALS = "credentials/gcp_checkpoint.secret"
GCS_CREDS_ENV_VAR = "COSMOS_GCP_CHECKPOINT_CREDS"


# ---------------------------------------------------------------------------
# GCS download utilities (uses s5cmd via uvx for parallel transfers)
# ---------------------------------------------------------------------------


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse an S3-compatible URI into (bucket, key_prefix)."""
    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.strip("/")
    return bucket, prefix


def _to_s3_uri(uri: str) -> str:
    """Normalize a gs:// or s3:// URI to the s3:// scheme expected by s5cmd."""
    bucket, prefix = _parse_s3_uri(uri)
    return f"s3://{bucket}/{prefix}"


def _load_gcs_credentials(credentials_path: str | Path) -> dict[str, Any]:
    """Load GCS S3-compatible credentials from the COSMOS_GCP_CHECKPOINT_CREDS
    env var (base64-encoded JSON) or fall back to a local file."""
    env_value = os.environ.get(GCS_CREDS_ENV_VAR)
    if env_value:
        return json.loads(base64.b64decode(env_value))
    with open(credentials_path) as f:
        return json.load(f)


def _count_remote_objects(
    s3_uri: str,
    env: dict[str, str],
    endpoint_url: str,
) -> int | None:
    """Count remote objects under *s3_uri* using ``s5cmd ls``.

    Returns ``None`` if the listing fails so callers can fall back to an
    indeterminate progress bar.
    """
    cmd = ["uvx", "s5cmd", "--endpoint-url", endpoint_url, "ls", f"{s3_uri}/*"]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return sum(1 for line in result.stdout.splitlines() if line.strip())


def _download_with_s5cmd(
    gcs_uri: str,
    local_dir: Path,
    credentials_path: str | Path,
) -> None:
    """Download all objects under S3-compatible URI using s5cmd via uvx.

    s5cmd performs massively parallel transfers and natively skips files that
    already exist at the destination with matching size (via ``sync``).
    """
    creds = _load_gcs_credentials(credentials_path)
    s3_uri = _to_s3_uri(gcs_uri)

    env = {
        **os.environ,
        "AWS_ACCESS_KEY_ID": creds["aws_access_key_id"],
        "AWS_SECRET_ACCESS_KEY": creds["aws_secret_access_key"],
        "AWS_REGION": creds.get("region_name", ""),
    }

    local_dir.mkdir(parents=True, exist_ok=True)

    endpoint_url = creds["endpoint_url"]
    cmd = [
        "uvx",
        "s5cmd",
        "--endpoint-url",
        endpoint_url,
        "sync",
        f"{s3_uri}/*",
        str(local_dir) + "/",
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    total = _count_remote_objects(s3_uri, env, endpoint_url)
    if total is not None:
        logger.info(f"Remote listing: {total} object(s)")

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    stderr_lines: list[str] = []
    with tqdm(total=total, desc="s5cmd sync", unit="file", dynamic_ncols=True) as pbar:
        for line in proc.stdout:
            if line.strip():
                pbar.update(1)
        proc.wait()
        if proc.stderr:
            stderr_lines = proc.stderr.readlines()

    if proc.returncode != 0:
        raise RuntimeError(f"s5cmd failed (exit {proc.returncode}):\n{''.join(stderr_lines)}")
    logger.success(f"s5cmd sync complete: {s3_uri} -> {local_dir}")


def _extract_lz4_archives(root_dir: Path, dataset_name_override: str = "") -> None:
    """Extract ``.zarr.tar.lz4`` archives found under *root_dir*.

    Each archive ``<name>.zarr.tar.lz4`` is extracted into a sibling
    ``<name>.zarr`` directory.  Already-extracted archives (where the
    target directory exists and is non-empty) are skipped.

    A per-prefix ``.lz4_extract_done.<prefix>`` marker is written after
    extraction so subsequent calls with the same ``dataset_name_override``
    skip the expensive ``rglob`` scan entirely.
    """
    if dataset_name_override:
        prefixes = sorted(name.strip() for name in dataset_name_override.split(",") if name.strip())
        marker_key = "_".join(p.replace("/", "--") for p in prefixes)
        extract_marker = root_dir / f".lz4_extract_done.{marker_key}"
    else:
        prefixes = []
        extract_marker = root_dir / ".lz4_extract_done"

    if extract_marker.exists():
        return

    archives = list(root_dir.rglob("*.zarr.tar.lz4"))
    if not archives:
        extract_marker.touch()
        return

    if prefixes:
        filtered = [a for a in archives if any(str(a.relative_to(root_dir)).startswith(p) for p in prefixes)]
        logger.info(
            f"Filtered {len(archives)} archive(s) to {len(filtered)} matching "
            f"dataset_name_override prefixes: {prefixes}"
        )
        archives = filtered

    if not archives:
        extract_marker.touch()
        return
    logger.info(f"Found {len(archives)} .zarr.tar.lz4 archive(s) to extract under {root_dir}")
    for archive in archives:
        zarr_dir = archive.with_name(archive.name.removesuffix(".tar.lz4"))
        done_marker = zarr_dir.with_suffix(zarr_dir.suffix + ".extract_done")
        if done_marker.exists():
            logger.debug(f"Already extracted: {zarr_dir}")
            continue
        zarr_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting {archive.name} -> {zarr_dir}")
        result = subprocess.run(
            f"lz4 -d -c {archive} | tar xf - -C {zarr_dir} --strip-components=1 --overwrite",
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"lz4/tar extraction failed for {archive}:\n{result.stderr}")
        done_marker.touch()
    logger.success(f"All archives extracted under {root_dir}")
    extract_marker.touch()


def download_from_gcs_uri(
    gcs_uri: str,
    cache_dir: Path,
    gcs_credentials: str = DEFAULT_GCS_CREDENTIALS,
    force_download: bool = False,
    dataset_name_override: str = "",
) -> Path:
    """Download a dataset from a GCS S3-compatible URI and return the local path."""
    bucket, prefix = _parse_s3_uri(gcs_uri)
    local_dataset_dir = cache_dir / bucket / prefix
    complete_marker = local_dataset_dir / ".download_complete"
    lock_path = Path(str(local_dataset_dir) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize concurrent workers so only one syncs; others wait and reuse.
    # The sentinel file is the source of truth for completeness.
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            if not force_download and complete_marker.exists():
                logger.info(f"Dataset already exists at {local_dataset_dir}, skipping download")
            else:
                logger.info(f"Downloading dataset from {gcs_uri} to {local_dataset_dir}")
                local_dataset_dir.mkdir(parents=True, exist_ok=True)
                _download_with_s5cmd(gcs_uri, local_dataset_dir, gcs_credentials)
                _extract_lz4_archives(local_dataset_dir, dataset_name_override=dataset_name_override)
                complete_marker.touch()
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
    return local_dataset_dir


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def set_dataset_mode(dataset: Any, mode: str) -> None:
    """Propagate *mode* through wrapper layers to the underlying dataset(s)."""
    if hasattr(dataset, "_datasets"):
        for entry in dataset._datasets:
            if isinstance(entry, dict):
                ds = entry.get("dataset")
            elif isinstance(entry, (list, tuple)):
                ds = entry[1] if len(entry) >= 2 else None
            else:
                ds = entry
            if ds is not None:
                set_dataset_mode(ds, mode)
    if hasattr(dataset, "datasets") and isinstance(dataset.datasets, dict):
        for ds in dataset.datasets.values():
            set_dataset_mode(ds, mode)
    if hasattr(dataset, "dataset"):
        set_dataset_mode(dataset.dataset, mode)
    if hasattr(dataset, "mode"):
        dataset.mode = mode
    if hasattr(dataset, "_mode"):
        dataset._mode = mode


def _apply_root_to_target(target: dict[str, Any], root: str | list[str]) -> None:
    """Set *root* on a single dataset config dict."""
    target["root"] = root
    logger.info(f"Overriding dataset root to {target['root']}")


def override_dataset_root(dataset_config: dict[str, Any], root: str | list[str]) -> None:
    """Inject a *root* override into the Hydra dataset config dict.

    The config is expected to be a ``wrap_dataset(list_of_datasets=<inner>)``
    structure.  We set ``root`` on the inner dataset config so the dataset
    class uses the provided local path instead of its default.

    When *root* is a list it is used directly (one root per repo_id).
    When *root* is a single string and the existing config has a list of roots,
    the single value is broadcast to every entry.

    When a single *root* string is given we also search for ``*normalizer*.json``
    files under that directory and, if found, set ``normalizer_dir`` on the inner
    config so that datasets whose normalizer was downloaded alongside the data
    (e.g. UMI from GCS) pick up the local copy instead of the original
    cluster-local path baked into the Hydra YAML.

    ``list_of_datasets`` may be a dict (single / narrowed entry) or a list of
    dataset-entry dicts.  In the list case we apply the override to each entry.
    """
    inner = dataset_config.get("list_of_datasets", dataset_config)

    if isinstance(inner, list):
        for entry in inner:
            if isinstance(entry, dict):
                target = entry.get("dataset", entry)
                if isinstance(target, dict):
                    _apply_root_to_target(target, root)
    elif isinstance(inner, dict):
        _apply_root_to_target(inner, root)
    else:
        _apply_root_to_target(dataset_config, root)
