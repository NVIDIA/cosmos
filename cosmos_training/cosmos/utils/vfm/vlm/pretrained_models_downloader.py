# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import filelock
from boto3.s3.transfer import TransferConfig
from loguru import logger as log

from cosmos.utils.flags import INTERNAL
from cosmos.utils.easy_io.backends.auto_auth import json_load_auth, open_auth

_LOCK_TIMEOUT_SECONDS = 1800  # 30 minutes


def _load_s3_credentials(credential_path: str) -> dict:
    """Resolve S3 credentials from file or PROD_* env vars.

    Mirrors cosmos.utils.easy_io.backends.boto3_client.Boto3Client so callers honor
    CI/OSS env-var auth (e.g. PROD_GCP_CHECKPOINT_*) instead of crashing with
    FileNotFoundError when ``credentials/*.secret`` is absent on disk.
    """
    with open_auth(credential_path, "r") as f:
        return json_load_auth(f)


def parallel_download_s3_prefix_to_dir(
    bucket: str,
    prefix: str,
    dest_dir: str,
    credential_path: str,
    max_workers: int = 4,
    skip_if_exists: bool = True,
    exclude_list: list[str] = [],
) -> list[str]:
    """
    Parallel download of all objects under s3_uri (prefix) to dest_dir,
    preserving relative paths. Returns list of downloaded (or skipped) local paths.
    Example of exclude_list: [".safetensors"]
    """
    os.makedirs(dest_dir, exist_ok=True)

    s3 = boto3.client("s3", **_load_s3_credentials(credential_path))

    # List all objects under prefix (paginated)
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    # Collect (key, size) for real objects (exclude "folders")
    objects: list[tuple[str, int]] = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip "directory placeholders"
            if key.endswith("/"):
                continue
            if any(exclude in key for exclude in exclude_list):
                log.info(f"Skipping {key} because it matches exclude_list {exclude_list}")
                continue
            objects.append((key, obj["Size"]))

    # Nothing to do
    if not objects:
        return []

    # Prepare download tasks
    tasks = []
    results: list[str] = []

    transfer_cfg = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,  # 64MB threshold
        multipart_chunksize=64 * 1024 * 1024,  # 64MB parts
        max_concurrency=max_workers,
        use_threads=True,
    )

    def submit_download(executor, key, size):
        rel_path = os.path.relpath(key, start=prefix) if prefix else key
        local_path = os.path.join(dest_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Skip if exists and size matches
        if skip_if_exists and os.path.exists(local_path):
            try:
                if os.path.getsize(local_path) == size:
                    results.append(local_path)
                    return None  # do not submit
            except OSError:
                pass
        log.info(f"Downloading s3://{bucket}/{key} to {local_path}")
        fut = executor.submit(
            s3.download_file,
            bucket,
            key,
            local_path,
            ExtraArgs={},  # you can add {"RequestPayer": "requester"} if needed
            Config=transfer_cfg,
        )
        fut._local_path = local_path  # type: ignore[attr-defined]
        return fut

    # Dispatch
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for key, size in objects:
            f = submit_download(ex, key, size)
            if f is not None:
                tasks.append(f)

        for f in as_completed(tasks):
            # propagate any exceptions immediately
            _ = f.result()
            results.append(getattr(f, "_local_path"))

    return results


def has_model_weights(cache_dir: str) -> bool:
    import glob

    return len(glob.glob(os.path.join(cache_dir, "*.safetensors"))) > 0


def s3_dir_exists(bucket, prefix, credentials):
    """
    Check whether a given prefix (directory) exists in an S3 bucket.

    Args:
        bucket (str): The name of the S3 bucket.
        prefix (str): The prefix (directory path) to check.

    Returns:
        bool: True if the prefix exists, False otherwise.
    """
    s3 = boto3.client("s3", **_load_s3_credentials(credentials))
    # Make sure prefix ends with "/" to represent a "directory"
    if not prefix.endswith("/"):
        prefix += "/"

    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return "Contents" in resp


def _download_from_hf_hub(model_name_or_path: str, include_model_weights: bool = True) -> str:
    """Download a model from HuggingFace Hub when no S3 credentials/bucket are configured.

    Mirrors the logic in cosmos_rl resolve_model_path: detects safetensors to set
    ignore_patterns, then calls snapshot_download into HF_HOME/hub.
    """
    from huggingface_hub import HfFileSystem, snapshot_download

    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    hf_token = os.environ.get("HF_TOKEN", None)
    hf_home = os.environ.get("HF_HOME", "")
    hf_cache_dir = os.path.join(hf_home, "hub") if hf_home else os.path.expanduser("~/.cache/huggingface/hub")

    ignore_patterns: list[str] = []
    try:
        hf_fs = HfFileSystem(token=hf_token)
        files = hf_fs.ls(model_name_or_path, detail=False)
        has_safetensors = any(
            f.endswith("model.safetensors.index.json") or f.endswith("model.safetensors") for f in files
        )
        if has_safetensors:
            ignore_patterns += ["*pytorch_model*", "*consolidated*"]
    except Exception as e:
        log.warning(f"Could not list HuggingFace repo {model_name_or_path}: {e}")

    if not include_model_weights:
        ignore_patterns.append("*.safetensors")

    log.info(f"Downloading {model_name_or_path} from HuggingFace Hub (ignore={ignore_patterns})")
    local_path = snapshot_download(
        model_name_or_path,
        token=hf_token,
        cache_dir=hf_cache_dir,
        ignore_patterns=ignore_patterns or None,
    )
    log.info(f"Downloaded {model_name_or_path} to {local_path}")
    return local_path


def maybe_download_hf_model_from_s3(
    model_name_or_path: str,
    credentials: str,
    bucket: str,
    include_model_weights: bool = False,
    cache_dir: str = None,
    s3_prefix: str = "cosmos_reason2/hf_models",
    require_s3_exists: bool = False,
) -> str:
    # Short-circuit when model_name_or_path is already a local directory — no
    # S3 or HF Hub fetch is needed. Prevents opening credentials/*.secret
    # in OSS/local-checkpoint smoke runs that already have the model on disk.
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    exclude_list = [".safetensors"] if not include_model_weights else []
    s3_prefix = os.path.join(s3_prefix, model_name_or_path)
    # download the model from s3 to local cache
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.getenv("IMAGINAIRE_CACHE_DIR", "~/.cache/imaginaire"))

    cache_dir = os.path.join(cache_dir, s3_prefix)

    if not credentials or not bucket:
        log.warning(
            f"No S3 credentials/bucket configured, trying to download from HuggingFace Hub for {model_name_or_path}"
        )
        return _download_from_hf_hub(model_name_or_path, include_model_weights)

    # In OSS/CI mode (not INTERNAL), route registered tokenizer/HF URIs through the
    # checkpoint registry so the HF Hub fallback runs without ever opening the S3
    # credential file. Mirrors the legacy download_tokenizer_files behavior on main.
    if not INTERNAL:
        from cosmos.utils.checkpoint_db import CheckpointConfig, sanitize_uri

        s3_uri = f"s3://{bucket}/{s3_prefix}"
        if CheckpointConfig.maybe_from_uri(sanitize_uri(s3_uri)) is not None:
            from cosmos.utils.checkpoint_db import download_checkpoint_v2

            local_path = download_checkpoint_v2(s3_uri)
            if "://" not in local_path:
                return local_path

    if not s3_dir_exists(bucket, s3_prefix, credentials):
        if require_s3_exists:
            raise FileNotFoundError(f"Model {model_name_or_path} not found in s3://{bucket}/{s3_prefix}")
        else:
            log.critical(
                f"Model {model_name_or_path} not found in s3://{bucket}/{s3_prefix} with credentials {credentials}",
                rank0_only=False,
            )
            return model_name_or_path

    lock_path = os.path.join(cache_dir, "lock.lock")
    lock = filelock.FileLock(lock_path, timeout=_LOCK_TIMEOUT_SECONDS)  # 1 minute timeout for download
    with lock:
        if (
            os.path.exists(cache_dir)
            and not include_model_weights
            and os.path.exists(os.path.join(cache_dir, "vocab.json"))
        ):
            return cache_dir
        elif os.path.exists(cache_dir) and include_model_weights and has_model_weights(cache_dir):
            return cache_dir
        else:
            os.makedirs(cache_dir, exist_ok=True)
            tic = time.time()
            parallel_download_s3_prefix_to_dir(bucket, s3_prefix, cache_dir, credentials, exclude_list=exclude_list)
            toc = time.time()
            print(f"Time taken to download model {model_name_or_path}: {toc - tic:.3f} seconds")

    return cache_dir


if __name__ == "__main__":
    """
    Usage:
    PYTHONPATH=. python3 cosmos/utils/vlm/pretrained_models_downloader.py
    """
    cache_dir = maybe_download_model(  # noqa: F821
        "eagle_er_qwen3_1p7b_siglip_400m", "credentials/s3_training.secret", "checkpoints-us-east-1"
    )
    print(f"Downloaded to {cache_dir}")
