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

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import filelock
from loguru import logger as log

_LOCK_TIMEOUT_SECONDS = 1800  # 30 minutes


def resolve_hf_model_store(credentials: str, bucket: str) -> tuple[str, str]:
    """
    Resolve checkpoint store credentials/bucket to the permanent HF model store.
    GCP training checkpoints → gcp_load_from_object_store_permanent (nv-00-10206-checkpoint)
    AWS training checkpoints → aws_load_from_object_store_permanent (nv-cosmos-vlm)
    Falls back to the provided credentials/bucket if neither matches.
    """
    from cosmos.utils.vlm.configs_defaults.checkpointer import (
        aws_load_from_object_store_permanent,
        gcp_load_from_object_store_permanent,
    )

    if credentials == gcp_load_from_object_store_permanent.credentials:
        return gcp_load_from_object_store_permanent.credentials, gcp_load_from_object_store_permanent.bucket
    elif credentials == aws_load_from_object_store_permanent.credentials:
        return aws_load_from_object_store_permanent.credentials, aws_load_from_object_store_permanent.bucket
    return credentials, bucket


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

    s3 = boto3.client("s3", **json.load(open(credential_path, "r")))

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

    def _stream_download(key, local_path):
        """Download via get_object streaming to bypass ETag checksum validation.

        s3.download_file uses s3transfer which validates ETags after download.
        GCS's S3-compatible API returns CRC32C-based ETags for composite objects,
        which don't match the MD5-based ETags boto3 expects, causing checksum errors.
        Using get_object directly skips that validation.
        """
        response = s3.get_object(Bucket=bucket, Key=key)
        with open(local_path, "wb") as f:
            for chunk in response["Body"].iter_chunks(chunk_size=8 * 1024 * 1024):
                f.write(chunk)

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
        fut = executor.submit(_stream_download, key, local_path)
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
    s3 = boto3.client("s3", **json.load(open(credentials, "r")))
    # Make sure prefix ends with "/" to represent a "directory"
    if not prefix.endswith("/"):
        prefix += "/"

    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return "Contents" in resp


def maybe_download_hf_model_from_s3(
    model_name_or_path: str,
    credentials: str,
    bucket: str,
    include_model_weights: bool = False,
    cache_dir: str = None,
    s3_prefix: str = "cosmos_reason2/hf_models",
    require_s3_exists: bool = False,
) -> str:
    exclude_list = [".safetensors"] if not include_model_weights else []
    s3_prefix = os.path.join(s3_prefix, model_name_or_path)
    # download the model from s3 to local cache
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.getenv("IMAGINAIRE_CACHE_DIR", "~/.cache/imaginaire"))

    cache_dir = os.path.join(cache_dir, s3_prefix)

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
            and (
                os.path.exists(os.path.join(cache_dir, "vocab.json"))
                or os.path.exists(os.path.join(cache_dir, "tokenizer.json"))
            )
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
