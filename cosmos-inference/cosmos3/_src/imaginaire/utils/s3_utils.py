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
from typing import Any, Optional

from cosmos3._src.imaginaire.utils import distributed, log
from cosmos3._src.imaginaire.utils.easy_io import easy_io


def download_from_s3_with_cache(
    s3_path: str,
    cache_fp: Optional[str] = None,
    cache_dir: Optional[str] = None,
    rank_sync: bool = True,
    backend_args: Optional[dict] = None,
    backend_key: Optional[str] = None,
) -> str:
    """download data from S3 with optional caching.

    This function first attempts to load the data from a local cache file. If
    the cache file doesn't exist, it downloads the data from S3 to the cache
    location. Caching is performed in a rank-aware manner
    using `distributed.barrier()` to ensure only one download occurs across
    distributed workers (if `rank_sync` is True).

    Args:
        s3_path (str): The S3 path of the data to load.
        cache_fp (str, optional): The path to the local cache file. If None,
            a filename will be generated based on `s3_path` within `cache_dir`.
        cache_dir (str, optional): The directory to store the cache file. If
            None, the environment variable `IMAGINAIRE_CACHE_DIR` (defaulting
            to "/tmp") will be used.
        rank_sync (bool, optional): Whether to synchronize download across
            distributed workers using `distributed.barrier()`. Defaults to True.
        backend_args (dict, optional): The backend arguments passed to easy_io to construct the backend.
        backend_key (str, optional): The backend key passed to easy_io to registry the backend or retrieve the backend if it is already registered.

    Returns:
        cache_fp (str): The path to the local cache file.

    Raises:
        FileNotFoundError: If the data cannot be found in S3 or the cache.
    """
    cache_dir = os.environ.get("TORCH_HOME") if cache_dir is None else cache_dir
    cache_dir = (
        os.environ.get("IMAGINAIRE_CACHE_DIR", os.path.expanduser("~/.cache/imaginaire"))
        if cache_dir is None
        else cache_dir
    )
    cache_dir = os.path.expanduser(cache_dir)
    if cache_fp is None:
        cache_fp = os.path.join(cache_dir, s3_path.replace("s3://", ""))
    if not cache_fp.startswith("/"):
        cache_fp = os.path.join(cache_dir, cache_fp)

    if distributed.get_rank() == 0:
        if os.path.exists(cache_fp):
            # check the size of cache_fp
            if os.path.getsize(cache_fp) < 1:
                os.remove(cache_fp)
                log.warning(f"Removed empty cache file {cache_fp}.")

    if rank_sync:
        if not os.path.exists(cache_fp):
            log.critical(f"Local cache {cache_fp} Not exist! Downloading {s3_path} to {cache_fp}.")
            log.info(f"backend_args: {backend_args}")
            log.info(f"backend_key: {backend_key}")

            easy_io.copyfile_to_local(
                s3_path, cache_fp, dst_type="file", backend_args=backend_args, backend_key=backend_key
            )
            log.info(f"Downloaded {s3_path} to {cache_fp}.")
        else:
            log.info(f"Local cache {cache_fp} already exist! {s3_path} -> {cache_fp}.")

        distributed.barrier()
    else:
        if not os.path.exists(cache_fp):
            easy_io.copyfile_to_local(
                s3_path, cache_fp, dst_type="file", backend_args=backend_args, backend_key=backend_key
            )

            log.info(f"Downloaded {s3_path} to {cache_fp}.")
    return cache_fp


def load_from_s3_with_cache(
    s3_path: str,
    cache_fp: Optional[str] = None,
    cache_dir: Optional[str] = None,
    rank_sync: bool = True,
    backend_args: Optional[dict] = None,
    backend_key: Optional[str] = None,
    easy_io_kwargs: Optional[dict] = None,
) -> Any:
    """Loads data from S3 with optional caching.

    This function first attempts to load the data from a local cache file. If
    the cache file doesn't exist, it downloads the data from S3 to the cache
    location and then loads it. Caching is performed in a rank-aware manner
    using `distributed.barrier()` to ensure only one download occurs across
    distributed workers (if `rank_sync` is True).

    Args:
        s3_path (str): The S3 path of the data to load.
        cache_fp (str, optional): The path to the local cache file. If None,
            a filename will be generated based on `s3_path` within `cache_dir`.
        cache_dir (str, optional): The directory to store the cache file. If
            None, the environment variable `IMAGINAIRE_CACHE_DIR` (defaulting
            to "/tmp") will be used.
        rank_sync (bool, optional): Whether to synchronize download across
            distributed workers using `distributed.barrier()`. Defaults to True.
        backend_args (dict, optional): The backend arguments passed to easy_io to construct the backend.
        backend_key (str, optional): The backend key passed to easy_io to registry the backend or retrieve the backend if it is already registered.

    Returns:
        Any: The loaded data from the S3 path or cache file.

    Raises:
        FileNotFoundError: If the data cannot be found in S3 or the cache.
    """
    cache_fp = download_from_s3_with_cache(s3_path, cache_fp, cache_dir, rank_sync, backend_args, backend_key)

    if easy_io_kwargs is None:
        easy_io_kwargs = {}
    return easy_io.load(cache_fp, **easy_io_kwargs)
