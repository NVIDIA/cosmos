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
import threading
import time
import traceback
import warnings
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Callable

import omegaconf
import torch.distributed as dist
import webdataset as wds
from webdataset.handlers import reraise_exception

from cosmos3._src.imaginaire.datasets.webdataset.config.schema import AugmentorConfig, DatasetConfig, DatasetInfo, TarSample, Wdinfo
from cosmos3._src.imaginaire.datasets.webdataset.utils.iterators import WebDataset
from cosmos3._src.imaginaire.datasets.webdataset.utils.misc import remove_extensions_from_keys, skip_keys, update_url
from cosmos3._src.imaginaire.lazy_config import instantiate
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.distributed import get_rank, get_world_size
from cosmos3._src.imaginaire.utils.object_store import ObjectStore


def wrap_augmentor_func_as_generator(func: Callable, data: Iterable):
    for data_dict in data:
        data_dict_out = func(data_dict)
        if data_dict_out is None:
            # Skip "unhealthy" samples
            continue
        yield data_dict_out


def _sample_timer(data: Iterable) -> Iterable:
    """Pipeline stage that measures total per-sample production time.

    Must be the LAST stage appended to the dataset pipeline.  When the
    DataLoader worker calls ``next()`` on this iterator the call propagates
    through the entire upstream chain (I/O -> decode -> augment -> ...),
    so the elapsed time captures the full cost of producing one sample.
    """
    it = iter(data)
    while True:
        t_start = time.monotonic()
        try:
            sample = next(it)
        except StopIteration:
            return
        sample["_sample_time"] = time.monotonic() - t_start
        yield sample


class Dataset:
    def __init__(
        self,
        config: DatasetConfig,
        handler: Callable = reraise_exception,
    ):
        r"""Webdataloader class

        Args:
            config: Dataset config
            world_size: Total number of GPUs
        """
        super().__init__()

        self.config = config

        self.world_size = get_world_size()

        dataset_info = config.dataset_info
        self.streaming_download = config.streaming_download

        self.s3_client = dict()
        self.bucket = dict()
        self.use_object_store = False
        self.data_keys = config.keys

        # Parse the metadata
        self.wdinfo = Wdinfo([], 0, 0)
        self.parse_dataset_info(dataset_info=dataset_info, use_multithread=True)
        self.handler = handler
        self.augmentors = dict()

    def parse_dataset_info(self, dataset_info: list[DatasetInfo], use_multithread: bool = True):
        r"""Parse metadata about the list of tar files.

        When ``torch.distributed`` is initialized, only rank 0 fetches the
        wdinfo JSONs (in parallel via a thread pool) and broadcasts the parsed
        metadata to every other rank.

        Args:
            dataset_info (list): List of dictionaries containing paths to metadata files.
            use_multithread (bool): Whether to use multi-threaded parsing across datasets. Default: True.
        """
        rank = get_rank()
        world_size = get_world_size()
        use_broadcast = world_size > 1 and dist.is_available() and dist.is_initialized()
        log.info(f"Start parsing dataset info with {len(dataset_info)} entries, use multithread = {use_multithread}")
        tic = time.time()

        # Thread-local ObjectStore cache for per-thread ObjectStore construction.
        thread_local_stores = threading.local()

        def get_thread_local_store(dset_info: DatasetInfo) -> ObjectStore:
            """Get or create a thread-local ObjectStore for a dataset."""
            cache = getattr(thread_local_stores, "cache", None)
            if cache is None:
                cache = thread_local_stores.cache = {}
            key = (dset_info.object_store_config.credentials, dset_info.object_store_config.bucket)
            if key not in cache:
                cache[key] = ObjectStore(config_object_storage=dset_info.object_store_config)
            return cache[key]

        def process_single_dataset(dset_num: int, dset_info: DatasetInfo):
            # For each dataset, we parse the file paths and store them as a list of TarSample.
            # TarSample will then be used by each worker to load the data.
            use_object_store = dset_info.object_store_config.enabled
            dset_id = "dset: {}".format(dset_num)
            if use_object_store:
                object_store_reader = get_thread_local_store(dset_info)
                # Create PBSS config if data is loaded from PBSS
                bucket_dset = dset_info.object_store_config.bucket
            else:
                object_store_reader = None
                bucket_dset = None

            tar_samples = []
            total_key_count = 0
            chunk_sizes = []

            # Read all wdinfo files and obtain the DataSample list
            for wdinfo_path in dset_info.wdinfo:
                if use_object_store:
                    if not object_store_reader.object_exists(wdinfo_path):
                        raise FileNotFoundError(f"{wdinfo_path} not found")
                    cur_dset_info = object_store_reader.load_object(key=wdinfo_path, type="json")  # type: ignore
                else:
                    with open(wdinfo_path, "r") as fp:
                        cur_dset_info = json.load(fp)

                data_root = cur_dset_info["root"]
                # Strip s3://bucket/ prefix from root if present, as the bucket is specified separately
                if data_root.startswith("s3://"):
                    # Remove s3://bucket/ prefix (e.g., "s3://debug/path/" -> "path/")
                    parts = data_root[5:].split("/", 1)  # Split after "s3://"
                    if len(parts) > 1:
                        data_root = parts[1]  # Take everything after bucket name
                    else:
                        data_root = ""
                tar_files_list = cur_dset_info["data_list"]
                # Use per-tar actual sample counts from data_list_key_count when available;
                # fall back to evenly distributing total_key_count across tars.
                # chunk_size is only the nominal tar capacity and is not reliable.
                per_tar_key_counts = cur_dset_info.get(
                    "data_list_key_count",
                    [cur_dset_info["total_key_count"] // max(len(tar_files_list), 1)] * len(tar_files_list),
                )
                local_tar_samples = [
                    TarSample(
                        path=tar_file,
                        root=data_root,
                        keys=(
                            dset_info.per_dataset_keys if dset_info.per_dataset_keys else self.data_keys
                        ),  # use per dataset keys if available
                        meta=dset_info,
                        dset_id=dset_id,
                        num_samples=n_samples,
                        sample_keys_full_list=None,
                    )
                    for tar_file, n_samples in zip(tar_files_list, per_tar_key_counts)
                ]
                tar_samples.extend(local_tar_samples)
                total_key_count += cur_dset_info["total_key_count"]
                # Fall back to average samples-per-tar when chunk_size is absent (e.g. SILA wdinfos).
                default_chunk_size = cur_dset_info["total_key_count"] // max(len(tar_files_list), 1)
                chunk_sizes.append(cur_dset_info.get("chunk_size", default_chunk_size))

            # boto3 clients are not picklable, so they can't ride along in the
            # broadcast payload; we rebuild them locally on every rank below.
            return {
                "dset_num": dset_num,
                "dset_id": dset_id,
                "tar_samples": tar_samples,
                "total_key_count": total_key_count,
                "chunk_sizes": chunk_sizes,
                "has_object_store": use_object_store,
                "bucket": bucket_dset,
            }

        # Step 1: rank 0 (or single-process runs) fetches every wdinfo JSON.
        fetch_elapsed = 0.0
        broadcast_elapsed = 0.0
        if rank == 0 or not use_broadcast:
            fetch_tic = time.time()
            try:
                dataset_results = []
                tasks: list[tuple[int, DatasetInfo]] = []
                for i, dset_info in enumerate(dataset_info):
                    if len(dset_info.wdinfo) == 0:
                        log.warning(f"No wdinfo found for dataset {i}, skipping...")
                        continue
                    tasks.append((i, dset_info))
                if use_multithread and len(tasks) > 1:
                    # Only rank 0 runs this in distributed mode, so we can
                    # over-subscribe the pool: wdinfo fetches are I/O-bound,
                    # so ~2x CPU count keeps the (per-thread) connection pools
                    num_workers = min(2 * (os.cpu_count() or 16), len(tasks))
                    log.info(f"Fetching {len(tasks)} datasets with {num_workers} threads")
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        futures = [executor.submit(process_single_dataset, *task) for task in tasks]
                        for future in as_completed(futures):
                            dataset_results.append(future.result())
                else:
                    for task in tasks:
                        dataset_results.append(process_single_dataset(*task))
                payload = {"ok": True, "dataset_results": dataset_results}
            except Exception as exc:
                payload = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            fetch_elapsed = time.time() - fetch_tic
        else:
            payload = None

        # Step 2: broadcast the parsed metadata (or error sentinel) to all ranks.
        if use_broadcast:
            obj_list = [payload]
            broadcast_tic = time.time()
            dist.broadcast_object_list(obj_list, src=0)
            broadcast_elapsed = time.time() - broadcast_tic
            payload = obj_list[0]

        assert payload is not None  # for type checkers
        if not payload["ok"]:
            raise RuntimeError(
                f"Rank 0 failed while fetching wdinfo metadata: "
                f"{payload['error_type']}: {payload['error_message']}\n"
                f"{payload['traceback']}"
            )
        dataset_results = payload["dataset_results"]

        # Step 3: every rank merges results and rebuilds ObjectStore instances
        # locally (boto3 clients aren't picklable, so they can't ride along in
        # the broadcast payload). Each cache entry holds a full ObjectStore;
        # we key by (credentials, bucket) so configs with thousands of
        # DatasetInfo entries sharing the same auth + bucket reuse a single
        # ObjectStore per rank instead of building one per DatasetInfo.
        self.use_object_store = any(result["has_object_store"] for result in dataset_results)
        local_object_stores: dict[tuple[str, str], ObjectStore] = {}
        for result in dataset_results:
            dset_id = result["dset_id"]
            self.wdinfo.tar_files.extend(result["tar_samples"])
            self.wdinfo.total_key_count += result["total_key_count"]
            if len(set(result["chunk_sizes"])) > 1:
                warnings.warn(
                    f"Multiple chunk_size values found in {dset_id}: {result['chunk_sizes']}. Using the first one."
                )
            self.wdinfo.chunk_size = result["chunk_sizes"][0]
            if result["has_object_store"]:
                dset_info = dataset_info[result["dset_num"]]
                cache_key = (dset_info.object_store_config.credentials, dset_info.object_store_config.bucket)
                if cache_key not in local_object_stores:
                    local_object_stores[cache_key] = ObjectStore(config_object_storage=dset_info.object_store_config)
                self.s3_client[dset_id] = local_object_stores[cache_key].client
                if result["bucket"]:
                    self.bucket[dset_id] = result["bucket"]

        toc = time.time()
        log.info(
            f"Parsed dataset info with {len(dataset_info)} wdinfos "
            f"(num_keys = {self.wdinfo.total_key_count}, num_tars = {len(self.wdinfo.tar_files)}) "
            f"and multithread = {use_multithread}, took {(toc - tic):.2f} seconds "
            f"(fetch = {fetch_elapsed:.2f}s [rank 0 only], broadcast = {broadcast_elapsed:.2f}s, "
            f"world_size = {world_size})"
        )

    @staticmethod
    # This is the function that calls each augmentor in sequence.
    def augmentor_fn(data, augmentations):
        def _stamp_pre_aug(upstream):
            for sample in upstream:
                sample["_pre_aug_time"] = time.monotonic()
                sample["_aug_step_last"] = sample["_pre_aug_time"]
                yield sample

        def _checkpoint(upstream, step_name):
            for sample in upstream:
                now = time.monotonic()
                last = sample.get("_aug_step_last", now)
                sample.setdefault("_aug_step_times", {})[step_name] = now - last
                sample["_aug_step_last"] = now
                yield sample

        # Build augmentor chain
        data = _stamp_pre_aug(data)
        for aug_fn in augmentations:
            # Use generator function as augmentor
            # (recommended, allows skipping or replicating samples inside the augmentor)
            name = getattr(aug_fn, "__name__", None) or type(aug_fn).__name__
            if getattr(aug_fn, "is_generator", False):
                data = aug_fn(data)
            else:  # Use regular function as augmentor (backward compatibility)
                data = wrap_augmentor_func_as_generator(aug_fn, data)
            data = _checkpoint(data, name)
        for sample in data:
            sample.pop("_aug_step_last", None)
            pre = sample.pop("_pre_aug_time", None)
            if pre is not None:
                sample["_aug_time"] = time.monotonic() - pre
            yield sample

    def build_data_augmentor(self, augmentor_cfg: dict[str, AugmentorConfig]) -> Callable:
        r"""Function for building data augmentors from augmentor config."""
        augmentations = []
        for aug in augmentor_cfg.keys():
            augmentations.append(instantiate(augmentor_cfg[aug]))

        # This is the function that calls each augmentor in sequence.
        return partial(Dataset.augmentor_fn, augmentations=augmentations)

    def build_dataset(self, **kwargs) -> WebDataset:
        tar_list = self.wdinfo.tar_files
        num_tars = len(tar_list)
        assert num_tars > 0, "Did not find any data."

        shuffle_buffer_size = getattr(self.config, "buffer_size", self.wdinfo.chunk_size)

        # update distributor urls and chunk size
        distributor_fn = self.config.distributor

        distributor_fn.set_urls(tar_list)
        distributor_fn.set_chunk_size(self.wdinfo.chunk_size)

        dataset = WebDataset(
            distributor_fn,
            load_from_object_store=self.use_object_store,
            s3_client=self.s3_client,
            s3_bucket_name=self.bucket,
            streaming_download=self.streaming_download,
            handler=self.handler,
        )

        # Creating a shuffle buffer
        if shuffle_buffer_size > 0:
            dataset.append(wds.shuffle(shuffle_buffer_size))

        # Adding decoders
        # Decoders are functions that decode the input IO stream
        decoder_list = getattr(self.config, "decoders", [])
        decoder_functions = []
        for decoder in decoder_list:
            # If the specified decoder is a string, use the webdataset decoder
            # If its a callable function, use the defined function to decode data
            assert isinstance(decoder, str) or callable(decoder), "Decoder should either be callable or a str"
            decoder_functions.append(decoder)
        dataset.append(wds.decode(*decoder_functions))

        # After the decoders are added, remove extension from the keys
        # Extensions in the data keys are needed for auto-detection of decoders in webdataset.
        if self.config.remove_extension_from_keys:
            dataset.append(remove_extensions_from_keys)

        # Function to skip keys
        dataset.append(skip_keys)
        # Building augmentors
        augmentor_cfg = getattr(self.config, "augmentation", None)
        assert isinstance(augmentor_cfg, (dict, omegaconf.dictconfig.DictConfig)), (
            f"getting type: {type(augmentor_cfg)}"
        )
        augmentation_fn = self.build_data_augmentor(augmentor_cfg)
        dataset.append(augmentation_fn)

        # Updates URL names so that the collate function can handle
        dataset.append(update_url)

        dataset.append(_sample_timer)

        dataset.total_images = self.wdinfo.total_key_count  # type: ignore
        log.info("Total number of training shards: %d" % num_tars)
        log.info("Total training key count: %d" % dataset.total_images)  # type: ignore

        return dataset
