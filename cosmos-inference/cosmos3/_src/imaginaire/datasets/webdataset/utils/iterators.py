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

import io
import os
import random
import sys
import time
from typing import IO, Any, BinaryIO, Callable, Dict, Iterable, Iterator, Optional, Tuple, Union
from urllib.parse import urlparse

import boto3
import botocore
import botocore.exceptions
import pandas as pd
import webdataset.gopen as gopen_webdata
import yaml
from webdataset import cache, filters, shardlists
from webdataset.compat import FluidInterface
from webdataset.handlers import reraise_exception
from webdataset.pipeline import DataPipeline
from webdataset.pytorch import IterableDataset
from webdataset.tariterators import group_by_keys, tar_file_iterator

from cosmos3._src.imaginaire.datasets.webdataset.config.schema import TarSample
from cosmos3._src.imaginaire.datasets.webdataset.utils.stream import RetryingStream
from cosmos3._src.imaginaire.utils import log

# Number of attempts to read s3 objects.
_NUM_OBJECT_STORE_READ_ATTEMPTS = 10


def gopen(url: Tuple, mode: str = "rb", bufsize: int = 8192, **kw) -> Union[io.BytesIO, RetryingStream, BinaryIO, IO]:
    r"""Open the URL.
    This uses the `gopen_schemes` dispatch table to dispatch based
    on scheme.
    Support for the following schemes is built-in: pipe, file,
    http, https, sftp, ftps, scp.
    When no scheme is given the url is treated as a file.
    You can use the OPEN_VERBOSE argument to get info about
    files being opened.
    Args:
        url (tuple): (source URL, dataset id)
        the source URL is join(TarSample.root, one of TarSample.keys, TarSample.path)
        e.g. join("openx_short_cmu_playing_with_food_202505/v2.3/resolution_lt_720/aspect_ratio_4_3/duration_5_10/", "videos", "part_000000/000000.tar")
        mode (str): the mode ("rb", "r")
        bufsize (int): the buffer size
    Returns:
        Byte streams
    """
    global fallback_gopen
    verbose = int(os.environ.get("GOPEN_VERBOSE", 0))
    if verbose:
        log.info("GOPEN", url, gopen_webdata.info, file=sys.stderr)

    assert mode in ["rb", "wb"], mode
    if url == "-":
        if mode == "rb":
            return sys.stdin.buffer
        elif mode == "wb":
            return sys.stdout.buffer
        else:
            raise ValueError(f"unknown mode {mode}")

    # If we specify 'object_store' in keyword arguments,
    # then we would load from s3.
    if "object_store" in kw and kw["object_store"]:
        assert isinstance(url, tuple)
        return gopen_s3(
            url,
            s3_clients=kw["s3_client"],
            s3_bucket_name=kw["s3_bucket_name"],
            streaming_download=kw["streaming_download"],
        )

    # For all other gopen schemes, use the native webdataset gopen functions.
    # pr = gopen_webdata.urlparse(url)
    # this should be a path to an existing file on local machine
    url = url[0]
    assert isinstance(url, str)
    pr = urlparse(url)
    if pr.scheme == "":
        bufsize = int(os.environ.get("GOPEN_BUFFER", -1))
        return open(url, mode, buffering=bufsize)
    if pr.scheme == "file":
        bufsize = int(os.environ.get("GOPEN_BUFFER", -1))
        return open(pr.path, mode, buffering=bufsize)
    handler = gopen_webdata.gopen_schemes["__default__"]
    handler = gopen_webdata.gopen_schemes.get(pr.scheme, handler)
    return handler(url, mode, bufsize, **kw)  # type: ignore


def gopen_s3(
    url: tuple,
    s3_clients: Dict[str, boto3.client],  # type: ignore
    s3_bucket_name: Dict[str, str],
    streaming_download=True,
) -> Union[io.BytesIO, RetryingStream]:
    r"""Gopen scheme for s3.
    Function for reading urls from s3
    Args:
        url (list[TarSample]): the source URL
        s3_client (boto3.client): Boto3 client for downloading from S3
        s3_bucket_name (str): Bucket name for the S3 data
    Returns:
        Byte streams
    """

    attempt = 0

    url_path = url[0]
    dset_id = url[1]
    s3_client = s3_clients[dset_id]
    bucket = s3_bucket_name[dset_id]

    while attempt < _NUM_OBJECT_STORE_READ_ATTEMPTS:
        try:
            if streaming_download:
                # Downloads in a streaming fashion
                s3_stream = RetryingStream(s3_client, bucket=bucket, key=url_path)
                return s3_stream
            else:
                # Downloads the entire file
                buffer = io.BytesIO()
                s3_client.download_fileobj(bucket, url_path, buffer)
                buffer.seek(0)
                return buffer
        except botocore.exceptions.ClientError as e:
            # If there is an exception (usually connectivity error or protocol error), read again
            attempt += 1
            retry_interval = min(
                0.1 * 2**attempt + random.uniform(0, 1), 30
            )  # sleep workers randomly to avoid burst of requests
            log.info(
                f"Got an exception while downloading data {url_path}: attempt={attempt} - {e}. {type(e)}",
                rank0_only=False,
            )
            log.info(f"Retrying tar file download after {retry_interval}s", rank0_only=False)
            time.sleep(retry_interval)
            continue
    raise ConnectionError("Unable to read {} from PBSS. {} attempts tried.".format(url, attempt))


def url_opener(data: Iterable, handler: Callable = reraise_exception, **kw) -> Iterator[dict]:
    r"""Given a stream of url names (packaged in `dict(url=url)`), yield opened streams.

    Args:
        data (Iterable): Iterator of dictionaires containing url paths.
        handler (Callable): Exception handler.

    Yields:
      Dictionaries with this structure:
        {"url": ...
         "stream": list[Union[io.BytesIO, RetryingStream]]}
    """
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample

        url = sample["url"]
        assert isinstance(url, TarSample), "URL should be of type TarSample"
        try:
            stream = []
            for data_key in url.keys:
                url_path_full = os.path.join(url.root, data_key, url.path)
                url_key = (url_path_full, url.dset_id)
                stream.append(gopen(url_key, **kw))

            sample.update(stream=stream)
            yield sample
        except Exception as exn:
            log.info(f"Got an exception while opening urls - {exn}", rank0_only=False)
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break


def process_sample(sample, url, key_idx):
    assert isinstance(sample, dict) and "data" in sample and "fname" in sample
    # Edit the url entries
    sample["__url__"] = url
    # This is the folder name
    data_key = url.keys[key_idx]
    # Handle the case where data_key has "/"
    data_key = data_key.replace("/", "_")
    # Edit the fname to include the data_key
    fname_splits = sample["fname"].split(".")
    if len(fname_splits) == 2:
        prefix, suffix = fname_splits  # {sample_key}.{suffix} e.g. "id_1410095.json"
    else:  # if the fname here contains more than one dot, we replace all the dots except the last one with "-"
        prefix = "-".join(fname_splits[:-1])
        suffix = fname_splits[-1]

    # e.g. "id_1410095.caption_ai_from_image.json"
    sample["fname"] = f"{prefix}.{data_key}.{suffix}"

    return sample


def tar_file_expander(
    data: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
    s3_client: Optional[Dict[str, boto3.client]] = None,  # type: ignore
    s3_bucket_name: Optional[Dict[str, str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Expand tar files.

    Args:
        data (Iterable[Iterable[Dict[str, Any]]]): iterator over opened tar file streams.
        handler (Callable[[Exception], bool]): exception handler.
        select_files (Optional[Callable[[str], bool]]): select files from tarfiles by name (permits skipping files).
        rename_files (Optional[Callable[[str], bool]]): Renaming tar files.

        Optional args if reading sample_keys_full_list:
        s3_clients (Dict[str, boto3.client]): If loading from object store, specify S3 client. Keys is the dset_id, i.e. dataset id since different dataset could use different s3 client and bucket
        s3_bucket_name (Dict[str, str]): If loading from object store, specify S3 bucket name.

    Yields:
        a stream of samples.
    """
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            tar_file_iterator_list = []
            for stream_id in range(len(source["stream"])):
                tar_file_iterator_list.append(
                    tar_file_iterator(
                        source["stream"][stream_id],
                        handler=handler,
                        select_files=select_files,
                        rename_files=rename_files,
                    )
                )
            if url.sample_keys_full_list is None:  # Original behavior
                # tar_file_iterator_list is a list of iterator: [tar_file_iterator_0, tar_file_iterator_1, ... tar_file_iterator_N]
                for sample in zip(*tar_file_iterator_list):
                    # Merging data from all streams
                    # sample is list of dictionaries, each dictionary contains data and fname
                    # sample [tar_file_iterator_0[0], tar_file_iterator_1[0], ... tar_file_iterator_N[0]], length = num_of_data_key
                    for key_idx, sample_key in enumerate(sample):
                        sample_key = process_sample(sample_key, url, key_idx)
                        yield sample_key
            else:
                # Read the index file from pbss
                s3_client_cur = s3_client[url.dset_id]
                bucket_cur = s3_bucket_name[url.dset_id]
                sample_keys_full_list = read_sample_keys_full_list(
                    url.sample_keys_full_list, s3_client_cur, bucket_cur
                )  # e.g. ["has_material_glb_from_obj_v4_1410095_0", "has_material_glb_from_obj_v4_1410095_1", ...]
                sample_keys_full_to_index = {element: index for index, element in enumerate(sample_keys_full_list)}

                # Start reading the tar files
                target_index = 0
                last_index = [-1] * len(tar_file_iterator_list)  # Keep track of the last index of each tar file
                sample_list = []  # List of samples from each tar file
                while True:  # Exit until target_index reach the max value
                    skip_offset = False
                    for key_idx, iterator in enumerate(tar_file_iterator_list):
                        if last_index[key_idx] >= target_index:
                            # This tar is moving faster than others, skip it and wait for others
                            continue

                        # Read the tar file until current_index >= target_index
                        sample, current_index = run_iterator_to_index(
                            iterator,
                            target_index,
                            sample_keys_full_to_index,
                            name=f"{url.sample_keys_full_list}.{url.keys[key_idx]}",
                        )
                        if sample is None:  # Iterator {key_idx} already reached the end, exit the for loop
                            if target_index < len(sample_keys_full_to_index):  # Missing keys
                                missing_info = f"index_path={url.sample_keys_full_list} | id={target_index}, sample_key={sample_keys_full_list[target_index]};"
                                log.info(
                                    f"[missing keys] found in tar file: data_key={url.keys[key_idx]} | {missing_info}",
                                    rank0_only=False,
                                )
                            sample_list = []  # Reset the sample_list
                            break

                        # Update the last_index
                        last_index[key_idx] = current_index

                        # Process sample dict
                        sample = process_sample(sample, url=url, key_idx=key_idx)

                        # Now check if the current index is matched or ahead
                        if current_index == target_index:  # Nice!
                            sample_list.append(sample)
                        elif current_index > target_index:
                            # This means there is missing keys in this tar, this tar is moving faster than others

                            # Log the missing info
                            missing_info = f"index_path={url.sample_keys_full_list} | "
                            for missing_idx in range(target_index, current_index):
                                missing_info += f" id={missing_idx}, sample_key={sample_keys_full_list[missing_idx]}; "
                            log.info(
                                f"[missing keys] found in tar file: data_key={url.keys[key_idx]} | {missing_info}",
                                rank0_only=False,
                            )

                            # Update the target_index to current_index, skip index inbetween old target_index and current_index
                            target_index = current_index

                            # Reset sample_list, save the sample from this tar into sample_list and wait for others
                            sample_list = [
                                sample
                            ]  # Attnetion: this will change the order of sample_list, we will put them in the right order later
                            skip_offset = True  # Skip the offset of target_index, since we are waiting for others
                            break
                        elif current_index < target_index:
                            # This should not happen
                            raise ValueError(
                                "Invalid output from run_iterator_to_index function. current_index should be equal or less than target_index"
                            )

                    # Decide where to yield the samples
                    if len(sample_list) == len(tar_file_iterator_list):
                        # Only yeild the samples if all the tars are preserved
                        all_prefix = [sample["fname"].split(".")[0] for sample in sample_list]
                        # Check all the prefix are the same
                        assert all(prefix == all_prefix[0] for prefix in all_prefix), (
                            f"prefixes are not the same: {all_prefix}"
                        )
                        # Correct the order of sample_list
                        sample_list = correct_order(sample_list, url.keys)
                        # Yield all the samples
                        for sample in sample_list:
                            assert isinstance(sample, dict) and "data" in sample and "fname" in sample
                            yield sample
                        sample_list = []  # Reset the sample_list
                    elif len(sample_list) > 1:
                        # Unexpected
                        raise ValueError(f"Unexpected length of sample_list: {len(sample_list)}")
                    elif len(sample_list) == 0 or len(sample_list) == 1:
                        # If the sample_list is empty, it means the tar file is exhausted
                        # If the sample_list has only one element, it means one tar file is moving faster than others
                        pass  # Do nothing

                    if not skip_offset:
                        # If sample_list has one element, we stay at current target_index until others catch up
                        target_index += 1  # Increase it by 1
                    if target_index == len(sample_keys_full_to_index):
                        break  # Reach the maximum index
                # Make sure all the iterator are closed
                for iterators in tar_file_iterator_list:
                    try:
                        next(iterators)
                    except StopIteration:
                        pass

        except Exception as exn:
            log.info(f"Got an exception while expanding tars - {exn}", rank0_only=False)
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


def correct_order(sample_list: list[Dict], expected_keys_order: list[str]) -> list[Dict]:
    """Make sure the order of samples are the same as the url.keys order."""
    data_keys_per_sample = [sample["fname"].split(".")[1] for sample in sample_list]
    expected_keys_order = [key.replace("/", "_") for key in expected_keys_order]
    if data_keys_per_sample == expected_keys_order:  # Correct order
        return sample_list
    # Order the sample_list based on the expected_keys_order
    sample_list_ordered = [None] * len(expected_keys_order)
    for data_key, sample in zip(data_keys_per_sample, sample_list):
        idx = expected_keys_order.index(data_key)
        sample_list_ordered[idx] = sample
    return sample_list_ordered


def load_func_parquet(buffer):
    data_list = pd.read_parquet(buffer).values.tolist()
    names = [data[0] for data in data_list]
    return names


def _read_sample_keys_full_list(key, s3_client: boto3.client, s3_bucket_name: str):
    with io.BytesIO() as buffer:
        s3_client.download_fileobj(Bucket=s3_bucket_name, Key=key, Fileobj=buffer)
        buffer.seek(0)
        sample_keys_full_list = load_func_parquet(buffer)
    sample_keys_full_list = [key.split(".")[0] for key in sample_keys_full_list]
    return sample_keys_full_list


def read_sample_keys_full_list(key: str, s3_client: boto3.client, s3_bucket_name: str, max_attempts=10):
    for attempt in range(max_attempts):
        try:
            return _read_sample_keys_full_list(key, s3_client, s3_bucket_name)
        except botocore.exceptions.ClientError as e:
            retry_interval = min(
                0.1 * 2**attempt + random.uniform(0, 1), 30
            )  # sleep workers randomly to avoid burst of requests
            log.exception(
                f"Failed to read sample_keys_full_list {key}, attempt {attempt}. {e}. Retrying after {retry_interval}s."
            )
            if attempt < max_attempts - 1:
                time.sleep(retry_interval)
    raise ConnectionError(f"Unable to read sample_keys_full_list {key} after {max_attempts} attempts.")


def run_iterator_to_index(iterator, target_index: int, sample_keys_full_to_index: dict, name: str = ""):
    """
    Iterates over samples from an iterator, checking against the index of current sample (current_index)
    to target_index, until it finds
    1) the sample key corresponds to the target index
    or 2) the target index is passed (i,e, the target keys are missing)
    or 3) until the iterator is exhausted.

    This function is designed to handle cases where there are unexpected, duplicated, or missing
    sample keys based on the index mapping provided.

    Args:
        iterator (iterator): An iterator yielding dictionaries that must include a key 'fname',
            which contains the filename. The filename should be in the format 'prefix.suffix',
            where 'prefix' will be used as the sample key.
        target_index (int): The index of the sample to be retrieved according to the dictionary
            mapping sample keys to indices.
        sample_keys_full_to_index (dict): A dictionary mapping sample keys (extracted from the
            'fname' prefix of the iterator's samples) to their respective indices. This mapping
            dictates the order in which samples are considered valid and should be found.
            e.g. {"name_0": 0, "name_1": 1, "name_2": 2}
        name (str): Names of the tar file, used to log the progress.

    Returns:
        tuple: A tuple containing:
            - sample (dict or None): The sample dictionary that matches the target index, or None
              if no such sample is found by the time the iterator is exhausted.
            - current_index (int or None): The index of the found sample according to the mapping,
              or None if no sample is found.

    Raises:
        StopIteration: If the iterator is exhausted without finding a matching sample, though this
        is caught internally and handled by returning None values.
    """
    sample, current_index = None, None
    skip_count = 0
    while True:
        try:
            sample = next(iterator)
            prefix, suffix = sample["fname"].split(".")
            sample_key = prefix

            if sample_key not in sample_keys_full_to_index:  # extra sample_key
                log.info(
                    f"Skipping ({skip_count}) unexpected key {sample_key}; not found in the sample_keys_full_to_index {name} {sample_keys_full_to_index.keys()}"
                )
                skip_count += 1
                continue
            current_index = sample_keys_full_to_index[sample_key]  # can be <,=,> target_index
            if current_index < target_index:
                # Note: current_index < target_index happens when duplicated keys or it's under catching up process
                # e.g.       [name_0, name_0, name_1] with target index = 1
                # Pointer at             ^
                # Current index is 0, which is less than target index 1
                # In this case, we keep iterating
                # log.info(f"[Skip] key {sample_key}; current_index={current_index} < target_index={target_index} {name}")
                continue
            elif current_index >= target_index:  # Note: current_index > targer_index happens when there is missing keys
                # Note: current_index > targer_index happens when there is missing keys
                # e.g.       [name_0, name_2, name_3] with target index 1
                # Pointer at             ^
                # Current index is 2, which is greater than target index 1
                # In this case, we return the current_index, set the target_index to 2 and tell other tars to catch up.
                # if current_index == target_index:  # Matched!
                #     log.info(f"[Pass!] current_index={current_index} == target_index={target_index}")
                # else:  # Missing keys
                #     log.info(f"[Missing key detected!] current_index={current_index} > target_index={target_index} {name}")
                break

        except StopIteration:
            sample = None
            current_index = None
            break
    return sample, current_index


def tarfile_samples(
    src: Iterable,
    handler: Callable = reraise_exception,
    load_from_object_store: bool = False,
    s3_client: Dict[str, boto3.client] = None,  # type: ignore
    s3_bucket_name: Optional[Dict[str, str]] = None,
    streaming_download: bool = True,
) -> Iterator[Dict]:
    r"""
    Given an iterator of filenames, this function opens the URL streams
    and groups data by keys.

    Args:
        src (Iterable): Iterator of TarSample.
        handler (Callable): Exception handler.
        load_from_object_store (bool): A boolean flag to specify whether to load from
            object store.
        s3_client (boto3.client): If loading from object store, specify S3 client.
        s3_bucket_name (str): If loading from object store, specify S3 bucket name.
        streaming_download(bool): If enabled, performs streaming download.
    """
    streams = url_opener(
        src,
        handler=handler,
        object_store=load_from_object_store,
        s3_client=s3_client,
        s3_bucket_name=s3_bucket_name,
        streaming_download=streaming_download,
    )
    files = tar_file_expander(streams, handler=handler, s3_client=s3_client, s3_bucket_name=s3_bucket_name)
    samples = group_by_keys(files, handler=handler)
    return samples


tarfile_to_samples = filters.pipelinefilter(tarfile_samples)


class WebDataset(DataPipeline, FluidInterface):
    r"""Webdataset class modified to support loading from object store."""

    def __init__(
        self,
        urls: list[TarSample],
        handler: Callable = reraise_exception,
        resampled: bool = False,
        shardshuffle: Optional[bool] = None,
        cache_size: int = -1,
        cache_dir: Optional[str] = None,
        detshuffle: bool = False,
        nodesplitter: Callable = shardlists.single_node_only,
        verbose: bool = False,
        load_from_object_store: bool = False,
        s3_client: Dict[str, boto3.client] = None,  # type: ignore
        s3_bucket_name: Optional[Dict[str, str]] = None,
        streaming_download: bool = True,
    ):
        r"""
        Args:
            urls (list[TarSample]): An iterator containing a list of url names.
            handler (Callable): Exception handler.
            resampled (bool): If true, sample shards from shard list with replacement.
            shardshuffle (bool): If true, shuffles the entire shard list.
            cache_size (int): Size of cache.
            cache_dir (str): Path to store cache.
            detshuffle (bool): Whether to use deterministic shuffling when shardshuffle is True.
            nodesplitter (Callable): Function for splitting urls among nodes.
            verbose (bool): If True, prints logs.
            load_from_object_store (bool): A boolean flag to specify whether to load from
                object store.
            s3_client (boto3.client): If loading from object store, specify S3 client.
            s3_bucket_name (str): If loading from object store, specify S3 bucket name.
            streaming_download (bool): Whether to do streaming download or full object download.
        """
        super().__init__()
        if isinstance(urls, IterableDataset):
            assert not resampled
            self.append(urls)
        elif isinstance(urls, str) and (urls.endswith(".yaml") or urls.endswith(".yml")):
            with open(urls) as stream:
                spec = yaml.safe_load(stream)
            assert "datasets" in spec
            self.append(shardlists.MultiShardSample(spec))
        elif isinstance(urls, dict):
            assert "datasets" in urls
            self.append(shardlists.MultiShardSample(urls))
        elif resampled:
            self.append(shardlists.ResampledShards(urls))
        else:
            self.append(shardlists.SimpleShardList(urls))
            self.append(nodesplitter)
            self.append(shardlists.split_by_worker)
            if shardshuffle is True:
                shardshuffle = 100  # type: ignore
            if shardshuffle is not None:
                if detshuffle:
                    self.append(filters.detshuffle(shardshuffle))
                else:
                    self.append(filters.shuffle(shardshuffle))
        if cache_dir is None or cache_size == 0:
            self.append(
                tarfile_to_samples(
                    handler=handler,
                    load_from_object_store=load_from_object_store,
                    s3_client=s3_client,
                    s3_bucket_name=s3_bucket_name,
                    streaming_download=streaming_download,
                )
            )
        else:
            # We dont use cache.
            assert cache_size == -1 or cache_size > 0
            self.append(
                cache.cached_tarfile_to_samples(
                    handler=handler,
                    verbose=verbose,
                    cache_size=cache_size,
                    cache_dir=cache_dir,
                )
            )
