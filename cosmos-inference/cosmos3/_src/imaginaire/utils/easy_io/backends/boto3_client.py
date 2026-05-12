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

import asyncio
import concurrent.futures
import io
import os
import time
from collections.abc import Generator
from math import ceil
from multiprocessing import shared_memory
from typing import Any, Optional

import boto3
import numpy as np
from botocore.config import Config as S3Config
from botocore.exceptions import ClientError

import cosmos3._src.imaginaire.utils.easy_io.backends.auto_auth as auto
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.env_parsers.cred_env_parser import CRED_ENVS

try:
    # pyrefly: ignore  # import-error
    import aioboto3

    # pyrefly: ignore  # import-error
    import aioboto3.session

    # pyrefly: ignore  # import-error
    from aiobotocore.config import AioConfig

    # pyrefly: ignore  # import-error
    from aiobotocore.session import AioSession
except ImportError:
    aioboto3 = None
    AioSession = None

MAX_RETRIES = 5
RETRY_DELAY = 1  # seconds


async def upload_single_part_async(
    s3: AioSession, bucket: str, key: str, part_number: int, data: bytes, upload_id: str
) -> dict[str, Any]:
    """
    Uploads a single part of a file asynchronously to S3.

    Args:
        s3 (S3): The S3 client.
        bucket (str): The S3 bucket name.
        key (str): The S3 key (file path).
        part_number (int): The part number of the upload.
        data (bytes): The data to upload.
        upload_id (str): The upload ID for the multipart upload.

    Returns:
        dict[str, Any]: A dictionary containing the part number and ETag.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = await s3.upload_part(
                Bucket=bucket, Key=key, PartNumber=part_number, UploadId=upload_id, Body=data
            )
            return {"PartNumber": part_number, "ETag": response["ETag"]}
        except (ClientError, asyncio.TimeoutError, Exception) as e:
            log.warning(f"Attempt {attempt + 1} failed for part {part_number}: {str(e)}", rank0_only=False)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))  # Exponential backoff
            else:
                log.error(f"Failed to upload part {part_number} after {MAX_RETRIES} attempts", rank0_only=False)
                raise


async def upload_parts_async(
    part_size: int,
    part_numbers: range,
    upload_id: str,
    data: bytes,
    bucket: str,
    key: str,
    client_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Uploads multiple parts of a file asynchronously to S3.

    Args:
        part_size (int): The size of each part in bytes.
        part_numbers (range): The range of part numbers to upload.
        upload_id (str): The upload ID for the multipart upload.
        data (bytes): The data to upload.
        bucket (str): The S3 bucket name.
        key (str): The S3 key (file path).
        client_config (dict[str, Any]): The S3 client configuration.

    Returns:
        list[dict[str, Any]]: A list of dictionaries containing part numbers and ETags.
    """
    session = aioboto3.Session()
    config = AioConfig(retries={"max_attempts": 3, "mode": "adaptive"}, connect_timeout=5, read_timeout=10)
    start_idx = part_numbers[0]
    async with session.client("s3", config=config, **client_config) as s3:
        tasks = []
        for part_number in part_numbers:
            start = (part_number - start_idx) * part_size
            end = min(start + part_size, len(data))
            part_data = data[start:end]
            tasks.append(upload_single_part_async(s3, bucket, key, part_number + 1, part_data, upload_id))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_parts = []
        failed_parts = []
        for part_number, result in enumerate(results, start=start_idx + 1):
            if isinstance(result, Exception):
                failed_parts.append(part_number)
            else:
                successful_parts.append(result)

        if failed_parts:
            log.error(f"Failed to upload parts: {failed_parts}", rank0_only=False)
            raise Exception(f"Failed to upload {len(failed_parts)} parts")

        successful_parts.sort(key=lambda part: part["PartNumber"])
        return successful_parts


def upload_parts_to_s3(args: tuple[range, str, int, bytes, str, str, dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Uploads parts of a file to S3 using a new event loop.

    Args:
        args (tuple[range, str, int, bytes, str, str, dict[str, Any]]): The arguments for uploading parts, including:
            part_numbers (range): The range of part numbers to upload.
            upload_id (str): The upload ID for the multipart upload.
            part_size (int): The size of each part in bytes.
            data (bytes): The data to upload.
            bucket (str): The S3 bucket name.
            key (str): The S3 key (file path).
            client_config (dict[str, Any]): The S3 client configuration.

    Returns:
        list[dict[str, Any]]: A list of dictionaries containing part numbers and ETags.
    """
    part_numbers, upload_id, part_size, data, bucket, key, client_config = args
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    parts = loop.run_until_complete(
        upload_parts_async(part_size, part_numbers, upload_id, data, bucket, key, client_config)
    )
    loop.close()
    return parts


async def download_single_part_async(
    s3, bucket: str, key: str, part_number: int, start: int, end: int, shm_name: str, part_size: int
) -> None:
    """
    Downloads a single part of a file asynchronously and writes it to shared memory.

    Args:
        s3 (S3): The S3 client.
        bucket (str): The S3 bucket name.
        key (str): The S3 key (file path).
        part_number (int): The part number.
        start (int): The start byte of the part.
        end (int): The end byte of the part.
        shm_name (str): The name of the shared memory block.
        part_size (int): The size of each part in bytes.
    """
    for attempt in range(MAX_RETRIES):
        try:
            range_header = f"bytes={start}-{end}"
            response = await s3.get_object(Bucket=bucket, Key=key, Range=range_header)
            data = await response["Body"].read()

            shm = shared_memory.SharedMemory(name=shm_name)
            offset = part_number * part_size
            shm.buf[offset : offset + len(data)] = data
            shm.close()
            return
        except (ClientError, asyncio.TimeoutError, Exception) as e:
            log.warning(f"Attempt {attempt + 1} failed for part {part_number}: {str(e)}", rank0_only=False)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))  # Exponential backoff
            else:
                log.error(f"Failed to download part {part_number} after {MAX_RETRIES} attempts", rank0_only=False)
                raise


async def download_parts_async(
    part_size: int, part_numbers: range, bucket: str, key: str, client_config: dict[str, Any], shm_name: str
) -> None:
    """
    Downloads multiple parts of a file asynchronously and writes them to shared memory.

    Args:
        part_size (int): The size of each part in bytes.
        part_numbers (range): The range of part numbers to download.
        bucket (str): The S3 bucket name.
        key (str): The S3 key (file path).
        client_config (dict[str, Any]): The S3 client configuration.
        shm_name (str): The name of the shared memory block.
    """
    session = aioboto3.Session()
    config = AioConfig(retries={"max_attempts": 5, "mode": "adaptive"}, connect_timeout=10, read_timeout=30)
    async with session.client("s3", config=config, **client_config) as s3:
        tasks = [
            download_single_part_async(
                s3,
                bucket,
                key,
                part_number,
                part_number * part_size,
                (part_number + 1) * part_size - 1,
                shm_name,
                part_size,
            )
            for part_number in part_numbers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failed_parts = [part for part, result in zip(part_numbers, results) if isinstance(result, Exception)]

        if failed_parts:
            log.error(f"Failed to download parts: {failed_parts}", rank0_only=False)
            raise Exception(f"Failed to download {len(failed_parts)} parts")


def download_parts_to_s3(args: tuple[range, int, str, str, dict[str, Any], str]) -> bytes:
    """
    Downloads parts of a file using a new event loop.

    Args:
        args (tuple[range, int, str, str, dict[str, Any]]): The arguments for downloading parts, including:
            part_numbers (range): The range of part numbers to download.
            part_size (int): The size of each part in bytes.
            bucket (str): The S3 bucket name.
            key (str): The S3 key (file path).
            client_config (dict[str, Any]): The S3 client configuration.

    Returns:
        bytes: The combined file data from all downloaded parts.
    """
    part_numbers, part_size, bucket, key, client_config, shm_name = args
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(download_parts_async(part_size, part_numbers, bucket, key, client_config, shm_name))
    loop.close()


class Boto3Client:
    """
    This class:

    - Provides higher-level S3 operations.
    - Serves as a wrapper around boto3.client in order to make boto3.client serializable.
        - It's required to use spawn method of creating DataLoader workers,
          which is in turn required to avoid segfaults when using Triton,
          e.g. for torch.compile or custom kernels.
    """

    def __init__(
        self,
        s3_credential_path: str,
        max_attempt: int = 3,
    ):
        self.max_attempt: int = max_attempt
        assert s3_credential_path, "s3_credential_path is required"
        assert os.path.exists(s3_credential_path) or CRED_ENVS.APP_ENV in [
            "prod",
            "dev",
            "stg",
        ], f"Credential file not found: {s3_credential_path}"

        # Keep track of S3 client constructor parameters so it can be recreated when pickling.
        with auto.open_auth(s3_credential_path, "r") as f:
            self._s3_cred_info = auto.json_load_auth(f)
        self._s3_config = S3Config(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},
            response_checksum_validation="when_required",
            request_checksum_calculation="when_required",
        )
        self._init_client()
        self._mc_kv_store = None

    def _init_client(self):
        """Initialize the S3 client."""
        self._client = boto3.client("s3", **self._s3_cred_info, config=self._s3_config)

    def __getstate__(self):
        state = self.__dict__.copy()
        # S3 client isn't pickleable.
        del state["_client"]
        return state

    def __setstate__(self, state: dict[str, Any]):
        self.__dict__.update(state)
        self._init_client()

    def size(self, filepath: str) -> int:
        filepath = self._check_path(filepath)

        if self._mc_kv_store and self._mc_kv_store.available:
            if self._mc_kv_store.has(filepath):
                return len(self._mc_kv_store.get(filepath))

        attempt: int = 0
        while attempt < self.max_attempt:
            try:
                return self._client.head_object(
                    Bucket=filepath.split("/")[0],
                    Key="/".join(filepath.split("/")[1:]),
                )["ContentLength"]
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    raise  # Object does not exist.
                else:
                    attempt += 1
                    log.error(f"Attempt {attempt} failed for {filepath}: {e}", rank0_only=False)
                    if attempt >= self.max_attempt:
                        raise  # Re-raise the exception after max attempt
                    time.sleep(2)  # Wait for 2 seconds before retrying
            except Exception as e:
                attempt += 1
                log.error(f"Attempt {attempt} failed for {filepath}: due to an unexpected error: {e}", rank0_only=False)
                if attempt >= self.max_attempt:
                    raise  # Re-raise the exception after max attempt
                time.sleep(2)  # Wait for 2 seconds before retrying

        raise ConnectionError("Unable to head {} from. {} attempts tried.".format(filepath, attempt))

    def get(self, filepath: str, offset: Optional[int] = None, size: Optional[int] = None) -> bytes:
        raw_filepath = filepath
        filepath = self._check_path(filepath)

        read_offset: Optional[int] = None
        read_size: Optional[int] = None
        byte_range: Optional[str] = None
        if offset is not None or size is not None:
            read_offset = offset or 0
            assert read_offset >= 0, "Read offset must be ≥ 0"

            # Try not to incur a remote call to get the file size. This can heavily slow down ranged reads.
            #
            # This means we won't always validate the read offset or read size against the file size.
            read_size = size or (self.size(filepath=raw_filepath) - read_offset)
            assert read_size >= 1, "Read size must be ≥ 1 or read offset must be < file size"

            byte_range = f"bytes={read_offset}-{read_offset + read_size - 1}"

        if self._mc_kv_store and self._mc_kv_store.available:
            if self._mc_kv_store.has(filepath):
                chunk: bytes = self._mc_kv_store.get(filepath)
                if read_offset is not None and read_size is not None:
                    return chunk[read_offset : read_offset + read_size]
                else:
                    return chunk

        attempt = 0
        while attempt < self.max_attempt:
            try:
                buffer = io.BytesIO()
                if byte_range is None:
                    self._client.download_fileobj(
                        Bucket=filepath.split("/")[0],
                        Key="/".join(filepath.split("/")[1:]),
                        Fileobj=buffer,
                    )
                else:
                    # The boto S3 Transfer Manager doesn't support ranged reads yet.
                    #
                    # https://github.com/boto/boto3/issues/1215
                    # https://github.com/boto/s3transfer/issues/248
                    resp = self._client.get_object(
                        Bucket=filepath.split("/")[0],
                        Key="/".join(filepath.split("/")[1:]),
                        Range=byte_range,
                    )
                    buffer.write(resp["Body"].read())
                buffer.seek(0)
                # Only cache full reads.
                if byte_range is None:
                    if self._mc_kv_store and self._mc_kv_store.available:
                        self._mc_kv_store.put(filepath, buffer.read())
                        buffer.seek(0)

                return buffer.read()
            except Exception as e:
                attempt += 1
                log.error(f"Got an exception: attempt={attempt} - {e} - {filepath}", rank0_only=False)

        raise ConnectionError("Unable to read {} from. {} attempts tried.".format(filepath, attempt))

    def put(self, obj, filepath):
        filepath = self._check_path(filepath)
        bucket_name = filepath.split("/")[0]
        key = "/".join(filepath.split("/")[1:])
        attempt = 0
        while attempt < self.max_attempt:
            try:
                # If obj is a string path to a local file, use upload_file instead
                if isinstance(obj, str) and os.path.isfile(obj):
                    self._client.upload_file(Filename=obj, Bucket=bucket_name, Key=key)
                    return
                if isinstance(obj, io.BytesIO):
                    obj.seek(0)
                    self._client.upload_fileobj(obj, Bucket=bucket_name, Key=key)
                    return
                if isinstance(obj, bytes):
                    self._client.put_object(Body=obj, Bucket=bucket_name, Key=key)
                    return
                else:
                    raise ValueError("Unsupported object type for upload")
            except ClientError as e:
                attempt += 1
                log.error(f"Got an exception: attempt={attempt} - {e} - {filepath}", rank0_only=False)

        raise ConnectionError("Unable to write {} to. {} attempts tried.".format(filepath, attempt))

    def fast_put(self, obj, filepath, num_processes: int = 32):
        assert aioboto3 is not None, "aioboto3 is required for fast_put"
        original_filepath = filepath
        filepath = self._check_path(filepath)
        bucket = filepath.split("/")[0]
        key = "/".join(filepath.split("/")[1:])
        part_size = 16 * 1024 * 1024  # 16 MB part size

        if isinstance(obj, bytes):
            data = obj
        elif isinstance(obj, str) and os.path.isfile(obj):
            with open(obj, "rb") as f:
                data = f.read()
        elif isinstance(obj, io.BytesIO):
            obj.seek(0)
            data = obj.read()
        else:
            raise ValueError("Unsupported object type for upload")

        file_size = len(data)
        if file_size <= part_size * num_processes:
            return self.put(data, original_filepath)
        num_parts = ceil(file_size / part_size)
        upload_id = self._client.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]

        part_numbers = np.array_split(np.arange(num_parts), num_processes)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            args = []
            for i in range(num_processes):
                cur_parts = part_numbers[i].tolist()
                cur_data = data[cur_parts[0] * part_size : min(cur_parts[-1] * part_size + part_size, file_size)]
                args.append((cur_parts, upload_id, part_size, cur_data, bucket, key, self._s3_cred_info))
            results = executor.map(upload_parts_to_s3, args)
            parts = []
            for result in results:
                parts.extend(result)

        parts = sorted(parts, key=lambda part: part["PartNumber"])
        self._client.complete_multipart_upload(
            Bucket=bucket, Key=key, UploadId=upload_id, MultipartUpload={"Parts": parts}
        )

    def contains(self, filepath: str, max_retries=10) -> bool:
        """
        Checks if the specified object exists in the S3 bucket with retry logic for errors.

        Args:
            filepath (str): The s3 path of the file to check, must start with "s3://".

        Returns:
            bool: True if the object exists in the S3 bucket, False otherwise.

        Raises:
            ClientError: If an error response other than "404 Not Found" is returned from the S3 service.
        """
        filepath = self._check_path(filepath)
        bucket = filepath.split("/")[0]
        key = "/".join(filepath.split("/")[1:])

        retries = 0
        while retries < max_retries:
            try:
                # Try to check if the object exists
                self._client.head_object(Bucket=bucket, Key=key)
                return True  # Object exists
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return False  # Object does not exist
                else:
                    retries += 1
                    print(f"Attempt {retries} failed with error: {e}")
                    if retries >= max_retries:
                        raise  # Re-raise the exception if max retries are reached
                    time.sleep(2)  # Wait for 2 seconds before retrying
            except Exception as e:
                retries += 1
                print(f"Attempt {retries} failed due to an unexpected error: {e}")
                if retries >= max_retries:
                    raise  # Re-raise the exception if max retries are reached
                time.sleep(2)  # Wait for 2 seconds before retrying

    def isdir(self, filepath: str, max_retries=10) -> bool:
        """
        Determines if the specified path corresponds to a directory in S3 with retry logic.

        A directory in S3 is implied if there are any objects stored with the given prefix,
        which means this function checks for the existence of any objects at or under the specified path.

        Args:
            filepath (str): The s3 path to check, must start with "s3://".

        Returns:
            bool: True if the specified path corresponds to a directory in S3, False otherwise.
                Directories in S3 are not physical entities but are implied by object keys.

        Raises:
            ClientError: An error from the S3 API that isn't related to the absence of the directory
                        (logged but not raised further).
        """
        filepath = self._check_path(filepath)
        if not filepath.endswith("/"):
            filepath += "/"

        bucket = filepath.split("/")[0]
        prefix = "/".join(filepath.split("/")[1:])

        retries = 0
        while retries < max_retries:
            try:
                # Try to check if any objects exist with the given prefix (i.e., directory in S3)
                resp = self._client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/", MaxKeys=1)
                # Check if any content or prefixes exist under the given path
                return "CommonPrefixes" in resp or "Contents" in resp
            except ClientError as e:
                retries += 1
                log.error(f"Attempt {retries} failed: {e}", rank0_only=False)
                if retries >= max_retries:
                    return False  # Return False if maximum retries are reached
                time.sleep(2)  # Wait for 2 seconds before retrying
            except Exception as e:
                retries += 1
                log.error(f"Attempt {retries} failed due to an unexpected error: {e}", rank0_only=False)
                if retries >= max_retries:
                    return False  # Return False if maximum retries are reached
                time.sleep(2)  # Wait for 2 seconds before retrying

    def delete(self, filepath):
        filepath = self._check_path(filepath)
        self._client.delete_object(Bucket=filepath.split("/")[0], Key="/".join(filepath.split("/")[1:]))

    def ls_dir(self, filepath: str) -> Generator[str, None, None]:
        """
        List all folders in an S3 bucket with a given prefix.

        Args:
            filepath (str): The S3 path of the folder to list.

        Yields:
            str: The keys of the folders in the S3 bucket.
        """
        filepath = self._check_path(filepath)
        bucket = filepath.split("/")[0]
        prefix = "/".join(filepath.split("/")[1:])
        continuation_token = None
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        while True:
            if continuation_token:
                resp = self._client.list_objects_v2(
                    Bucket=bucket, Prefix=prefix, Delimiter="/", ContinuationToken=continuation_token
                )
            else:
                resp = self._client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")

            if "CommonPrefixes" in resp:
                for item in resp["CommonPrefixes"]:
                    yield item["Prefix"][len(prefix) :]

            # Check if there are more keys to retrieve
            if resp.get("IsTruncated"):  # If IsTruncated is True, there are more keys
                continuation_token = resp.get("NextContinuationToken")
            else:
                break

    def list(self, filepath: str, exclude_prefix: Optional[str] = None) -> Generator[str, None, None]:
        """
        List all keys in an S3 bucket with a given prefix, excluding files that start with
        specified prefix.

        Args:
            filepath (str): The S3 path of the file to list.
            exclude_prefix (str): Files starting with this prefix will be excluded from results.
                                Defaults to "real".

        Yields:
            str: The keys of the files in the S3 bucket that don't start with exclude_prefix.
        """
        filepath = self._check_path(filepath)
        bucket = filepath.split("/")[0]
        prefix = "/".join(filepath.split("/")[1:])

        continuation_token = None

        while True:
            if continuation_token:
                resp = self._client.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=continuation_token)
            else:
                resp = self._client.list_objects_v2(Bucket=bucket, Prefix=prefix)

            if "Contents" in resp:
                for item in resp["Contents"]:
                    key = item["Key"][len(prefix) :]
                    # Skip files that start with the excluded prefix
                    if exclude_prefix is None or not key.startswith(exclude_prefix):
                        yield key

            # Check if there are more keys to retrieve
            if resp.get("IsTruncated"):  # If IsTruncated is True, there are more keys
                continuation_token = resp.get("NextContinuationToken")
            else:
                break

    def _check_path(self, filepath: str):
        assert filepath.startswith("s3://")
        filepath = filepath[5:]
        return filepath
