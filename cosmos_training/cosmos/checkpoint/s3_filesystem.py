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
import time
from contextlib import contextmanager
from typing import Generator, Union
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.filesystem import FileSystemBase

from cosmos.utils import log
from cosmos.utils.easy_io import easy_io


class S3Stream(io.BytesIO):
    """
    Workaround for PyTorch manually closing the stream before we can upload it to S3. We override the close() as noop
    and instead call our own _true_close() method to close the stream after we are done using it.
    The commit at fault is https://github.com/pytorch/pytorch/commit/9c909bf3bb122db2cce95e2eb7459bbe50dfa15a
    """

    def close(self):
        self.flush()
        # No close

    def _true_close(self):
        super().close()


class S3FileSystem(FileSystemBase):
    """Implementation of FileSystemBase for AWS S3 storage."""

    def __init__(
        self,
        credential_path: str,
        max_attempts: int = 20,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        backoff_factor: float = 2.0,
        enable_gcs_patch_in_boto3: bool = False,
    ) -> None:
        """
        Initialize S3FileSystem with retry configuration.

        Args:
            credential_path: Path to AWS credentials JSON file
            max_attempts: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            backoff_factor: Multiplicative factor for backoff time
            enable_gcs_patch_in_boto3: Whether to enable GCS patch in boto3
        """
        self.easy_io_backend = easy_io.get_file_backend(
            backend_args={
                "backend": "s3",
                "s3_credential_path": credential_path,
                "path_mapping": None,
            }
        )
        self.max_attempts = max_attempts
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_factor = backoff_factor
        self.enable_gcs_patch_in_boto3 = enable_gcs_patch_in_boto3
        if enable_gcs_patch_in_boto3:
            log.info("enable_gcs_patch_in_boto3: True")

    def _retry_with_backoff(self, operation_func, *args, **kwargs):
        """
        Execute an operation with exponential backoff retry logic.

        Args:
            operation_func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the operation function

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        backoff = self.initial_backoff

        for attempt in range(self.max_attempts):
            try:
                return operation_func(*args, **kwargs)
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                log.info(f"S3 Filesystem: Received ClientError: {error_code}", rank0_only=False)

                # Handle specific error cases
                if error_code in ["SlowDown", "ThrottlingException", "RequestLimitExceeded", "InternalError"]:
                    last_exception = e
                    if attempt < self.max_attempts - 1:  # Don't sleep on last attempt
                        current_backoff = min(backoff, self.max_backoff)
                        log.info(f"S3 Filesystem: Retrying in {current_backoff} seconds", rank0_only=False)
                        time.sleep(current_backoff)
                        backoff *= self.backoff_factor
                        continue
                # For other client errors, raise immediately
                raise
            except Exception as e:
                log.info(f"S3 Filesystem: Received Exception: {str(e)}", rank0_only=False)
                last_exception = e
                if attempt < self.max_attempts - 1:
                    current_backoff = min(backoff, self.max_backoff)
                    log.info(f"S3 Filesystem: Retrying in {current_backoff} seconds", rank0_only=False)
                    time.sleep(current_backoff)
                    backoff *= self.backoff_factor
                    continue

        # pyrefly: ignore [bad-raise]
        raise last_exception

    @contextmanager
    def create_stream(self, path: Union[str, os.PathLike], mode: str) -> Generator[io.IOBase, None, None]:
        """Create a stream for reading from or writing to S3 with retry logic."""
        path_str = str(path)
        bucket, key = self._parse_s3_uri(path_str)
        log.info(f"S3 Filesystem: Creating stream for {key} in bucket {bucket}", rank0_only=False)

        if mode == "rb":
            stream = io.BytesIO()
            try:

                def download_operation():
                    stream.write(self.easy_io_backend.get(filepath=path_str))
                    stream.seek(0)

                log.info(f"S3 Filesystem: Downloading {key} from bucket {bucket}", rank0_only=False)
                self._retry_with_backoff(download_operation)
                log.info("S3 Filesystem: Download complete", rank0_only=False)
                yield stream
            finally:
                stream.close()
        elif mode == "wb":
            stream = S3Stream()
            try:
                yield stream

                def upload_operation():
                    stream.seek(0)
                    self.easy_io_backend.put(obj=stream, filepath=path_str)

                log.info(f"S3 Filesystem: Uploading {key} to bucket {bucket}", rank0_only=False)
                self._retry_with_backoff(upload_operation)
                log.info("S3 Filesystem: Upload complete", rank0_only=False)
            finally:
                stream._true_close()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def concat_path(self, path: Union[str, os.PathLike], suffix: str) -> Union[str, os.PathLike]:
        """Concatenate S3 path with suffix."""
        path_str = str(path)
        if path_str.endswith("/"):
            return f"{path_str}{suffix}"
        return f"{path_str}/{suffix}"

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        """Initialize and validate S3 path."""
        path_str = str(path)
        if not path_str.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {path_str}. Must start with 's3://'")
        return path_str

    def rename(self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]) -> None:
        """Rename (move) an object in S3 with retry logic."""
        src_path = str(path)
        dst_path = str(new_path)

        def copy_operation():
            self.easy_io_backend.copyfile(src=src_path, dst=dst_path)

        self._retry_with_backoff(copy_operation)

        def delete_operation():
            self.easy_io_backend.remove(filepath=src_path)

        self._retry_with_backoff(delete_operation)

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        """
        Create a "directory" in S3.

        Note: S3 doesn't have real directories, but we can create an empty object
        with a trailing slash to simulate a directory.
        """
        # Creating same buckets from different ranks can cause rate limit issues in GCP.
        # In object store, we don't need to create a directory.
        pass

    def ls(self, path: Union[str, os.PathLike]) -> list[str]:
        """List objects under the given S3 path (prefix) and return s3:// URIs."""
        path_str = str(path)
        return [
            f"{path_str.removesuffix('/')}/{obj_suffix}"
            for obj_suffix in self.easy_io_backend.list_dir_or_file(dir_path=path_str, list_dir=False, list_file=True)
        ]

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """Validate if the checkpoint_id is a valid S3 URI."""
        checkpoint_id_str = str(checkpoint_id)
        try:
            if not checkpoint_id_str.startswith("s3://"):
                return False
            parsed = urlparse(checkpoint_id_str)
            return bool(parsed.netloc and parsed.path)  # Must have bucket and key
        except Exception:
            return False

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        """Check if an object exists in S3 with retry logic."""
        try:

            def head_operation() -> bool:
                return self.easy_io_backend.exists(filepath=str(path))

            return self._retry_with_backoff(head_operation)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") == "404":
                return False
            raise

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        """Remove a file from S3 with retry logic."""

        def delete_operation():
            self.easy_io_backend.remove(filepath=str(path))

        self._retry_with_backoff(delete_operation)

    def _parse_s3_uri(self, uri: str) -> tuple[str, str]:
        """
        Parse an S3 URI into bucket and key.

        Args:
            uri: S3 URI in the format s3://bucket-name/key

        Returns:
            Tuple of (bucket_name, key)

        Raises:
            ValueError: If the URI is invalid
        """
        uri = uri if isinstance(uri, str) else str(uri)
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}. Must start with 's3://'")

        parsed = urlparse(uri)
        bucket = parsed.netloc

        # Remove leading slash from key
        key = parsed.path.lstrip("/")

        if not bucket:
            raise ValueError(f"Invalid S3 URI: {uri}. No bucket specified")

        return bucket, key


class S3StorageWriter(FileSystemWriter):
    def __init__(
        self,
        credential_path: str,
        path: str,
        enable_gcs_patch_in_boto3: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize an S3 writer for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (str): The S3 URI to write checkpoints to.
            kwargs (dict): Keyword arguments to pass to the parent :class:`FileSystemWriter`.
            enable_gcs_patch_in_boto3 (bool): Whether to enable GCS patch in boto3
        """
        super().__init__(
            path=path,
            sync_files=False,
            **kwargs,
        )
        self.fs = S3FileSystem(credential_path, enable_gcs_patch_in_boto3=enable_gcs_patch_in_boto3)  # type: ignore
        self.path = self.fs.init_path(path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)


class S3StorageReader(FileSystemReader):
    def __init__(
        self, credential_path: str, path: Union[str, os.PathLike], enable_gcs_patch_in_boto3: bool = False
    ) -> None:
        """
        Initialize an S3 reader for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (Union[str, os.PathLike]): The S3 path to read checkpoints from.
            enable_gcs_patch_in_boto3 (bool): Whether to enable GCS patch in boto3
        """
        super().__init__(path)
        self.fs = S3FileSystem(credential_path, enable_gcs_patch_in_boto3=enable_gcs_patch_in_boto3)  # type: ignore
        self.path = self.fs.init_path(path)
        self.sync_files = False

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)
