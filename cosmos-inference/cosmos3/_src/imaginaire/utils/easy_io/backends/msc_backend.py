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

import copy
import io
import os
import re
import tempfile
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from shutil import SameFileError
from typing import Any, Optional, Union
from urllib.parse import urlparse

import yaml
from multistorageclient import StorageClient, StorageClientConfig
from multistorageclient.types import Range

import cosmos3._src.imaginaire.utils.easy_io.backends.auto_auth as auto
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.easy_io.backends.base_backend import BaseStorageBackend, mkdir_or_exist

# {scheme}://
_URL_PREFIX_REGEX = r"[a-zA-Z0-9+.-]*:\/\/"


def _get_telemetry_config_from_msc_secret() -> Optional[dict[str, Any]]:
    """Generate MSC telemetry configuration from credentials/msc.secret file if available.

    Reads OpenTelemetry configuration from the ``credentials/msc.secret`` YAML file.
    The file should contain an ``msc.opentelemetry`` section with the required fields
    (vault_endpoint, vault_namespace, approle_id, approle_secret, mount_point, secret_path).
    The endpoint field is optional and defaults to the production OTLP endpoint.

    Returns:
        Optional[dict]: OpenTelemetry configuration dictionary if file exists and contains
            required fields, None otherwise.
    """
    msc_secret_path = Path("credentials/msc.secret")

    if not msc_secret_path.exists():
        log.debug(f"MSC secret file not found at {msc_secret_path}", rank0_only=True)
        return None

    try:
        with open(msc_secret_path, "r") as f:
            msc_config = yaml.safe_load(f)

        if not msc_config or not isinstance(msc_config, dict):
            log.warning(f"Invalid MSC secret file format at {msc_secret_path}", rank0_only=True)
            return None

        msc_section = msc_config.get("msc", {})
        opentelemetry_section = msc_section.get("opentelemetry", {})

        required_fields = (
            "vault_endpoint",
            "vault_namespace",
            "approle_id",
            "approle_secret",
            "mount_point",
            "secret_path",
        )
        missing = [f for f in required_fields if not opentelemetry_section.get(f)]
        if missing:
            log.warning(
                f"MSC secret file at {msc_secret_path} missing required fields: {', '.join(missing)}",
                rank0_only=True,
            )
            return None

        vault_endpoint = opentelemetry_section["vault_endpoint"]
        vault_namespace = opentelemetry_section["vault_namespace"]
        approle_id = opentelemetry_section["approle_id"]
        approle_secret = opentelemetry_section["approle_secret"]
        mount_point = opentelemetry_section["mount_point"]
        secret_path = opentelemetry_section["secret_path"]
        cert_key = opentelemetry_section["cert_key"]
        key_key = opentelemetry_section["key_key"]
        ca_key = opentelemetry_section["ca_key"]
        endpoint = opentelemetry_section["endpoint"]
    except Exception as e:
        log.warning(f"Failed to load MSC secret file at {msc_secret_path}: {e}", rank0_only=True)
        return None

    # Construct OpenTelemetry configuration dictionary.
    opentelemetry_config = {
        "opentelemetry": {
            "metrics": {
                "attributes": [
                    # All environments.
                    {"type": "static", "options": {"attributes": {"msc.ppp": "COSMOS", "msc.job": "unknown"}}},
                    {"type": "host", "options": {"attributes": {"msc.cluster": "name", "msc.node": "name"}}},
                    {"type": "process", "options": {"attributes": {"msc.process": "pid"}}},
                    {
                        "type": "msc_config",
                        "options": {
                            "attributes": {
                                "msc.approle_id": {
                                    "expression": "opentelemetry.metrics.exporter.options.auth.approle_id"
                                },
                                "msc.approle_secret": {
                                    "expression": (
                                        "hash('sha3-224', opentelemetry.metrics.exporter.options.auth.approle_secret)"
                                    )
                                },
                            }
                        },
                    },
                    # Progressive enhancement for Lepton environments.
                    #
                    # https://docs.nvidia.com/dgx-cloud/lepton/features/batch-jobs/predefined-env-vars
                    {
                        "type": "environment_variables",
                        "options": {
                            "attributes": {
                                "msc.job": "LEPTON_JOB_NAME",
                                "msc.job_user": "LEPTON_USERID",
                                "msc.job_nodes": "LEPTON_JOB_TOTAL_WORKERS",
                                "msc.cluster": "LEPTON_WORKER_CLUSTER_NAME",
                                "msc.node": "LEPTON_WORKER_ID",
                                "msc.node_gpus": "LEPTON_RESOURCE_ACCELERATOR_NUM",
                            }
                        },
                    },
                    # Progressive enhancement for Slurm environments.
                    #
                    # https://slurm.schedmd.com/prolog_epilog.html#environment_variables
                    {
                        "type": "environment_variables",
                        "options": {
                            "attributes": {
                                "msc.ppp": "SLURM_JOB_ACCOUNT",
                                "msc.job": "SLURM_JOB_ID",
                                "msc.job_user": "SLURM_JOB_USER",
                                "msc.job_nodes": "SLURM_JOB_NUM_NODES",
                                "msc.job_gpus": "SLURM_GPUS",
                                "msc.cluster": "SLURM_CLUSTER_NAME",
                                "msc.node": "SLURMD_NODENAME",
                                "msc.node_gpus": "SLURM_GPUS_ON_NODE",
                                "msc.slurm_job_partition": "SLURM_JOB_PARTITION",
                            }
                        },
                    },
                ],
                "reader": {
                    "options": {
                        # ≤ 100 Hz collect frequency.
                        "collect_interval_millis": 10,
                        "collect_timeout_millis": 100,
                        # ≤ 1 Hz export frequency.
                        "export_interval_millis": 1000,
                        "export_timeout_millis": 500,
                    }
                },
                "exporter": {
                    "type": "_otlp_mtls_vault",
                    "options": {
                        "exporter": {"endpoint": endpoint},
                        "auth": {
                            "vault_endpoint": vault_endpoint,
                            "vault_namespace": vault_namespace,
                            "approle_id": approle_id,
                            "approle_secret": approle_secret,
                            "mount_point": mount_point,
                            "secret_path": secret_path,
                            "cert_key": cert_key,
                            "key_key": key_key,
                            "ca_key": ca_key,
                        },
                    },
                },
            },
        }
    }

    return opentelemetry_config


class MSCBackend(BaseStorageBackend):
    """Multi-Storage Client (MSC) backend.

    Uses MSC storage clients instead of MSC shortcuts.

    URL file paths (e.g. 's3://path/of/file') are handled transparently. Using URL file paths
    as input will return URL file path outputs when appropriate to match Boto3Backend behavior.

    **If using URL file paths, the storage provider's base path option must be empty!**

    Get/put concurrency can be set for certain providers in the MSC configuration file.

    Examples:
        >>> backend = MSCBackend()
        >>> filepath = "path/of/file"  # or "s3://path/of/file"
        >>> backend.get(filepath)
    """

    _storage_client: StorageClient
    _path_mapping: dict[str, str]

    def __init__(
        self,
        config_path: Optional[str] = "credentials/msc_config.yaml",
        profile: Optional[str] = None,
        s3_credential_path: Optional[str] = None,
        path_mapping: Optional[dict[str, str]] = None,
    ):
        """Initialize a backend.

        Args:
            config_path (str, optional): MSC config path (e.g. ``credentials/msc_config.yaml``).
            profile (str, optional): MSC profile from the MSC config to use.
                Mutually exclusive with ``s3_credential_path``.
            s3_credential_path (str, optional): Legacy Boto3 config path (e.g. ``credentials/s3_training.secret``).
                Translated into an MSC profile that's merged with the MSC config at ``config_path`` with:

                - The profile name set to ``s3_credential_path`` verbatim.
                - The storage and credentials provider types determined by the file contents.

                Mutually exclusive with ``profile``.
            path_mapping (dict, optional): Path mapping dict from src path to dst path.
                When ``path_mapping={'src': 'dst'}``, ``src`` in ``filepath`` will be replaced by ``dst``.
                Doesn't apply to the local path in ``copy{file,tree}_{from,to}_local`` methods.
        """
        if all(_ is None for _ in (profile, s3_credential_path)) or all(
            _ is not None for _ in (profile, s3_credential_path)
        ):
            raise ValueError("Must specify exactly one of profile or s3_credential_path")

        msc_config_dict: dict[str, Any] = {}

        # Use an existing MSC config file as the base MSC config.
        if config_path is not None:
            config_dict, _ = StorageClientConfig.read_msc_config(config_file_paths=[config_path])
            if config_dict is None:
                log.info(f"No MSC config at {config_path}, using empty base MSC config", rank0_only=True)
            else:
                msc_config_dict = config_dict

        # Create an MSC profile from the legacy Boto3 config.
        if s3_credential_path is not None:
            with auto.open_auth(s3_credential_path, "r") as unloaded_legacy_boto3_config:
                legacy_boto3_config = auto.json_load_auth(unloaded_legacy_boto3_config)
                if len(legacy_boto3_config) > 0:
                    profile = s3_credential_path

                    # Merge with any existing profiles.
                    msc_config_dict["profiles"] = msc_config_dict.get("profiles", {})
                    # Merge with the existing profile, replacing `storage_provider` and `credentials_provider` completely.
                    msc_config_dict["profiles"][profile] = msc_config_dict["profiles"].get(profile, {})

                    storage_provider_type: str = "s3"
                    parsed_endpoint_url = urlparse(legacy_boto3_config["endpoint_url"])
                    # Handle regional SwiftStack endpoints.
                    if parsed_endpoint_url.hostname.endswith(".s8k.io"):
                        storage_provider_type = "s8k"
                    # Handle global and regional GCS endpoints.
                    elif parsed_endpoint_url.hostname.startswith("storage.") and parsed_endpoint_url.hostname.endswith(
                        ".googleapis.com"
                    ):
                        storage_provider_type = "gcs_s3"

                    msc_config_dict["profiles"][profile]["storage_provider"] = {
                        "type": storage_provider_type,
                        "options": {
                            "base_path": "",
                            "endpoint_url": legacy_boto3_config["endpoint_url"],
                            "region_name": legacy_boto3_config["region_name"],
                        },
                    }

                    if all(_ in legacy_boto3_config for _ in ("aws_access_key_id", "aws_secret_access_key")):
                        msc_config_dict["profiles"][profile]["credentials_provider"] = {
                            "type": "S3Credentials",
                            "options": {
                                "access_key": legacy_boto3_config["aws_access_key_id"],
                                "secret_key": legacy_boto3_config["aws_secret_access_key"],
                            },
                        }
                else:
                    raise ValueError("Cannot create profile from empty legacy Boto3 config")

        assert profile is not None, "Failed to resolve MSC profile"

        # Add OpenTelemetry configuration if credentials/msc.secret file is provided.
        otel_config = _get_telemetry_config_from_msc_secret()
        if otel_config:
            msc_config_dict.update(otel_config)
            log.debug("MSC Observability is configured from credentials/msc.secret", rank0_only=True)
        else:
            log.debug(
                "MSC Observability is not configured (credentials/msc.secret not found or invalid)", rank0_only=True
            )

        # easy_io needs backend args to be JSON-serializable for backend instance cache keys.
        #
        # StorageClientConfig isn't, so we need to construct it here instead of receiving one.
        self._storage_client = StorageClient(
            config=StorageClientConfig.from_dict(config_dict=msc_config_dict, profile=profile)
        )

        assert isinstance(path_mapping, dict) or path_mapping is None
        # Make a deep copy of the path mapping to prevent external mutation.
        self._path_mapping = {} if path_mapping is None else copy.deepcopy(path_mapping)
        for src, dst in self._path_mapping.items():
            log.info(f"Path mapping: {src} -> {dst}", rank0_only=False)

    def _translate_filepath(self, filepath: Union[str, Path], translate_url: bool = True) -> str:
        """Translate a `filepath` to a string.

        Paths are of the form 'path/to/file' (path form) or '{protocol}://path/to/file' (URL form).

        Args:
            filepath (str): File path to be translated.
            translate_url (bool): Strip '{scheme}://' prefixes. Needed for paths passed directly to MSC storage clients.
        """
        assert isinstance(filepath, (str, Path))

        # Change to a POSIX path string.
        if isinstance(filepath, str):
            # If the ``filepath`` is concatenated by ``os.path.join`` in a Windows
            # environment, the ``filepath`` will be the format of 'prefix\file.txt'.
            filepath = re.sub(r"\\+", "/", filepath)
        elif isinstance(filepath, Path):
            # These should only be filesystem paths (e.g. '/path/of/file').
            # URL paths (e.g. ``Path('s3://profile/path/of/file')``) collapse '://' to ':/'.
            filepath = filepath.as_posix()
        else:
            raise ValueError(f"Unhandled filepath type: {type(filepath)}")

        # Remap path.
        #
        # If there's multiple matching srcs, use the longest src (i.e. the most specific).
        longest_src: str = ""
        for src in self._path_mapping.keys():
            if filepath.startswith(src) and len(src) > len(longest_src):
                longest_src = src
        if len(longest_src) > 0:
            filepath = filepath.replace(longest_src, self._path_mapping[longest_src], 1)

        # Optionally strip URL prefix then return.
        #
        # Don't use urlparse in case filepath is an invalid URL.
        return re.sub(rf"^{_URL_PREFIX_REGEX}", "", filepath) if translate_url else filepath

    def size(self, filepath: Union[str, Path]) -> int:
        """Get the file size in bytes for a given ``filepath``.

        Args:
            filepath (str or Path): Path to get file size in bytes.

        Returns:
            int: File size in bytes for filepath.

        Examples:
            >>> backend = MSCBackend()
            >>> filepath = "path/of/file"  # or "s3://path/of/file"
            >>> backend.size(filepath)  # file containing "hello world"
            11
        """
        path = self._translate_filepath(filepath=filepath)
        return self._storage_client.info(path=path, strict=False).content_length

    def get(self, filepath: Union[str, Path], offset: Optional[int] = None, size: Optional[int] = None) -> bytes:
        """Read bytes from a given ``filepath`` with 'rb' mode in range [offset, offset + size).

        Args:
            filepath (str or Path): Path to read data.
            offset (int, optional): Read offset in bytes (0-index). Defaults to 0.
            size (int, optional): Read size in bytes. Defaults to the file size.

        Returns:
            bytes: Return bytes read from filepath.

        Examples:
            >>> backend = MSCBackend()
            >>> filepath = "path/of/file"  # or "s3://path/of/file"
            >>> backend.get(filepath)
            b'hello world'
        """
        path = self._translate_filepath(filepath=filepath)
        byte_range: Optional[Range] = None
        if offset is not None or size is not None:
            read_offset = offset or 0
            assert read_offset >= 0, "Read offset must be ≥ 0"

            # Try not to incur a remote call to get the file size. This can heavily slow down ranged reads.
            #
            # This means we won't always validate the read offset or read size against the file size.
            read_size = size or (self.size(filepath=filepath) - read_offset)
            assert read_size >= 1, "Read size must be ≥ 1 or read offset must be < file size"

            byte_range = Range(offset=read_offset, size=read_size)

        if byte_range is None:
            buffer = io.BytesIO()
            # `StorageClient.read()` defers to `StorageProvider.get_object()` while
            # `StorageClient.download_file()` defers to `StorageProvider.download_file()`.
            #
            # Currently, only `StorageProvider.download_file()` supports parallel downloads
            # in some storage providers (e.g. boto S3 transfer manager for S3 storage providers)
            # so it's often much faster.
            self._storage_client.download_file(remote_path=path, local_path=buffer)
            buffer.seek(0)
            return buffer.read()
        else:
            return self._storage_client.read(path=path, byte_range=byte_range)

    def get_text(
        self,
        filepath: Union[str, Path],
        encoding: str = "utf-8",
    ) -> str:
        """Read text from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.

        Examples:
            >>> backend = MSCBackend()
            >>> filepath = "path/of/file"  # or "s3://path/of/file"
            >>> backend.get_text(filepath)
            'hello world'
        """
        return str(self.get(filepath=filepath), encoding=encoding)

    def put(self, obj: Union[bytes, io.BytesIO], filepath: Union[str, Path]) -> None:
        """Write bytes to a given ``filepath``.

        Args:
            obj (bytes): Data to be saved.
            filepath (str or Path): Path to write data.

        Examples:
            >>> backend = MSCBackend()
            >>> filepath = "path/of/file"  # or "s3://path/of/file"
            >>> backend.put(b"hello world", filepath)
        """
        path = self._translate_filepath(filepath=filepath)
        buffer = io.BytesIO()
        if isinstance(obj, bytes):
            buffer.write(obj)
            buffer.seek(0)
        elif isinstance(obj, io.BytesIO):
            buffer = obj
        else:
            raise ValueError(f"Unhandled obj type: {type(obj)}")
        # `StorageClient.write()` defers to `StorageProvider.put_object()` while
        # `StorageClient.upload_file()` defers to `StorageProvider.upload_file()`.
        #
        # Currently, only `StorageProvider.upload_file()` supports parallel uploads
        # in some storage providers (e.g. boto S3 transfer manager for S3 storage providers)
        # so it's often much faster.
        self._storage_client.upload_file(remote_path=path, local_path=buffer)

    def put_text(
        self,
        obj: str,
        filepath: Union[str, Path],
        encoding: str = "utf-8",
    ) -> None:
        """Write text to a given ``filepath``.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to encode the ``obj``.
                Defaults to 'utf-8'.

        Examples:
            >>> backend = MSCBackend()
            >>> filepath = "path/of/file"  # or "s3://path/of/file"
            >>> backend.put_text("hello world", filepath)
        """
        self.put(obj=bytes(obj, encoding=encoding), filepath=filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.

        Examples:
            >>> backend = MSCBackend()
            >>> filepath = "path/of/file"  # or "s3://path/of/file"
            >>> backend.exists(filepath)
            True
        """
        path = self._translate_filepath(filepath=filepath)
        try:
            # Include directories and files.
            self._storage_client.info(path=path, strict=True)
            return True
        except FileNotFoundError:
            return False

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.

        Examples:
            >>> backend = MSCBackend()
            >>> filepath = "path/of/dir"  # or "s3://path/of/file"
            >>> backend.isdir(filepath)
            True
        """
        path = self._translate_filepath(filepath=filepath)
        try:
            # Include directories and files.
            metadata = self._storage_client.info(path=path, strict=True)
            return metadata.type == "directory"
        except FileNotFoundError:
            return False

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.

        Examples:
            >>> backend = MSCBackend()
            >>> filepath = "path/of/file"  # or "s3://path/of/file"
            >>> backend.isfile(filepath)
            True
        """
        path = self._translate_filepath(filepath=filepath)
        try:
            return self._storage_client.is_file(path=path)
        except FileNotFoundError:
            return False

    def join_path(
        self,
        filepath: Union[str, Path],
        *filepaths: Union[str, Path],
    ) -> str:
        r"""Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of \*filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result after concatenation.

        Examples:
            >>> backend = MSCBackend()
            >>> filepath = "path/of/file"  # or "s3://path/of/file"
            >>> backend.join_path(filepath, "another/path")
            'path/of/file/another/path'  # or "s3://path/of/file/another/path"
            >>> backend.join_path(filepath, "/another/path")
            'path/of/file/another/path'  # or "s3://path/of/file/another/path"
        """
        filepath = self._translate_filepath(filepath=filepath, translate_url=False)
        if filepath.endswith("/") and not filepath.endswith("://"):
            filepath = filepath[:-1]
        formatted_paths = [filepath]
        for path in filepaths:
            formatted_path = self._translate_filepath(filepath=path)
            formatted_paths.append(formatted_path.lstrip("/"))

        return "/".join(formatted_paths)

    @contextmanager
    def get_local_path(
        self,
        filepath: Union[str, Path],
    ) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath`` to a local temporary directory,
        and return the temporary path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str or Path): Download a file from ``filepath``.

        Yields:
            Iterable[str]: Only yield one temporary path.

        Examples:
            >>> backend = MSCBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> filepath = "path/of/file"  # or "s3://path/of/file"
            >>> with backend.get_local_path(filepath) as path:
            ...     # do something here
        """
        assert self.isfile(filepath=filepath)
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath=filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def copyfile(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy a file src to dst and return the destination file.

        If dst specifies a file that already exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: The destination file.

        Raises:
            SameFileError: If src and dst are the same file, a SameFileError
                will be raised.

        Examples:
            >>> backend = MSCBackend()
            >>> # dst is a file
            >>> src = "path/of/file"  # or "s3://path/of/file"
            >>> dst = "path/of/file1"  # or "s3://path/of/file1"
            >>> backend.copyfile(src, dst)
            'path/of/file1'  # or "s3://path/of/file1"

            >>> # dst is a directory
            >>> dst = "path/of/dir"  # or "s3://path/of/dir"
            >>> backend.copyfile(src, dst)
            'path/of/dir/file'  # or "s3://path/of/dir/file"
        """
        if not self.isfile(filepath=src):
            raise FileNotFoundError("src does not exist or is not a file")
        if self.isdir(filepath=dst):
            dst = self.join_path(dst, self._translate_filepath(filepath=src).split("/")[-1])
        if self._translate_filepath(filepath=src) == self._translate_filepath(filepath=dst):
            raise SameFileError("src and dst should not be same")

        self.put(obj=self.get(filepath=src), filepath=dst)

        return self._translate_filepath(filepath=dst, translate_url=False)

    def copytree(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to dst.

        Returns:
            str: The destination directory.

        Raises:
            FileExistsError: If dst had already existed, a FileExistsError will
                be raised.

        Examples:
            >>> backend = MSCBackend()
            >>> src = "path/of/dir"  # or "s3://path/of/dir"
            >>> dst = "path/of/dir1"  # or "s3://path/of/dir1"
            >>> backend.copytree(src, dst)
            'path/of/dir1'  # or "s3://path/of/dir1"
        """
        if not self.isdir(filepath=src):
            raise FileNotFoundError("src does not exist or is not a directory")
        if self.exists(filepath=dst):
            raise FileExistsError("dst should not exist")

        for path in self.list_dir_or_file(src, list_dir=False, recursive=True):
            src_path = self.join_path(src, path)
            dst_path = self.join_path(dst, path)
            self.put(obj=self.get(filepath=src_path), filepath=dst_path)

        return self._translate_filepath(filepath=dst, translate_url=False)

    def copyfile_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Upload a local file src to dst and return the destination file.

        Args:
            src (str or Path): A local file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Examples:
            >>> backend = MSCBackend()
            >>> # dst is a file
            >>> src = "path/of/your/file"
            >>> dst = "path/of/file1"  # or "s3://path/of/file1"
            >>> backend.copyfile_from_local(src, dst)
            'path/of/file1'  # or "s3://path/of/file1"

            >>> # dst is a directory
            >>> dst = "path/of/dir"
            >>> backend.copyfile_from_local(src, dst)
            'path/of/dir/file'  # or "s3://path/of/dir/file"
        """
        if self.isdir(filepath=dst):
            dst = self.join_path(dst, os.path.basename(src))

        with open(src, "rb") as f:
            self.put(obj=f.read(), filepath=dst)

        return self._translate_filepath(filepath=dst, translate_url=False)

    def copytree_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory.

        Args:
            src (str or Path): A local directory to be copied.
            dst (str or Path): Copy directory to dst.

        Returns:
            str: The destination directory.

        Raises:
            FileExistsError: If dst had already existed, a FileExistsError will
                be raised.

        Examples:
            >>> backend = MSCBackend()
            >>> src = "path/of/your/dir"
            >>> dst = "path/of/dir1"  # or "s3://path/of/dir1"
            >>> backend.copytree_from_local(src, dst)
            'path/of/dir1'  # or "s3://path/of/dir1"
        """
        if self.exists(filepath=dst):
            raise FileExistsError("dst should not exist")

        src = str(src)

        for cur_dir, _, files in os.walk(src):
            for f in files:
                src_path = os.path.join(cur_dir, f)
                dst_path = self.join_path(dst, src_path.replace(src, ""))
                self.copyfile_from_local(src=src_path, dst=dst_path)

        return self._translate_filepath(filepath=dst, translate_url=False)

    def copyfile_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        dst_type: str,  # Choose from ["file", "dir"]
    ) -> Union[str, Path]:
        """Copy the file src to local dst and return the destination file.

        If dst specifies a directory, the file will be copied into dst using
        the base filename from src. If dst specifies a file that already
        exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to to local dst.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Examples:
            >>> backend = MSCBackend()
            >>> # dst is a file
            >>> src = "path/of/file"  # or "s3://path/of/file"
            >>> dst = "path/of/your/file"
            >>> backend.copyfile_to_local(src, dst)
            'path/of/your/file'

            >>> # dst is a directory
            >>> dst = "path/of/your/dir"
            >>> backend.copyfile_to_local(src, dst)
            'path/of/your/dir/file'
        """
        assert dst_type in ["file", "dir"]
        # There is no good way to detect whether dst is a directory or a file, so we make dst_type required
        if dst_type == "dir":
            basename = os.path.basename(self._translate_filepath(filepath=src))
            if isinstance(dst, str):
                dst = os.path.join(dst, basename)
            else:
                assert isinstance(dst, Path)
                dst = dst / basename

        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(dst)
        os.makedirs(parent_dir, exist_ok=True)

        try:
            with open(dst, "wb") as f:
                data = self.get(filepath=src)
                f.write(data)
        except Exception as e:
            log.error(f"Failed to write file: {e}")
            raise

        return dst

    def copytree_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> Union[str, Path]:
        """Recursively copy an entire directory tree rooted at src to a local
        directory named dst and return the destination directory.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to local dst.

        Returns:
            str: The destination directory.

        Examples:
            >>> backend = MSCBackend()
            >>> src = "path/of/dir"  # or "s3://path/of/dir"
            >>> dst = "path/of/your/dir"
            >>> backend.copytree_to_local(src, dst)
            'path/of/your/dir'
        """
        for path in self.list_dir_or_file(dir_path=src, list_dir=False, recursive=True):
            dst_path = os.path.join(dst, path)
            mkdir_or_exist(os.path.dirname(dst_path))
            with open(dst_path, "wb") as f:
                f.write(self.get(filepath=self.join_path(src, path)))

        return dst

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.

        Raises:
            FileNotFoundError: If filepath does not exist, an FileNotFoundError
                will be raised.
            IsADirectoryError: If filepath is a directory, an IsADirectoryError
                will be raised.

        Examples:
            >>> backend = MSCBackend()
            >>> filepath = "path/of/file"  # or "s3://path/of/file"
            >>> backend.remove(filepath)
        """
        if not self.exists(filepath=filepath):
            raise FileNotFoundError(f"filepath {filepath} does not exist")

        if self.isdir(filepath=filepath):
            raise IsADirectoryError("filepath should be a file")

        self._storage_client.delete(path=self._translate_filepath(filepath=filepath), recursive=False)

    def rmtree(self, dir_path: Union[str, Path]) -> None:
        """Recursively delete a directory tree.

        Args:
            dir_path (str or Path): A directory to be removed.

        Examples:
            >>> backend = MSCBackend()
            >>> dir_path = "path/of/dir"  # or "s3://path/of/dir"
            >>> backend.rmtree(dir_path)
        """
        self._storage_client.delete(path=self._translate_filepath(filepath=dir_path), recursive=True)

    def copy_if_symlink_fails(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> bool:
        """Create a symbolic link pointing to src named dst.

        Directly copy src to dst because MSCBackend does not support creating
        a symbolic link.

        Args:
            src (str or Path): A file or directory to be copied.
            dst (str or Path): Copy a file or directory to dst.

        Returns:
            bool: Return False because MSCBackend does not support create
            a symbolic link.

        Examples:
            >>> backend = MSCBackend()
            >>> src = "path/of/file"  # or "s3://path/of/file"
            >>> dst = "path/of/your/file"  # or "s3://path/of/your/file"
            >>> backend.copy_if_symlink_fails(src, dst)
            False
            >>> src = "path/of/dir"  # or "s3://path/of/dir"
            >>> dst = "path/of/your/dir"  # or "s3://path/of/your/dir"
            >>> backend.copy_if_symlink_fails(src, dst)
            False
        """
        if self.isfile(filepath=src):
            self.copyfile(src=src, dst=dst)
        else:
            self.copytree(src=src, dst=dst)
        return False

    def list_dir(self, dir_path: Union[str, Path]) -> Generator[str, None, None]:
        """List all folders in a storage location with a given prefix.

        Args:
            dir_path (str | Path): Path of the directory.

        Examples:
            >>> backend = MSCBackend()
            >>> dir_path = "path/of/dir"  # or "s3://path/of/dir"
            >>> list(backend.list_dir(dir_path))
            ["subdir1/", "subdir2/"]
        """
        path = self._translate_filepath(filepath=dir_path).removesuffix("/") + "/"
        for metadata in self._storage_client.list(path=path, include_directories=True, include_url_prefix=False):
            if metadata.type == "directory":
                yield metadata.key.removeprefix(path).removesuffix("/") + "/"

    def list_dir_or_file(  # pylint: disable=too-many-arguments
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, tuple[str]]] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.

        Note:
            Most object stores have no concept of directories but it simulates
            the directory hierarchy in the filesystem through public prefixes.
            In addition, if the returned path ends with '/', it means the path
            is a public prefix which is a logical directory.

        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
            In addition, the returned path of directory will not contains the
            suffix '/' which is consistent with other backends.

        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Defaults to True.
            list_file (bool): List the path of files. Defaults to True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Defaults to None.
            recursive (bool): If set to True, recursively scan the
                directory. Defaults to False.

        Yields:
            Iterable[str]: A relative path to ``dir_path``.

        Examples:
            >>> backend = MSCBackend()
            >>> dir_path = "path/of/dir"  # or "s3://path/of/dir"
            >>> # list those files and directories in current directory
            >>> list(backend.list_dir_or_file(dir_path))
            ["file.txt", "subdir", "subdir/cat.png", "subdir/subsubdir/dog.jpg"]
            >>> # only list files
            >>> list(backend.list_dir_or_file(dir_path, list_dir=False))
            ["file.txt", "subdir/cat.png", "subdir/subsubdir/dog.jpg"]
            >>> # only list directories
            >>> list(backend.list_dir_or_file(dir_path, list_file=False))
            ["subdir"]
            >>> # only list files ending with specified suffixes
            >>> list(backend.list_dir_or_file(dir_path, suffix=".txt"))
            ["file.txt"]
            >>> # list all files and directory recursively
            >>> list(backend.list_dir_or_file(dir_path, recursive=True))
            ["file.txt", "subdir", "subdir/cat.png", "subdir/subsubdir", "subdir/subsubdir/dog.png"]
        """
        dir_path = self._translate_filepath(filepath=dir_path).removesuffix("/") + "/"

        if list_dir and suffix is not None:
            raise TypeError("`list_dir` should be False when `suffix` is not None")

        if list_dir and not list_file and not recursive:
            raise TypeError(
                "Please use `list_dir` instead of `list_dir_or_file` "
                "when you only want to list the first level directories."
            )

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError("`suffix` must be a string or tuple of strings")

        yielded_subdir_paths: set[str] = set()
        # In the MSC, the `include_directories` option switches between flat and hierarchical for both files and "directories".
        #
        # In the Boto3Backend, however, the `recursive` option only applies to "directories" (seems like a bug).
        #
        # Construct directories from file paths to match the Boto3Backend behavior.
        #
        # If this behavior needs to be fixed, switch to `include_directories=(not recursive)` and adjust metadata processing.
        for metadata in self._storage_client.list(path=dir_path, include_directories=False, include_url_prefix=False):
            # Only files should be returned with `include_directories=False`, but just in case.
            if metadata.type == "file":
                rel_path: str = metadata.key.removeprefix(dir_path)
                if list_dir:
                    rel_path_fragments = rel_path.split("/")
                    if len(rel_path_fragments) > 1:
                        for i in range(len(rel_path_fragments) - 1 if recursive else 1):
                            subdir_path = "/".join(rel_path_fragments[: i + 1])
                            if subdir_path not in yielded_subdir_paths:
                                yielded_subdir_paths.add(subdir_path)
                                yield subdir_path
                if list_file:
                    if suffix is None or rel_path.endswith(suffix):
                        yield rel_path

    def generate_presigned_url(self, url: str, client_method: str = "get_object", expires_in: int = 3600) -> str:
        """Generate the presigned url of video stream which can be passed to
        mmcv.VideoReader. Now only work on Boto3 backend.

        Note:
            Now only work on Boto3 backend.

        Args:
            url (str): Url of video stream.
            client_method (str): Method of client, 'get_object' or
                'put_object'. Default: 'get_object'.
            expires_in (int): expires, in seconds. Default: 3600.

        Returns:
            str: Generated presigned url.
        """
        raise NotImplementedError("generate_presigned_url is not supported in MSCBackend")
