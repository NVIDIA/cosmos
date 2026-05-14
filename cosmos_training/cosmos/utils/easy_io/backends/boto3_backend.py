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
import re
import tempfile
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from shutil import SameFileError
from typing import Optional, Union

from cosmos.utils import log
from cosmos.utils.easy_io.backends.base_backend import BaseStorageBackend, has_method, mkdir_or_exist
from cosmos.utils.easy_io.backends.boto3_client import Boto3Client


class Boto3Backend(BaseStorageBackend):
    """boto3 storage backend (for internal usage).

    **Deprecated**. Use the MSC backend instead.

    Boto3Backend supports reading and writing data to multiple clusters.
    If the file path contains the cluster name, Boto3Backend will read data
    from specified cluster or write data to it. Otherwise, Boto3Backend will
    access the default cluster.

    Args:
        path_mapping (dict, optional): Path mapping dict from local path to
            Boto3 path. When ``path_mapping={'src': 'dst'}``, ``src`` in
            ``filepath`` will be replaced by ``dst``. Defaults to None.
        s3_credential_path (str, optional): Config path of Boto3 client. Default: None.
            `New in version 0.3.3`.

    Examples:
        >>> backend = Boto3Backend()
        >>> filepath1 = 's3://path/of/file'
        >>> filepath2 = 'cluster-name:s3://path/of/file'
        >>> backend.get(filepath1)  # get data from default cluster
        >>> client.get(filepath2)  # get data from 'cluster-name' cluster
    """

    def __init__(
        self,
        s3_credential_path: str = "",
        path_mapping: Optional[dict] = None,
    ):
        self._client = Boto3Client(s3_credential_path=s3_credential_path)
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping
        if path_mapping:
            for k, v in path_mapping.items():
                log.critical(f"Path mapping: {k} -> {v}", rank0_only=False)

    def _map_path(self, filepath: Union[str, Path]) -> str:
        """Map ``filepath`` to a string path whose prefix will be replaced by
        :attr:`self.path_mapping`.

        Args:
            filepath (str or Path): Path to be mapped.
        """
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v, 1)
        return filepath

    def _format_path(self, filepath: str) -> str:
        """Convert a ``filepath`` to standard format of s3 oss.

        If the ``filepath`` is concatenated by ``os.path.join``, in a Windows
        environment, the ``filepath`` will be the format of
        's3://bucket_name\\image.jpg'. By invoking :meth:`_format_path`, the
        above ``filepath`` will be converted to 's3://bucket_name/image.jpg'.

        Args:
            filepath (str): Path to be formatted.
        """
        return re.sub(r"\\+", "/", filepath)

    def _replace_prefix(self, filepath: Union[str, Path]) -> str:
        filepath = str(filepath)
        return filepath
        # return filepath.replace('s3://', 's3://')

    def size(self, filepath: Union[str, Path]) -> int:
        """Get the file size in bytes for a given ``filepath``.

        Args:
            filepath (str or Path): Path to get file size in bytes.

        Returns:
            int: File size in bytes for filepath.

        Examples:
            >>> backend = Boto3Backend()
            >>> filepath = 's3://path/of/file'
            >>> backend.size(filepath)  # file containing 'hello world'
            11
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.size(filepath)

    def get(self, filepath: Union[str, Path], offset: Optional[int] = None, size: Optional[int] = None) -> bytes:
        """Read bytes from a given ``filepath`` with 'rb' mode in range [offset, offset + size).

        Args:
            filepath (str or Path): Path to read data.
            offset (int, optional): Read offset in bytes (0-index). Defaults to 0.
            size (int, optional): Read size in bytes. Defaults to the file size.

        Returns:
            bytes: Return bytes read from filepath.

        Examples:
            >>> backend = Boto3Backend()
            >>> filepath = 's3://path/of/file'
            >>> backend.get(filepath)
            b'hello world'
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        value = self._client.get(filepath=filepath, offset=offset, size=size)
        return value

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
            >>> backend = Boto3Backend()
            >>> filepath = 's3://path/of/file'
            >>> backend.get_text(filepath)
            'hello world'
        """
        return str(self.get(filepath), encoding=encoding)

    def put(self, obj: Union[bytes, io.BytesIO], filepath: Union[str, Path]) -> None:
        """Write bytes to a given ``filepath``.

        Args:
            obj (bytes): Data to be saved.
            filepath (str or Path): Path to write data.

        Examples:
            >>> backend = Boto3Backend()
            >>> filepath = 's3://path/of/file'
            >>> backend.put(b'hello world', filepath)
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        self._client.put(obj, filepath)

    def fast_put(self, obj: Union[bytes, io.BytesIO], filepath: Union[str, Path], num_processes: int = 32) -> None:
        """Write bytes to a given ``filepath`` with multiple processes and async"""
        assert num_processes > 1
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        self._client.fast_put(obj, filepath, num_processes=num_processes)

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
            >>> backend = Boto3Backend()
            >>> filepath = 's3://path/of/file'
            >>> backend.put_text('hello world', filepath)
        """
        self.put(bytes(obj, encoding=encoding), filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.

        Examples:
            >>> backend = Boto3Backend()
            >>> filepath = 's3://path/of/file'
            >>> backend.exists(filepath)
            True
        """
        if not (has_method(self._client, "contains") and has_method(self._client, "isdir")):
            raise NotImplementedError(
                "Current version of Boto3 Python SDK has not supported "
                "the `contains` and `isdir` methods, please use a higher"
                "version or dev branch instead."
            )

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.contains(filepath) or self._client.isdir(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.

        Examples:
            >>> backend = Boto3Backend()
            >>> filepath = 's3://path/of/dir'
            >>> backend.isdir(filepath)
            True
        """
        if not has_method(self._client, "isdir"):
            raise NotImplementedError(
                "Current version of Boto3 Python SDK has not supported "
                "the `isdir` method, please use a higher version or dev"
                " branch instead."
            )

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.

        Examples:
            >>> backend = Boto3Backend()
            >>> filepath = 's3://path/of/file'
            >>> backend.isfile(filepath)
            True
        """
        if not has_method(self._client, "contains"):
            raise NotImplementedError(
                "Current version of Boto3 Python SDK has not supported "
                "the `contains` method, please use a higher version or "
                "dev branch instead."
            )

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.contains(filepath)

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
            >>> backend = Boto3Backend()
            >>> filepath = 's3://path/of/file'
            >>> backend.join_path(filepath, 'another/path')
            's3://path/of/file/another/path'
            >>> backend.join_path(filepath, '/another/path')
            's3://path/of/file/another/path'
        """
        filepath = self._format_path(self._map_path(filepath))
        if filepath.endswith("/"):
            filepath = filepath[:-1]
        formatted_paths = [filepath]
        for path in filepaths:
            formatted_path = self._format_path(self._map_path(path))
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
            >>> backend = Boto3Backend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> filepath = 's3://path/of/file'
            >>> with backend.get_local_path(filepath) as path:
            ...     # do something here
        """
        assert self.isfile(filepath)
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
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

        src and dst should have the same prefix. If dst specifies a directory,
        the file will be copied into dst using the base filename from src. If
        dst specifies a file that already exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: The destination file.

        Raises:
            SameFileError: If src and dst are the same file, a SameFileError
                will be raised.

        Examples:
            >>> backend = Boto3Backend()
            >>> # dst is a file
            >>> src = 's3://path/of/file'
            >>> dst = 's3://path/of/file1'
            >>> backend.copyfile(src, dst)
            's3://path/of/file1'

            >>> # dst is a directory
            >>> dst = 's3://path/of/dir'
            >>> backend.copyfile(src, dst)
            's3://path/of/dir/file'
        """
        src = self._format_path(self._map_path(src))
        dst = self._format_path(self._map_path(dst))
        if self.isdir(dst):
            dst = self.join_path(dst, src.split("/")[-1])

        if src == dst:
            raise SameFileError("src and dst should not be same")

        self.put(self.get(src), dst)
        return dst

    def copytree(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory.

        src and dst should have the same prefix.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            str: The destination directory.

        Raises:
            FileExistsError: If dst had already existed, a FileExistsError will
                be raised.

        Examples:
            >>> backend = Boto3Backend()
            >>> src = 's3://path/of/dir'
            >>> dst = 's3://path/of/dir1'
            >>> backend.copytree(src, dst)
            's3://path/of/dir1'
        """
        src = self._format_path(self._map_path(src))
        dst = self._format_path(self._map_path(dst))

        if self.exists(dst):
            raise FileExistsError("dst should not exist")

        for path in self.list_dir_or_file(src, list_dir=False, recursive=True):
            src_path = self.join_path(src, path)
            dst_path = self.join_path(dst, path)
            self.put(self.get(src_path), dst_path)

        return dst

    def copyfile_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Upload a local file src to dst and return the destination file.

        Args:
            src (str or Path): A local file to be copied.
            dst (str or Path): Copy file to dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Examples:
            >>> backend = Boto3Backend()
            >>> # dst is a file
            >>> src = 'path/of/your/file'
            >>> dst = 's3://path/of/file1'
            >>> backend.copyfile_from_local(src, dst)
            's3://path/of/file1'

            >>> # dst is a directory
            >>> dst = 's3://path/of/dir'
            >>> backend.copyfile_from_local(src, dst)
            's3://path/of/dir/file'
        """
        dst = self._format_path(self._map_path(dst))
        if self.isdir(dst):
            dst = self.join_path(dst, os.path.basename(src))

        with open(src, "rb") as f:
            self.put(f.read(), dst)

        return dst

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
            >>> backend = Boto3Backend()
            >>> src = 'path/of/your/dir'
            >>> dst = 's3://path/of/dir1'
            >>> backend.copytree_from_local(src, dst)
            's3://path/of/dir1'
        """
        dst = self._format_path(self._map_path(dst))
        if self.exists(dst):
            raise FileExistsError("dst should not exist")

        src = str(src)

        for cur_dir, _, files in os.walk(src):
            for f in files:
                src_path = os.path.join(cur_dir, f)
                dst_path = self.join_path(dst, src_path.replace(src, ""))
                self.copyfile_from_local(src_path, dst_path)

        return dst

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
            >>> backend = Boto3Backend()
            >>> # dst is a file
            >>> src = 's3://path/of/file'
            >>> dst = 'path/of/your/file'
            >>> backend.copyfile_to_local(src, dst)
            'path/of/your/file'

            >>> # dst is a directory
            >>> dst = 'path/of/your/dir'
            >>> backend.copyfile_to_local(src, dst)
            'path/of/your/dir/file'
        """
        assert dst_type in ["file", "dir"]
        # There is no good way to detect whether dst is a directory or a file, so we make dst_type required
        if dst_type == "dir":
            basename = os.path.basename(src)
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
                data = self.get(src)
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
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            str: The destination directory.

        Examples:
            >>> backend = Boto3Backend()
            >>> src = 's3://path/of/dir'
            >>> dst = 'path/of/your/dir'
            >>> backend.copytree_to_local(src, dst)
            'path/of/your/dir'
        """
        for path in self.list_dir_or_file(src, list_dir=False, recursive=True):
            dst_path = os.path.join(dst, path)
            mkdir_or_exist(os.path.dirname(dst_path))
            with open(dst_path, "wb") as f:
                f.write(self.get(self.join_path(src, path)))

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
            >>> backend = Boto3Backend()
            >>> filepath = 's3://path/of/file'
            >>> backend.remove(filepath)
        """
        if not has_method(self._client, "delete"):
            raise NotImplementedError(
                "Current version of Boto3 Python SDK has not supported "
                "the `delete` method, please use a higher version or dev "
                "branch instead."
            )

        if not self.exists(filepath):
            raise FileNotFoundError(f"filepath {filepath} does not exist")

        if self.isdir(filepath):
            raise IsADirectoryError("filepath should be a file")

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        self._client.delete(filepath)

    def rmtree(self, dir_path: Union[str, Path]) -> None:
        """Recursively delete a directory tree.

        Args:
            dir_path (str or Path): A directory to be removed.

        Examples:
            >>> backend = Boto3Backend()
            >>> dir_path = 's3://path/of/dir'
            >>> backend.rmtree(dir_path)
        """
        for path in self.list_dir_or_file(dir_path, list_dir=False, recursive=True):
            filepath = self.join_path(dir_path, path)
            self.remove(filepath)

    def copy_if_symlink_fails(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> bool:
        """Create a symbolic link pointing to src named dst.

        Directly copy src to dst because PetrelBacekend does not support create
        a symbolic link.

        Args:
            src (str or Path): A file or directory to be copied.
            dst (str or Path): Copy a file or directory to dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            bool: Return False because Boto3Backend does not support create
            a symbolic link.

        Examples:
            >>> backend = Boto3Backend()
            >>> src = 's3://path/of/file'
            >>> dst = 's3://path/of/your/file'
            >>> backend.copy_if_symlink_fails(src, dst)
            False
            >>> src = 's3://path/of/dir'
            >>> dst = 's3://path/of/your/dir'
            >>> backend.copy_if_symlink_fails(src, dst)
            False
        """
        if self.isfile(src):
            self.copyfile(src, dst)
        else:
            self.copytree(src, dst)
        return False

    def list_dir(self, dir_path: Union[str, Path]):
        """List all folders in an S3 bucket with a given prefix.

        Args:
            dir_path (str | Path): Path of the directory.

        Examples:
            >>> backend = Boto3Backend()
            >>> dir_path = 's3://path/of/dir'
            >>> backend.list_dir(dir_path)
        """
        dir_path = self._map_path(dir_path)
        dir_path = self._format_path(dir_path)
        dir_path = self._replace_prefix(dir_path)
        return self._client.ls_dir(dir_path)

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
            Boto3 has no concept of directories but it simulates the directory
            hierarchy in the filesystem through public prefixes. In addition,
            if the returned path ends with '/', it means the path is a public
            prefix which is a logical directory.

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
            >>> backend = Boto3Backend()
            >>> dir_path = 's3://path/of/dir'
            >>> # list those files and directories in current directory
            >>> for file_path in backend.list_dir_or_file(dir_path):
            ...     print(file_path)
            >>> # only list files
            >>> for file_path in backend.list_dir_or_file(dir_path, list_dir=False):
            ...     print(file_path)
            >>> # only list directories
            >>> for file_path in backend.list_dir_or_file(dir_path, list_file=False):
            ...     print(file_path)
            >>> # only list files ending with specified suffixes
            >>> for file_path in backend.list_dir_or_file(dir_path, suffix='.txt'):
            ...     print(file_path)
            >>> # list all files and directory recursively
            >>> for file_path in backend.list_dir_or_file(dir_path, recursive=True):
            ...     print(file_path)
        """  # noqa: E501
        if not has_method(self._client, "list"):
            raise NotImplementedError(
                "Current version of Boto3 Python SDK has not supported "
                "the `list` method, please use a higher version or dev"
                " branch instead."
            )

        dir_path = self._map_path(dir_path)
        dir_path = self._format_path(dir_path)
        dir_path = self._replace_prefix(dir_path)
        if list_dir and suffix is not None:
            raise TypeError("`list_dir` should be False when `suffix` is not None")

        if list_dir and not list_file and not recursive:
            raise TypeError(
                "Please use `list_dir` instead of `list_dir_or_file` when you only want to list the first level directories."
            )

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError("`suffix` must be a string or tuple of strings")

        # Boto3's simulated directory hierarchy assumes that directory paths
        # should end with `/`
        if not dir_path.endswith("/"):
            dir_path += "/"

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive):
            # Keep track of directories we've already yielded to avoid duplicates
            yielded_dirs = set() if list_dir else None

            for path in self._client.list(dir_path):
                # All paths returned by S3 list are file paths, never directory paths
                absolute_path = self.join_path(dir_path, path)
                rel_path = absolute_path[len(root) :]

                # If we want directories, extract directory prefixes from file paths
                # boto3 client actually never return dir, it only return file paths
                if list_dir and "/" in rel_path:
                    if not recursive:
                        # Non-recursive: only yield immediate child directory (first level)
                        first_slash_pos = rel_path.find("/")
                        immediate_child_dir = rel_path[:first_slash_pos]

                        if immediate_child_dir not in yielded_dirs:
                            yielded_dirs.add(immediate_child_dir)
                            yield immediate_child_dir
                    else:
                        # Recursive: yield all directory levels
                        path_parts = rel_path.split("/")[:-1]  # Exclude filename
                        current_dir = ""
                        for part in path_parts:
                            if current_dir:
                                current_dir += "/" + part
                            else:
                                current_dir = part

                            if current_dir not in yielded_dirs:
                                yielded_dirs.add(current_dir)
                                yield current_dir

                # Handle file listing
                if (suffix is None or rel_path.endswith(suffix)) and list_file:
                    yield rel_path

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive)

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
        raise NotImplementedError("generate_presigned_url is not supported in Boto3Backend")
        return self._client.generate_presigned_url(url, client_method, expires_in)
