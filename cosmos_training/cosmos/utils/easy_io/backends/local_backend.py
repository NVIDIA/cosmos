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
import os.path as osp
import shutil
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

from cosmos.utils.easy_io.backends.base_backend import BaseStorageBackend, mkdir_or_exist


class LocalBackend(BaseStorageBackend):
    """Raw local storage backend."""

    _allow_symlink = True

    def size(self, filepath: Union[str, Path]) -> int:
        """Get the file size in bytes for a given ``filepath``.

        Args:
            filepath (str or Path): Path to get file size in bytes.

        Returns:
            int: File size in bytes for filepath.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.size(filepath)  # file containing 'hello world'
            11
        """
        return osp.getsize(filepath)

    def get(self, filepath: Union[str, Path], offset: Optional[int] = None, size: Optional[int] = None) -> bytes:
        """Read bytes from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.
            offset (int, optional): Read offset in bytes (0-index). Defaults to 0.
            size (int, optional): Read size in bytes. Defaults to the file size.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.get(filepath)
            b'hello world'
        """
        read_offset: Optional[int] = None
        read_size: Optional[int] = None
        if offset is not None or size is not None:
            read_offset = offset or 0
            assert read_offset >= 0, "Read offset must be ≥ 0"

            read_size = size or (self.size(filepath=filepath) - read_offset)
            assert read_size >= 1, "Read size must be ≥ 1 or read offset must be < file size"

        with open(filepath, "rb") as f:
            if read_offset is not None:
                f.seek(read_offset)
            value = f.read(read_size)
        return value

    def get_text(self, filepath: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read text from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.get_text(filepath)
            'hello world'
        """
        with open(filepath, encoding=encoding) as f:
            text = f.read()
        return text

    def put(self, obj: Union[bytes, io.BytesIO], filepath: Union[str, Path]) -> None:
        """Write bytes to a given ``filepath`` with 'wb' mode.

        Note:
            ``put`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.put(b'hello world', filepath)
        """
        mkdir_or_exist(osp.dirname(filepath))
        if isinstance(obj, io.BytesIO):
            obj.seek(0)
            obj = obj.getvalue()
        with open(filepath, "wb") as f:
            f.write(obj)

    def put_text(self, obj: str, filepath: Union[str, Path], encoding: str = "utf-8") -> None:
        """Write text to a given ``filepath`` with 'w' mode.

        Note:
            ``put_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.put_text('hello world', filepath)
        """
        mkdir_or_exist(osp.dirname(filepath))
        with open(filepath, "w", encoding=encoding) as f:
            f.write(obj)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.exists(filepath)
            True
        """
        return osp.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/dir'
            >>> backend.isdir(filepath)
            True
        """
        return osp.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.isfile(filepath)
            True
        """
        return osp.isfile(filepath)

    def join_path(self, filepath: Union[str, Path], *filepaths: Union[str, Path]) -> str:
        r"""Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of \*filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result of concatenation.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath1 = '/path/of/dir1'
            >>> filepath2 = 'dir2'
            >>> filepath3 = 'path/of/file'
            >>> backend.join_path(filepath1, filepath2, filepath3)
            '/path/of/dir/dir2/path/of/file'
        """
        # TODO, if filepath or filepaths are Path, should return Path
        return osp.join(filepath, *filepaths)

    @contextmanager
    def get_local_path(
        self,
        filepath: Union[str, Path],
    ) -> Generator[Union[str, Path], None, None]:
        """Only for unified API and do nothing.

        Args:
            filepath (str or Path): Path to be read data.
            backend_args (dict, optional): Arguments to instantiate the
                corresponding backend. Defaults to None.

        Examples:
            >>> backend = LocalBackend()
            >>> with backend.get_local_path('s3://bucket/abc.jpg') as path:
            ...     # do something here
        """
        yield filepath

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
            >>> backend = LocalBackend()
            >>> # dst is a file
            >>> src = '/path/of/file'
            >>> dst = '/path1/of/file1'
            >>> # src will be copied to '/path1/of/file1'
            >>> backend.copyfile(src, dst)
            '/path1/of/file1'

            >>> # dst is a directory
            >>> dst = '/path1/of/dir'
            >>> # src will be copied to '/path1/of/dir/file'
            >>> backend.copyfile(src, dst)
            '/path1/of/dir/file'
        """
        return shutil.copy(src, dst)

    def copytree(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory.

        src and dst should have the same prefix and dst must not already exist.

        TODO: Whether to support dirs_exist_ok parameter.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to dst.

        Returns:
            str: The destination directory.

        Raises:
            FileExistsError: If dst had already existed, a FileExistsError will
                be raised.

        Examples:
            >>> backend = LocalBackend()
            >>> src = '/path/of/dir1'
            >>> dst = '/path/of/dir2'
            >>> backend.copytree(src, dst)
            '/path/of/dir2'
        """
        return shutil.copytree(src, dst)

    def copyfile_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy a local file src to dst and return the destination file. Same
        as :meth:`copyfile`.

        Args:
            src (str or Path): A local file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Raises:
            SameFileError: If src and dst are the same file, a SameFileError
                will be raised.

        Examples:
            >>> backend = LocalBackend()
            >>> # dst is a file
            >>> src = '/path/of/file'
            >>> dst = '/path1/of/file1'
            >>> # src will be copied to '/path1/of/file1'
            >>> backend.copyfile_from_local(src, dst)
            '/path1/of/file1'

            >>> # dst is a directory
            >>> dst = '/path1/of/dir'
            >>> # src will be copied to
            >>> backend.copyfile_from_local(src, dst)
            '/path1/of/dir/file'
        """
        return self.copyfile(src, dst)

    def copytree_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory. Same as
        :meth:`copytree`.

        Args:
            src (str or Path): A local directory to be copied.
            dst (str or Path): Copy directory to dst.

        Returns:
            str: The destination directory.

        Examples:
            >>> backend = LocalBackend()
            >>> src = '/path/of/dir1'
            >>> dst = '/path/of/dir2'
            >>> backend.copytree_from_local(src, dst)
            '/path/of/dir2'
        """
        return self.copytree(src, dst)

    def copyfile_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        dst_type: Optional[str] = None,
    ) -> str:
        """Copy the file src to local dst and return the destination file. Same
        as :meth:`copyfile`.

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
            >>> backend = LocalBackend()
            >>> # dst is a file
            >>> src = '/path/of/file'
            >>> dst = '/path1/of/file1'
            >>> # src will be copied to '/path1/of/file1'
            >>> backend.copyfile_to_local(src, dst)
            '/path1/of/file1'

            >>> # dst is a directory
            >>> dst = '/path1/of/dir'
            >>> # src will be copied to
            >>> backend.copyfile_to_local(src, dst)
            '/path1/of/dir/file'
        """
        return self.copyfile(src, dst)

    def copytree_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
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
            >>> backend = LocalBackend()
            >>> src = '/path/of/dir1'
            >>> dst = '/path/of/dir2'
            >>> backend.copytree_from_local(src, dst)
            '/path/of/dir2'
        """
        return self.copytree(src, dst)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.

        Raises:
            IsADirectoryError: If filepath is a directory, an IsADirectoryError
                will be raised.
            FileNotFoundError: If filepath does not exist, an FileNotFoundError
                will be raised.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.remove(filepath)
        """
        if not self.exists(filepath):
            raise FileNotFoundError(f"filepath {filepath} does not exist")

        if self.isdir(filepath):
            raise IsADirectoryError("filepath should be a file")

        os.remove(filepath)

    def rmtree(self, dir_path: Union[str, Path]) -> None:
        """Recursively delete a directory tree.

        Args:
            dir_path (str or Path): A directory to be removed.

        Examples:
            >>> dir_path = '/path/of/dir'
            >>> backend.rmtree(dir_path)
        """
        shutil.rmtree(dir_path)

    def copy_if_symlink_fails(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> bool:
        """Create a symbolic link pointing to src named dst.

        If failed to create a symbolic link pointing to src, directly copy src
        to dst instead.

        Args:
            src (str or Path): Create a symbolic link pointing to src.
            dst (str or Path): Create a symbolic link named dst.

        Returns:
            bool: Return True if successfully create a symbolic link pointing
            to src. Otherwise, return False.

        Examples:
            >>> backend = LocalBackend()
            >>> src = '/path/of/file'
            >>> dst = '/path1/of/file1'
            >>> backend.copy_if_symlink_fails(src, dst)
            True
            >>> src = '/path/of/dir'
            >>> dst = '/path1/of/dir1'
            >>> backend.copy_if_symlink_fails(src, dst)
            True
        """
        try:
            os.symlink(src, dst)
            return True
        except Exception:
            if self.isfile(src):
                self.copyfile(src, dst)
            else:
                self.copytree(src, dst)
            return False

    def list_dir(self, dir_path: Union[str, Path]) -> Generator[str, None, None]:
        """List all folders in a storage location with a given prefix.

        Args:
            dir_path (str | Path): Path of the directory.

        Examples:
            >>> backend = LocalBackend()
            >>> dir_path = 'path/of/dir'
            >>> list(backend.list_dir(dir_path))
            ['subdir1/', 'subdir2/']
        """
        for entry in os.scandir(dir_path):
            if entry.is_dir():
                yield f"{entry.name}/"

    def list_dir_or_file(
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
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.

        Args:
            dir_path (str or Path): Path of the directory.
            list_dir (bool): List the directories. Defaults to True.
            list_file (bool): List the path of files. Defaults to True.
            suffix (str or tuple[str], optional): File suffix that we are
                interested in. Defaults to None.
            recursive (bool): If set to True, recursively scan the directory.
                Defaults to False.

        Yields:
            Iterable[str]: A relative path to ``dir_path``.

        Examples:
            >>> backend = LocalBackend()
            >>> dir_path = '/path/of/dir'
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
        if list_dir and suffix is not None:
            raise TypeError("`suffix` should be None when `list_dir` is True")

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError("`suffix` must be a string or tuple of strings")

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive):
            for entry in os.scandir(dir_path):
                if not entry.name.startswith(".") and entry.is_file():
                    rel_path = osp.relpath(entry.path, root)
                    if (suffix is None or rel_path.endswith(suffix)) and list_file:
                        yield rel_path
                elif osp.isdir(entry.path):
                    if list_dir:
                        rel_dir = osp.relpath(entry.path, root)
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(entry.path, list_dir, list_file, suffix, recursive)

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive)
