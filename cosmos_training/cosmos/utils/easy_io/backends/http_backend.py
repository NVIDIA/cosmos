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
import tempfile
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union
from urllib.request import Request, urlopen

from cosmos.utils.easy_io.backends.base_backend import BaseStorageBackend


class HTTPBackend(BaseStorageBackend):
    """HTTP and HTTPS storage bachend."""

    def size(self, filepath: Union[str, Path]) -> int:
        """Get the file size in bytes for a given ``filepath``.

        Args:
            filepath (str or Path): Path to get file size in bytes.

        Returns:
            int: File size in bytes for filepath.

        Examples:
            >>> backend = HTTPBackend()
            >>> filepath = 'http://path/of/file'
            >>> backend.size(filepath)  # file containing 'hello world'
            11
        """
        request = Request(url=str(filepath), method="HEAD")
        with urlopen(request) as response:
            if response.status == 200:
                return int(response.headers["Content-Length"])
            else:
                raise RuntimeError(f"Unexpected response: {response}")

    def get(self, filepath: Union[str, Path], offset: Optional[int] = None, size: Optional[int] = None) -> bytes:
        """Read bytes from a given ``filepath`` with 'rb' mode in range [offset, offset + size).

        Args:
            filepath (str): Path to read data.
            offset (int, optional): Read offset in bytes (0-index). Defaults to 0.
            size (int, optional): Read size in bytes. Defaults to the file size.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = HTTPBackend()
            >>> backend.get('http://path/of/file')
            b'hello world'
        """
        request = Request(url=str(filepath), method="GET")
        if offset is not None or size is not None:
            read_offset = offset or 0
            assert read_offset >= 0, "Read offset must be ≥ 0"

            # Try not to incur a remote call to get the file size. This can heavily slow down ranged reads.
            #
            # This means we won't always validate the read offset or read size against the file size.
            read_size = size or (self.size(filepath=filepath) - read_offset)
            assert read_size >= 1, "Read size must be ≥ 1 or read offset must be < file size"

            request.add_header("Range", f"bytes={read_offset}-{read_offset + read_size - 1}")
        with urlopen(request) as response:
            if response.status in {200, 206}:
                return response.read()
            else:
                raise RuntimeError(f"Unexpected response: {response}")

    def get_text(self, filepath: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read text from a given ``filepath``.

        Args:
            filepath (str): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.

        Examples:
            >>> backend = HTTPBackend()
            >>> backend.get_text('http://path/of/file')
            'hello world'
        """
        return self.get(filepath=filepath).decode(encoding)

    def put(self, obj: Union[bytes, io.BytesIO], filepath: Union[str, Path]) -> None:
        raise NotImplementedError(f"put not supported in {self.name}")

    def put_text(self, obj: str, filepath: Union[str, Path], encoding: str = "utf-8") -> None:
        raise NotImplementedError(f"put_text not supported in {self.name}")

    def exists(self, filepath: Union[str, Path]) -> bool:
        request = Request(url=str(filepath), method="HEAD")
        with urlopen(request) as response:
            if response.status == 404:
                return False
            elif response.status == 200:
                return True
            else:
                raise RuntimeError(f"Unexpected response: {response}")

    def isdir(self, filepath: Union[str, Path]) -> bool:
        raise NotImplementedError(f"isdir not supported in {self.name}")

    def isfile(self, filepath: Union[str, Path]) -> bool:
        raise NotImplementedError(f"isfile not supported in {self.name}")

    def join_path(self, filepath: Union[str, Path], *filepaths: Union[str, Path]) -> str:
        raise NotImplementedError(f"join_path not supported in {self.name}")

    @contextmanager
    def get_local_path(self, filepath: Union[str, Path]) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath`` to a local temporary directory,
        and return the temporary path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str): Download a file from ``filepath``.

        Yields:
            Iterable[str]: Only yield one temporary path.

        Examples:
            >>> backend = HTTPBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with backend.get_local_path('http://path/of/file') as path:
            ...     # do something here
        """
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def copyfile(self, src: Union[str, Path], dst: Union[str, Path]) -> str:
        raise NotImplementedError(f"copyfile not supported in {self.name}")

    def copytree(self, src: Union[str, Path], dst: Union[str, Path]) -> str:
        raise NotImplementedError(f"copytree not supported in {self.name}")

    def copyfile_from_local(self, src: Union[str, Path], dst: Union[str, Path]) -> str:
        raise NotImplementedError(f"copyfile_from_local not supported in {self.name}")

    def copytree_from_local(self, src: Union[str, Path], dst: Union[str, Path]) -> str:
        raise NotImplementedError(f"copytree_from_local not supported in {self.name}")

    def copyfile_to_local(self, src: Union[str, Path], dst: Union[str, Path], dst_type: str) -> Union[str, Path]:
        raise NotImplementedError(f"copyfile_to_local not supported in {self.name}")

    def copytree_to_local(self, src: Union[str, Path], dst: Union[str, Path]) -> Union[str, Path]:
        raise NotImplementedError(f"copytree_to_local not supported in {self.name}")

    def remove(self, filepath: Union[str, Path]) -> None:
        raise NotImplementedError(f"remove not supported in {self.name}")

    def rmtree(self, dir_path: Union[str, Path]) -> None:
        raise NotImplementedError(f"rmtree not supported in {self.name}")

    def copy_if_symlink_fails(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        raise NotImplementedError(f"copy_if_symlink_fails not supported in {self.name}")

    def list_dir(self, dir_path: Union[str, Path]) -> Generator[str, None, None]:
        raise NotImplementedError(f"list_dir not supported in {self.name}")

    def list_dir_or_file(  # pylint: disable=too-many-arguments
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, tuple[str]]] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        raise NotImplementedError(f"list_dir_or_file not supported in {self.name}")
