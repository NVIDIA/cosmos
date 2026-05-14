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
from abc import ABCMeta, abstractmethod
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == "":
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def has_method(obj, method):
    return hasattr(obj, method) and callable(getattr(obj, method))


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends."""

    # a flag to indicate whether the backend can create a symlink for a file
    # This attribute will be deprecated in future.
    _allow_symlink: bool = False

    @property
    def allow_symlink(self) -> bool:
        return self._allow_symlink

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def size(self, filepath: Union[str, Path]) -> int:
        pass

    @abstractmethod
    def get(self, filepath: Union[str, Path], offset: Optional[int] = None, size: Optional[int] = None) -> bytes:
        pass

    @abstractmethod
    def get_text(self, filepath: Union[str, Path], encoding: str = "utf-8") -> str:
        pass

    @abstractmethod
    def put(self, obj: Union[bytes, io.BytesIO], filepath: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def put_text(self, obj: str, filepath: Union[str, Path], encoding: str = "utf-8") -> None:
        pass

    @abstractmethod
    def exists(self, filepath: Union[str, Path]) -> bool:
        pass

    @abstractmethod
    def isdir(self, filepath: Union[str, Path]) -> bool:
        pass

    @abstractmethod
    def isfile(self, filepath: Union[str, Path]) -> bool:
        pass

    @abstractmethod
    def join_path(self, filepath: Union[str, Path], *filepaths: Union[str, Path]) -> str:
        pass

    @abstractmethod
    @contextmanager
    def get_local_path(self, filepath: Union[str, Path]) -> Generator[Union[str, Path], None, None]:
        pass

    @abstractmethod
    def copyfile(self, src: Union[str, Path], dst: Union[str, Path]) -> str:
        pass

    @abstractmethod
    def copytree(self, src: Union[str, Path], dst: Union[str, Path]) -> str:
        pass

    @abstractmethod
    def copyfile_from_local(self, src: Union[str, Path], dst: Union[str, Path]) -> str:
        pass

    @abstractmethod
    def copytree_from_local(self, src: Union[str, Path], dst: Union[str, Path]) -> str:
        pass

    @abstractmethod
    def copyfile_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        dst_type: str,  # Choose from ["file", "dir"]
    ) -> Union[str, Path]:
        pass

    @abstractmethod
    def copytree_to_local(self, src: Union[str, Path], dst: Union[str, Path]) -> Union[str, Path]:
        pass

    @abstractmethod
    def remove(self, filepath: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def rmtree(self, dir_path: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def copy_if_symlink_fails(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        pass

    @abstractmethod
    def list_dir(self, dir_path: Union[str, Path]) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def list_dir_or_file(  # pylint: disable=too-many-arguments
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, tuple[str]]] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        pass
