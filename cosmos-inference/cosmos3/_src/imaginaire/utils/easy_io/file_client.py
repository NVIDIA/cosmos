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

import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, Optional, Tuple, Union

from cosmos3._src.imaginaire.flags import TRAINING
from cosmos3._src.imaginaire.utils.easy_io.backends import BaseStorageBackend, HTTPBackend, LocalBackend


def is_filepath(filepath):
    return isinstance(filepath, (str, Path))


class HardDiskBackend(LocalBackend):
    """Raw hard disks storage backend."""

    @property
    def name(self):
        return self.__class__.__name__


class FileClient:
    """A general file client to access files in different backends.

    The client loads a file or text in a specified backend from its path
    and returns it as a binary or text file. There are two ways to choose a
    backend, the name of backend and the prefix of path. Although both of them
    can be used to choose a storage backend, ``backend`` has a higher priority
    that is if they are all set, the storage backend will be chosen by the
    backend argument. If they are all `None`, the disk backend will be chosen.
    Note that It can also register other backend accessor with a given name,
    prefixes, and backend class. In addition, We use the singleton pattern to
    avoid repeated object creation. If the arguments are the same, the same
    object will be returned.

    Warning:
        `FileClient` will be deprecated in future. Please use io functions
        in https://mmengine.readthedocs.io/en/latest/api/fileio.html#file-io

    Args:
        backend (str, optional): The storage backend type. Options are "disk",
            "memcached", "lmdb", "http" and "s3". Defaults to None.
        prefix (str, optional): The prefix of the registered storage backend.
            Options are "s3", "http", "https". Defaults to None.

    Examples:
        >>> # only set backend
        >>> file_client = FileClient(backend='s3')
        >>> # only set prefix
        >>> file_client = FileClient(prefix='s3')
        >>> # set both backend and prefix but use backend to choose client
        >>> file_client = FileClient(backend='s3', prefix='s3')
        >>> # if the arguments are the same, the same object is returned
        >>> file_client1 = FileClient(backend='s3')
        >>> file_client1 is file_client
        True

    Attributes:
        client (:obj:`BaseStorageBackend`): The backend object.
    """

    _backends = {
        "disk": HardDiskBackend,
        "http": HTTPBackend,
    }

    _prefix_to_backends: dict = {
        "http": HTTPBackend,
        "https": HTTPBackend,
    }

    if TRAINING:
        from cosmos3._src.imaginaire.utils.easy_io.backends.msc_backend import MSCBackend

        _backends["s3"] = MSCBackend
        _backends["msc"] = MSCBackend
        _prefix_to_backends["s3"] = MSCBackend

    _instances: dict = {}

    client: Any

    def __new__(cls, backend=None, prefix=None, **kwargs):
        if backend is None and prefix is None:
            backend = "disk"
        if backend is not None and backend not in cls._backends:
            raise ValueError(
                f"Backend {backend} is not supported. Currently supported ones are {list(cls._backends.keys())}"
            )
        if prefix is not None and prefix not in cls._prefix_to_backends:
            raise ValueError(
                f"prefix {prefix} is not supported. Currently supported ones are {list(cls._prefix_to_backends.keys())}"
            )

        # concatenate the arguments to a unique key for determining whether
        # objects with the same arguments were created
        arg_key = f"{backend}:{prefix}"
        for key, value in kwargs.items():
            arg_key += f":{key}:{value}"

        # if a backend was overridden, it will create a new object
        if arg_key in cls._instances:
            _instance = cls._instances[arg_key]
        else:
            # create a new object and put it to _instance
            _instance = super().__new__(cls)
            if backend is not None:
                _instance.client = cls._backends[backend](**kwargs)
            else:
                _instance.client = cls._prefix_to_backends[prefix](**kwargs)

            cls._instances[arg_key] = _instance

        return _instance

    @property
    def name(self):
        return self.client.name

    @property
    def allow_symlink(self):
        return self.client.allow_symlink

    @staticmethod
    def parse_uri_prefix(uri: Union[str, Path]) -> Optional[str]:
        """Parse the prefix of a uri.

        Args:
            uri (str | Path): Uri to be parsed that contains the file prefix.

        Examples:
            >>> FileClient.parse_uri_prefix('s3://path/of/your/file')
            's3'

        Returns:
            str | None: Return the prefix of uri if the uri contains '://' else
            ``None``.
        """
        assert is_filepath(uri)
        uri = str(uri)
        if "://" not in uri:
            return None
        else:
            prefix, _ = uri.split("://")
            # In the case of MSCBackend, the prefix may contains the cluster
            # name like clusterName:s3
            if ":" in prefix:
                _, prefix = prefix.split(":")
            return prefix

    @classmethod
    def infer_client(
        cls,
        file_client_args: Optional[dict] = None,
        uri: Optional[Union[str, Path]] = None,
    ) -> "FileClient":
        """Infer a suitable file client based on the URI and arguments.

        Args:
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. Defaults to None.
            uri (str | Path, optional): Uri to be parsed that contains the file
                prefix. Defaults to None.

        Examples:
            >>> uri = 's3://path/of/your/file'
            >>> file_client = FileClient.infer_client(uri=uri)
            >>> file_client_args = {'backend': 's3'}
            >>> file_client = FileClient.infer_client(file_client_args)

        Returns:
            FileClient: Instantiated FileClient object.
        """
        assert file_client_args is not None or uri is not None
        if file_client_args is None:
            file_prefix = cls.parse_uri_prefix(uri)  # type: ignore
            return cls(prefix=file_prefix)
        else:
            return cls(**file_client_args)

    @classmethod
    def _register_backend(cls, name, backend, force=False, prefixes=None):
        if not isinstance(name, str):
            raise TypeError(f"the backend name should be a string, but got {type(name)}")
        if not inspect.isclass(backend):
            raise TypeError(f"backend should be a class but got {type(backend)}")
        if not issubclass(backend, BaseStorageBackend):
            raise TypeError(f"backend {backend} is not a subclass of BaseStorageBackend")
        if not force and name in cls._backends:
            raise KeyError(
                f'{name} is already registered as a storage backend, add "force=True" if you want to override it'
            )

        if name in cls._backends and force:
            for arg_key, instance in list(cls._instances.items()):
                if isinstance(instance.client, cls._backends[name]):
                    cls._instances.pop(arg_key)
        cls._backends[name] = backend

        if prefixes is not None:
            if isinstance(prefixes, str):
                prefixes = [prefixes]
            else:
                assert isinstance(prefixes, (list, tuple))
            for prefix in prefixes:
                if prefix not in cls._prefix_to_backends:
                    cls._prefix_to_backends[prefix] = backend
                elif (prefix in cls._prefix_to_backends) and force:
                    overridden_backend = cls._prefix_to_backends[prefix]
                    for arg_key, instance in list(cls._instances.items()):
                        if isinstance(instance.client, overridden_backend):
                            cls._instances.pop(arg_key)
                else:
                    raise KeyError(
                        f"{prefix} is already registered as a storage backend,"
                        ' add "force=True" if you want to override it'
                    )

    @classmethod
    def register_backend(cls, name, backend=None, force=False, prefixes=None):
        """Register a backend to FileClient.

        This method can be used as a normal class method or a decorator.

        .. code-block:: python

            class NewBackend(BaseStorageBackend):

                def get(self, filepath):
                    return filepath

                def get_text(self, filepath):
                    return filepath

            FileClient.register_backend('new', NewBackend)

        or

        .. code-block:: python

            @FileClient.register_backend('new')
            class NewBackend(BaseStorageBackend):

                def get(self, filepath):
                    return filepath

                def get_text(self, filepath):
                    return filepath

        Args:
            name (str): The name of the registered backend.
            backend (class, optional): The backend class to be registered,
                which must be a subclass of :class:`BaseStorageBackend`.
                When this method is used as a decorator, backend is None.
                Defaults to None.
            force (bool, optional): Whether to override the backend if the name
                has already been registered. Defaults to False.
            prefixes (str or list[str] or tuple[str], optional): The prefixes
                of the registered storage backend. Defaults to None.
                `New in version 1.3.15.`
        """
        if backend is not None:
            cls._register_backend(name, backend, force=force, prefixes=prefixes)
            return

        def _register(backend_cls):
            cls._register_backend(name, backend_cls, force=force, prefixes=prefixes)
            return backend_cls

        return _register

    def get(self, filepath: Union[str, Path]) -> Union[bytes, memoryview]:
        """Read data from a given ``filepath`` with 'rb' mode.

        Note:
            There are two types of return values for ``get``, one is ``bytes``
            and the other is ``memoryview``. The advantage of using memoryview
            is that you can avoid copying, and if you want to convert it to
            ``bytes``, you can use ``.tobytes()``.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes | memoryview: Expected bytes object or a memory view of the
            bytes object.
        """
        return self.client.get(filepath)

    def get_text(self, filepath: Union[str, Path], encoding="utf-8") -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        return self.client.get_text(filepath, encoding)

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``put`` should create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        self.client.put(obj, filepath)

    def put_text(self, obj: str, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``put_text`` should create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str, optional): The encoding format used to open the
                `filepath`. Defaults to 'utf-8'.
        """
        self.client.put_text(obj, filepath)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str, Path): Path to be removed.
        """
        self.client.remove(filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        return self.client.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.
        """
        return self.client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.
        """
        return self.client.isfile(filepath)

    def join_path(self, filepath: Union[str, Path], *filepaths: Union[str, Path]) -> str:
        r"""Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of \*filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result of concatenation.
        """
        return self.client.join_path(filepath, *filepaths)

    @contextmanager
    def get_local_path(self, filepath: Union[str, Path]) -> Generator[Union[str, Path], None, None]:
        """Download data from ``filepath`` and write the data to local path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Note:
            If the ``filepath`` is a local path, just return itself.

        .. warning::
            ``get_local_path`` is an experimental interface that may change in
            the future.

        Args:
            filepath (str or Path): Path to be read data.

        Examples:
            >>> file_client = FileClient(prefix='s3')
            >>> with file_client.get_local_path('s3://bucket/abc.jpg') as path:
            ...     # do something here

        Yields:
            Iterable[str]: Only yield one path.
        """
        with self.client.get_local_path(str(filepath)) as local_path:
            yield local_path

    def list_dir_or_file(  # pylint: disable=too-many-arguments
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, Tuple[str]]] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.

        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.

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
        """
        yield from self.client.list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive)
