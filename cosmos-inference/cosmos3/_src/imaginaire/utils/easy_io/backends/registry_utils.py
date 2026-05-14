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
from typing import Optional, Type, Union

from cosmos3._src.imaginaire.flags import TRAINING
from cosmos3._src.imaginaire.utils.easy_io.backends.base_backend import BaseStorageBackend
from cosmos3._src.imaginaire.utils.easy_io.backends.http_backend import HTTPBackend
from cosmos3._src.imaginaire.utils.easy_io.backends.local_backend import LocalBackend

backends: dict = {}
prefix_to_backends: dict = {}


def _register_backend(
    name: str,
    backend: Type[BaseStorageBackend],
    force: bool = False,
    prefixes: Union[str, list, tuple, None] = None,
):
    """Register a backend.

    Args:
        name (str): The name of the registered backend.
        backend (BaseStorageBackend): The backend class to be registered,
            which must be a subclass of :class:`BaseStorageBackend`.
        force (bool): Whether to override the backend if the name has already
            been registered. Defaults to False.
        prefixes (str or list[str] or tuple[str], optional): The prefix
            of the registered storage backend. Defaults to None.
    """
    global backends, prefix_to_backends

    if not isinstance(name, str):
        raise TypeError(f"the backend name should be a string, but got {type(name)}")

    if not inspect.isclass(backend):
        raise TypeError(f"backend should be a class, but got {type(backend)}")
    if not issubclass(backend, BaseStorageBackend):
        raise TypeError(f"backend {backend} is not a subclass of BaseStorageBackend")

    if name in backends and not force:
        raise ValueError(
            f'{name} is already registered as a storage backend, add "force=True" if you want to override it'
        )
    backends[name] = backend

    if prefixes is not None:
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))

        for prefix in prefixes:
            if prefix in prefix_to_backends and not force:
                raise ValueError(
                    f'{prefix} is already registered as a storage backend, add "force=True" if you want to override it'
                )

            prefix_to_backends[prefix] = backend


def register_backend(
    name: str,
    backend: Optional[Type[BaseStorageBackend]] = None,
    force: bool = False,
    prefixes: Union[str, list, tuple, None] = None,
):
    """Register a backend.

    Args:
        name (str): The name of the registered backend.
        backend (class, optional): The backend class to be registered,
            which must be a subclass of :class:`BaseStorageBackend`.
            When this method is used as a decorator, backend is None.
            Defaults to None.
        force (bool): Whether to override the backend if the name has already
            been registered. Defaults to False.
        prefixes (str or list[str] or tuple[str], optional): The prefix
            of the registered storage backend. Defaults to None.

    This method can be used as a normal method or a decorator.

    Examples:

        >>> class NewBackend(BaseStorageBackend):
        ...     def get(self, filepath):
        ...         return filepath
        ...
        ...     def get_text(self, filepath):
        ...         return filepath
        >>> register_backend('new', NewBackend)

        >>> @register_backend('new')
        ... class NewBackend(BaseStorageBackend):
        ...     def get(self, filepath):
        ...         return filepath
        ...
        ...     def get_text(self, filepath):
        ...         return filepath
    """
    if backend is not None:
        _register_backend(name, backend, force=force, prefixes=prefixes)
        return

    def _register(backend_cls):
        _register_backend(name, backend_cls, force=force, prefixes=prefixes)
        return backend_cls

    return _register


register_backend("local", LocalBackend, prefixes="")
register_backend("http", HTTPBackend, prefixes=["http", "https"])

if TRAINING:
    from cosmos3._src.imaginaire.utils.easy_io.backends.msc_backend import MSCBackend

    # To avoid breaking backward Compatibility, 's3' is also used as a
    # prefix for MSCBackend
    register_backend("s3", MSCBackend, prefixes=["s3"])
