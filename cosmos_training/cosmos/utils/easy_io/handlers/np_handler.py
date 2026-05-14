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

from io import BytesIO
from typing import IO, Any

import numpy as np

from cosmos.utils.easy_io.handlers.base import BaseFileHandler


class NumpyHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file: IO[bytes], **kwargs) -> Any:
        """
        Load a NumPy array from a file-like object.

        Parameters:
            file (IO[bytes]): The file-like object containing the NumPy array data.
            **kwargs: Additional keyword arguments passed to `np.load`.

        Returns:
            numpy.ndarray: The loaded NumPy array.
        """
        return np.load(file, **kwargs)

    def load_from_path(self, filepath: str, **kwargs) -> Any:
        """
        Load a NumPy array from a file path.

        Parameters:
            filepath (str): The path to the file to load.
            **kwargs: Additional keyword arguments passed to `np.load`.

        Returns:
            numpy.ndarray: The loaded NumPy array.
        """
        return super().load_from_path(filepath, mode="rb", **kwargs)

    def dump_to_str(self, obj: np.ndarray, **kwargs) -> str:
        """
        Serialize a NumPy array to a string in binary format.

        Parameters:
            obj (np.ndarray): The NumPy array to serialize.
            **kwargs: Additional keyword arguments passed to `np.save`.

        Returns:
            str: The serialized NumPy array as a string.
        """
        with BytesIO() as f:
            np.save(f, obj, **kwargs)
            return f.getvalue()

    def dump_to_fileobj(self, obj: np.ndarray, file: IO[bytes], **kwargs):
        """
        Dump a NumPy array to a file-like object.

        Parameters:
            obj (np.ndarray): The NumPy array to dump.
            file (IO[bytes]): The file-like object to which the array is dumped.
            **kwargs: Additional keyword arguments passed to `np.save`.
        """
        np.save(file, obj, **kwargs)

    def dump_to_path(self, obj: np.ndarray, filepath: str, **kwargs):
        """
        Dump a NumPy array to a file path.

        Parameters:
            obj (np.ndarray): The NumPy array to dump.
            filepath (str): The file path where the array should be saved.
            **kwargs: Additional keyword arguments passed to `np.save`.
        """
        with open(filepath, "wb") as f:
            np.save(f, obj, **kwargs)
