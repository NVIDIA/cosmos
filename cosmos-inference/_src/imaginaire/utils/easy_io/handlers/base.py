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

from abc import ABCMeta, abstractmethod


class BaseFileHandler(metaclass=ABCMeta):
    # `str_like` is a flag to indicate whether the type of file object is
    # str-like object or bytes-like object. Pickle only processes bytes-like
    # objects but json only processes str-like object. If it is str-like
    # object, `StringIO` will be used to process the buffer.
    str_like = True

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        pass

    def load_from_path(self, filepath, mode="r", **kwargs):
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath, mode="w", **kwargs):
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)
