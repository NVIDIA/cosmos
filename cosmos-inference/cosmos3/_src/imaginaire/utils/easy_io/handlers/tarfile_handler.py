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

import tarfile

from cosmos3._src.imaginaire.utils.easy_io.handlers.base import BaseFileHandler


class TarHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, mode="r|*", **kwargs):
        return tarfile.open(fileobj=file, mode=mode, **kwargs)

    def load_from_path(self, filepath, mode="r|*", **kwargs):
        return tarfile.open(filepath, mode=mode, **kwargs)

    def dump_to_fileobj(self, obj, file, mode="w", **kwargs):
        with tarfile.open(fileobj=file, mode=mode) as tar:
            tar.add(obj, **kwargs)

    def dump_to_path(self, obj, filepath, mode="w", **kwargs):
        with tarfile.open(filepath, mode=mode) as tar:
            tar.add(obj, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError
