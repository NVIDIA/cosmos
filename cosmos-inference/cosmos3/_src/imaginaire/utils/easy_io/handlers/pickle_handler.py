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

import pickle
from io import BytesIO
from typing import Any

from cosmos3._src.imaginaire.utils.easy_io.handlers.base import BaseFileHandler


class PickleHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file: BytesIO, **kwargs):
        return pickle.load(file, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        return super().load_from_path(filepath, mode="rb", **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault("protocol", 2)
        return pickle.dumps(obj, **kwargs)

    def dump_to_fileobj(self, obj: Any, file: BytesIO, **kwargs):
        kwargs.setdefault("protocol", 2)
        pickle.dump(obj, file, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        with open(filepath, "wb") as f:
            pickle.dump(obj, f, **kwargs)
