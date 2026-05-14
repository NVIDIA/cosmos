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

try:
    import torch
except ImportError:
    torch = None

from cosmos.utils.easy_io.handlers.base import BaseFileHandler


class TorchJitHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        return torch.jit.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        torch.jit.save(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError
