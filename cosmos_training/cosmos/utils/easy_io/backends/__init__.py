# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cosmos.utils.flags import TRAINING
from cosmos.utils.easy_io.backends.base_backend import BaseStorageBackend
from cosmos.utils.easy_io.backends.http_backend import HTTPBackend
from cosmos.utils.easy_io.backends.local_backend import LocalBackend
from cosmos.utils.easy_io.backends.registry_utils import backends, prefix_to_backends, register_backend

__all__ = [
    "BaseStorageBackend",
    "LocalBackend",
    "HTTPBackend",
    "register_backend",
    "backends",
    "prefix_to_backends",
]

if TRAINING:
    from cosmos.utils.easy_io.backends.boto3_backend import Boto3Backend
    from cosmos.utils.easy_io.backends.msc_backend import MSCBackend

    __all__ += [
        "Boto3Backend",
        "MSCBackend",
    ]
