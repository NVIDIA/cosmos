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


def is_primitive(value):
    return isinstance(value, (int, float, str, bool, type(None)))


def convert_to_primitive(value):
    if isinstance(value, (list, tuple)):
        return [convert_to_primitive(v) for v in value if is_primitive(v) or isinstance(v, (list, dict))]
    elif isinstance(value, dict):
        return {k: convert_to_primitive(v) for k, v in value.items() if is_primitive(v) or isinstance(v, (list, dict))}
    elif is_primitive(value):
        return value
    else:
        return "non-primitive"  # Skip non-primitive types
