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

import base64
import json
import os

from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.validator import JsonDict, Validator

"""
Base class for parsing environment variables using validators.
Class will go through its list of validators and retrieve values from same named environment variables.
Validators provide:
- default value
- typed parsing
- enforments of mandatory values

Additionally the environment variables can be passed as single base64 encoded string.

we cannot enforce that a component isn't directly using the environment variables.
so evaluation of params should throw error to make sure actual env var is correct.
"""


class EnvParser:
    def __init__(self, b64_str=None):
        if b64_str:
            log.critical(f"b64_str recieved: {b64_str}")
            self.from_b64(b64_str)
        else:
            self.from_env()

    def from_env(self):
        validators = self.get_val_dict()
        for key in validators.keys():
            val = os.getenv(key.upper())
            # log.debug(f"getting env var {key.upper()}: {val}")
            if val:
                setattr(self, key, val)
        self.check_mandatory_values()

    def from_json(self, file_name):
        with open(file_name, "r") as f:
            log.info(f"Reading env params from {file_name}")
            dict = json.load(f)
            for key, value in dict.items():
                setattr(self, key, value)
            self.check_mandatory_values()

    def to_b64(self):
        json_str = self.to_json()
        # create bytes-like object for b64 encoder
        json_str_bytes = json_str.encode()
        b64_str = base64.b64encode(json_str_bytes).decode()

        print(b64_str)
        return b64_str

    def from_b64(self, b64_str):
        json_str = base64.b64decode(b64_str).decode()
        dict = json.loads(json_str)
        for key, value in dict.items():
            setattr(self, key, value)
        self.check_mandatory_values()

    def check_mandatory_values(self):
        for key, validator in self.get_val_dict().items():
            if getattr(self, key) is None and validator.default is None:
                raise ValueError(f"Missing mandatory env var: {key}")

    @classmethod
    def get_val_dict(cls):
        log.debug(f"getting val dict of {cls.__name__}")
        val_dict = {}
        val_dict.update({key: value for key, value in cls.__dict__.items() if isinstance(value, Validator)})

        return val_dict

    def dump_validators(self):
        validators = self.get_val_dict()
        for key, value in validators.items():
            log.debug(f"{key}: {value.__get__(self)}")

    def to_json(self, file_name=None):
        dict = {
            key.upper(): value.__get__(self)
            for key, value in EnvParser.__dict__.items()
            if isinstance(value, Validator)
        }
        json_str = json.dumps(dict, indent=4)
        print(json_str)

        if file_name:
            with open(file_name, "w") as f:
                log.info(f"Writing env params to {file_name}")
                f.write(json_str)

        return json_str

    def to_string_dict(self):
        result = {}
        for key, validator in self.get_val_dict().items():
            value = getattr(self, key)
            if value is None:
                value = validator.default
            if isinstance(validator, JsonDict):
                value = json.dumps(value)
            else:
                value = str(value)
            result[key] = value
        return result

    def __str__(self):
        return ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
