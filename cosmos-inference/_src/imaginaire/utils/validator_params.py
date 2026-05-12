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

import json
import pprint
import shlex

from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.validator import _UNSET, Validator

"""
Base class for all model parameter classes.

The primary purpose is to fully validate any input parameter including type, range, etc.
By using custom validators, we can additionally validate complex parameters such as images, text, etc.
Additioally, the class can parse command line arguments into a dictionary of parameters
and create a model parameter class from a dictionary of parameters.

if default of a validator is _UNSET, the parameter is mandatory and must be provided by the user.
Hence validators without explicit defaults require user input.
"""


class ValidatorParams:
    """
    factory method to create a model params class from a given api and a dictionary of args
    in comparison to createFromCmd, the server can first parse and modify some args,
    finally use this factory method to create the model params
    """

    @classmethod
    def create(cls, kwargs):
        instance = cls()
        log.info(f"creating model params class={cls}")
        instance.from_kwargs(kwargs)

        val_dict = cls.get_val_dict()

        for key, validator in val_dict.items():
            # Check if validator has no user-provided default (_UNSET) and no value was assigned
            if validator.default is _UNSET:
                value = getattr(instance, key, _UNSET)
                if value is _UNSET:
                    raise ValueError(
                        f"mandatory parameter {key} is missing - no default provided and no value assigned by user"
                    )

        return instance

    """
    factory method to create a model params class from a command string
    """

    @classmethod
    def createFromCmd(cls, cmd: str) -> object:
        kwargs = cls.parse(cmd)
        return cls.create(kwargs)

    def from_kwargs(self, kwargs):
        # most attributes of this class are validators,
        # but dervied class could add non-validators
        # or some validators might be hidden
        # therefore only allow exposed params to be set
        for key, value in kwargs.items():
            if key in self.get_exposed_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"unknown parameter {key} in command line")

    def to_kwargs(self) -> dict:
        """for a given config return a dictionary of all the parameters and their values"""
        param_names = self.get_exposed_params()
        return {key: getattr(self, key) for key in param_names}

    @classmethod
    def validate_kwargs(cls, kwargs) -> dict:
        """validate a dictionary of args and return the validated dictionary"""
        instance = cls.create(kwargs)
        return instance.to_kwargs()

    @staticmethod
    def parse(cmd: str) -> dict:
        """parse a command string into an api command (e.g. text2image) and a dictionary of args"""
        args = {}
        pairs = shlex.split(cmd)

        for arg in pairs:
            key, value = arg.split("=", 1)  # Split only on the first '='
            value = value.strip().strip("'")
            key = key.strip("--")
            args[key.strip()] = value

        log.debug(f"parsed cmd-line: {args}")
        return args

    @classmethod
    def probe(cls) -> list[str]:
        params = cls.get_exposed_params()
        log.info(f"exposed params for {cls}: {params}")
        return params

    """
    extened version of probe will query from each validator extended information.
    This will include default parameters, min, max, step, etc.
    """

    @classmethod
    def probe_ex(cls) -> dict:
        validator_dict = cls.get_val_dict()

        parameter_info = {key: value.json() for key, value in validator_dict.items() if not value.hidden}
        log.info(f"exposed params for {cls}: {json.dumps(parameter_info, indent=4)}")
        return parameter_info

    # a model parameter class can also have non exposed parameters:
    # we can hide parameters as needed from public API (compare to former exposed_params list in yaml configs in imaginaire3)
    # class can also have non-validator attributes
    @classmethod
    def get_exposed_params(cls) -> list[str]:
        # log.debug(f"getting exposed params of {cls.__name__}")

        # the exposed params are repeatedly used for parsing so we cache them
        # note that we are caching the exposed params per class in the class hierarchy!
        # each class has its own set of exposed params.
        # instances of the class will have the same set of exposed params
        if "_exposed_params" not in cls.__dict__:
            # log.debug(f"creating cache exposed params of {cls.__name__}")
            validator_dict = cls.get_val_dict()

            # if a parameter is hidden then probe() can't expose the param
            # and the param can't be set anymore
            cls._exposed_params = [key for key, value in validator_dict.items() if not value.hidden]
        return cls._exposed_params

    def exposed_params_dict(self):
        keys = self.get_exposed_params()
        out_dict = {key: getattr(self, key) for key in keys}
        return out_dict

    """
    returns a dictionary of all validators in the class hierarchy, e.g. for a string validator:

        prompt_validator = String()

    so prompt_validator is the instance of the String validator. the dictionary will be:

        {'prompt_validator': prompt_validator}
    """

    @classmethod
    def get_val_dict(cls) -> dict[str, Validator]:
        # log.debug(f"getting val dict of {cls.__name__}")
        val_dict = {}
        if cls is not ValidatorParams:
            val_dict.update(cls.__bases__[0].get_val_dict())

        val_dict.update({key: value for key, value in cls.__dict__.items() if isinstance(value, Validator)})

        return val_dict

    @classmethod
    def debug_print(cls):
        pp = pprint.PrettyPrinter(indent=4)
        print(f"*********** validator dict for {cls.__name__} ***********")
        val_dict = cls.get_val_dict()
        pp.pprint(val_dict)

    def __str__(self):
        return ", ".join(f"{key}={value}" for key, value in self.__dict__.items())

    def __repr__(self):
        return ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
