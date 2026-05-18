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
import binascii
import itertools
import json
import os
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any

# Sentinel value to indicate that no default was explicitly set by the user
# we want to mimic usage of function parameters: if no default is provided, the parameter is mandatory
_UNSET = object()


# from https://docs.python.org/3/howto/descriptor.html#validator-class


# validators can be customized to very specific needs, e.g. see HumanAttributes below
class Validator(ABC):
    def __init__(self, default=_UNSET, hidden=False):
        self.default = default
        self.hidden = hidden

    # set name is called when the validator is created as class variable
    # name is the name of the variable in the owner class, so here we create the name for the backing variable
    def __set_name__(self, owner, name):
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        value = getattr(obj, self.private_name, self.default)
        if value is _UNSET:
            # If we reach here, it means a mandatory parameter was accessed without being set
            attr_name = getattr(self, "private_name", "unknown").lstrip("_")
            raise ValueError(
                f"Parameter '{attr_name}' is mandatory but has not been set. "
                f"No default value was provided and no value was assigned."
            )
        return value

    def __set__(self, obj, value):
        value = self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        pass

    def json(self):
        pass


class Bool(Validator):
    def __init__(self, default=_UNSET, hidden=False, tooltip=None):
        super().__init__(default, hidden)
        self.default = default
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value):
        if isinstance(value, int):
            value = value != 0
        elif isinstance(value, str):
            value = value.lower()
            if value in ["true", "1"]:
                value = True
            elif value in ["false", "0"]:
                value = False
            else:
                raise ValueError(f"Expected {value!r} to be one of ['True', 'False', '1', '0']")
        elif not isinstance(value, bool):
            raise TypeError(f"Expected {value!r} to be an bool")

        return value

    def get_range_iterator(self):
        return [True, False]

    def __repr__(self) -> str:
        return f"Bool({self.private_name=} {self.default=} {self.hidden=})"

    def json(self):
        return {
            "type": bool.__name__,
            "default": self.default,
            "tooltip": self.tooltip,
        }


class Int(Validator):
    def __init__(self, default=_UNSET, min=None, max=None, step=1, hidden=False, tooltip=None):
        self.min = min
        self.max = max
        self.default = default
        self.step = step
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value):
        if isinstance(value, str):
            value = int(value)
        elif not isinstance(value, int):
            raise TypeError(f"Expected {value!r} to be an int")

        if self.min is not None and value < self.min:
            raise ValueError(f"Expected {value!r} to be at least {self.min!r}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Expected {value!r} to be no more than {self.max!r}")
        return value

    def get_range_iterator(self):
        if self.default is _UNSET:
            default_val = 0
        else:
            default_val = int(self.default) if isinstance(self.default, (int, float, str)) else 0
        iter_min = self.min if self.min is not None else default_val
        iter_max = self.max if self.max is not None else (default_val + 100)
        return itertools.takewhile(lambda x: x <= iter_max, itertools.count(iter_min, self.step))

    def __repr__(self) -> str:
        return f"Int({self.private_name=} {self.default=}, {self.min=}, {self.max=} {self.hidden=})"

    def json(self):
        return {
            "type": int.__name__,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "tooltip": self.tooltip,
        }


class Float(Validator):
    def __init__(self, default=_UNSET, min=None, max=None, step=0.5, hidden=False, tooltip=None):
        self.min = min
        self.max = max
        self.default = default
        self.step = step
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value):
        if isinstance(value, str) or isinstance(value, int):
            value = float(value)
        elif not isinstance(value, float):
            raise TypeError(f"Expected {value!r} to be float")

        if self.min is not None and value < self.min:
            raise ValueError(f"Expected {value!r} to be at least {self.min!r}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Expected {value!r} to be no more than {self.max!r}")
        return value

    def get_range_iterator(self):
        if self.default is _UNSET:
            default_val = 0.0
        else:
            default_val = float(self.default) if isinstance(self.default, (int, float, str)) else 0.0
        iter_min = self.min if self.min is not None else default_val
        iter_max = self.max if self.max is not None else (default_val + 100.0)
        return itertools.takewhile(lambda x: x <= iter_max, itertools.count(iter_min, self.step))

    def __repr__(self) -> str:
        return f"Float({self.private_name=} {self.default=}, {self.min=}, {self.max=} {self.hidden=})"

    def json(self):
        return {
            "type": float.__name__,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "tooltip": self.tooltip,
        }


class String(Validator):
    def __init__(self, default=_UNSET, min=None, max=None, predicate=None, hidden=False, tooltip=None):
        self.min = min
        self.max = max
        self.predicate = predicate
        self.default = default
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value):
        if value is None:
            return value  # Allow None as a valid value to be compatible with existing code
            # this breaks strict typing, so do this only for strings
        if not isinstance(value, str):
            raise TypeError(f"Expected {value!r} to be an str or None")
        if self.min is not None and len(value) < self.min:
            raise ValueError(f"Expected {value!r} to be no smaller than {self.min!r}")
        if self.max is not None and len(value) > self.max:
            raise ValueError(f"Expected {value!r} to be no bigger than {self.max!r}")
        if self.predicate is not None and not self.predicate(value):
            raise ValueError(f"Expected {self.predicate} to be true for {value!r}")
        return value

    def get_range_iterator(self):
        return iter([self.default])

    def __repr__(self) -> str:
        return f"String({self.private_name=} {self.default=}, {self.min=}, {self.max=} {self.hidden=})"

    def json(self):
        return {
            "type": str.__name__,
            "default": self.default,
            "tooltip": self.tooltip,
        }


class Path(Validator):
    def __init__(self, default=_UNSET, hidden=False, tooltip=None):
        self.default = default
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value):
        if value is None:
            return value
        if not isinstance(value, str):
            raise TypeError(f"{self.private_name} validator: Expected {value!r} to be an str")
        if not os.path.exists(value):
            raise ValueError(f"{self.private_name} validator: Expected {value!r} to be a valid path")

        return value

    def get_range_iterator(self):
        return iter([self.default])

    def __repr__(self) -> str:
        return f"String({self.private_name=} {self.default=}, {self.hidden=})"


class InputImage(Validator):
    def __init__(
        self, default=_UNSET, hidden=False, tooltip=None, supported_formats=["jpeg", "jpg", "png", "bmp", "gif"]
    ):
        self.default = default
        self.hidden = hidden
        self.tooltip = tooltip
        self.supported_formats = supported_formats

    def validate(self, value):
        ext = os.path.splitext(value)[1].lower()

        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {ext}")

        if not isinstance(value, str):
            raise TypeError(f"Expected {value!r} to be an str")
        if not os.path.exists(value):
            raise ValueError(f"Expected {value!r} to be a valid path")
        return value

    def get_range_iterator(self):
        return iter([self.default])

    def __repr__(self) -> str:
        return f"String({self.private_name=} {self.default=} {self.hidden=})"

    def json(self):
        return {
            "type": InputImage.__name__,
            "default": self.default,
            "values": self.supported_formats,
            "tooltip": self.tooltip,
        }


class JsonDict(Validator):
    """
    JSON stringified version of a python dict.
    Example: '{"ema_customization_iter.pt": "ema_customization_iter.pt"}'
    """

    def __init__(self, default=_UNSET, hidden=False):
        self.default = default
        self.hidden = hidden

    def validate(self, value):
        if not value:
            return {}
        try:
            dict = json.loads(value)
            return dict
        except json.JSONDecodeError as e:
            raise ValueError(f"Expected {value!r} to be json  stringified dict. Error: {str(e)}")

    def __repr__(self) -> str:
        return f"Dict({self.default=} {self.hidden=})"


class Dict(Validator):
    """
    Python dict.
    Example: {'key': 'value'}

    This allows a single level of parameter nesting, but not a full nested dict.
    For now we validate the individual keys here and store the dict as is.
    Alternatively, we could have a validator that gets/sets another ValidatorParams class.
    """

    def __init__(self, default=_UNSET, hidden=False):
        self.default = default
        self.hidden = hidden

    def validate(self, value):
        if not isinstance(value, dict):
            raise TypeError(f"Expected {value!r} to be an dict")
        return value

    def __repr__(self) -> str:
        value = getattr(self, self.private_name, self.default)

        return f"Dict({self.private_name=} {self.default=} {self.hidden=} value={json.dumps(value, indent=4)})"


class OneOf(Validator):
    def __init__(self, default=_UNSET, options=None, type_cast=None, hidden=False, tooltip=None):
        self.options = set(options) if options is not None else set()
        self.default = default
        self.type_cast = type_cast  # Cast the value to this type before checking if it's in options
        self.tooltip = tooltip
        self.hidden = hidden

    def validate(self, value):
        if self.type_cast:
            try:
                value = self.type_cast(value)
            except ValueError:
                raise ValueError(f"Expected {value!r} to be castable to {self.type_cast!r}")

        if value not in self.options:
            raise ValueError(f"Expected {value!r} to be one of {self.options!r}")

        return value

    def get_range_iterator(self):
        return self.options

    def __repr__(self) -> str:
        return f"OneOf({self.private_name=} {self.options=} {self.hidden=})"

    def json(self):
        return {
            "type": OneOf.__name__,
            "default": self.default,
            "values": list(self.options),
            "tooltip": self.tooltip,
        }


class MultipleOf(Validator):
    def __init__(self, default=_UNSET, multiple_of: int = 1, type_cast=None, hidden=False, tooltip=None):
        if type(multiple_of) is not int:
            raise ValueError(f"Expected {multiple_of!r} to be an int")
        self.multiple_of = multiple_of
        self.default = default
        self.type_cast = type_cast

        # if a parameter is hidden then probe() can't expose the param
        # and the param can't be set anymore
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value):
        if self.type_cast:
            try:
                value = self.type_cast(value)
            except ValueError:
                raise ValueError(f"Expected {value!r} to be castable to {self.type_cast!r}")

        if value % self.multiple_of != 0:
            raise ValueError(f"Expected {value!r} to be a multiple of {self.multiple_of!r}")

        return value

    def get_range_iterator(self):
        return itertools.count(0, self.multiple_of)

    def __repr__(self) -> str:
        return f"MultipleOf({self.private_name=} {self.multiple_of=} {self.hidden=})"

    def json(self):
        return {
            "type": MultipleOf.__name__,
            "default": self.default,
            "multiple_of": self.multiple_of,
            "tooltip": self.tooltip,
        }


class HumanAttributes(Validator):
    def __init__(self, default=_UNSET, hidden=False, tooltip=None):
        self.default = default
        self.hidden = hidden
        self.tooltip = tooltip

    # hard code the options for now
    # we extend this to init parameter as needed
    valid_attributes = {
        "emotion": ["angry", "contemptful", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"],
        "race": ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"],
        "gender": ["male", "female"],
        "age group": [
            "young",
            "teen",
            "adult early twenties",
            "adult late twenties",
            "adult early thirties",
            "adult late thirties",
            "adult middle aged",
            "older adult",
        ],
    }

    def get_range_iterator(self):
        # create a list of all possible combinations
        l1 = self.valid_attributes["emotion"]
        l2 = self.valid_attributes["race"]
        l3 = self.valid_attributes["gender"]
        l4 = self.valid_attributes["age group"]
        all_combinations = list(itertools.product(l1, l2, l3, l4))
        return iter(all_combinations)

    def validate(self, value):
        human_attributes = value.lower()
        if human_attributes not in ["none", "random"]:
            # In this case, we need for custom attribute string

            attr_string = human_attributes
            for attr_key in ["emotion", "race", "gender", "age group"]:
                attr_detected = False
                for attr_label in self.valid_attributes[attr_key]:
                    if attr_string.startswith(attr_label):
                        attr_string = attr_string[len(attr_label) + 1 :]  # noqa: E203
                        attr_detected = True
                        break

                if attr_detected is False:
                    raise ValueError(f"Expected {value!r} to be one of {self.valid_attributes!r}")

        return value

    def __repr__(self) -> str:
        return f"HumanAttributes({self.private_name=} {self.hidden=})"

    def json(self):
        return {
            "type": HumanAttributes.__name__,
            "default": self.default,
            "values": self.valid_attributes,
            "tooltip": self.tooltip,
        }


class BytesIOType(Validator):
    """
    Validator class for BytesIO. Valid inputs are either:
    - bytes
    - objects of class BytesIO
    - str which can be successfully  decoded into BytesIO
    """

    def __init__(self, default=_UNSET, hidden=False, tooltip=None):
        self.default = default
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value: Any) -> BytesIO:
        if isinstance(value, str):
            try:
                # Decode the Base64 string
                decoded_bytes = base64.b64decode(value)
                # Create a BytesIO stream from the decoded bytes
                return BytesIO(decoded_bytes)
            except (binascii.Error, ValueError) as e:
                raise ValueError(f"Invalid Base64 encoded string: {e}")
        elif isinstance(value, bytes):
            return BytesIO(value)
        elif isinstance(value, BytesIO):
            return value
        else:
            raise TypeError(f"Expected {value!r} to be a Base64 encoded string, bytes, or BytesIO")

    def __repr__(self) -> str:
        return f"BytesIOValidator({self.default=}, {self.hidden=})"

    def json(self):
        return {
            "type": BytesIO.__name__,
            "default": self.default,
            "tooltip": self.tooltip,
        }
