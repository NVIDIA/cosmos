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

import collections.abc as abc
import inspect
from dataclasses import is_dataclass
from typing import ClassVar

import attrs
from omegaconf import DictConfig

from cosmos3._src.imaginaire.lazy_config.registry import convert_target_to_string

__all__ = ["LazyCall"]


def get_default_params(cls_or_func):
    if callable(cls_or_func):
        # inspect signature for function
        signature = inspect.signature(cls_or_func)
    else:
        # inspect signature for class
        signature = inspect.signature(cls_or_func.__init__)
    params = signature.parameters
    default_params = {
        name: param.default for name, param in params.items() if param.default is not inspect.Parameter.empty
    }
    return default_params


_CONVERT_TARGET_TO_STRING: ClassVar[bool] = False
"""Used by tests to enforce conversion of target to string."""


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed,
    but returns a dict that describes the call.

    LazyCall object has to be called with only keyword arguments. Positional
    arguments are not yet supported.

    Examples:
    ::
        from detectron2.config import instantiate, LazyCall

        layer_cfg = LazyCall(nn.Conv2d)(in_channels=32, out_channels=32)
        layer_cfg.out_channels = 64   # can edit it afterwards
        layer = instantiate(layer_cfg)
    """

    def __init__(self, target):
        if not (callable(target) or isinstance(target, (str, abc.Mapping))):
            raise TypeError(f"target of LazyCall must be a callable or defines a callable! Got {target}")
        self._target = target

    def __call__(self, **kwargs):
        if _CONVERT_TARGET_TO_STRING or is_dataclass(self._target) or attrs.has(self._target):
            # omegaconf object cannot hold dataclass type
            # https://github.com/omry/omegaconf/issues/784
            target = convert_target_to_string(self._target)
        else:
            target = self._target
        kwargs["_target_"] = target

        _final_params = get_default_params(self._target)
        _final_params.update(kwargs)

        return DictConfig(content=_final_params, flags={"allow_objects": True})
