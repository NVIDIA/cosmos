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

import abc
import functools
import importlib
import json
import os
from collections.abc import Callable as Callable2
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from types import UnionType
from typing import Any, List, Optional, TypeVar, Union, get_args, get_origin

import attrs
import torch
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from cosmos3._src.imaginaire.lazy_config import LazyCall, LazyDict, instantiate
from cosmos3._src.imaginaire.lazy_config.lazy import get_default_params

T = TypeVar("T")


def from_dict(
    x: dict, clazz: str | type | None = None, force_construct_target: bool | None = None, field_name: str = ""
) -> T: ...
def to_dict(x: T, field_name: str = "", hydra_compat: bool = True) -> dict: ...
def from_yaml(path: str | None = None, clazz: type | None = None, file_like_or_str=None) -> T:
    if path:
        assert os.path.exists(path), f"{path} does not exist"
        with open(path) as in_f:
            return from_dict(yaml.safe_load(in_f), clazz=clazz)
    elif file_like_or_str:
        return from_dict(yaml.safe_load(file_like_or_str), clazz=clazz)
    else:
        raise ValueError("expected file_like_or_str or path to not be None")


def _yaml_safe(obj: Any) -> Any:
    # primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # dict-like
    if isinstance(obj, Mapping):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}

    # list/tuple-like (but not strings/bytes)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_yaml_safe(v) for v in obj]

    # classes / functions / bound methods -> import path
    if hasattr(obj, "__module__") and hasattr(obj, "__qualname__"):
        return f"{obj.__module__}.{obj.__qualname__}"

    # torch dtype, Path, enums, dataclasses, etc.
    return str(obj)


def to_yaml(config: T, out_path: str | None = None) -> str | None:
    config_dict = to_dict(config)
    safe_dict = _yaml_safe(config_dict)

    if out_path is not None:
        with open(out_path, "w") as f:
            yaml.safe_dump(
                safe_dict,
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
        return None

    return yaml.safe_dump(
        safe_dict,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )


def load_callable(name: str) -> Callable2 | None:
    if not name:
        return None

    idx = name.rfind(".")
    assert idx != -1, "expected <module_name>.<name>"
    module_name = name[0:idx]
    fn_name = name[idx + 1 :]
    mod = importlib.import_module(module_name)
    return getattr(mod, fn_name)


def maybe_load_callable(name: str | Callable2 | None) -> Callable2 | None:
    if isinstance(name, str):
        return load_callable(name)

    return name


def maybe_idx(x: Any, idx: int) -> Any:
    if idx < 0 or idx >= len(x):
        return None
    return x[idx]


def is_attrs(x: Any) -> bool:
    return hasattr(x, "__attrs_attrs__")


def to_qualitified_name(x) -> str:
    # Handle functools.partial explicitly
    if isinstance(x, functools.partial):
        fn = x.func
        fn_name = to_qualitified_name(fn)

        # args/keywords may contain non-serializable stuff; stringify safely
        args = []
        if x.args:
            args = [repr(a) for a in x.args]

        kwargs = {}
        if x.keywords:
            kwargs = {str(k): repr(v) for k, v in x.keywords.items()}

        if args or kwargs:
            return f"functools.partial({fn_name}, args={args}, kwargs={kwargs})"
        return f"functools.partial({fn_name})"

    # Normal callable/class/module qualified name
    mod = getattr(x, "__module__", None)
    qn = getattr(x, "__qualname__", None)

    if mod and qn:
        return f"{mod}.{qn}"

    # Some callables only have __name__
    name = getattr(x, "__name__", None)
    if mod and name:
        return f"{mod}.{name}"

    # Fallback: repr
    return repr(x)


def is_optional(x: type) -> bool:
    origin = get_origin(x)
    args = get_args(x)
    return origin is Optional or (origin in (Union, UnionType) and len(args) == 2 and type(None) in args)


def _to_dict_value(x: T, field_type: type, metadata: dict, field_name: str = ""):

    t = type(x)

    # attrs specific
    if x is attrs.NOTHING or x is None:
        return None
    # torch specifics
    elif field_type in (torch.memory_format, torch.dtype):
        return str(x)
    # i4 specific types
    elif field_type == LazyCall:
        result = _to_dict_value(x, field_type._target, metadata, field_name)
        return result
    elif field_type in (DictConfig, LazyDict):
        if "_target_" in x:
            default_params = get_default_params(x["_target_"])
            for default_key, default_v in default_params.items():
                if default_key not in x:
                    x[default_key] = default_v
        result = _to_dict_value(x, dict, metadata, field_name)
        object_type = getattr(x._metadata, "object_type", None)
        if object_type and (is_dataclass(object_type) or is_attrs(object_type)):
            result.setdefault("_target_", to_qualitified_name(object_type))
        return result
    elif field_type == ListConfig:
        return _to_dict_value(x, list, metadata, field_name)
    # general python types + dataclasses + attrs
    # * meta types
    elif field_type == type or field_type == abc.ABCMeta:

        return to_qualitified_name(x)
    elif get_origin(field_type) is type:
        return to_qualitified_name(x)
    elif callable(x) or get_origin(field_type) is Callable2:
        if callable(x):
            return to_qualitified_name(x)
        else:
            assert isinstance(x, str), f"{x.__class__=}"
            return x
    elif is_dataclass(t) or is_attrs(t):
        return to_dict(x, field_name=field_name)
    # * built-in composites types
    elif is_optional(field_type):
        return _to_dict_value(x, get_args(field_type)[0], metadata)
    elif get_origin(field_type) in (Union, UnionType):
        raise AssertionError("unions are not implemented yet!")
    # * primitives
    elif t in (dict,) or field_type in (dict,) or get_origin(field_type) in (dict,):
        return {
            _to_dict_value(
                k,
                maybe_idx(get_args(field_type), 0) or type(k),
                metadata,
                field_name=f"{field_name}.{k}.key",
            ): _to_dict_value(
                v,
                maybe_idx(get_args(field_type), 1) or type(v),
                metadata,
                field_name=f"{field_name}.{k}",
            )
            for k, v in x.items()
        }
    elif (
        t
        in (
            tuple,
            list,
        )
        or field_type
        in (
            tuple,
            list,
        )
        or get_origin(field_type) in (tuple, list)
    ):
        if field_type is None or field_type not in (
            tuple,
            list,
        ):
            field_type = list

        return field_type(
            [
                _to_dict_value(xx, maybe_idx(get_args(field_type), 0) or type(xx), metadata, field_name + f"[{i}]")
                for i, xx in enumerate(x)
            ]
        )
    elif field_type in (int, str, float, bool):
        result = field_type(x)
        return result
    else:  # catch all for everything else
        return x


def to_dict(x: T, field_name: str = "", hydra_compat: bool = True) -> dict:
    if is_dataclass(x):
        result = {}
        if hydra_compat:
            result["_target_"] = to_qualitified_name(x.__class__)
        for f in fields(x):

            if hydra_compat and f.name == "defaults":
                continue
            result[f.name] = _to_dict_value(
                x.__dict__[f.name],
                f.type,
                f.metadata,
                field_name=field_name + f".{f.name}" if field_name else f.name,
            )
        return result
    elif is_attrs(x):
        # references:
        # - https://github.com/python-attrs/attrs/blob/main/src/attr/_funcs.py
        attrs.resolve_types(x.__class__)

        result = {}
        if hydra_compat:
            result["_target_"] = to_qualitified_name(x.__class__)
        for f in attrs.fields(x.__class__):

            if hydra_compat and f.name == "defaults":
                continue
            result[f.name] = _to_dict_value(
                getattr(x, f.name),
                f.type,
                f.metadata,
                field_name=field_name + f".{f.name}" if field_name else f.name,
            )
        return result


def _from_dict_value(
    x: T,
    field_type: type,
    concrete_type: type,
    field_name: str,
    force_construct_target: bool | None = None,
):


    is_dc_type = is_dataclass(field_type)
    is_attrs_type = is_attrs(field_type)
    origin = get_origin(field_type) or field_type
    args = get_args(field_type)

    if x is None:
        return None
    elif field_type in (torch.memory_format, torch.dtype):
        return maybe_load_callable(x)
    elif field_type == LazyCall:
        return _from_dict_value(x, field_type._target, concrete_type, field_name=field_name)
    elif is_dc_type or is_attrs_type:
        if concrete_type == str:
            assert isinstance(x, str)
            if x.endswith(".json"):
                json_value = json.loads(x)
                return from_dict(
                    json_value, field_type, force_construct_target=force_construct_target, field_name=field_name
                )
            elif x.endswith(".yaml"):
                yaml_value = yaml.safe_load(x)
                return from_dict(
                    yaml_value, field_type, force_construct_target=force_construct_target, field_name=field_name
                )
            else:
                raise AssertionError(f"unexpected string: {x}")
        else:
            assert not isinstance(x, str)
            return from_dict(x, field_type, field_name=field_name)
    elif field_type in (DictConfig, LazyDict) or origin in (dict,):

        construct_target = x.get("_recursive_", field_type == DictConfig)
        if force_construct_target is not None:
            construct_target = force_construct_target

        target_value = x.get("_target_")
        target_cls = maybe_load_callable(target_value)

        if target_value and construct_target and (is_dataclass(target_cls) or is_attrs(target_cls)):
            result = from_dict(x, target_cls, force_construct_target=force_construct_target, field_name=field_name)
        else:
            result = {
                _from_dict_value(
                    k,
                    maybe_idx(get_args(field_type), 0) or type(k),
                    type(k),
                    field_name=f"{field_name}.{k}.key",
                    force_construct_target=construct_target,
                ): _from_dict_value(
                    v,
                    maybe_idx(get_args(field_type), 1) or type(v),
                    type(v),
                    field_name=f"{field_name}.{k}",
                    force_construct_target=construct_target,
                )
                for k, v in x.items()
            }
            if field_type in (DictConfig, LazyDict):
                result = OmegaConf.structured(result, flags={"allow_objects": True})
                if construct_target:
                    result = instantiate(result)
                if "_target_" in result:
                    result["_target_"] = maybe_load_callable(result["_target_"])
            elif construct_target and target_cls:  # instantiate a regular class from a dict
                special_keys = {
                    "_target_",
                    "_recursive_",
                    "_convert_",
                    "_args_",
                    "_kwargs_",
                }
                constructable_items = {
                    k: v for k, v in result.items() if not (isinstance(k, str) and k in special_keys)
                }
                result = target_cls(**constructable_items)
        return result
    elif field_type is ListConfig or origin in (
        list,
        List,
    ):
        return [
            _from_dict_value(
                xx, maybe_idx(get_args(field_type), 0) or type(xx), type(xx), field_name=f"{field_type}[{i}]"
            )
            for i, xx in enumerate(x)
        ]
    elif is_optional(field_type):
        return _from_dict_value(x, args[0], type(x), field_name=field_name)
    elif origin in (Union, UnionType):
        raise AssertionError("unions are not implemented yet!")
    elif origin is Callable2 or origin is type:
        return maybe_load_callable(x)
    elif field_type in (int, float, str, bool):
        return x
    elif field_type is type(None) or field_type == Any:  # no typing
        return x
    else:
        raise TypeError(
            f"unexpected type: {field_type} (origin={origin}, concrete_type={concrete_type}, args={args}, x={x})"
        )


def from_dict(
    x: dict, clazz: type | None = None, force_construct_target: bool | None = None, field_name: str = ""
) -> T:
    if clazz is None:
        assert "_target_" in x
        clazz = maybe_load_callable(x["_target_"])

    assert is_dataclass(clazz) or is_attrs(clazz), f"{clazz} is not a dataclass or attrs"
    if is_dataclass(clazz):
        construct_args = {}
        for f in fields(clazz):
            if f.name in x:
                construct_args[f.name] = _from_dict_value(
                    x[f.name],
                    f.type,
                    type(x[f.name]),
                    field_name=field_name + "." + f.name if field_name else f.name,
                    force_construct_target=force_construct_target,
                )
            elif is_optional(f.type):
                construct_args[f.name] = None
        return clazz(**construct_args)
    elif is_attrs(clazz):
        attrs.resolve_types(clazz)

        construct_args = {}
        for f in attrs.fields(clazz):
            if f.name in x:
                construct_args[f.name] = _from_dict_value(
                    x[f.name],
                    f.type,
                    type(x[f.name]),
                    field_name=field_name + "." + f.name if field_name else f.name,
                    force_construct_target=force_construct_target,
                )
            elif is_optional(f.type):
                construct_args[f.name] = None
        return clazz(**construct_args)
