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

import abc
import functools
import importlib
import json
import re
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import attrs
import cattrs
import cattrs.preconf.json
import omegaconf
import pydantic
import tomllib
import torch
import yaml
from typing_extensions import assert_never

from cosmos3.common.init import is_rank0
from cosmos3._src.imaginaire.flags import TRAINING
from cosmos3._src.imaginaire.lazy_config.registry import convert_target_to_string, locate
from cosmos3._src.imaginaire.utils import log

if TYPE_CHECKING:
    from cosmos3._src.imaginaire.config import Config

ROOT_DIR = Path(__file__).parents[2].absolute()
PACKAGE_DIR = ROOT_DIR / "cosmos3"
CONFIG_DIR = PACKAGE_DIR / "configs"


def load_config(
    config_file: str,
    experiment: str,
    *,
    overrides: list[str] = [],
) -> "Config":
    """Load config from config store."""
    assert TRAINING
    from cosmos3._src.imaginaire.utils import config_helper

    config_module = importlib.import_module(config_helper.get_config_module(config_file))
    config = config_module.make_config()
    config = config_helper.override(config, ["--", f"experiment={experiment}", *overrides])
    return config


def save_config(config: Any, output_dir: Path) -> None:
    """Save config to output directory for debugging."""
    if not is_rank0():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / "config.yaml"
    serialize_config(config, config_file, invalid="ignore")
    log.info(f"Saved config to {config_file}")


InvalidMode = Literal["error", "warn", "ignore"]
"""How to handle invalid fields.

* error: raise an error.
* warn: log a warning.
* ignore:
    * When unstructuring, convert to string.
    * When structuring, leave unstructured.
"""
_InvalidMode = InvalidMode  # For backward compatibility.
_InvalidModeValidator = pydantic.TypeAdapter(_InvalidMode)


@attrs.define
class _UnstructureOptions:
    converter: cattrs.Converter

    invalid: _InvalidMode


@attrs.define
class _FixOptions:
    converter: cattrs.Converter

    invalid: _InvalidMode
    add_defaults: bool


def unstructure_config(config: Any, *, invalid: _InvalidMode = "error") -> dict:
    """Unstructure config to primitive types."""
    _InvalidModeValidator.validate_python(invalid)
    options = _UnstructureOptions(converter=config_converter, invalid=invalid)
    return _unstructure(config, prefix=(), options=options)


def structure_config(config_dict: Any, target: type | str | None = None, /, *, invalid: _InvalidMode = "error") -> Any:
    """Structure config from primitive types."""
    if target is None:
        target = config_dict[_TYPE_KEY]
    if isinstance(target, str):
        target = locate(target)
    config_dict = fix_config_dict(config_dict, invalid=invalid, add_defaults=True)
    return config_converter.structure(config_dict, target)


def serialize_config_dict(config_dict: dict, config_file: Path) -> None:
    """Serialize config dict to a file."""
    match config_file.suffix.lower():
        case ".yaml" | ".yml":
            config_str = yaml.safe_dump(config_dict, sort_keys=True)
        case ".json":
            config_str = json.dumps(config_dict, indent=2, sort_keys=True)
        case _:
            raise ValueError(f"Unsupported file extension '{config_file.suffix}'")
    config_str = apply_config_replacements(config_str)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(config_str)


def serialize_config(config: Any, config_file: Path, /, *, invalid: _InvalidMode = "error") -> None:
    """Serialize config to a file."""
    config_dict = unstructure_config(config, invalid=invalid)
    serialize_config_dict(config_dict, config_file)


def deserialize_config_dict(config_file: Path) -> dict:
    """Deserialize config dict from a file."""
    config_str = config_file.read_text()
    config_str = undo_config_replacements(config_str)
    match config_file.suffix.lower():
        case ".yaml" | ".yml":
            config_dict = yaml.safe_load(config_str)
        case ".json":
            config_dict = json.loads(config_str)
        case _:
            raise ValueError(f"Unsupported file extension '{config_file.suffix}'")
    config_dict = fix_config_dict(config_dict)
    return config_dict


def deserialize_config(
    config_file: Path, target: type | str | None = None, /, *, invalid: _InvalidMode = "error"
) -> Any:
    """Deserialize config from a file."""
    config_dict = deserialize_config_dict(config_file)
    return structure_config(config_dict, target, invalid=invalid)


def fix_config_dict(config_dict: Any, *, invalid: _InvalidMode = "error", add_defaults: bool = False) -> dict:
    """Fix config dict.

    * Unstructure.
    * Fix legacy fields.
    * Optional: Add missing default fields.
    """
    _InvalidModeValidator.validate_python(invalid)
    config_dict = unstructure_config(config_dict)
    return _fix(
        config_dict,
        prefix=(),
        options=_FixOptions(converter=config_converter, invalid=invalid, add_defaults=add_defaults),
    )


_TYPE_KEY = "_type"
_TARGET_KEY = "_target_"


def _join_prefix(prefix: tuple[Any, ...]) -> str:
    return ".".join(str(p) for p in prefix)


def get_default_params(tp: type) -> dict:
    from cosmos3._src.imaginaire.lazy_config.lazy_call import get_default_params

    return {k: v for k, v in get_default_params(tp).items() if v is not attrs.NOTHING}


def _fix(obj: Any, prefix: tuple[Any, ...], options: _FixOptions) -> Any:
    # Handle primitive types.
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, list):
        return [_fix(item, prefix=(*prefix, i), options=options) for i, item in enumerate(obj)]

    # Handle objects.
    if isinstance(obj, dict):
        # Fix metadata.
        metadata = obj.pop("_metadata", {})
        tp_str = obj.pop(_TYPE_KEY, None)
        tp_str = metadata.pop("object_type", None) or tp_str
        if tp_str is not None and tp_str.startswith("omegaconf."):
            tp_str = None
        if tp_str is not None:
            obj[_TYPE_KEY] = tp_str

        # Recurse.
        data = {k: _fix(v, prefix=(*prefix, k), options=options) for k, v in obj.items()}

        # Add missing default fields.
        target_str = obj.get(_TARGET_KEY) or obj.get(_TYPE_KEY)
        if options.add_defaults and target_str is not None:
            target = locate(target_str)
            data = get_default_params(target) | data

        return data
    raise ValueError(f"Unsupported type: {type(obj)}")


def _unstructure(obj: Any, prefix: tuple[Any, ...], options: _UnstructureOptions) -> Any:
    """Wrapper around cattrs 'unstructure' to handle missing type annotations.

    For values missing type annotations, the type of the object is included in
    the unstructured data.
    """
    # Handle primitive types.
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, list):
        return [_unstructure(item, prefix=(*prefix, i), options=options) for i, item in enumerate(obj)]
    if isinstance(obj, dict):
        return {k: _unstructure(v, prefix=(*prefix, k), options=options) for k, v in obj.items()}

    # Handle objects.
    try:
        # Use cattrs to unstructure the object.
        data = options.converter.unstructure(obj)
        if data is obj:
            raise ValueError(f"Unsupported type: {type(obj)}")
        if isinstance(data, dict) and not isinstance(obj, omegaconf.DictConfig):
            if _TYPE_KEY in data:
                raise ValueError(f"Already has a {_TYPE_KEY} field: {data[_TYPE_KEY]}")
            data[_TYPE_KEY] = convert_target_to_string(type(obj))
    except Exception as e:
        if options.invalid == "ignore":
            return str(obj)
        msg = f"Invalid value '{_join_prefix(prefix)}': {e}"
        if options.invalid == "warn":
            log.warning(msg)
            return str(obj)
        if options.invalid == "error":
            raise ValueError(msg) from e
        assert_never(options.invalid)
    # Recursively unstructure the data.
    return _unstructure(data, prefix=prefix, options=options)


config_converter = cattrs.preconf.json.make_converter()


# type
def _is_type_cls(cls: Any) -> bool:
    return cls in [type, abc.ABCMeta] or typing.get_origin(cls) is type


def _unstructure_type(cls: Any | None) -> str | None:
    if cls is None:
        return None
    return convert_target_to_string(cls)


def _structure_type(data: Any, _cls: Any) -> type:
    assert isinstance(data, str)
    return locate(data)


config_converter.register_unstructure_hook_func(_is_type_cls, _unstructure_type)
config_converter.register_structure_hook_func(_is_type_cls, _structure_type)


# functools.partial
def _unstructure_functools_partial(obj: functools.partial | None) -> dict | None:
    if obj is None:
        return None

    # Convert to omegaconf: https://hydra.cc/docs/advanced/instantiate_objects/overview/
    return {
        _TARGET_KEY: convert_target_to_string(obj.func),
        "_args_": obj.args,
        **obj.keywords,
    }


def _structure_functools_partial(data: Any, _cls: Any) -> functools.partial:
    assert isinstance(data, dict)
    func = locate(data.pop(_TARGET_KEY))
    args = data.pop("_args_")
    return functools.partial(func, *args, **data)


config_converter.register_unstructure_hook(functools.partial, _unstructure_functools_partial)
config_converter.register_structure_hook(functools.partial, _structure_functools_partial)

# We need allow objects, because we add default fields to the config.
_OMEGACONF_FLAGS = dict(allow_objects=True)


# omegaconf.DictConfig
def _is_omegaconf_dict_cls(cls: Any) -> bool:
    return isinstance(cls, type) and issubclass(cls, omegaconf.DictConfig)


def _unstructure_omegaconf_dict(obj: omegaconf.DictConfig | None) -> dict | None:
    if obj is None or obj._is_none():
        return None

    # Create a shallow copy without recursion or resolution.
    data = dict(obj.items_ex(resolve=False))

    target_str = data.get(_TARGET_KEY)
    if target_str is not None and not isinstance(target_str, str):
        data[_TARGET_KEY] = convert_target_to_string(target_str)
    if obj._metadata.object_type not in [None, dict]:
        data[_TYPE_KEY] = convert_target_to_string(obj._metadata.object_type)

    return data


def _structure_omegaconf_dict(data: Any, _cls: Any) -> omegaconf.DictConfig | None:
    if data is None:
        return None
    assert isinstance(data, dict), type(data)
    # Ideally, we should use omegaconf.structured if _TYPE_KEY is present.
    return omegaconf.OmegaConf.create(data, flags=_OMEGACONF_FLAGS)


config_converter.register_unstructure_hook_func(_is_omegaconf_dict_cls, _unstructure_omegaconf_dict)
config_converter.register_structure_hook_func(_is_omegaconf_dict_cls, _structure_omegaconf_dict)


# omegaconf.ListConfig
def _is_omegaconf_list_cls(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, omegaconf.ListConfig)


def _unstructure_omegaconf_list(obj: omegaconf.ListConfig | None) -> list | None:
    if obj is None or obj._is_none():
        return None
    # Create a shallow copy without recursion or resolution.
    return list(obj._iter_ex(resolve=False))


def _structure_omegaconf_list(data: Any, _: Any) -> omegaconf.ListConfig | None:
    if data is None:
        return None
    assert isinstance(data, list), type(data)
    return omegaconf.OmegaConf.create(data, flags=_OMEGACONF_FLAGS)


config_converter.register_unstructure_hook_func(_is_omegaconf_list_cls, _unstructure_omegaconf_list)
config_converter.register_structure_hook_func(_is_omegaconf_list_cls, _structure_omegaconf_list)


# torch types
def _unstructure_torch_type(obj: Any | None) -> str | None:
    if obj is None:
        return None
    return str(obj).removeprefix("torch.")


def _structure_torch_type(data: Any, _cls: Any) -> Any | None:
    if data is None:
        return None
    assert isinstance(data, str), type(data)
    return getattr(torch, data)


for _torch_type in [
    torch.dtype,
    torch.layout,
    torch.memory_format,
]:
    config_converter.register_unstructure_hook(_torch_type, _unstructure_torch_type)
    config_converter.register_structure_hook(_torch_type, _structure_torch_type)


# torch.device
def _unstructure_torch_device(obj: torch.device | None) -> str | None:
    if obj is None:
        return None
    return str(obj)


def _structure_torch_device(data: Any, _cls: Any) -> torch.device | None:
    if data is None:
        return None
    assert isinstance(data, str), type(data)
    return torch.device(data)


config_converter.register_unstructure_hook(torch.device, _unstructure_torch_device)
config_converter.register_structure_hook(torch.device, _structure_torch_device)


# torch.Tensor
def _unstructure_torch_tensor(obj: torch.Tensor | None) -> list | None:
    if obj is None:
        return None
    return obj.detach().cpu().tolist()


def _structure_torch_tensor(data: Any, _cls: Any) -> torch.Tensor | None:
    if data is None:
        return None
    assert isinstance(data, list), type(data)
    return torch.tensor(data)


config_converter.register_unstructure_hook(torch.Tensor, _unstructure_torch_tensor)
config_converter.register_structure_hook(torch.Tensor, _structure_torch_tensor)



def replace_case_preserving(text: str, old: str, new: str) -> str:
    """Similar to `str.replace()`, but preserves the case of the matched text."""

    def replace_func(match: re.Match[str]) -> str:
        original = match.group()
        if original.isupper():
            return new.upper()
        if original.islower():
            return new.lower()
        if original == old.capitalize():
            return new.capitalize()
        if original[0].isupper():
            return new.upper()
        return new.lower()

    pattern = re.compile(re.escape(old), re.IGNORECASE)
    return pattern.sub(replace_func, text)


def apply_config_replacements(config_str: str) -> str:
    """Apply config replacements to a config string."""
    return config_str


def undo_config_replacements(config_str: str) -> str:
    """Undo config replacements to a config string."""
    return config_str


def undo_config_dict_replacements(config_dict: dict) -> dict:
    """Undo config replacements to a config dict."""
    config_str = json.dumps(config_dict)
    config_str = undo_config_replacements(config_str)
    return json.loads(config_str)
