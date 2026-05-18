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

import ast
import builtins
import importlib
import inspect
import logging
import os
import pickle
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import attrs
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

try:
    import dill as dill_pickle
except ImportError:
    dill_pickle = None

try:
    import cloudpickle
except ImportError:
    cloudpickle = None

from cosmos.utils.lazy_config.file_io import PathManager
from cosmos.utils.lazy_config.lazy_call import LazyCall, get_default_params

__all__ = ["LazyCall", "LazyConfig"]


def sort_dict(d: Dict[str, Any]) -> OrderedDict[str, Any]:
    return OrderedDict(sorted(d.items(), key=lambda x: x[0]))


def dict_representer(dumper: yaml.Dumper, data: OrderedDict[str, Any]) -> yaml.nodes.MappingNode:
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


def sort_recursive(obj: Union[Dict[str, Any], List[Any], Any]) -> Union[OrderedDict[str, Any], List[Any], Any]:
    if isinstance(obj, dict):
        return sort_dict({k: sort_recursive(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [sort_recursive(item) for item in obj]
    return obj


yaml.add_representer(OrderedDict, dict_representer)

OmegaConf.register_new_resolver("add", lambda *vals: sum(vals))
OmegaConf.register_new_resolver("subtract", lambda *vals: vals[0] - sum(vals[1:]))


def _visit_dict_config(cfg, func):
    """
    Apply func recursively to all DictConfig in cfg.
    """
    if isinstance(cfg, DictConfig):
        func(cfg)
        for v in cfg.values():
            _visit_dict_config(v, func)
    elif isinstance(cfg, ListConfig):
        for v in cfg:
            _visit_dict_config(v, func)


def _validate_py_syntax(filename):
    # see also https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
    with PathManager.open(filename, "r") as f:
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError(f"Config file {filename} has syntax error!") from e


def _cast_to_config(obj):
    # if given a dict, return DictConfig instead
    if isinstance(obj, dict):
        return DictConfig(obj, flags={"allow_objects": True})
    return obj


_CFG_PACKAGE_NAME = "detectron2._cfg_loader"
"""
A namespace to put all imported config into.
"""


def _random_package_name(filename):
    # generate a random package name when loading config files
    return _CFG_PACKAGE_NAME + str(uuid.uuid4())[:4] + "." + os.path.basename(filename)


@contextmanager
def _patch_import():
    """
    Enhance relative import statements in config files, so that they:
    1. locate files purely based on relative location, regardless of packages.
       e.g. you can import file without having __init__
    2. do not cache modules globally; modifications of module states has no side effect
    3. support other storage system through PathManager, so config files can be in the cloud
    4. imported dict are turned into omegaconf.DictConfig automatically
    """
    old_import = builtins.__import__

    def find_relative_file(original_file, relative_import_path, level):

        # if such import should produce `x` as a python module or DictConfig.
        # This can be discussed further if needed.
        relative_import_err = """
Relative import of directories is not allowed within config files.
Within a config file, relative import can only import other config files.
""".replace("\n", " ")
        if not len(relative_import_path):
            raise ImportError(relative_import_err)

        cur_file = os.path.dirname(original_file)
        for _ in range(level - 1):
            cur_file = os.path.dirname(cur_file)
        cur_name = relative_import_path.lstrip(".")
        for part in cur_name.split("."):
            cur_file = os.path.join(cur_file, part)
        if not cur_file.endswith(".py"):
            cur_file += ".py"
        if not PathManager.isfile(cur_file):
            cur_file_no_suffix = cur_file[: -len(".py")]
            if PathManager.isdir(cur_file_no_suffix):
                raise ImportError(f"Cannot import from {cur_file_no_suffix}." + relative_import_err)
            else:
                raise ImportError(
                    f"Cannot import name {relative_import_path} from {original_file}: {cur_file} does not exist."
                )
        return cur_file

    def new_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            # Only deal with relative imports inside config files
            level != 0 and globals is not None and (globals.get("__package__", "") or "").startswith(_CFG_PACKAGE_NAME)
        ):
            cur_file = find_relative_file(globals["__file__"], name, level)
            _validate_py_syntax(cur_file)
            spec = importlib.machinery.ModuleSpec(_random_package_name(cur_file), None, origin=cur_file)
            module = importlib.util.module_from_spec(spec)
            module.__file__ = cur_file
            with PathManager.open(cur_file) as f:
                content = f.read()
            exec(compile(content, cur_file, "exec"), module.__dict__)
            for name in fromlist:  # turn imported dict into DictConfig automatically
                val = _cast_to_config(module.__dict__[name])
                module.__dict__[name] = val
            return module
        return old_import(name, globals, locals, fromlist=fromlist, level=level)

    builtins.__import__ = new_import
    yield new_import
    builtins.__import__ = old_import


class LazyConfig:
    """
    Provide methods to save, load, and overrides an omegaconf config object
    which may contain definition of lazily-constructed objects.
    """

    @staticmethod
    def load_rel(filename: str, keys: Union[None, str, Tuple[str, ...]] = None):
        """
        Similar to :meth:`load()`, but load path relative to the caller's
        source file.

        This has the same functionality as a relative import, except that this method
        accepts filename as a string, so more characters are allowed in the filename.
        """
        caller_frame = inspect.stack()[1]
        caller_fname = caller_frame[0].f_code.co_filename
        assert caller_fname != "<string>", "load_rel Unable to find caller"
        caller_dir = os.path.dirname(caller_fname)
        filename = os.path.join(caller_dir, filename)
        return LazyConfig.load(filename, keys)

    @staticmethod
    def load(filename: str, keys: Union[None, str, Tuple[str, ...]] = None):
        """
        Load a config file.

        Args:
            filename: absolute path or relative path w.r.t. the current working directory
            keys: keys to load and return. If not given, return all keys
                (whose values are config objects) in a dict.
        """
        has_keys = keys is not None
        filename = filename.replace("/./", "/")  # redundant
        if os.path.splitext(filename)[1] not in [".py", ".yaml", ".yml"]:
            raise ValueError(f"Config file {filename} has to be a python or yaml file.")
        if filename.endswith(".py"):
            _validate_py_syntax(filename)

            with _patch_import():
                # Record the filename
                module_namespace = {
                    "__file__": filename,
                    "__package__": _random_package_name(filename),
                }
                with PathManager.open(filename) as f:
                    content = f.read()
                # Compile first with filename to:
                # 1. make filename appears in stacktrace
                # 2. make load_rel able to find its parent's (possibly remote) location
                exec(compile(content, filename, "exec"), module_namespace)

            ret = module_namespace
        else:
            with PathManager.open(filename) as f:
                obj = yaml.unsafe_load(f)
            ret = OmegaConf.create(obj, flags={"allow_objects": True})

        if has_keys:
            if isinstance(keys, str):
                return _cast_to_config(ret[keys])
            else:
                return tuple(_cast_to_config(ret[a]) for a in keys)
        else:
            if filename.endswith(".py"):
                # when not specified, only load those that are config objects
                ret = DictConfig(
                    {
                        name: _cast_to_config(value)
                        for name, value in ret.items()
                        if isinstance(value, (DictConfig, ListConfig, dict)) and not name.startswith("_")
                    },
                    flags={"allow_objects": True},
                )
            return ret

    @staticmethod
    def save_pkl(cfg, filename: str) -> str:
        """
        Saves a Config object to a file using pickle serialization. This method is typically used
        when the configuration object contains complex objects, such as lambdas, that are not supported by
        simpler serialization methods like YAML. The function attempts to create a deep copy of the configuration
        object before serialization to ensure that the original object remains unmodified.

        Args:
            cfg: A Config object to be serialized and saved.
            filename: The path and name of the file where the configuration should be saved. The function
                      assumes the file extension indicates a pickle format (e.g., .pkl).

        Returns:
            str: The filename to which the configuration was saved. This can be used to verify the file location
                 or log the outcome.

        Notes:
            - The function logs a warning if the configuration is successfully saved using pickle.
            - If saving fails, an error is logged with the exception details.
        """
        logger = logging.getLogger(__name__)
        try:
            cfg = deepcopy(cfg)
        except Exception:
            pass

        try:
            with PathManager.open(filename, "wb") as f:
                pickle.dump(cfg, f)
            logger.warning(f"Config is saved using pickle at {filename}.")
        except Exception as e:
            logger.error(f"Failed to save config to {filename}: {e}. Trying dill or cloudpickle instead")
            if dill_pickle:
                try:
                    with PathManager.open(filename, "wb") as f:
                        pickle.dump(dill_pickle.dumps(cfg, recurse=True), f)
                        logger.warning(f"Config is saved using dill at {filename}.")
                except Exception as e:
                    logger.error(f"Failed to save config to {filename}: {e}.")
                    if cloudpickle:
                        try:
                            with PathManager.open(filename, "wb") as f:
                                pickle.dump(cloudpickle.dumps(cfg), f)
                            logger.warning(f"Config is saved using cloudpickle at {filename}.")
                        except Exception as e:
                            logger.error(f"Failed to save config to {filename}: {e}.")
                    else:
                        logger.error("cloudpickle is not available. Cannot save the config.")
                        raise e

        return filename

    @staticmethod
    def save_yaml(cfg, filename: str) -> str:
        """
        Saves a Config object to a file using YAML serialization. This method is beneficial when the configuration object's content needs to be human-readable and easily editable. YAML is suitable for configurations that do not contain complex types like lambdas, which must be handled differently. The function converts unserializable items to strings before saving to ensure compatibility with YAML serialization.

        Args:
            cfg: A Config object to be serialized and saved. It handles both DictConfig and ListConfig types.
            filename: The path and name of the file where the configuration should be saved. The function does not require a specific file extension but typically uses '.yaml'.

        Returns:
            str: The filename to which the configuration was saved. This can be used to verify the file location or log the outcome.

        Notes:
            - The function logs a warning if the configuration is successfully saved using YAML.
            - If saving fails, an error is logged with the exception details.
        """
        logger = logging.getLogger(__name__)
        try:
            cfg = deepcopy(cfg)
        except Exception:
            pass

        # Define a function to check if an item is serializable to YAML
        def is_serializable(item):
            try:
                OmegaConf.to_yaml(item)
                return True
            except Exception as e:
                return False

        # Function to convert unserializable items to strings
        def serialize_config(config):
            if isinstance(config, DictConfig):
                for key, value in config.items():
                    if isinstance(value, (DictConfig, ListConfig)):
                        try:
                            if "_target_" in value:
                                default_params = get_default_params(value["_target_"])
                                for default_key, default_v in default_params.items():
                                    if default_key not in value:
                                        value[default_key] = default_v
                        except Exception as e:
                            logger.error(f"Failed to add default argument values: {e}")

                        serialize_config(value)
                    else:
                        if not is_serializable(value) and value is not None:
                            config[key] = str(value)
            elif isinstance(config, ListConfig):
                for i, item in enumerate(config):
                    if isinstance(item, (DictConfig, ListConfig)):
                        serialize_config(item)
                    else:
                        if not is_serializable(item) and item is not None:
                            config[i] = str(item)
            else:
                raise NotImplementedError("Input config must be a DictConfig or ListConfig.")
            return config

        # Convert Config object to a DictConfig object.
        config_dict = attrs.asdict(cfg)
        config_omegaconf = DictConfig(content=config_dict, flags={"allow_objects": True})

        # Serialize the DictConfig object by converting non-serializable objects to strings.
        config_omegaconf = serialize_config(config_omegaconf)

        config_dict: Dict[str, Any] = OmegaConf.to_container(config_omegaconf, resolve=True)
        sorted_config: OrderedDict[str, Any] = sort_recursive(config_dict)
        with open(filename, "w") as f:
            yaml.dump(sorted_config, f, default_flow_style=False)
        logger.warning(f"Config is saved using omegaconf at {filename}.")
        return filename
