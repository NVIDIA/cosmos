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

import importlib
import os
import sys
from datetime import datetime
from types import ModuleType
from typing import Any, Optional

from cosmos.utils.log import LEVEL, get_datetime_format, get_machine_format, logger, make_new_logger

__enable_telemetry: bool = os.getenv("ENABLE_TELEMETRY", "FALSE").upper() == "TRUE"
__telemetry_backends: set[str] = set(os.getenv("TELEMETRY_BACKENDS", "logger").split(","))
__training_telemetry_module: Optional[ModuleType] = None
__provider = None


def import_training_telemetry() -> Optional[ModuleType]:
    """Lazy import of the training_telemetry package to avoid a hard dependency."""
    global __training_telemetry_module
    if __training_telemetry_module is not None:
        return __training_telemetry_module

    try:
        __training_telemetry_module = importlib.import_module("training_telemetry")
        return __training_telemetry_module
    except ImportError as e:
        logger.error(f"Heimdall telemetry is enabled but package is not installed: {e}")
        logger.info(
            "Please install the package using `pip install aidot-training-telemetry --index-url=https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple`"
        )
        return None


def set_telemetry_provider(local_path: str) -> Optional[Any]:
    """
    Set the telemetry provider if telemetry is enabled, and if we can import the necessary modules, otherwise return None.
    """
    global __enable_telemetry
    if not __enable_telemetry:
        logger.info(
            "Heimdall telemetry is disabled, if using Heimdall,consider setting ENABLE_TELEMETRY=true to enable it"
        )
        return None

    global __provider
    if __provider is not None:
        return __provider

    training_telemetry = import_training_telemetry()
    if training_telemetry is None:
        logger.error(
            "Heimdall telemetry is enabled but package is not installed, consider setting ENABLE_TELEMETRY=false to disable it, or install the package using `pip install aidot-training-telemetry --index-url=https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple`"
        )
        __enable_telemetry = False
        return None

    rank = training_telemetry.get_rank()
    log_only_errors = rank != 0
    logger.info(
        f"Setting up telemetry provider, rank: {rank}, log only errors: {log_only_errors}, backends: {__telemetry_backends}"
    )
    backends = []
    for backend_name in __telemetry_backends:
        backend_name = backend_name.strip().lower()
        if not backend_name:
            continue
        logger.debug(f"Setting up telemetry backend: {backend_name}")
        if backend_name == "logger":
            backends.append(
                training_telemetry.LoggerBackendConfig(rank_aware=True, errors_only=log_only_errors),
            )
        elif backend_name == "file":
            backends.append(training_telemetry.FileBackendConfig(output_file_path=local_path + "/telemetry/events"))
        elif backend_name == "nvtx":
            backends.append(training_telemetry.NVTXBackendConfig())
        else:
            logger.error(f"Unknown telemetry backend will be ignored: {backend_name}")
    if not backends:
        logger.error("No telemetry backends configured, telemetry will be disabled")
        __enable_telemetry = False
        return None

    config = training_telemetry.TelemetryConfig(backends=backends)

    # Use a telemetry-specific logger because we don't want to report information that is not useful for telemetry
    # such as the file path to the library and the function name of the logging backend
    telemetry_logger = make_new_logger(depth=0)
    message_format = "<level>{level}</level>|training-telemetry] {message}"
    telemetry_logger.remove()
    telemetry_logger.add(
        sys.stdout,
        level=LEVEL,
        format=f"{get_datetime_format()}{get_machine_format()}{message_format}",
    )
    telemetry_logger.add(
        f"{local_path}/telemetry/stdout.log",
        encoding="utf8",
        level=LEVEL,
        format=f"{get_datetime_format()}{get_machine_format()}{message_format}",
        rotation="100 MB",
        enqueue=True,
    )
    __provider = training_telemetry.Provider.set_provider(config=config, logger=telemetry_logger)
    return __provider


def get_telemetry_recorder() -> Optional[Any]:
    """
    Get the telemetry recorder if it is available, otherwise return None.
    """
    global __provider
    return __provider.recorder if __provider else None


def get_timezone_name() -> str:
    """Get the timezone name, tzlocal is installed with the training-telemetry package, if it is not installed, use the TZ environment variable, otherwise use the local timezone"""
    try:
        # pyrefly: ignore  # import-error
        from tzlocal import get_localzone

        return str(get_localzone())
    except ImportError:
        tz_name = os.environ.get("TZ")
        if tz_name is None:
            tz_name = datetime.now().astimezone().tzname()
        return tz_name


def get_checkpoint_strategy(checkpoint_config: dict[str, Any]) -> str:
    """
    Return the checkpoint strategy, sync or async, based on the checkpoint config.

    FIXME: it seems that the config only reports the type of checkpoint, sync or async,
    for the cosmos distributed checkpointer, but I don't want a dependency  on it here
    so I'm looking at the class name. It seems that the default Checkpointer uses an async thread
    all the time, so the default is async. The FIX should involve either having a config that is
    always followed by all checkpointers, or adding a function to the base checkpointer class,
    indicating if checkpoint is sync or async, but this means that the checkpoint class could
    only be retrieved after the checkpointer is created, and not from the config.
    """
    if checkpoint_config.type is not None and "DistributedCheckpointer" in checkpoint_config.type["_target_"].__name__:
        return "async" if checkpoint_config.dcp_async_mode_enabled else "sync"
    else:
        return "async"
