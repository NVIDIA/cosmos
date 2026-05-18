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

import os

from omegaconf import OmegaConf, omegaconf

from cosmos.utils.lazy_config import PLACEHOLDER
from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.callback import OneLoggerCallback
from cosmos.utils.log import logger


def override_one_logger_callback(config) -> None:
    """Add OneLoggerCallback to imaginaire config"""

    # Enable OneLogger by environment variable.
    enable_onelogger = os.environ.get("ENABLE_ONELOGGER", "FALSE").lower() == "true"

    # Check if OneLoggerCallback already exists (by an explicit input argument)
    one_logger_callback_exists = False
    for _callback in config.trainer.callbacks:
        if isinstance(config.trainer.callbacks, (list, omegaconf.ListConfig)):  # old format
            logger.warning("Using old list format for callbacks. Please use registry-compatible dict format.")
            callback_target = _callback._target_
        else:  # omegaconf.dictconfig.DictConfig, registry-compatible format
            if "_target_" not in config.trainer.callbacks[_callback]:
                continue
            callback_target = config.trainer.callbacks[_callback]._target_

        if callback_target is OneLoggerCallback:
            assert enable_onelogger, "OneLoggerCallback should only be used when ENABLE_ONELOGGER is TRUE"
            one_logger_callback_exists = True
            break

    # Add OneLoggerCallback
    if enable_onelogger and not one_logger_callback_exists:
        one_logger_lazy_callback = L(OneLoggerCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER)
        if isinstance(config.trainer.callbacks, list):  # old format
            config.trainer.callbacks.append(one_logger_lazy_callback)
        else:
            ONELOGGER_CALLBACK = dict(one_logger=one_logger_lazy_callback)

            OmegaConf.set_struct(config.trainer.callbacks, False)
            config.trainer.callbacks = OmegaConf.merge(config.trainer.callbacks, ONELOGGER_CALLBACK)
            OmegaConf.set_struct(config.trainer.callbacks, True)

    return config
