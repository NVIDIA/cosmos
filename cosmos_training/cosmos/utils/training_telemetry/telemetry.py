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

import functools
import socket
from typing import Callable

from omegaconf import OmegaConf, omegaconf

from cosmos.utils.config import Config
from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.log import logger
from cosmos.utils.training_telemetry.utils import (
    get_checkpoint_strategy,
    get_timezone_name,
    import_training_telemetry,
    set_telemetry_provider,
)


def _add_callback_to_config(config: Config) -> Config:
    """Add TelemetryCallback to the config"""
    from cosmos.utils.training_telemetry.callback import TelemetryCallback

    # Check if callback already exists (by an explicit input argument)
    callback_exists = False
    for _callback in config.trainer.callbacks:
        if isinstance(config.trainer.callbacks, (list, omegaconf.ListConfig)):  # old format
            logger.warning("Using old list format for callbacks. Please use registry-compatible dict format.")
            callback_target = _callback._target_
        else:  # omegaconf.dictconfig.DictConfig, registry-compatible format
            if "_target_" not in config.trainer.callbacks[_callback]:
                continue
            callback_target = config.trainer.callbacks[_callback]._target_

        if callback_target is TelemetryCallback:
            callback_exists = True
            break

    # Add TelemetryCallback
    if not callback_exists:
        telemetry_lazy_callback = L(TelemetryCallback)()
        if isinstance(config.trainer.callbacks, list):  # old format
            config.trainer.callbacks.append(telemetry_lazy_callback)
        else:
            TELEMETRY_CALLBACK = dict(telemetry_callback=telemetry_lazy_callback)

            OmegaConf.set_struct(config.trainer.callbacks, False)
            config.trainer.callbacks = OmegaConf.merge(config.trainer.callbacks, TELEMETRY_CALLBACK)
            OmegaConf.set_struct(config.trainer.callbacks, True)

    return config


def monitor(func: Callable) -> Callable:
    """
    Decorator to wrap a function with telemetry tracking.
    The wrapped function must take a config argument.
    """

    @functools.wraps(func)
    def wrapper(config: Config, *args, **kwargs):
        provider = set_telemetry_provider(local_path=config.job.path_local)
        if provider is None:
            return func(config, *args, **kwargs)
        else:
            training_telemetry = import_training_telemetry()
            metrics = training_telemetry.metrics.ApplicationMetrics.create(
                rank=training_telemetry.torch.utils.get_rank(),
                world_size=training_telemetry.torch.utils.get_world_size(),
                node_name=socket.gethostname(),
                timezone=get_timezone_name(),
                total_iterations=config.trainer.max_iter,
                checkpoint_enabled=True,
                checkpoint_strategy=get_checkpoint_strategy(config.checkpoint),
            )
            with training_telemetry.context.running(start_time=None) as span:
                config = _add_callback_to_config(config)
                event_name = training_telemetry.events.EventName.SPAN_ATTRIBUTES
                provider.recorder.event(training_telemetry.events.Event.create(event_name, metrics), span)
                return func(config, *args, **kwargs)

    return wrapper
