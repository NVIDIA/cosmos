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
import time
import uuid
from functools import wraps
from typing import Any, Dict, Optional

import torch

from cosmos.utils.distributed import get_rank, get_world_size
from cosmos.utils.log import logger
from cosmos.utils.one_logger.one_logger_global_vars import (
    CHECKPOINT_CALLBACK_ARGS,
    MINIMAL_SCHEMA_CALLBACK_ARGS,
    THROUGHPUT_CALLBACK_ARGS,
    VERSION,
)

try:
    from megatron.core import parallel_state
except ImportError:
    parallel_state = None


one_logger = None  # globally unique OneLoggerUtils object

rankpulse = None
try:
    if os.environ.get("RANKPULSE_ENABLE", "0").lower() in ["1", "true", "yes", "y"]:
        # pyrefly: ignore  # import-error
        import one_logger.rankpulse as rankpulse
except ImportError:
    print(
        "WARNING: RANKPULSE_ENABLE is set but rankpulse module is not available. Please install one-logger package with rankpulse support."
    )


#######################################################
# Timer implementation
#######################################################


class _NamedTimer:
    """
    A timer class that supports multiple named timers.
    dt will be returned.
    A named timer cannot be started if it is already currently running.
    Use case: measuring execution of multiple code blocks.
    Note that this is only an internal class to log time info.
    """

    def __init__(self, barrier: Optional[callable] = None):
        """
        Initializes a new instance of the _NamedTimer class.

        :param barrier: A function to call for synchronization. Default to None.
        :type barrier: callable
        """
        self.barrier = barrier
        self.reset()

    def __getitem__(self, name):
        """
        Gets the timer data for a specified timer name.

        Args:
            name (str): The name of the timer.

        Returns:
            dict: Timer data for the specified timer name.
        """
        return self.get(name)

    def reset(self, name=None):
        """
        Resets all / specific timer

        Args:
            name (str): timer name to reset (if None all timers are reset)
        """
        if name is None:
            self.timers = {}
        else:
            self.timers[name] = {}

    def start(self, name: str, set_barrier: bool = False):
        """
        Starts measuring a named timer.

        :param name: timer name to start
        :type name: str
        :param set_barrier: Synchronize ranks before starting. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool
        """
        timer_data = self.timers.get(name, {})

        if "is_active" in timer_data:
            raise RuntimeError(f"Cannot start timer = '{name}' since it is already active")

        if set_barrier:
            if not callable(self.barrier):
                raise RuntimeError("No barrier function to call in the _NamedTimer")
            self.barrier()

        timer_data["start"] = time.time()
        timer_data["is_active"] = True
        timer_data["count"] = timer_data.get("count", 0) + 1

        self.timers[name] = timer_data

    def stop(self, name: str, set_barrier: bool = False):
        """
        Stops measuring a named timer.

        :param name: timer name to stop.
        :type name: str
        :param set_barrier: Synchronize ranks before starting. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool
        """
        timer_data = self.timers.get(name)
        if (timer_data is None) or ("is_active" not in timer_data):
            raise RuntimeError(f"Cannot end timer = '{name}' since it is not active")

        if set_barrier:
            if not callable(self.barrier):
                raise RuntimeError("No barrier function to call in the _NamedTimer")
            self.barrier()

        # compute dt and make timer inactive
        timer_data["finish"] = time.time()
        timer_data.pop("is_active")

        # update latest,avg,min,max,total
        timer_data["latest"] = timer_data["finish"] - timer_data["start"]
        if timer_data.get("min", float("inf")) > timer_data["latest"]:
            timer_data["min"] = timer_data["latest"]

        if timer_data.get("max", 0) < timer_data["latest"]:
            timer_data["max"] = timer_data["latest"]

        timer_data["total"] = timer_data.get("total", 0) + timer_data["latest"]
        timer_data["avg"] = timer_data["total"] / timer_data["count"]

        self.timers[name] = timer_data

    def is_active(self, name=""):
        """
        Checks if a named timer is currently active.

        Args:
            name (str): The name of the timer to check.

        Returns:
            bool: True if the timer is active, False otherwise.
        """
        timer_data = self.timers.get(name, {})
        if "is_active" in timer_data:
            return True
        return False

    def active_timers(self):
        """
        Return list of all active named timers

        Returns:
            list: A list of names of active timers.
        """
        return [k for k, v in self.timers.items() if ("is_active" in v)]

    def get(self, name):
        """
        Retrieves the data for a specified timer name.

        Args:
            name (str): The name of the timer.

        Returns:
            dict: The data associated with the specified timer name.
        """
        if name not in self.timers:
            self.timers[name] = {}
        return self.timers[name]


#######################################################
# Callback implementation
#######################################################


class OneLoggerUtils:
    O_MINIMAL = 0x00000001
    """The mode must include this bit to enable the minimal schema."""

    O_THROUGHPUT = 0x00000002
    """The mode must include this bit to enable throughput metrics."""

    O_CKPT = 0x00000004
    """The mode must include this bit to enable saving checkpoint-related metrics."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Construct TrainingCallbacks to log metrics of training jobs.

        :param config: Configuration of OneLogger callback for E2E training metrics logging.  This dictionary contains: 1. application-specific values for logging various metrics.  such as minimal schema, throughput metrics, checkpoint metrics etc.  2. control settings for callbacks behavior. E.g., barrier function, log interval etc.
        :type config: dict

        ****

        **config** (for callback control) must contain:
            - **enable_for_current_rank** (bool): Whether to enable logging for the current rank in distributed training.
            - **one_logger_project** (str): The project name for the OneLogger system.
            - **one_logger_run_name** (str): The name for the current run, used for identifying the log entries.
            - **one_logger_async** (bool): Whether to enable asynchronous logging.
            - **log_every_n_train_iterations** (int): Frequency of logging, specified as the number of steps between logs. NOTE: this value will only affect the on_train_batch_end callback
            - **barrier** (callable, optional): Function to synchronize all ranks, optional. Default to None.  NOTE: If no barrier function provided, OneLogger won't set any barrier for any timestamp calculation across ranks. So only the timestamp calculated in last rank will be used.

        **config** (for Minimal Schema) must contain:
            - **app_tag_run_name** (str or callable): Tag (or callable to generate the tag) for the run name. Jobs belonging to same training run, suppose to have the same app_tag_run_name. NOTE: Please review this value with HWInf-MLWFO-E2E-Dev before enabling instrumentation in production.
            - **app_tag_run_version** (str or callable): Tag version (or callable to generate version) for the run.  NOTE: It will be used to track the changes in the application side which might change the performance baseline, suggesting we should do separate baseline calculation even if the runs belong to the same training that continues.
            - **app_start_time** (int or callable): Timestamp (or callable to get timestamp) when the application started.
            - **world_size** (int or callable): Number (or callable to get number) of processes participating in the training.
            - **global_batch_size** (int or callable): Global batch size or function to compute it.
            - **batch_size** (int or callable): Batch size in each batch iteration or function to calculate the batch size.
            - **app_tag** (str or list or callable): App_tag or function to compute the app tag. The app_tag is array of strings/single string  of unique tag values, calculated from user provided input to be used to identify app use cases expected to have similar perf. NOTE: Please review this value with HWInf-MLWFO-E2E-Dev before enabling instrumentation in production.
            - **train_iterations_target** (int or callable): Target number of training iterations or callable to generate it.
            - **train_samples_target** (int or callable): Target number of training samples or function to generate the number, optional.
            - **is_baseline_run** (bool or callable): Flag (or callable to return flag) that indicates if this is a baseline run for comparison purposes, optional. Default to False.
            - **is_train_iterations_enabled** (bool or callable): Flag (or callable to return flag) that whether to log training iterations. Default to True.
            - **is_test_iterations_enabled** (bool or callable): Flag (or callable to return flag) that whether to log test iterations. Default to True.
            - **is_save_checkpoint_enabled** (bool or callable): Flag (or callable to return flag) that whether to log metrics related to saving checkpoints. Default to True.
            - **is_log_throughput_enabled** (bool or callable): Flag (or callable to return flag) that whether to log throughput-related metrics. Default to True.
            - **micro_batch_size** (int or callable, optional): Size (or callable to generate the size) of each micro-batch in training.
            - **summary_data_schema_version** (str or callable, optional): Version (or callable to return version) of the data schema used for summarizing metrics. Default to "1.0.0"
            - **app_run_type** (str or callable, optional): Type (or callable to return type) of the application run (e.g., training, validation), optional. Default to "training".
            - **app_metrics_feature_tags** (str or callable, optional): Feature tags (or callable to return tags) used to categorize metrics, optional. Default to "full"
            - **seq_length** (int or callable, optional): Sequence length of a training sample or function to calculate the length. Default to None.
            - **metadata** (dict or callable, optional): Other static metadata to track or callable to generate the metadata dict. Default to None. NOTE: the metadata name should be different with key names in `config`.

        **config** (for Throughput Metrics) must contain:
            - **flops_per_sample** (int or callable, optional): FLOPs per sample or function to compute FLOPs per sample. NOTE: this must be set if `is_log_throughput_enabled` is set to `True` or exception would be raised from OneLogger.

        **config** (for Saving Checkpoint-related Metrics) must contain:
            - **save_checkpoint_strategy** (str): Strategy used for saving checkpoints.

        """
        self.enable_for_current_rank = config.get("enable_for_current_rank", False)
        self.one_logger_project = config.get("one_logger_project", "e2e-tracking")
        self.one_logger_run_name = config.get("one_logger_run_name", str(uuid.uuid4()))
        self.quiet = config.get("quiet", False)
        self.one_logger_async = config.get("one_logger_async", True)
        self.log_every_n_train_iterations = config.get("log_every_n_train_iterations", 50)
        self.barrier = config.get("barrier", None)

        try:
            self._set_one_logger()
            self.timer = _NamedTimer(barrier=self.barrier)
        except Exception as e:
            logger.info(e)
            self.one_logger = None

        self.mode = self.O_MINIMAL
        if self._evaluate_value(config.get("is_log_throughput_enabled", True)):
            self.mode |= self.O_THROUGHPUT
        if self._evaluate_value(config.get("is_save_checkpoint_enabled", True)):
            self.mode |= self.O_CKPT

        self._initialize(**config)

    def _no_exception_raise(func):
        """
        A decorator function that wraps a given function to handle exceptions gracefully.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "quiet") or not self.quiet:
                return func(self, *args, **kwargs)
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.info(e)
                # Skipping execution if error happens
                self.one_logger = None
                return None

        return wrapper

    @_no_exception_raise
    def __getattr__(self, name):
        """The __getattr__ method is called only when an attribute is not found in the class"""
        message = (
            f"Attribute '{name}' is not found in the current version of the library. "
            f"Please ensure you are using the latest version."
        )
        raise AttributeError(message)

    @_no_exception_raise
    def _set_one_logger(self):
        """
        Initializes the `one_logger` instance if the `enable_for_current_rank` attribute is set to True.
        """
        # Mark: Get OneLogger
        if self.enable_for_current_rank:
            try:
                # pyrefly: ignore  # import-error
                from one_logger.core import OneLogger

                config = {
                    "project": self.one_logger_project,
                    "name": self.one_logger_run_name,
                    "quiet": self.quiet,
                    "async": self.one_logger_async,
                    "no_exception": False,
                }
                self.one_logger = OneLogger(config=config)
            except BaseException:
                logger.info(
                    "WARNING: one_logger package is required to enable e2e metrics "
                    "tracking. please go to "
                    "https://confluence.nvidia.com/display/MLWFO/Package+Repositories"
                    " for details to install it"
                )
        else:
            self.one_logger = None

    @_no_exception_raise
    def _get_one_logger(self):
        """
        Returns the `one_logger` instance.
        """
        return self.one_logger

    @_no_exception_raise
    def _check_enabled(func):
        """
        A decorator to ensure `one_logger` is enabled before executing the function.

        The wrapped function only executes if `one_logger` is initialized; otherwise, it returns None.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, "one_logger") and self.one_logger:
                return func(self, *args, **kwargs)
            else:
                # Skipping execution because OneLogger is not enabled.
                return None

        return wrapper

    @_check_enabled
    def _evaluate_value(self, input_val):
        """
        Resolves the input variable to a value. If the input is callable, it calls it;
        otherwise, it returns the input directly.
        """
        if callable(input_val):
            return input_val()
        return input_val

    @_check_enabled
    def _store_has_key(self, key):
        """
        A wrapper function to reuse one_logger.store_has_key.
        """
        return self.one_logger.store_has_key(key)

    @_check_enabled
    def _store_set(self, key, value):
        """
        A wrapper function to reuse one_logger.store_set.
        """
        self.one_logger.store_set(key, value)

    @_check_enabled
    def _store_get(self, key, default_value=None):
        """
        A wrapper function to reuse one_logger.store_get and evaluate the value at the same time.
        """
        if not self._store_has_key(key):
            if default_value is None:
                raise ValueError(f"Missing required key '{key}' from store")
            else:
                return default_value
        else:
            return self._evaluate_value(self.one_logger.store_get(key))

    @_check_enabled
    def _log_metrics(self, metrics_to_log):
        """
        Logs metrics using the `one_logger`.

        :param metrics_to_log: A dictionary of metrics to log.
        :type metrics_to_log: dict
        """
        self.one_logger.log_metrics(metrics_to_log)

    @_check_enabled
    def _log_app_tag(self, app_tag):
        """
        Logs an application tag using the `one_logger`.

        :param app_tag: The application tag to log.
        :type app_tag: str
        """
        self.one_logger.log_app_tag(app_tag)

    @_check_enabled
    def _on_start(self, name: str, set_barrier: bool = False):
        """
        Starts a timer with the given name. Resets the timer if already active.

        :param name: The name of the timer.
        :type name: str

        :param set_barrier: Synchronize ranks before starting the timer. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :return: None
        """
        if self.timer.is_active(name):
            logger.info(
                f"Timer `{name}` was not correctly stopped, suggesting a "
                "possible issue. The timer will be reset for now."
            )
            self.timer.reset(name)

        self.timer.start(name, set_barrier)

    # Method to do timer related implemention when we want to stop a timer
    @_check_enabled
    def _on_end(self, name, set_barrier=False):
        """
        Stops a timer with the given name.

        :param name: The name of the timer.
        :type name: str

        :param set_barrier: Synchronize ranks before stopping the timer. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :return: None
        """
        self.timer.stop(name, set_barrier)

    @_check_enabled
    def _validate_and_store_args(self, callback_name, kwargs):
        """
        Validates the arguments passed to a callback against expected schema.

        Checks for required, optional, and default arguments for the specified callback. Raises
        an error if required arguments are missing or additional unexpected arguments are provided.

        :param callback_name: The name of the callback.
        :type callback_name: str
        :param kwargs: The arguments provided to the callback.
        :type kwargs: dict

        :raises ValueError: If required arguments are missing or unexpected arguments are provided.
        """
        args_mappings = []
        if self.mode & self.O_MINIMAL:
            args_mappings.append(MINIMAL_SCHEMA_CALLBACK_ARGS)
        if self.mode & self.O_THROUGHPUT:
            args_mappings.append(THROUGHPUT_CALLBACK_ARGS)
        if self.mode & self.O_CKPT:
            args_mappings.append(CHECKPOINT_CALLBACK_ARGS)

        required_args = set()
        optional_args = set()
        for args_mapping in args_mappings:
            required_args |= args_mapping.get(callback_name, {}).get("required", set())
            optional_args |= args_mapping.get(callback_name, {}).get("optional", set())
        missing_args = required_args - kwargs.keys()
        if len(missing_args) != 0:
            raise ValueError(
                f"Missing required arguments for callback '{self.__class__.__name__}.{callback_name}': {missing_args}"
            )
        additional_args = kwargs.keys() - (required_args | optional_args)
        if len(additional_args) != 0:
            raise ValueError(
                f"Additional arguments provided for callback '{self.__class__.__name__}.{callback_name}': {additional_args}"
            )
        for k, v in kwargs.items():
            self._store_set(k, v)

    @_check_enabled
    def _initialize(self, **config):
        """
        Initializes the logger with initial values and stores metric settings.
        """
        self._validate_and_store_args("initialize", config)

        self._store_set("train_iterations", 0)
        self._store_set("validation_iterations", 0)
        self._store_set("train_samples", 0)
        self._store_set("train_epochs", 0)
        self._store_set("train_iterations_time_total", 0)
        self._store_set("validation_iterations_time_total", 0)
        if self.mode & self.O_CKPT:
            self._store_set("save_checkpoint_count", 0)
            self._store_set("save_checkpoint_sync_count", 0)
            if self._store_get("save_checkpoint_strategy") == "async":
                self._store_set("save_checkpoint_async_count", 0)

        metrics_to_log = {"one_logger_utils_version": VERSION}
        if self.mode & self.O_MINIMAL:
            self._log_app_tag(self._store_get("app_tag"))

            # Metrics with default values provided
            metrics_to_log["summary_data_schema_version"] = self._store_get(
                "summary_data_schema_version", "1.0.0"
            )  # Associate the code with the data schema version
            metrics_to_log["app_run_type"] = self._store_get("app_run_type", "training")  # Hard coded this to traininig
            metrics_to_log["app_metrics_feature_tags"] = self._store_get(
                "app_metrics_feature_tags", "full"
            )  # Hard coded this to full
            metrics_to_log["is_baseline_run"] = self._store_get("is_baseline_run")
            metrics_to_log["is_train_iterations_enabled"] = self._store_get("is_train_iterations_enabled")
            metrics_to_log["is_validation_iterations_enabled"] = self._store_get("is_validation_iterations_enabled")
            metrics_to_log["is_test_iterations_enabled"] = self._store_get("is_test_iterations_enabled")
            metrics_to_log["is_save_checkpoint_enabled"] = self._store_get("is_save_checkpoint_enabled")
            metrics_to_log["is_log_throughput_enabled"] = self._store_get("is_log_throughput_enabled")
            # Required metrics that must be specified
            metrics_to_log["app_tag_run_version"] = self._store_get("app_tag_run_version")
            metrics_to_log["world_size"] = self._store_get("world_size")
            metrics_to_log["micro_batch_size"] = self._store_get("micro_batch_size", None)
            metrics_to_log["global_batch_size"] = self._store_get("global_batch_size")
            metrics_to_log["app_tag_run_name"] = self._store_get("app_tag_run_name")
            metrics_to_log["train_iterations_target"] = self._store_get("train_iterations_target")
            train_samples_target = self._store_get("train_samples_target")
            metrics_to_log["train_samples_target"] = train_samples_target

            # Log seq length if available
            if self._store_has_key("seq_length"):
                seq_length = self._store_get("seq_length")
                train_tokens_target = seq_length * train_samples_target
                metrics_to_log["model_seq_length"] = seq_length
                metrics_to_log["train_tokens_target"] = train_tokens_target

        if self.mode & self.O_CKPT:
            metrics_to_log["save_checkpoint_strategy"] = self._store_get("save_checkpoint_strategy")

        if self._store_has_key("metadata"):
            self._validate_metadata_and_update(metrics_to_log, self._store_get("metadata"))

        self._log_metrics(metrics_to_log)

    def _validate_metadata_and_update(self, target_to_update: dict, metadata: dict) -> dict:
        """Validate metadata and update target dict.

        :param target_to_update: target dictionary to be updated with metadata.
        :type target_update: dict
        :param metadata: metadata of the current run
        :type metadata: dict
        :return: updated dict with metadata
        :rtype: dict
        """
        overlap_keys = set(target_to_update.keys()).intersection(metadata.keys())

        if overlap_keys:
            raise ValueError(f"Metadata overlap found with keys: {overlap_keys}")

        target_to_update.update(metadata)
        return target_to_update

    @_check_enabled
    def on_model_init_start(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics at the start of the model initialization.

        :param set_barrier: synchronize ranks before executing the callback. default to false. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: metrics needed for callback function invocation.
        :type metrics_input_kwargs: dict

        ****

        **metrics_input_kwargs** could contain (optional):
            - **app_model_init_start_time**: The timestamp of starting model initialization. If provide, OneLogger will use this value or it will use internal timer to get the timestamp.
        """
        self._on_start("model_init", set_barrier)
        self._validate_and_store_args("on_model_init_start", metrics_input_kwargs)
        if self._store_has_key("app_model_init_start_time"):
            self._log_metrics({"app_model_init_start_time": self._store_get("app_model_init_start_time")})
        else:
            self._log_metrics({"app_model_init_start_time": self.timer.get("model_init").get("start") * 1000})

    @_check_enabled
    def on_model_init_end(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics at the end of the model initialization.

        :param set_barrier: synchronize ranks before executing the callback. default to false. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: metrics needed for callback function invocation.
        :type metrics_input_kwargs: dict

        ****

        **metrics_input_kwargs** could contain (optional):
            - **app_model_init_finish_time**: The timestamp of finishing model initialization. If provide, OneLogger will use this value or it will use internal timer to get the timestamp.
        """
        self._on_end("model_init", set_barrier)
        self._validate_and_store_args("on_model_init_end", metrics_input_kwargs)

        if self._store_has_key("app_model_init_finish_time"):
            self._log_metrics({"app_model_init_finish_time": self._store_get("app_model_init_finish_time")})
        else:
            self._log_metrics({"app_model_init_finish_time": self.timer.get("model_init").get("finish") * 1000})

    @_check_enabled
    def on_dataloader_init_start(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics at the start of the dataloader initialization.

        :param set_barrier: synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: metrics needed for callback function invocation.
        :type metrics_input_kwargs: dict

        ****

        **metrics_input_kwargs** could contain (optional):
            - **app_build_dataiter_start_time**: The timestamp of starting dataloader initialization. If provide, OneLogger will use this value or it will use internal timer to get the timestamp.

        """
        self._on_start("dataloader_init", set_barrier)
        self._validate_and_store_args("on_dataloader_init_start", metrics_input_kwargs)

        if self._store_has_key("app_build_dataiter_start_time"):
            self._log_metrics({"app_build_dataiter_start_time": self._store_get("app_build_dataiter_start_time")})
        else:
            self._log_metrics({"app_build_dataiters_start_time": self.timer.get("dataloader_init").get("start") * 1000})

    @_check_enabled
    def on_dataloader_init_end(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics at the end of the dataloader initialization.

        :param set_barrier: synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: metrics needed for callback function invocation.
        :type metrics_input_kwargs: dict

        ****

        **metrics_input_kwargs** could contain (optional):
            - **app_build_dataiter_finish_time**: The timestamp of finishing dataloader initialization. If provide, OneLogger will use this value or it will use internal timer to get the timestamp.

        """
        self._on_end("dataloader_init", set_barrier)
        self._validate_and_store_args("on_dataloader_init_end", metrics_input_kwargs)
        if self._store_has_key("app_build_dataiters_finish_time"):
            self._log_metrics({"app_build_dataiters_finish_time": self._store_get("app_build_dataiters_finish_time")})
        else:
            self._log_metrics(
                {"app_build_dataiters_finish_time": self.timer.get("dataloader_init").get("finish") * 1000}
            )

    @_check_enabled
    def on_load_checkpoint_start(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics when start loading checkpoint.

        :param set_barrier: synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: metrics needed for callback function invocation.
        :type metrics_input_kwargs: dict

        ****

        **metrics_input_kwargs** could contain (optional):
            - **load_checkpoint_start_time**: The timestamp of starting checkpoint loading. If provide, OneLogger will use this value or it will use internal timer to get the timestamp.

        """
        self._on_start("load_checkpoint", set_barrier)
        self._validate_and_store_args("on_load_checkpoint_start", metrics_input_kwargs)
        if self._store_has_key("load_checkpoint_start_time"):
            self._log_metrics({"load_checkpoint_start_time": self._store_get("load_checkpoint_start_time")})
        else:
            self._log_metrics({"load_checkpoint_start_time": self.timer.get("load_checkpoint").get("start") * 1000})

    @_check_enabled
    def on_load_checkpoint_end(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics when finish loading checkpoint.

        :param set_barrier: synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: metrics needed for callback function invocation.
        :type metrics_input_kwargs: dict

        ****

        **metrics_input_kwargs** could contain (optional):
            - **load_checkpoint_finish_time**: The timestamp of finishing checkpoint loading. If provide, OneLogger will use this value or it will use internal timer to get the timestamp.

        """
        self._on_end("load_checkpoint", set_barrier)
        self._validate_and_store_args("on_load_checkpoint_end", metrics_input_kwargs)

        if self._store_has_key("load_checkpoint_finish_time"):
            load_checkpoint_finish_time = self._store_get("load_checkpoint_finish_time")
        else:
            load_checkpoint_finish_time = self.timer.get("load_checkpoint").get("finish") * 1000

        if self._store_has_key("load_checkpoint_start_time"):
            load_checkpoint_time = (
                load_checkpoint_finish_time - self._store_get("load_checkpoint_start_time")
            ) / 1000  # in sec
        else:
            load_checkpoint_time = self.timer.get("load_checkpoint").get("total")  # in sec

        self._log_metrics(
            {
                "load_checkpoint_finish_time": load_checkpoint_finish_time,
                "load_checkpoint_time": load_checkpoint_time,
            }
        )

    @_check_enabled
    def on_optimizer_init_start(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics at the start of building optimizer

        :param set_barrier: synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: metrics needed for callback function invocation.
        :type metrics_input_kwargs: dict

        ****

        **metrics_input_kwargs** could contain (optional):
            - **app_build_optimizer_start_time**: The timestamp of starting optimizer build. If provide, OneLogger will use this value or it will use internal timer to get the timestamp.

        """
        self._on_start("optimizer_init", set_barrier)
        self._validate_and_store_args("on_optimizer_init_start", metrics_input_kwargs)

        if self._store_has_key("app_build_optimizer_start_time"):
            self._log_metrics({"app_build_optimizer_start_time": self._store_get("app_build_optimizer_start_time")})
        else:
            self._log_metrics({"app_build_optimizer_start_time": self.timer.get("optimizer_init").get("start") * 1000})

    @_check_enabled
    def on_optimizer_init_end(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics at the end of building optimizer

        :param set_barrier: synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: metrics needed for callback function invocation.
        :type metrics_input_kwargs: dict

        ****

        **metrics_input_kwargs** could contain (optional):
            - **app_build_optimizer_finish_time**: The timestamp of finishing optimizer build. If provide, OneLogger will use this value or it will use internal timer to get the timestamp.

        """
        self._on_end("optimizer_init", set_barrier)
        self._validate_and_store_args("on_optimizer_init_end", metrics_input_kwargs)

        if self._store_has_key("app_build_optimizer_finish_time"):
            self._log_metrics({"app_build_optimizer_finish_time": self._store_get("app_build_optimizer_finish_time")})
        else:
            self._log_metrics(
                {"app_build_optimizer_finish_time": self.timer.get("optimizer_init").get("finish") * 1000}
            )

    @_check_enabled
    def on_train_start(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, int]) -> None:
        """
        Log metrics at the beginning of the train loop.

        :param set_barrier: synchronize ranks before executing the callback. default to false. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: metrics needed for callback function invocation.
        :type metrics_input_kwargs: dict

        ****

        **metrics_input_kwargs** (for minimal schema) must contain:
            - **train_iterations_start**: the start iteration number.
            - **train_samples_start**: the start sample number.

        """
        if rankpulse:
            try:
                rankpulse.start(
                    interval_seconds=int(os.getenv("RANKPULSE_INTERVAL_SECONDS", "15")),
                    twindow_seconds=int(os.getenv("RANKPULSE_TWINDOW_SECONDS", "300")),
                    enable_gpu_debug_info=True
                    if os.environ.get("RANKPULSE_GPU_DEBUG_INFO", "0").lower() in ["1", "true", "yes", "y"]
                    else False,
                )
            except Exception as e:
                print(f"WARNING: Failed to start rankpulse: {e}")
        self._on_start("app_train_loop", set_barrier)
        self._validate_and_store_args("on_train_start", metrics_input_kwargs)

        metrics_to_log = {}
        if self.mode & self.O_MINIMAL:
            batch_size = self._store_get("batch_size")
            train_iterations_start = self._store_get("train_iterations_start")
            self._store_set("train_iterations_start", train_iterations_start)
            self._store_set("train_iterations_end", train_iterations_start)
            if self._store_has_key("train_samples_start"):
                self._store_set("train_samples_end", self._store_get("train_samples_start"))
            else:
                self._store_set(
                    "train_samples_start",
                    train_iterations_start * batch_size,
                )
                self._store_set(
                    "train_samples_end",
                    train_iterations_start * batch_size,
                )
            metrics_to_log["train_iterations_start"] = train_iterations_start
            metrics_to_log["train_iterations_end"] = train_iterations_start
            metrics_to_log["train_samples_start"] = self._store_get("train_samples_start")
            metrics_to_log["train_samples_end"] = self._store_get("train_samples_end")
            metrics_to_log["app_train_loop_start_time"] = self.timer.get("app_train_loop").get("start") * 1000
            if self._store_has_key("app_start_time"):
                metrics_to_log["app_start_time"] = self._store_get("app_start_time")
        if self.mode & self.O_THROUGHPUT:
            batch_size = self._store_get("batch_size")
            flops_per_sample = self._store_get("flops_per_sample")
            # The initial value of num_floating_point_operations_so_far is nonzero if loading ckpt, while total_flops is always zero.
            self._store_set(
                "num_floating_point_operations_so_far",
                train_iterations_start * batch_size * flops_per_sample,
            )
            self._store_set("total_flops", 0)
            metrics_to_log["train_tflop_start"] = float(self._store_get("num_floating_point_operations_so_far")) / (
                10**12
            )
        self._log_metrics(metrics_to_log)

    @_check_enabled
    def on_train_end(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics at the end of the train loop

        :param set_barrier: Synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: Metrics needed for callback function invocation. Currently no input metrics needed.
        :type metrics_input_kwargs: dict
        """
        self._on_end("app_train_loop", set_barrier)
        self._validate_and_store_args("on_train_end", metrics_input_kwargs)
        metrics_to_log = {
            "app_train_loop_finish_time": self.timer.get("app_train_loop").get("finish") * 1000,
        }
        self._log_metrics(metrics_to_log)

    @_check_enabled
    def on_train_batch_start(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics at the beginning of each train iteraion
        :param set_barrier: Synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: Metrics needed for callback function invocation. Currently no input metrics needed.
        :type metrics_input_kwargs: dict
        """
        self._on_start("train_iterations", set_barrier)
        self._validate_and_store_args("on_train_batch_start", metrics_input_kwargs)

    @_check_enabled
    def on_train_batch_end(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics at the end of each train iteraion
        :param set_barrier: Synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: Metrics needed for callback function invocation. Currently no input metrics needed.
        :type metrics_input_kwargs: dict
        """
        self._on_end("train_iterations", set_barrier)
        self._validate_and_store_args("on_train_batch_end", metrics_input_kwargs)

        if self.mode & self.O_MINIMAL:
            global_batch_size = self._store_get("global_batch_size")
            self._store_set("train_iterations", self._store_get("train_iterations") + 1)
            self._store_set(
                "train_iterations_end",
                self._store_get("train_iterations_start") + self._store_get("train_iterations"),
            )
            self._store_set(
                "train_samples",
                self._store_get("train_samples") + global_batch_size,
            )
            self._store_set(
                "train_samples_end",
                self._store_get("train_samples_start") + self._store_get("train_samples"),
            )
            self._store_set(
                "train_iterations_time_total",
                self.timer.get("train_iterations").get("total"),
            )
            if not self._store_has_key("first_logged_train_iterations_finish_time"):
                self._store_set(
                    "first_logged_train_iterations_finish_time",
                    self.timer.get("train_iterations").get("finish") * 1000,
                )
            if self._store_has_key("seq_length"):
                self._store_set(
                    "train_tokens",
                    self._store_get("seq_length") * self._store_get("train_samples"),
                )
        if self.mode & self.O_THROUGHPUT:
            global_batch_size = self._store_get("global_batch_size")
            flops_per_sample = self._store_get("flops_per_sample")
            flops = global_batch_size * flops_per_sample
            self._store_set(
                "num_floating_point_operations_so_far",
                self._store_get("num_floating_point_operations_so_far") + flops,
            )
            self._store_set("total_flops", self._store_get("total_flops") + flops)

        metrics_to_log = self._get_metrics_on_train_batch_end()

        if self._store_get("train_iterations_end") % self.log_every_n_train_iterations == 0:
            self._log_metrics(metrics_to_log)

    def _get_metrics_on_train_batch_end(self):
        """Helper function to get all metrics needed to be tracked on_train_batch_end"""
        metrics_to_log = {}
        if self.mode & self.O_MINIMAL:
            metrics_to_log["train_iterations"] = self._store_get("train_iterations")
            metrics_to_log["train_iterations_end"] = self._store_get("train_iterations_start") + self._store_get(
                "train_iterations"
            )
            metrics_to_log["train_samples"] = self._store_get("train_samples")
            metrics_to_log["train_samples_end"] = self._store_get("train_samples_start") + self._store_get(
                "train_samples"
            )
            metrics_to_log["train_iterations_time_total"] = self._store_get("train_iterations_time_total")
            timer_data = self.timer.get("train_iterations")
            timer_avg = timer_data.get("avg")
            timer_min = timer_data.get("min")
            timer_finish = timer_data.get("finish")
            metrics_to_log["train_iterations_time_msecs_avg"] = timer_avg * 1000 if timer_avg is not None else 0
            metrics_to_log["train_iterations_time_msecs_min"] = timer_min * 1000 if timer_min is not None else 0
            metrics_to_log["first_logged_train_iterations_finish_time"] = self._store_get(
                "first_logged_train_iterations_finish_time", default_value=0
            )
            metrics_to_log["last_logged_train_iterations_finish_time"] = (
                timer_finish * 1000 if timer_finish is not None else 0
            )
            if self._store_has_key("train_tokens"):
                metrics_to_log["train_tokens"] = self._store_get("train_tokens")
            self._log_app_tag(self._store_get("app_tag"))
        if self.mode & self.O_THROUGHPUT:
            train_iterations_time_total = self._store_get("train_iterations_time_total")
            if train_iterations_time_total:
                train_throughput_per_gpu = self._store_get("total_flops") / (
                    train_iterations_time_total * 10**12 * self._store_get("world_size")
                )
            else:
                train_throughput_per_gpu = 0.0
            if not self._store_has_key("train_throughput_per_gpu_max"):
                self._store_set("train_throughput_per_gpu_max", train_throughput_per_gpu)
            else:
                self._store_set(
                    "train_throughput_per_gpu_max",
                    max(
                        train_throughput_per_gpu,
                        self._store_get("train_throughput_per_gpu_max"),
                    ),
                )
            metrics_to_log["train_tflop_end"] = float(self._store_get("num_floating_point_operations_so_far")) / (
                10**12
            )
            metrics_to_log["train_tflop"] = float(self._store_get("total_flops")) / (10**12)
            metrics_to_log["train_throughput_per_gpu"] = train_throughput_per_gpu
            metrics_to_log["train_throughput_per_gpu_max"] = self._store_get("train_throughput_per_gpu_max")

        return metrics_to_log

    @_check_enabled
    def on_validation_start(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics at the begininig of the validation loop

        :param set_barrier: Synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: Metrics needed for callback function invocation. Currently no input metrics needed.
        :type metrics_input_kwargs: dict
        """
        self._on_start("validation_loop", set_barrier)
        self._validate_and_store_args("on_validation_start", metrics_input_kwargs)

    @_check_enabled
    def on_validation_batch_start(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics at the beginning of each validation iteraion

        :param set_barrier: Synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: Metrics needed for callback function invocation. Currently no input metrics needed.
        :type metrics_input_kwargs: dict
        """
        self._on_start("validation_iterations", set_barrier)
        self._validate_and_store_args("on_validation_batch_start", metrics_input_kwargs)

    @_check_enabled
    def on_validation_batch_end(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]):
        """Log metrics at the end of each validation iteraion

        :param set_barrier: Synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: Metrics needed for callback function invocation. Currently no input metrics needed.
        :type metrics_input_kwargs: dict
        """
        self._on_end("validation_iterations", set_barrier)
        self._validate_and_store_args("on_validation_batch_end", metrics_input_kwargs)
        self._store_set(
            "validation_iterations",
            self._store_get("validation_iterations") + 1,
        )

    @_check_enabled
    def on_validation_end(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics at the end of the validation loop

        :param set_barrier: Synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: Metrics needed for callback function invocation. Currently no input metrics needed.
        :type metrics_input_kwargs: dict
        """
        self._on_end("validation_loop", set_barrier)
        self._validate_and_store_args("on_validation_end", metrics_input_kwargs)
        self._store_set(
            "validation_iterations_time_total",
            self.timer.get("validation_iterations").get("total"),
        )

        metrics_to_log = {
            "validation_iterations_time_total": self.timer.get("validation_iterations").get("total"),
            "validation_iterations_time_msecs_avg": self.timer.get("validation_iterations").get("avg") * 1000,
            "validation_iterations_time_msecs_min": self.timer.get("validation_iterations").get("min") * 1000,
        }
        self._log_metrics(metrics_to_log)

    @_check_enabled
    def on_save_checkpoint_start(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """Log metrics before saving the chekpoint

        :param set_barrier: Synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: Metrics needed for callback function invocation. Currently no input metrics needed.
        :type metrics_input_kwargs: dict

        ****

        **metrics_input_kwargs** must contain:
            - **global_step**: The saving iteration number.

        """
        assert self.mode & self.O_CKPT, (
            "Checkpoint saving is not enabled. Please ensure that `is_save_checkpoint_enabled = True` "
            "in the config to enable this feature."
        )

        self._on_start("save_checkpoint", set_barrier)
        self._validate_and_store_args("on_save_checkpoint_start", metrics_input_kwargs)

        # Make sure iteration related metrics is updated in DB for each checkpointing
        metrics_to_update = self._get_metrics_on_train_batch_end()
        self._log_metrics(metrics_to_update)

        global_step = self._store_get("global_step")

        self._store_set(
            "save_checkpoint_count",
            self._store_get("save_checkpoint_count") + 1,
        )

        productive_metrics = {
            "train_iterations_productive_end": self._store_get("train_iterations_end"),
            "train_samples_productive_end": self._store_get("train_samples_end"),
            "train_iterations_time_total_productive": self._store_get("train_iterations_time_total"),
            "validation_iterations_time_total_productive": self._store_get("validation_iterations_time_total"),
        }
        if self.mode & self.O_THROUGHPUT:
            productive_metrics.update(
                {
                    "train_tflop_productive_end": float(self._store_get("num_floating_point_operations_so_far"))
                    / (10**12)
                }
            )
        self._store_set(f"productive_metrics:{global_step}", productive_metrics)
        metrics_to_log = {
            "train_iterations_save_checkpoint_end": self._store_get("train_iterations_end"),
            "save_checkpoint_count": self._store_get("save_checkpoint_count"),
        }
        self._log_metrics(metrics_to_log)

    @_check_enabled
    def on_save_checkpoint_end(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, int]) -> None:
        """
        Log metrics after saving the chekpoint

        :param set_barrier: Synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: Metrics needed for callback function invocation.
        :type metrics_input_kwargs: dict

        ****

        **metrics_input_kwargs** (for Saving Checkpoint-related Metrics) must contain:
            - **global_step**: The saving iteration number.

        """
        assert self.mode & self.O_CKPT, (
            "Checkpoint saving is not enabled. Please ensure that `is_save_checkpoint_enabled = True` "
            "in the config to enable this feature."
        )

        self._on_end("save_checkpoint", set_barrier)
        self._validate_and_store_args("on_save_checkpoint_end", metrics_input_kwargs)

        global_step = self._store_get("global_step")

        metrics_to_log = {}

        self._store_set(
            "save_checkpoint_sync_count",
            self._store_get("save_checkpoint_sync_count") + 1,
        )

        productive_time = {
            "save_checkpoint_sync_time_total_productive": self.timer.get("save_checkpoint").get("total"),
            "successful_save_checkpoint_sync_finish_time": self.timer.get("save_checkpoint").get("finish") * 1000,
        }

        self._store_set(f"productive_time:{global_step}", productive_time)


        if self._store_has_key(f"on_save_checkpoint_success:{global_step}"):
            successful_save_checkpoint_sync_finish_time = productive_time.pop(
                "successful_save_checkpoint_sync_finish_time"
            )
            metrics_to_log.update(productive_time)

            # Check and track first/last_successful_save_checkpoint_sync_finish_time
            if not self._store_has_key("first_save_checkpoint_success"):
                self._store_set("first_save_checkpoint_success", True)
                metrics_to_log.update(
                    {
                        "first_saved_train_iterations_start_time": self.timer.get("app_train_loop").get("start") * 1000,
                        "first_successful_save_checkpoint_sync_finish_time": successful_save_checkpoint_sync_finish_time,
                    }
                )
            metrics_to_log.update(
                {
                    "last_successful_save_checkpoint_sync_finish_time": successful_save_checkpoint_sync_finish_time,
                }
            )
            self.one_logger.store_pop(f"on_save_checkpoint_success:{global_step}")

        metrics_to_log.update(
            {
                "save_checkpoint_sync_time_total": self.timer.get("save_checkpoint").get("total"),
                "save_checkpoint_sync_time_min": self.timer.get("save_checkpoint").get("min"),
                "save_checkpoint_sync_time_max": self.timer.get("save_checkpoint").get("max"),
                "save_checkpoint_sync_count": self._store_get("save_checkpoint_sync_count"),
            }
        )

        self._log_metrics(metrics_to_log)

        # Set flag for on_save_checkpoint_end call done
        self._store_set(f"on_save_checkpoint_end:{global_step}", True)

    @_check_enabled
    def on_save_checkpoint_success(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, int]) -> None:
        """
        Log metrics after saving the chekpoint successfully

        :param set_barrier: Synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: Metrics needed for callback function invocation.
        :type metrics_input_kwargs: dict

        ****

        **metrics_input_kwargs** (for Saving Checkpoint-related Metrics) must contain:
            - **global_step**: The saving iteration number.

        """
        assert self.mode & self.O_CKPT, (
            "Checkpoint saving is not enabled. Please ensure that `is_save_checkpoint_enabled = True` "
            "in the config to enable this feature."
        )

        self._validate_and_store_args("on_save_checkpoint_success", metrics_input_kwargs)

        metrics_to_log = {}

        global_step = self._store_get("global_step")

        # Fetch productivity metrics cached on_save_checkpoint_start
        productive_metrics = self.one_logger.store_pop(f"productive_metrics:{global_step}")


        # Check if on_save_checkpoint_end is called.
        if self._store_has_key(f"on_save_checkpoint_end:{global_step}"):
            productive_time = self._store_get(f"productive_time:{global_step}")
            successful_save_checkpoint_sync_finish_time = productive_time.pop(
                "successful_save_checkpoint_sync_finish_time"
            )
            productive_metrics.update(productive_time)

            # Check and track first/last_successful_save_checkpoint_sync_finish_time
            if not self._store_has_key("first_save_checkpoint_success"):
                self._store_set("first_save_checkpoint_success", True)
                metrics_to_log.update(
                    {
                        "first_saved_train_iterations_start_time": self.timer.get("app_train_loop").get("start") * 1000,
                        "first_successful_save_checkpoint_sync_finish_time": successful_save_checkpoint_sync_finish_time,
                    }
                )
            metrics_to_log.update(
                {
                    "last_successful_save_checkpoint_sync_finish_time": successful_save_checkpoint_sync_finish_time,
                }
            )
            self.one_logger.store_pop(f"on_save_checkpoint_end:{global_step}")

        # Check need to update productive metrics for current step
        need_update = True
        if self._store_has_key("global_step_max"):
            need_update = global_step > self._store_get("global_step_max")

        if need_update:
            self._store_set("global_step_max", global_step)
            metrics_to_log.update(productive_metrics)

        if self.mode & self.O_CKPT:
            if self._store_get("save_checkpoint_strategy") == "async":
                self._store_set(
                    "save_checkpoint_async_count",
                    self._store_get("save_checkpoint_async_count") + 1,
                )
                metrics_to_log.update({"save_checkpoint_async_count": self._store_get("save_checkpoint_async_count")})

        self._log_metrics(metrics_to_log)

        # Set flag for on_save_checkpoint_success call done
        self._store_set(f"on_save_checkpoint_success:{global_step}", True)

    @_check_enabled
    def on_app_start(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """
        Log metrics at the begining of application run.

        :param set_barrier: Synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: Metrics needed for callback function invocation.
        :type metrics_input_kwargs: dict

        ****

        **metrics_input_kwargs** must contain:
            - **app_start_time**: The application start timestamp in ms. If provide, OneLogger will use this value or it will use internal timer to get the timestamp.

        """
        self._on_start("app_run", set_barrier)
        self._validate_and_store_args("on_app_start", metrics_input_kwargs)

        if self._store_has_key("app_start_time"):
            self._log_metrics({"app_start_time": self._store_get("app_start_time")})
        else:
            self._log_metrics({"app_start_time": self.timer.get("app_run").get("start") * 1000.0})

    @_check_enabled
    def on_app_end(self, set_barrier: bool = False, **metrics_input_kwargs: Dict[str, Any]) -> None:
        """
        Log metrics at the end of application run.

        :param set_barrier: Synchronize ranks before executing the callback. Default to False. NOTE: if this is set to True, `barrier` in `OneLoggerUtils` constructor must be set with correct callable object.
        :type set_barrier: bool

        :param metrics_input_kwargs: Metrics needed for callback function invocation.
        :type metrics_input_kwargs: dict

        ****

        **metrics_input_kwargs** could contain (optional):
            - **app_finish_time**: The timestamp of finishing application run. If provide, OneLogger will use this value or it will use internal timer to get the timestamp.
        """
        self._on_end("app_run", set_barrier)
        self._validate_and_store_args("on_app_end", metrics_input_kwargs)
        if self._store_has_key("app_finish_time"):
            self._log_metrics({"app_finish_time": self._store_get("app_finish_time")})
        else:
            self._log_metrics({"app_finish_time": self.timer.get("app_run").get("finish") * 1000.0})

    @_check_enabled
    def finish(self) -> None:
        """
        Mark a OneLogger tracking as finished, and finish uploading all data with clean up. NOTE: Please remember to call this function explicitly to avoid potential data loss.
        """
        if rankpulse:
            try:
                rankpulse.stop(timeout_seconds=3.0)
            except Exception as e:
                print(f"WARNING: Failed to stop rankpulse: {e}")
        self.one_logger.finish()


def initialize_one_logger(config: dict) -> None:
    """Initialize OneLoggerUtils object as a global variable"""
    global one_logger
    assert one_logger is None, "one_logger is already initialized"

    one_logger = OneLoggerUtils(config)

    return


def initialize_one_logger_from_imaginaire_config(config: Any) -> None:
    """Initialize OneLoggerUtils object from imaginaire4 config"""

    try:
        batch_size = config.dataloader_train.batch_size
    except Exception as e:
        logger.warning(e)
        logger.warning("config.dataloader_train.batch_size does not exist. We substitute it with 1.")
        # For edify video, image & video joint trianing configs do not have a fixed batch_size.
        # We use a substitute batch_size = 1.
        batch_size = 1

    if parallel_state is None or not parallel_state.is_initialized():
        # Support environment without mcore installed or not initialized.
        cp_size = 1
        tp_size = 1
        data_parallel_size = 1
        pp_size = 1
    else:
        cp_size = parallel_state.get_context_parallel_world_size()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()

    micro_batch_size = batch_size / cp_size / tp_size
    global_batch_size = batch_size * data_parallel_size

    job_size = (
        f"BS_{batch_size}"
        + f"_microBS_{micro_batch_size}"
        + f"_nGPUs_{get_world_size()}"
        + f"_TP_{tp_size}"
        + f"_CP_{cp_size}"
        + f"_PP_{pp_size}"
    )

    # check if the environment is local / slurm / NGC.
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    ngc_job_id = os.environ.get("NGC_JOB_ID")
    if slurm_job_id is not None:
        job_environment = "slurm"
    elif ngc_job_id is not None:
        job_environment = "ngc"
    else:  # usually a local machine.
        job_environment = os.uname().nodename

    # Check GPU information.
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name().replace(" ", "-")  # ex) NVIDIA-A100-SXM4-80GB
    else:
        gpu_name = "none"

    app_tag_run_name = f"{config.job.project}/{config.job.group}/{config.job.name}"
    app_tag_run_version = "0.0.0"  # hard-coded app_tag_run version.
    app_tag = f"{app_tag_run_name}/{job_size}/ENV_{job_environment}_GPU_{gpu_name}"
    # Allow disabling async mode via environment variable to avoid BrokenPipeError issues
    # Set ONE_LOGGER_ASYNC=false to use synchronous mode (more stable with wandb online)
    one_logger_async = os.environ.get("ONE_LOGGER_ASYNC", "true").lower() == "true"
    one_logger_config = dict(
        enable_for_current_rank=(get_rank() == 0),  # global master rank
        one_logger_project="imaginaire4",  # wandb project name for OneLogger
        one_logger_run_name=f"{config.job.project}/{config.job.group}/{config.job.name}",
        one_logger_async=one_logger_async,  # Set ONE_LOGGER_ASYNC=false env var to avoid pipe issues
        log_every_n_train_iterations=config.trainer.logging_iter,
        barrier=None,
        app_tag=app_tag,  # jobs with same app_tag are expected to have similar perf.
        app_tag_run_name=app_tag_run_name,  # variations within an app_tag
        app_tag_run_version=app_tag_run_version,
        app_start_time=round(time.time() * 1000),  # timestamp in ms
        world_size=get_world_size(),
        micro_batch_size=micro_batch_size,
        batch_size=batch_size,
        global_batch_size=global_batch_size,
        train_iterations_target=config.trainer.max_iter,
        train_samples_target=config.trainer.max_iter * global_batch_size,
        is_baseline_run=False,
        is_train_iterations_enabled=True,
        is_validation_iterations_enabled=True,
        is_test_iterations_enabled=False,
        is_save_checkpoint_enabled=True,
        save_checkpoint_strategy="async",
        is_log_throughput_enabled=False,
    )
    initialize_one_logger(one_logger_config)

    return


def one_logger_is_initialized() -> bool:
    return one_logger is not None


def destroy_one_logger() -> None:
    global one_logger
    one_logger = None

    return


def get_one_logger() -> OneLoggerUtils:
    return one_logger
