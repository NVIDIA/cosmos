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

from __future__ import annotations

import collections
import collections.abc
import functools
import json
import os
import random
from contextlib import ContextDecorator, nullcontext
from dataclasses import fields
from typing import Any, Callable, List, Tuple, TypeVar, Union

import numpy as np
from loguru import logger as logging

try:
    # pyrefly: ignore  # import-error
    import straggler
except ImportError:
    straggler = None
import termcolor
import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed._tensor.api import DTensor

from cosmos3._src.imaginaire.utils import distributed, log
from cosmos3._src.imaginaire.utils.distributed import all_gather_tensor
from cosmos3._src.imaginaire.utils.easy_io import easy_io
from cosmos3._src.imaginaire.utils.timer import Timer


def requires_grad(model: torch.nn.Module, value: bool = True) -> None:
    """Set a model to require gradients or not.

    Args:
        model (torch.nn.Module): Neural network model.
        value (bool): Whether the network requires gradients or not.
    """
    for p in model.parameters():
        p.requires_grad = value


def to(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    memory_format: torch.memory_format = torch.preserve_format,
) -> Any:
    """Recursively cast data into the specified device, dtype, and/or memory_format.

    The input data can be a tensor, a list of tensors, a dict of tensors.
    See the documentation for torch.Tensor.to() for details.

    Args:
        data (Any): Input data.
        device (str | torch.device): GPU device (default: None).
        dtype (torch.dtype): data type (default: None).
        memory_format (torch.memory_format): memory organization format (default: torch.preserve_format).

    Returns:
        data (Any): Data cast to the specified device, dtype, and/or memory_format.
    """
    assert device is not None or dtype is not None or memory_format is not None, (
        "at least one of device, dtype, memory_format should be specified"
    )

    if isinstance(data, torch.Tensor):
        if (
            memory_format == torch.channels_last
            and data.dim() != 4
            or memory_format == torch.channels_last_3d
            and data.dim() != 5
        ):
            memory_format = torch.preserve_format  # do not change the memory format
        is_cpu = (isinstance(device, str) and device == "cpu") or (
            isinstance(device, torch.device) and device.type == "cpu"
        )
        data = data.to(
            device=device,
            dtype=dtype,
            memory_format=memory_format,
            non_blocking=(not is_cpu),
        )
        return data
    elif isinstance(data, collections.abc.Mapping):
        return type(data)({key: to(data[key], device=device, dtype=dtype, memory_format=memory_format) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return type(data)([to(elem, device=device, dtype=dtype, memory_format=memory_format) for elem in data])
    else:
        return data


def serialize(data: Any) -> Any:
    """Serialize data by hierarchically traversing through iterables.

    Args:
        data (Any): Input data.

    Returns:
        data (Any): Serialized data.
    """
    if isinstance(data, collections.abc.Mapping):
        return type(data)({key: serialize(data[key]) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return type(data)([serialize(elem) for elem in data])
    else:
        try:
            json.dumps(data)
        except TypeError:
            data = str(data)
        return data


def print_environ_variables(env_vars: list[str]) -> None:
    """Print a specific list of environment variables.

    Args:
        env_vars (list[str]): List of specified environment variables.
    """
    for env_var in env_vars:
        if env_var in os.environ:
            log.info(f"Environment variable {Color.green(env_var)}: {Color.yellow(os.environ[env_var])}")
        else:
            log.warning(f"Environment variable {Color.green(env_var)} not set!")


def set_random_seed(seed: int, by_rank: bool = False) -> None:
    """Set random seed. This includes random, numpy, Pytorch.

    Args:
        seed (int): Random seed.
        by_rank (bool): if true, each GPU will use a different random seed.
    """
    if by_rank:
        seed += distributed.get_rank()
    log.info(f"Using random seed {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets seed on the current CPU & all GPUs


def arch_invariant_rand(
    shape: List[int] | Tuple[int], dtype: torch.dtype, device: str | torch.device, seed: int | None = None
):
    """Produce a GPU-architecture-invariant randomized Torch tensor.

    Args:
        shape (list or tuple of ints): Output tensor shape.
        dtype (torch.dtype): Output tensor type.
        device (torch.device): Device holding the output.
        seed (int): Optional randomization seed.

    Returns:
        tensor (torch.tensor): Randomly-generated tensor.
    """
    # Create a random number generator, optionally seeded
    rng = np.random.RandomState(seed)

    # Generate random numbers using the generator
    random_array = rng.standard_normal(shape).astype(np.float32)  # Use standard_normal for normal distribution

    # Convert to torch tensor and return
    return torch.from_numpy(random_array).to(dtype=dtype, device=device)


def get_data_batch_size(data: dict[str, torch.Tensor] | torch.Tensor) -> int:
    """Get the batch size from a data batch, a (possibly hierarchical) dictionary of tensors.

    Args:
        data (dict[str, torch.Tensor]): Data batch (dictionary of tensors).

    Returns:
        batch_size (int): Data batch size.
    """

    def _get_batch_size(input_data: Any) -> Union[int, None]:
        """
        Helper function that recursively finds a tensor in the input data
        (could be a nested dictionary or list of tensors) and returns its batch size.
        """
        if isinstance(input_data, torch.Tensor):
            return len(input_data)
        elif isinstance(input_data, collections.abc.Mapping):
            for key, value in input_data.items():
                batch_size = _get_batch_size(value)
                if batch_size is not None:
                    return batch_size
        elif isinstance(input_data, (list, tuple)) and len(input_data) > 0:
            # Handle list/tuple of tensors (variable-length batches)
            # The batch size is the length of the list
            # We are verifying if input_data[0] is indeed a tensor. If so, return the length of the list.
            if isinstance(input_data[0], torch.Tensor):
                return len(input_data)
            # Recurse into first element if it's a nested structure
            return _get_batch_size(input_data[0])
        return None

    batch_size = _get_batch_size(data)
    if not isinstance(batch_size, int):
        raise ValueError(f"Batch size ({batch_size}) obtained from invalid data: {data}")
    return batch_size


def parameters_to_buffer(module: torch.nn.Module, persistent: bool = True):
    """Convert parameters in a module to buffers.
    Buffers do not have its own gradients and thus not updated by backpropagation.

    Args:
        module (torch.nn.Module): a module to convert parameters
        persistent (bool): If True, buffers are included in state_dict.
    """
    named_params = dict()

    for name, param in module.named_parameters():
        named_params[name] = param

    for name, param in named_params.items():
        module_hierarchy = name.split(".")
        submodule_name = ".".join(module_hierarchy[:-1])
        submodule = module.get_submodule(submodule_name)
        subname = module_hierarchy[-1]
        delattr(submodule, subname)
        submodule.register_buffer(subname, param, persistent=persistent)

    return


T = TypeVar("T", bound=Callable[..., Any])


class timer(Timer):
    """Simple CPU timer for timing the execution of code.

    It can be used as either a context manager or a function decorator. The timing result will be logged upon exit.

    Example:
        def func_a():
            time.sleep(1)
        with timer("func_a"):
            func_a()

        @timer("func_b)
        def func_b():
            time.sleep(1)
        func_b()
    """

    def __init__(self, context: str, debug: bool = False):
        super().__init__(
            tag=context,
            measure_cpu=True,
            measure_cuda=False,
            unit="s",
            debug=debug,
        )


class memory_checker(ContextDecorator):  # noqa: N801
    """Simple memory checker for a given block of code.

    It can be used as either a context manager or a function decorator. The memory usage will be logged upon exit.
    Example:
        def func_a():
            torch.rand([int(1024**2)]).float().cuda()
        with memory_checker("func_a"):
            func_a()
        >>> 0.004GB memory used

        @memory_checker("func_b")
        def func_b():
            random_var = torch.rand([int(1024**2)]).cuda()
        func_b()
    """

    def __init__(self, context: str, debug: bool = False):
        self.context = context
        self.debug = debug

    def __enter__(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.initial_memory = torch.cuda.max_memory_allocated()

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        torch.cuda.synchronize()
        final_memory = torch.cuda.max_memory_allocated()
        message = f"Memory used within {self.context}: {(final_memory - self.initial_memory) / 1024**3:.4f} GB"
        if self.debug:
            log.debug(message)
        else:
            log.info(message)

    def __call__(self, func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN202
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.max_memory_allocated()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            final_memory = torch.cuda.max_memory_allocated()
            message = f"Memory used within {self.context}: {(final_memory - initial_memory) / 1024**3:.4f} GB"
            if self.debug:
                log.debug(message)
            else:
                log.info(message)
            return result

        return wrapper  # type: ignore


class TrainingTimer:
    """Timer for timing the execution of code, aggregating over multiple training iterations.

    It is used as a context manager to measure the execution time of code and store the timing results
    for each function. The context managers can be nested.

    Attributes:
        results (dict): A dictionary to store timing results for various code.

    Example:
        timer = Timer()
        for i in range(100):
            with timer("func_a"):
                func_a()
        avg_time = sum(timer.results["func_a"]) / len(timer.results["func_a"])
        print(f"func_a() took {avg_time} seconds.")
    """

    def __init__(self) -> None:
        self.results = dict()
        self.average_results = dict()
        self.timers = []
        self.func_stack = []
        self.reset()

    def reset(self) -> None:
        self.results = {key: [] for key in self.results}

    def __enter__(self) -> TrainingTimer:
        timer = Timer(measure_cpu=True, measure_cuda=False, debug=True, unit="s")
        self.timers.append(timer)
        timer.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        timer = self.timers.pop()
        timer.end()
        result = timer.get_cpu_time()
        key = self.func_stack.pop()
        self.results.setdefault(key, [])
        self.results[key].append(result)

    def __call__(self, func_name: str) -> TrainingTimer:
        self.func_stack.append(func_name)
        return self

    def __getattr__(self, func_name: str) -> TrainingTimer:
        return self.__call__(func_name)

    def nested(self, func_name: str) -> TrainingTimer:
        return self.__call__(func_name)

    def compute_average_results(self) -> dict[str, float]:
        results = dict()
        for key, value_list in self.results.items():
            results[key] = sum(value_list) / len(value_list)
        return results


def timeout_handler(timeout_period: float, signum: int, frame: int) -> None:
    # What to do when the process gets stuck. For now, we simply end the process.
    error_message = f"Timeout error: more than {timeout_period} seconds passed since the last iteration."
    if distributed.is_rank0():
        import wandb

        wandb.alert(title="Timeout error!", text=error_message, level=wandb.AlertLevel.ERROR)
    raise TimeoutError(error_message)


class Color:
    """A convenience class to colorize strings in the console.

    Example:
        import
        print("This is {Color.red('important')}.")
    """

    @staticmethod
    def red(x: str) -> str:
        return termcolor.colored(str(x), color="red")

    @staticmethod
    def green(x: str) -> str:
        return termcolor.colored(str(x), color="green")

    @staticmethod
    def blue(x: str) -> str:
        return termcolor.colored(str(x), color="blue")

    @staticmethod
    def cyan(x: str) -> str:
        return termcolor.colored(str(x), color="cyan")

    @staticmethod
    def yellow(x: str) -> str:
        return termcolor.colored(str(x), color="yellow")

    @staticmethod
    def magenta(x: str) -> str:
        return termcolor.colored(str(x), color="magenta")

    @staticmethod
    def grey(x: str) -> str:
        return termcolor.colored(str(x), color="grey")


class BufferCnt:
    """
    Buffer counter which keeps track of the condition when called and returns True when the condition in met "thres"
    amount of times, otherwise returns False.

    Example usage:
        buf = BufferCnt(thres=3)
        for _ in range(5):
            if buf(random.random() > 0.5):
                print("We got lucky 3 times out of 5.")

    Args:
        thres (int): The amount of times the expression needs to be True before returning True.
        reset_over_thres (bool): Whether to reset the buffer after returning True.
    """

    def __init__(self, thres=10, reset_over_thres=False):
        self._cnt = 0
        self.thres = thres
        self.reset_over_thres = reset_over_thres

    def __call__(self, expre, thres=None):
        if expre is True:
            self._cnt += 1
        else:
            self._cnt = 0

        if thres is None:
            thres = self.thres

        if self._cnt >= thres:
            if self.reset_over_thres:
                self.reset()
            return True

        return False

    @property
    def cnt(self):
        return self._cnt

    def reset(self):
        self._cnt = 0


def dataclass_instance_to_dict(dataclass: Any) -> dict:
    """Convert a dataclass to a dictionary.

    Args:
        dataclass (Any): Dataclass object.

    Returns:
        dict: Dictionary representation of the dataclass.
    """
    return {f.name: getattr(dataclass, f.name) for f in fields(dataclass)}


def get_local_tensor_if_DTensor(tensor: torch.Tensor | DTensor) -> torch.tensor:
    if isinstance(tensor, DTensor):
        local = tensor.to_local()
        # As per PyTorch documentation, if the communication is not finished yet, we need to wait for it to finish
        # https://pytorch.org/docs/stable/distributed.tensor.html#torch.distributed.tensor.DTensor.to_local
        if isinstance(local, AsyncCollectiveTensor):
            return local.wait()
        else:
            return local
    return tensor


def set_torch_compile_options(recompile_limit: int = 8, use_duck_shape: bool = True):
    """
    Set some of the torch compile config options. The default values of arguments are default config values in PyTorch as of 2.10 version.
    The value recompile_limit=32 is useful for Wan Tokenizer encoding compilation, as the standard value of 8 can easily overflow.
    The value of use_duck_shape=False is useful for Cosmos3 MoT training to reduce recompilations.

    Args:
        recompile_limit (int): Controls the maximum number of cache entries with a guard on same ID_MATCH'd object.
        use_duck_shape (bool): This flag changes whether we should use the same symbolic variable to represent input sizes that are the same
    """
    try:
        # PyTorch >= 2.7
        torch._dynamo.config.recompile_limit = recompile_limit
        torch.fx.experimental._config.use_duck_shape = use_duck_shape
    except AttributeError:
        try:
            torch._dynamo.config.cache_size_limit = recompile_limit
            torch.fx.experimental._config.use_duck_shape = use_duck_shape
        except AttributeError as e:
            log.warning("torch.compile is not available due to missing config options.")
            raise e


class NVTXRangeContext:
    """
    Context manager which inserts NVTX range around the current context and optionally calls torch.cuda.synchronize
    at the start and the end of the context.

    Args:
        name (str): Name of the NVTX range.
        enabled (bool): Whether the context manager is enabled. When disabled, it does nothing. Default: True.
        synchronize (bool): Whether to call torch.cuda.synchronize() at the start and the end of the context. Default: True.
    """

    def __init__(self, name: str, enabled: bool = True, synchronize: bool = True):
        self.name = name
        self.enabled = enabled
        self.synchronize = synchronize

    def __enter__(self):
        if not self.enabled:
            return
        if self.synchronize:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        if self.synchronize:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()


class StragglerDetectorV2:
    """StragglerDetectorV2 is a class that allows you to easily integrate "straggler" tool:
    https://gitlab-master.nvidia.com/dl/gwe/fault_tolerance_related/straggler/-/tree/cupti?ref_type=heads.

    This tool detects stragglers using low-level CUPTI tool, which can gather kernel execution time with very low overhead.
    The execution times are compared across different ranks, as well as to the execution time of the exact same kernels in the past.
    This tool can be easily integrated, as it's resilient to any synchronizations, since it captures kernels execution time.
    It means that we can wrap the entire  forward or backward passes and the stragglers will be identified regardless
    of synchronizations happening during the iteration.

    Args:
        enabled (bool): Whether the straggler detection is enabled. When disabled, it does nothing. Default: True.
        report_freq (int): Generate a report each report_freq iterations that analyzes the GPUs performance. Defaults to 100.
        profile_freq (int): Enable the CUPTI profiling each profile_freq iterations. Since the overhead is very low,
                            the default value is 1.
        max_diff (float): Defines the maximum relative difference between the fastest and the slowest rank to determine the slowdown. Defaults to 2.0
        raise_error (bool): Whether to raise error when stragglers are detected enough times. Defaults to True."""

    def __init__(
        self,
        enabled: bool = True,
        report_freq: int = 100,
        profile_freq: int = 1,
        max_diff: float = 2.0,
        raise_error: bool = True,
        save_s3: bool = False,
    ):
        self.enabled = enabled
        self.report_freq = report_freq
        self.profile_freq = profile_freq
        self.name = self.__class__.__name__
        self.slowdown_count = BufferCnt(thres=10, reset_over_thres=True)
        self.max_diff = max_diff
        self.raise_error = raise_error
        self.save_s3 = save_s3

    def initialize(self):
        if self.enabled:
            if not straggler:

                raise RuntimeError(
                    "Please install straggler package before using StragglerDetectionV2."
                    "Package can be installed from here: https://gitlab-master.nvidia.com/dl/osiris/straggler"
                )

            straggler.Detector.initialize(
                scores_to_compute=["relative_perf_scores", "individual_perf_scores"],
                gather_on_rank0=False,  # all ranks results will be available on rank 0
                profiling_interval=self.profile_freq,
            )

    def profile_section(self, name: str, section_enabled: bool, profile_cuda: bool = True):
        if section_enabled and self.enabled:
            return straggler.Detector.detection_section(name, profile_cuda=profile_cuda)
        else:
            return nullcontext()

    def _aggregate_section_results(self, local_section_summaries):
        data = []
        for key in local_section_summaries:
            # straggler reports time in ms
            data.append(local_section_summaries[key][straggler.Statistic.MAX] / 1000)
        return distributed.all_gather_tensor(torch.tensor(data).cuda())

    def generate_report(self, iteration):
        if self.enabled and iteration % self.report_freq == 0:
            report = straggler.Detector.generate_report()
            gpu_relative_perf_score = report.gpu_relative_perf_scores[distributed.get_rank()]
            gpu_relative_perf_score_gather_list = distributed.all_gather_tensor(
                torch.tensor([gpu_relative_perf_score]).cuda()
            )
            local_section_data = self._aggregate_section_results(report.local_section_summaries)
            if distributed.get_rank() == 0:
                stragglers = report.identify_stragglers(gpu_rel_threshold=1 / self.max_diff)
                wandb_info = {
                    f"{self.name}/relative_gpu_perf_{rank}": perf[0].item()
                    for rank, perf in enumerate(gpu_relative_perf_score_gather_list)
                }
                for key_id, key in enumerate(report.local_section_summaries):
                    wandb_info.update(
                        {f"{self.name}/{key}_{rank:03d}": v[key_id].item() for rank, v in enumerate(local_section_data)}
                    )

                data_tensor = torch.tensor(gpu_relative_perf_score_gather_list)
                slowest_rank_id = torch.argmin(data_tensor)
                wandb_info.update(
                    {
                        f"slowest_rank/{self.name}_rank": slowest_rank_id.item(),
                        f"slowest_rank/{self.name}_relative_perf": torch.min(data_tensor).item(),
                    }
                )

                for key_id, key in enumerate(report.local_section_summaries):
                    data_tensor = torch.tensor([v[key_id] for v in local_section_data])
                    wandb_info.update(
                        {
                            f"slowest_rank/slowest_{key}_rank": torch.argmax(data_tensor).item(),
                            f"slowest_rank/slowest_{key}_time": torch.max(data_tensor).item(),
                        }
                    )

                import wandb

                if wandb.run:
                    wandb.log(wandb_info, step=iteration)

                import cosmos3._src.imaginaire.utils.launch

                if cosmos3._src.imaginaire.utils.launch.S3_READY and (iteration % (5 * self.report_freq) == 0) and self.save_s3:
                    easy_io.dump(
                        wandb_info,
                        f"s3://rundir/{self.__class__.__name__}/iter_{iteration:09d}.yaml",
                    )
                    easy_io.dump(
                        report,
                        f"s3://rundir/{self.__class__.__name__}/report_iter_{iteration:09d}.pkl",
                    )

                # Which GPUs are slower than other GPUs, based on the execution time of kernels
                relative_stragglers = stragglers["straggler_gpus_relative"]
                # Which GPUs are slower than itself in the past, based on the past execution time of kernels.
                individual_stragglers = stragglers["straggler_gpus_individual"]
                is_slowdown = relative_stragglers or individual_stragglers
                if is_slowdown:
                    hostname = torch.ByteTensor(bytearray(os.uname().nodename, "utf-8")).cuda()
                    whole_hostname = all_gather_tensor(hostname)
                    slowest_hostname = whole_hostname[slowest_rank_id].cpu().numpy().tobytes().decode("utf-8")
                    logging.critical(f"Slowest rank hostname: {slowest_hostname}")

                if self.slowdown_count(is_slowdown) and self.raise_error:
                    raise RuntimeError(
                        f"Detected GPU {slowest_rank_id} to be too slow compared to other GPUs."
                        f" The relative performance of {slowest_rank_id} rank was {report.gpu_relative_perf_scores[slowest_rank_id]}. Terminating the training."
                    )
