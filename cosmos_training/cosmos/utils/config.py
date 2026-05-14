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

"""Training config system for Imaginare4"""

from __future__ import annotations

import importlib
import os
import time
from typing import Any, Dict, Optional, Type, TypeVar, Union

import attrs
import torch
from loguru import logger as logging

from cosmos.utils.flags import TRAINING

try:
    from megatron.core import ModelParallelConfig

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False

from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.lazy_config import LazyDict
from cosmos.utils import distributed
from cosmos.utils.misc import Color

T = TypeVar("T")


def _is_attrs_instance(obj: object) -> bool:
    """
    Helper function to check if an object is an instance of an attrs-defined class.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is an instance of an attrs-defined class, False otherwise.
    """
    return hasattr(obj, "__attrs_attrs__")


def make_freezable(cls: T) -> T:
    """
    A decorator that adds the capability to freeze instances of an attrs-defined class.

    NOTE: This requires the wrapped attrs to be defined with attrs.define(slots=False) because we need
    to hack on a "_is_frozen" attribute.

    This decorator enhances an attrs-defined class with the ability to be "frozen" at runtime.
    Once an instance is frozen, its attributes cannot be changed. It also recursively freezes
    any attrs-defined objects that are attributes of the class.

    Usage:
        @make_freezable
        @attrs.define(slots=False)
        class MyClass:
            attribute1: int
            attribute2: str

        obj = MyClass(1, 'a')
        obj.freeze()  # Freeze the instance
        obj.attribute1 = 2  # Raises AttributeError

    Args:
        cls: The class to be decorated.

    Returns:
        The decorated class with added freezing capability.
    """

    if not hasattr(cls, "__dict__"):
        raise TypeError(
            "make_freezable cannot be used with classes that do not define __dict__. Make sure that the wrapped "
            "class was defined with `@attrs.define(slots=False)`"
        )

    original_setattr = cls.__setattr__

    def setattr_override(self, key, value) -> None:  # noqa: ANN001
        """
        Override __setattr__ to allow modifications during initialization
        and prevent modifications once the instance is frozen.
        """
        if hasattr(self, "_is_frozen") and self._is_frozen and key != "_is_frozen":
            raise AttributeError("Cannot modify frozen instance")
        original_setattr(self, key, value)  # type: ignore

    cls.__setattr__ = setattr_override  # type: ignore

    def freeze(self: object) -> None:
        """
        Freeze the instance and all its attrs-defined attributes.
        """
        for _, value in attrs.asdict(self, recurse=False).items():
            if _is_attrs_instance(value) and hasattr(value, "freeze"):
                value.freeze()
        self._is_frozen = True  # type: ignore

    cls.freeze = freeze  # type: ignore

    return cls


def _pretty_print_attrs_instance(obj: object, indent: int = 0, use_color: bool = False) -> str:
    """
    Recursively pretty prints attrs objects with color.
    """

    assert attrs.has(obj.__class__)

    lines: list[str] = []
    for attribute in attrs.fields(obj.__class__):
        value = getattr(obj, attribute.name)
        if attrs.has(value.__class__):
            if use_color:
                lines.append("   " * indent + Color.cyan("* ") + Color.green(attribute.name) + ":")
            else:
                lines.append("   " * indent + "* " + attribute.name + ":")
            lines.append(_pretty_print_attrs_instance(value, indent + 1, use_color))
        else:
            if use_color:
                lines.append(
                    "   " * indent + Color.cyan("* ") + Color.green(attribute.name) + ": " + Color.yellow(value)
                )
            else:
                lines.append("   " * indent + "* " + attribute.name + ": " + str(value))
    return "\n".join(lines)


def pretty_print_overrides(overrides: Optional[list[str]] = None, use_color: bool = False) -> str:
    """
    Pretty prints overrides.
    """

    lines: list[str] = []
    lines.append(Color.cyan("* ") + Color.green("overrides") + ": ")
    for override in overrides:
        if override == "--":
            continue
        if override.startswith("~"):
            attribute_name = override[1:]
            attribute_value = None
        else:
            attribute_name, attribute_value = override.split("=")
        if use_color:
            lines.append("   " + Color.cyan("* ") + Color.green(attribute_name) + ": " + Color.yellow(attribute_value))
        else:
            lines.append("   " + "* " + attribute_name + ": " + str(attribute_value))

    return "\n".join(lines)


@make_freezable
@attrs.define(slots=False)  # slots=False is required for make_freezable. See the make_freezable notes for more info.
class ObjectStoreConfig:
    # Whether the file I/O is from object store instead of local disk.
    enabled: bool = False
    # Path to the object store credentials file.
    credentials: str = ""
    # Object store bucket to read from / write to the objects.
    bucket: str = ""


@make_freezable
@attrs.define(slots=False)
class HFExportConfig:
    # Whether to enable HuggingFace safetensors export after each DCP checkpoint.
    enabled: bool = False
    # HuggingFace Hub repo ID to push exported weights to (e.g. "nvidia/cosmos3-qwen3-8b").
    # None means local/S3 only.
    hf_repo_id: Optional[str] = None
    # Object store for uploading exports. If not enabled, upload is skipped.
    # To reuse the DCP checkpoint bucket, copy checkpoint.save_to_object_store here.
    upload_to_object_store: ObjectStoreConfig = attrs.field(factory=ObjectStoreConfig)
    # Export every N DCP checkpoints. Must be >= 1 and ideally a multiple of checkpoint.save_iter.
    export_every_n: int = attrs.field(default=1, validator=attrs.validators.ge(1))


@make_freezable
@attrs.define(slots=False)
class JobConfig:
    # Project name.
    project: str = ""
    # Experiment name.
    group: str = ""
    # Run/job name.
    name: str = ""
    # W&B mode, can be "online", or "disabled".
    wandb_mode: str = "online"
    # Cluster configuration (optional, for cluster-specific settings).
    cluster: Optional[Any] = None

    @property
    def path(self) -> str:
        return f"{self.project}/{self.group}/{self.name}"

    @property
    def path_local(self) -> str:
        local_root = os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output")
        return f"{local_root}/{self.path}"


@make_freezable
@attrs.define(slots=False)
class EMAConfig:
    # Enable tracking a set of exponential moving average (EMA) weights.
    enabled: bool = False
    # EMA decay rate.
    beta: float = 0.9999
    # Enable removing "_orig_mod-" from buffer names that is added by torch.compile
    torch_compile_buffer_renaming: bool = False


@make_freezable
@attrs.define(slots=False)
class PowerEMAConfig:
    # Enable tracking a set of exponential moving average (EMA) weights.
    enabled: bool = False
    # EDM2 paper EMA decay rate.
    s: float = 0.1
    # Enable removing "_orig_mod-" from buffer names that is added by torch.compile
    torch_compile_buffer_renaming: bool = False


@make_freezable
@attrs.define(slots=False)
class DDPConfig:
    # Traverse the computation graph to find parameters that don't receive gradients.
    find_unused_parameters: bool = False
    # Set to True if the computation graph does not change during the whole training loop.
    static_graph: bool = True
    # Set to True if we want to synchronize buffers. Set to False if the sync is going to be handled elsewhere.
    broadcast_buffers: bool = True


@make_freezable
@attrs.define(slots=False)
class CuDNNConfig:
    # Set to True for better reproducibility of the results (only using deterministic cudnn functions).
    deterministic: bool = False
    # If set to True, cudnn will benchmark several algorithms and pick the fastest one.
    benchmark: bool = True


@make_freezable
@attrs.define(slots=False)
class JITConfig:
    # Enable exporting a JIT compiled model.
    enabled: bool = False
    # Input tensor shape, for example input.
    input_shape: Union[list[int], None] = None
    # Device to compile onto.
    device: str = "cuda"
    # # Data type to compile onto.
    dtype: str = "bfloat16"
    # Strict mode for PyTorch JIT.
    strict: bool = True


@make_freezable
@attrs.define(slots=False)
class CheckpointConfig:
    # possible checkpoint class
    type: Optional[Dict] = None

    # for dcp, whether to use async mode
    dcp_async_mode_enabled: bool = False

    # Configs for saving the checkpoints to object store.
    save_to_object_store: ObjectStoreConfig = attrs.field(factory=ObjectStoreConfig)

    # Save the checkpoint every N iterations.
    save_iter: int = 999999999

    # Load state_dict to the models in strict mode. If True, `allow_partial_load` in dcp
    # planner will be set to False. DCP will raise an error if there are missing keys.
    # If False, `allow_partial_load` in dcp planner will be set to True. DCP will not
    # raise an error if there are missing keys.
    strict_resume: bool = True

    # Configs for loading the checkpoints from object store.
    load_from_object_store: ObjectStoreConfig = attrs.field(factory=ObjectStoreConfig)

    # Path of model weights to resume the checkpoint from.
    load_path: str = ""

    # The following 3 flags (load_training_state, only_load_scheduler_state, keys_to_skip_loading)
    # only take effect when the checkpoints are loaded from `load_path`. If loading happens from
    # the previous checkpoint of the same model, these flags are ignored.

    # Whether to load the training states (optimizer/scheduler/grad-scaler) from the checkpoint path.
    load_training_state: bool = False

    # Whether to load the scheduler state only from the checkpoint path. If
    # load_training_state is True, this will be ignored.
    only_load_scheduler_state: bool = False

    # When loading checkpoints from `load_path`, this list serves as a filter
    # to bypass the loading for specific model parameters. A key is considered
    # a match—and thus its loading is skipped—if it contains any element of this
    # list as a substring. This mechanism allows for broad suppression of entire
    # modules or parameter groups without requiring the specification of fully
    # qualified names (FQNs). Skipping loading of keys is useful when the new model
    # has a different component architecture, e.g. different RoPE embeddings than
    # the current model.
    keys_to_skip_loading: list[str] = []

    # Configs for JIT compiling EMA model.
    jit: JITConfig = attrs.field(factory=JITConfig)

    # Print detailed information during checkpoint saving/loading.
    verbose: bool = True

    # Keys not to resume from the checkpoint, choices: ["model", "optim", "scheduler", "trainer", "dataloader"]
    keys_not_to_resume: list[str] = []

    # Whether to use the local filesystem for broadcasting checkpoint data (used for Tensor Parallel Checkpointer).
    broadcast_via_filesystem: bool = False
    load_ema_to_reg: bool = False

    # Enable GCS patch in boto3 for loading/saving checkpoints from/to GCS
    enable_gcs_patch_in_boto3: bool = False

    # Config for exporting HuggingFace-compatible safetensors after each DCP checkpoint.
    hf_export: HFExportConfig = attrs.field(factory=HFExportConfig)


@make_freezable
@attrs.define(slots=False)
class NVTXConfig:
    """Config for NVTX ranges used in the main training loop.

    See tutorials/nanogpt for more details on how to integrate profiling into your model."""

    # Enable the NVTX ranges.
    enabled: bool = False
    # Synchronize everything in each NVTX range.
    cuda_synchronize: bool = False


@make_freezable
@attrs.define(slots=False)
class StragglerDetectionConfig:
    """Config for Straggler detection tool: https://gitlab-master.nvidia.com/dl/gwe/fault_tolerance_related/straggler/-/tree/cupti?ref_type=heads"""

    # Enable the Straggler Detection.
    enabled: bool = False
    # How frequently should the Straggler reports be generated.
    report_freq: int = 100
    # How frequently iterations should be profiled
    profile_freq: int = 1
    # What is the maximum relative difference between GPUs after they are considered stragglers
    max_diff: float = 2.0
    # Should the error be raised when straggler is detected
    raise_error: bool = True
    # Analyze kernels in the forward pass.
    analyze_forward: bool = True
    # Analyze kernels in the backward pass.
    analyze_backward: bool = True
    # Analyze kernels in the optimizer.
    analyze_optimizer: bool = True
    # Analyze dataloading time.
    analyze_dataloading: bool = True
    # Whether to save logs to S3
    save_s3: bool = False


@make_freezable
@attrs.define(slots=False)
class Profiling:
    # Torch profiler: set this True to dump chrome traces.
    enable_profiling: bool = False
    # Nsight Systems: set this True AND launch under `nsys profile --capture-range=cudaProfilerApi`.
    enable_nsys: bool = False
    # CUDA memory snapshot: set this True to dump allocator snapshots.
    enable_memory_snapshot: bool = False
    save_s3: bool = False
    profile_freq: int = 1
    # Number of warmup iterations before the active profile iteration.
    profile_warmup: int = 3
    # Target ranks for profiling, each entry must be >=0 and < world_size.
    target_ranks: list[int] = list(range(8))
    # The options below apply only to the torch profiler (enable_profiling).
    # Set `record_shape` and `profile_memory` to False to reduce profile size.
    record_shape: bool = False
    profile_memory: bool = False
    with_stack: bool = True
    with_modules: bool = True


@make_freezable
@attrs.define(slots=False)
class CompileConfig:
    """
    torch.compile config options passed to set_torch_compile_options function.
    """

    recompile_limit: int = 8
    use_duck_shape: bool = True


@make_freezable
@attrs.define(slots=False)
class TrainerConfig:
    if TRAINING:
        from cosmos.trainer import ImaginaireTrainer
        from cosmos.utils import callback

        type: Type[ImaginaireTrainer] = ImaginaireTrainer
        # Set the callback class.
        # Defaults to the callbacks below.
        callbacks: LazyDict = LazyDict(
            dict(
                ema=L(callback.EMAModelCallback)(),
                progress_bar=L(callback.ProgressBarCallback)(),
                wandb=L(callback.WandBCallback)(),
            )
        )

    # distributed parallelism strategy
    distributed_parallelism: str = "ddp"
    # Distributed data parallel configs.
    ddp: DDPConfig = attrs.field(factory=DDPConfig)
    # cuDNN configs.
    cudnn: CuDNNConfig = attrs.field(factory=CuDNNConfig)
    # Set the random seed.
    seed: int = 0
    # Gradient scaler arguments (for torch.amp.GradScaler).
    grad_scaler_args: dict = attrs.field(factory=lambda: dict(enabled=False))
    # Maximum number of iterations to train the model.
    max_iter: int = 999999999
    # Maximum number of iterations to validate the model. If None, validate on the entire dataset.
    max_val_iter: int | None = None
    # How often we log the training stats.
    logging_iter: int = 100
    # Whether we want to run the validation routines.
    run_validation: bool = True
    # How often we evaluate on the validation set.
    validation_iter: int = 999999999
    # Whether to run the validation on the start of the training.
    run_validation_on_start: bool = False
    # Kill the process after N seconds since the last iteration (usually means dead job).
    timeout_period: int = 999999999
    # Tensor memory organization format.
    memory_format: torch.memory_format = torch.preserve_format
    # Gradient accumulation (update step every N iteration).
    grad_accum_iter: int = 1
    # Straggler Detection config
    straggler_detection: StragglerDetectionConfig = attrs.field(factory=StragglerDetectionConfig)
    # Profiling config
    profiling: Profiling = attrs.field(factory=Profiling)
    compile_config: CompileConfig = attrs.field(factory=CompileConfig)

    # Whether to save the checkpoint at iteration 0.
    save_zero_checkpoint: bool = False


@make_freezable
@attrs.define(slots=False)
class Config:
    """Config for an imaginaire4 job.

    See /README.md/Configuration System for more info.
    """

    # Model configs.
    model: LazyDict
    # Optimizer configs.
    optimizer: LazyDict
    # Scheduler configs.
    scheduler: LazyDict
    # Training data configs.
    dataloader_train: LazyDict | None
    # Validation data configs.
    dataloader_val: LazyDict | None

    # Training job configs.
    job: JobConfig = attrs.field(factory=JobConfig)

    # Trainer configs.
    trainer: TrainerConfig = attrs.field(factory=TrainerConfig)

    if USE_MEGATRON:
        # Megatron-Core configs
        model_parallel: ModelParallelConfig = attrs.field(factory=ModelParallelConfig)
    else:
        model_parallel: None = None

    # Checkpointer configs.
    checkpoint: CheckpointConfig = attrs.field(factory=CheckpointConfig)

    # enable upload reproducible setup to s3
    upload_reproducible_setup: bool = False

    def pretty_print(self, use_color: bool = False) -> str:
        return _pretty_print_attrs_instance(self, 0, use_color)

    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    def model_init_kwargs(self) -> dict[str, Any]:
        """Live root-level sub-configs to pass into instantiate(self.model).

        Override in subclasses whose model __init__ expects fully-composed
        top-level configs (e.g. policy/checkpoint/train) rather than the stale
        LazyCall snapshot stored under config.model.* at make_config() time.
        """
        return {}


    def validate(self) -> None:
        """Validate that the config has all required fields."""

        # broadcast job.name across all ranks to make sure it is consistent
        # otherwise, unaligned job names leads unaligned path to save checkpoints
        job_name_tensor = torch.ByteTensor(bytearray(self.job.name, "utf-8")).cuda()
        distributed.broadcast(job_name_tensor, 0)
        self.job.name = job_name_tensor.cpu().numpy().tobytes().decode("utf-8")


        assert self.job.project != ""
        assert self.job.group != ""
        assert self.job.name != ""


def load_config(config_path: str, opts: list[str], enable_one_logger: bool = False) -> Config:
    from cosmos.utils.serialization import from_yaml, load_callable

    t1 = time.monotonic_ns()
    if config_path.endswith(".yaml"):
        config = from_yaml(config_path)
        # for registration of dataloaders, etc.
        _ = load_callable(config.__module__).make_config()

        from cosmos.utils.config_helper import override

        config = override(config, opts, remove_defaults=True)
    else:
        config = _load_py_config(config_path, opts, validate=False)

    if enable_one_logger:
        try:
            # pyrefly: ignore  # missing-import
            from cosmos.utils.one_logger.one_logger_override_utils import override_one_logger_callback

            ol_t1 = time.monotonic_ns()
            config = override_one_logger_callback(config)
            ol_t2 = time.monotonic_ns()
            logging.debug(f"override_one_logger_callback: took {(ol_t2 - ol_t1) / 1e6:.2f}ms")
        except ImportError:
            pass

    t2 = time.monotonic_ns()
    logging.debug(f"total time to load config: {(t2 - t1) / 1e6:.2f}ms")
    return config


def _load_py_config(config_path: str, opts: list[str], validate: bool = True) -> Config:

    from cosmos.utils.config_helper import get_config_module, override

    t1 = time.monotonic_ns()
    config_module = get_config_module(config_path)
    t2 = time.monotonic_ns()
    logging.debug(f"get_config_module: took {(t2 - t1) / 1e6:.2f}ms")

    t1 = time.monotonic_ns()
    config = importlib.import_module(config_module).make_config()
    t2 = time.monotonic_ns()
    logging.debug(f"importlib.import_module: took {(t2 - t1) / 1e6:.2f}ms")

    t1 = time.monotonic_ns()
    config = override(config, opts)
    t2 = time.monotonic_ns()
    logging.debug(f"override: took {(t2 - t1) / 1e6:.2f}ms")

    if validate:
        t1 = time.monotonic_ns()
        config.validate()
        t2 = time.monotonic_ns()
        logging.debug(f"config.validate: took {(t2 - t1) / 1e6:.2f}ms")

    return config
