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

"""
Distributed checkpoint (DCP) directory structure and storage backends.

The checkpointer saves model state in a sharded format across multiple processes:

self.save_dirname/
├── iter_000000005/                    # Checkpoint at iteration 5
│   ├── model/                         # Model state shards
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   ├── optim/                        # Optimizer state shards
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   ├── scheduler/                    # Learning rate scheduler state
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   └── trainer/                      # Additional training state
│       ├── __0_0.distcp              # Shard 0 from rank 0
│       └── __1_0.distcp              # Shard 1 from rank 1
│   └── dataloader/                   # Optional per-rank dataloader state
│       ├── rank_0.pkl
│       └── rank_1.pkl
└── latest_checkpoint.txt             # Points to most recent checkpoint folder, e.g. iter_000000005

Storage backends:
- Local filesystem:
  self.save_dirname = "{config_job.path_local}/checkpoints"

- S3 object store:
  self.save_dirname = "s3://{bucket}/{config_job.path}/checkpoints"
  where bucket = self.config_checkpoint.save_to_object_store.bucket

The sharded format enables efficient distributed saving/loading by:
1. Parallelizing I/O across processes
2. Reducing memory usage per process
3. Supporting both local and cloud storage backends
"""

import enum
import multiprocessing
import os
import re
import time
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.metadata import (
    STATE_DICT_TYPE,
    Metadata,
    StorageMeta,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from cosmos3._src.imaginaire.checkpointer.base import AbstractCheckpointer
from cosmos3._src.imaginaire.checkpointer.s3_filesystem import S3StorageReader, S3StorageWriter
from cosmos3._src.imaginaire.config import CheckpointConfig, JobConfig
from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.utils import callback, distributed, log, misc
from cosmos3._src.imaginaire.utils.easy_io import easy_io
from cosmos3._src.vfm.utils.rand_state import get_rand_state_dict, set_rand_state_dict


class ModelWrapper(Stateful):
    """
    Wrapper for model state dict handling. Strips away the _orig_mod. prefix
    among other things from the state dict keys.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        return get_model_state_dict(self.model)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=True),
        )


@runtime_checkable
class _DataloaderStateHandler(Protocol):
    """Structural contract for callbacks that participate in dataloader-state checkpointing."""

    checkpoint_component: str

    def has_checkpoint_state(self) -> bool: ...
    def state_dict(self) -> dict[Any, Any]: ...
    def load_state_dict(self, state_dict: dict[Any, Any]) -> None: ...


class _DataloaderWrapper:
    """Adapter that surfaces a dataloader-state callback's checkpoint API.

    Walks the registered callbacks at construction time and binds to the
    first callback that:

    1. Declares ``checkpoint_component == "dataloader"``, AND
    2. Returns ``True`` from ``has_checkpoint_state()``.

    The bound callback's ``state_dict`` / ``load_state_dict`` methods are
    re-exposed via :meth:`state_dict` / :meth:`load_state_dict`.  Callers
    must gate those on :meth:`has_state` — invoking them when nothing was
    bound raises :class:`RuntimeError`.

    Note: only the first callback tagged ``checkpoint_component=="dataloader"``
    is considered; if it does not currently want its state checkpointed,
    no further callbacks are searched.  In practice there is at most one
    such callback (see ``DataLoaderStateCallback``).
    """

    def __init__(self, callbacks: callback.CallBackGroup | None) -> None:
        self._callback: _DataloaderStateHandler | None = None
        if callbacks is None:
            return
        for current_callback in callbacks._callbacks:
            if getattr(current_callback, "checkpoint_component", None) != "dataloader":
                continue
            if current_callback.has_checkpoint_state():
                self._callback = current_callback
            return

    def has_state(self) -> bool:
        return self._callback is not None

    def state_dict(self) -> dict[Any, Any]:
        if self._callback is None:
            raise RuntimeError("No dataloader state handler is registered, cannot save dataloader state.")
        return self._callback.state_dict()

    def load_state_dict(self, state_dict: dict[Any, Any]) -> None:
        if self._callback is None:
            raise RuntimeError("No dataloader state handler is registered, cannot load dataloader state.")
        self._callback.load_state_dict(state_dict)


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class Terminate:
    pass


class SaveDone:
    def __init__(self, iteration: int, elapsed_time: float, succeeded: bool):
        self.iteration = iteration
        self.elapsed_time = elapsed_time
        self.succeeded = succeeded

    def __str__(self):
        return f"SaveDone(iteration={self.iteration}, elapsed_time={self.elapsed_time}, succeeded={self.succeeded})"


def save_checkpoint_in_background(
    receiver_queue: multiprocessing.Queue,
    sender_queue: multiprocessing.Queue,
    config_checkpoint: CheckpointConfig,
    config_job: JobConfig,
) -> None:
    """
    Handles model checkpoint saving in a separate background process using PyTorch's distributed functionality.
    This function runs in a dedicated process to avoid blocking the main training loop.

    Args:
        receiver_queue: Queue to receive state dictionaries and commands from the main process
        sender_queue: Queue to send completion signals back to the main process
        config_checkpoint: Configuration settings for checkpoint saving behavior
        config_job: Configuration settings for the training job

    Flow:
        1. Initializes distributed processing environment
        2. Continuously waits for state dictionaries to save
        3. Saves checkpoints asynchronously
        4. Signals completion back to main process
        5. Terminates when receiving a Terminate signal

    Raises:
        AssertionError: If received object is neither Terminate signal nor valid state dict tuple

    Note:
        - Uses a different port than the main process to avoid conflicts
        - Disables TorchElastic agent store for checkpoint operations
        - Automatically cleans up distributed process group on exit
    """
    # Configure distributed environment
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"

    # Set up GPU device and distributed processing
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(backend="gloo")

    # Initialize checkpointing mechanism
    checkpoint_handler = DistributedCheckpointer(
        config_checkpoint=config_checkpoint,
        config_job=config_job,
        callbacks=None,
        disable_async=True,
    )

    while True:
        log.info(f"Checkpoint background process is ready for next task, waiting for new state_dict")
        received_data = receiver_queue.get()
        log.info(f"Checkpoint background process received new state_dict")

        if isinstance(received_data, Terminate):
            log.info(f"Checkpoint background process received termination signal, closing sender queue")
            break

        assert isinstance(received_data, tuple), "Received data must be a tuple of (state_dict, checkpoint_path)"
        state_dict, checkpoint_path = received_data

        # Save checkpoint and measure time taken.
        start_time = time.monotonic()
        iteration = state_dict["trainer"][0]["iteration"]
        succeeded = False

        try:
            log.info(f"Saving checkpoint to {checkpoint_path}")
            checkpoint_handler.save_state_dict_worker(state_dict, checkpoint_path)
            succeeded = True
        except Exception as e:
            log.error(f"Error saving checkpoint to {checkpoint_path}: {e}")
            # continue because if the thread exits, the main thread keeps on adding to the queue
        finally:
            elapsed_time = time.monotonic() - start_time
            log.info(
                f"Checkpoint save completed in background process. "
                f"Time taken: {elapsed_time:.2f} seconds, iteration: {iteration}, "
                f"status: {'SUCCESS' if succeeded else 'FAILURE'}"
            )
            sender_queue.put(SaveDone(iteration, elapsed_time, succeeded))

    log.info("Cleaning up: destroying distributed process group")
    dist.destroy_process_group()


def _replace_keys_with_ema_keys(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    """
    Renames model parameters from "net." to "net_ema.".
    """
    if not all(k.startswith("net.") for k in state_dict.keys()):
        raise ValueError("State dict must start with net. keys when load_ema_to_reg is True")
    return {k.replace("net.", "net_ema."): v for k, v in state_dict.items()}


class CustomLoadPlanner(dcp.DefaultLoadPlanner):
    """
    CustomLoadPlanner that supports ignoring keys during checkpoint load.
    This is useful when the checkpoint is saved with a different component
    architecture, e.g. different RoPE embeddings than the current model.
    """

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        allow_partial_load: bool = False,
        keys_to_skip_loading: List[str] = [],
        load_ema_to_reg: bool = False,
    ) -> None:
        super().__init__(
            flatten_state_dict=flatten_state_dict,
            flatten_sharded_tensors=flatten_sharded_tensors,
            allow_partial_load=allow_partial_load,
        )
        self.keys_to_skip_loading = keys_to_skip_loading
        self.load_ema_to_reg = load_ema_to_reg
        if len(keys_to_skip_loading) > 0:
            log.info(f"Skipping loading of keys that match the following patterns: {keys_to_skip_loading}")

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        state_dict = self._skip_keys_if_found(state_dict)

        if self.load_ema_to_reg:
            state_dict = _replace_keys_with_ema_keys(state_dict)

        super().set_up_planner(
            state_dict=state_dict,
            metadata=metadata,
            is_coordinator=is_coordinator,
        )

    def _skip_keys_if_found(
        self,
        state_dict: STATE_DICT_TYPE,
    ) -> Dict[str, Any]:
        """
        While loading the checkpoint, skip the weight loading for the keys
        that contain any element of `self.keys_to_skip_loading` as a substring.
        """
        if len(self.keys_to_skip_loading) == 0:
            return state_dict

        new_state_dict = {}
        for fqn, obj in state_dict.items():
            if any(skip_key in fqn for skip_key in self.keys_to_skip_loading):
                log.warning(f"Skipping loading of key: {fqn}")
                continue
            new_state_dict[fqn] = obj
        return new_state_dict


class CustomSavePlanner(dcp.DefaultSavePlanner):
    """
    Custom save planner that enables an override for cache_plans_key when
    caching of save plans is enabled. Caching of save plans reduces checkpointing
    time by reusing the same save plan across checkpoints. This reduces the
    checkpointing time by ~60% (benchmarked using the 235B-A22B Qwen3-VL model
    on 64 GB200 nodes).
    """

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        dedup_save_to_lowest_rank: bool = False,
        save_reg_to_ema: bool = False,
        enable_plan_caching: bool = False,
        cache_plans_key: str | None = None,
    ) -> None:
        super().__init__(
            flatten_state_dict=flatten_state_dict,
            flatten_sharded_tensors=flatten_sharded_tensors,
            dedup_save_to_lowest_rank=dedup_save_to_lowest_rank,
            enable_plan_caching=enable_plan_caching,
        )
        if cache_plans_key is not None:
            self._cached_plans_key = cache_plans_key

        self.save_reg_to_ema = save_reg_to_ema

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        storage_meta: StorageMeta | None = None,
        is_coordinator: bool = False,
    ) -> None:
        if self.save_reg_to_ema:
            state_dict = _replace_keys_with_ema_keys(state_dict)

        super().set_up_planner(
            state_dict=state_dict,
            storage_meta=storage_meta,
            is_coordinator=is_coordinator,
        )


class DistributedCheckpointer(AbstractCheckpointer):
    CHECKPOINT_KEYS = ["model", "optim", "scheduler", "trainer", "dataloader"]

    def __init__(
        self,
        config_checkpoint: CheckpointConfig,
        config_job: JobConfig,
        callbacks: Optional[callback.CallBackGroup] = None,
        disable_async: bool = False,
    ):
        super().__init__(config_checkpoint, config_job, callbacks)
        self.config_checkpoint = config_checkpoint
        if config_checkpoint.dcp_async_mode_enabled and not disable_async:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            self.async_mode = AsyncMode.DISABLED

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ctx = get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=save_checkpoint_in_background,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                    config_checkpoint,
                    config_job,
                ),
                daemon=True,
            )
            self.mp.start()
            self.cpu_offload_state_dict = None
            self.staging_ckpt_file = None
            self.staging_stream = torch.cuda.Stream()
            self.checkpoint_in_progress = False

    def keys_to_resume_during_load(self) -> tuple[set[str], str | None, bool | None]:
        """
        Determines the keys to resume from the checkpoint and the checkpoint path.
        If the checkpoint is the latest checkpoint of the same model, then it is a
        normal resume. If the checkpoint is a different model's checkpoint, then it is
        a warm start.

        Args:
            None

        Returns:
            resume_keys: The keys to resume from the checkpoint.
            checkpoint_path: The path to the checkpoint. If the checkpoint is a different
            warm_start: Whether to warm start the training from a different model's checkpoint.
                If the checkpoint is a different model's checkpoint, then this is True.
                If the checkpoint is the latest checkpoint of the same model, then this is False.
        """
        latest_checkpoint_file = self._read_latest_checkpoint_file()

        resume_keys = []
        warm_start = None

        if latest_checkpoint_file is not None:
            # 1. Resume training from the latest checkpoint of the same model.
            warm_start = False
            checkpoint_path = os.path.join(self.load_dirname, latest_checkpoint_file)
            resume_keys.extend(self.CHECKPOINT_KEYS)

        else:
            if self.load_path and not str(self.load_path).endswith(".pt"):
                # 2. Warm Start: Resume training from a different model's checkpoint
                # specified by `load_path`.
                warm_start = True
                checkpoint_path = self.load_path

                if self.load_s3_backend_key:
                    checkpoint_path = f"s3://{self.config_checkpoint.load_from_object_store.bucket}/{checkpoint_path}"

                    # If the path doesn't end with specific checkpoint, read the latest
                    # checkpoint file to determine the most recent checkpoint iteration.
                    if not re.search(r"/checkpoints/iter_\d{9}/?$", checkpoint_path):
                        old_ckpt_path = checkpoint_path
                        latest_ckpt_path = os.path.join(checkpoint_path, "checkpoints/latest_checkpoint.txt")

                        # If the latest checkpoint file exists, use it to determine the
                        # checkpoint path. Otherwise, use the original path.
                        if easy_io.exists(latest_ckpt_path, backend_key=self.load_s3_backend_key):
                            checkpoint_file = easy_io.load(
                                latest_ckpt_path, backend_key=self.load_s3_backend_key
                            ).strip()
                            checkpoint_path = f"{checkpoint_path}/checkpoints/{checkpoint_file}"
                        else:
                            log.warning(
                                f"Latest checkpoint file {latest_ckpt_path} not found, load from {old_ckpt_path}"
                            )
                            checkpoint_path = old_ckpt_path

                if self.load_training_state:
                    resume_keys.extend(self.CHECKPOINT_KEYS)
                else:
                    resume_keys.append("model")
                    if self.only_load_scheduler_state:
                        resume_keys.append("scheduler")
            else:
                checkpoint_path = None

        if len(self.keys_not_to_resume) > 0:
            for key in self.keys_not_to_resume:
                assert key in self.CHECKPOINT_KEYS, f"Invalid key to resume: {key} not in {self.CHECKPOINT_KEYS}"
            resume_keys = [key for key in resume_keys if key not in self.keys_not_to_resume]

        return set(resume_keys), checkpoint_path, warm_start

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_start(model)

        resume_keys, checkpoint_path, warm_start = self.keys_to_resume_during_load()
        resume_keys = sorted(resume_keys)
        log.critical(f"Resuming ckpt {checkpoint_path} with keys: {resume_keys}")

        iteration = 0

        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)

            for key in resume_keys:
                dist.barrier()

                cur_key_ckpt_full_path = os.path.join(checkpoint_path, key)
                log.critical(f"Start loading checkpoint from {cur_key_ckpt_full_path}")

                storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                strict_resume = self.config_checkpoint.strict_resume

                # Note that we only allow skipping loading of keys during warm start. If the checkpoint is
                # the latest checkpoint of the same model, then we don't need to skip any keys.
                keys_to_skip_loading = self.config_checkpoint.keys_to_skip_loading if warm_start else []

                load_planner = CustomLoadPlanner(
                    allow_partial_load=not strict_resume,
                    keys_to_skip_loading=keys_to_skip_loading,
                )

                if key == "model":
                    log.info("- Loading the model...")
                    _model_wrapper = ModelWrapper(model)
                    _state_dict = _model_wrapper.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    if self.config_checkpoint.load_ema_to_reg:
                        # The model has both net.* and net_ema.* submodules, so _state_dict
                        # contains both sets of keys after dcp.load(). Copy EMA weights into
                        # regular model weights so we can resume from EMA and reset EMA.
                        for sd_key in list(_state_dict.keys()):
                            if sd_key.startswith("net."):
                                key_ema = "net_ema." + sd_key.removeprefix("net.")
                                assert key_ema in _state_dict, (
                                    f"EMA key {key_ema} not found in state_dict. "
                                    "Ensure the model has net_ema submodule."
                                )
                                _state_dict[sd_key] = _state_dict[key_ema]
                    results = _model_wrapper.load_state_dict(_state_dict)
                    if results is not None:
                        if len(results.missing_keys) > 0:
                            raise ValueError(f"Missing keys (not found in checkpoint): {results.missing_keys}")
                        if len(results.unexpected_keys) > 0:
                            raise ValueError(
                                f"Unexpected keys (found in checkpoint but not in model): {results.unexpected_keys}"
                            )

                elif key == "optim":
                    log.info("- Loading the optimizer...")
                    _state_dict = optimizer.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    optimizer.load_state_dict(_state_dict)

                elif key == "scheduler":
                    log.info("- Loading the scheduler...")
                    _state_dict = scheduler.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    scheduler.load_state_dict(_state_dict)

                elif key == "trainer":
                    log.info("- Loading the trainer...")

                    # Use rank-specific key for RNG state to support correct per-rank restoration
                    rng_key = f"rng_state_{dist.get_rank()}"
                    current_rng_state = get_rand_state_dict()
                    _state_dict = {
                        "grad_scaler": grad_scaler.state_dict(),
                        "iteration": iteration,
                    }
                    # Check if rng_key exists in checkpoint metadata to avoid failure with strict_resume=True
                    metadata = storage_reader.read_metadata()
                    rng_key_exists = any(
                        k.startswith(f"{rng_key}.") or k == rng_key for k in metadata.state_dict_metadata.keys()
                    )
                    if rng_key_exists:
                        _state_dict[rng_key] = current_rng_state

                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    grad_scaler.load_state_dict(_state_dict["grad_scaler"])
                    iteration = _state_dict["iteration"]
                    set_rand_state_dict(_state_dict.get(rng_key, current_rng_state))

                elif key == "dataloader":
                    if not easy_io.exists(cur_key_ckpt_full_path, backend_key=self.load_s3_backend_key):
                        log.info(
                            f"Checkpoint {cur_key_ckpt_full_path} does not exist, skip loading dataloader.",
                            rank0_only=False,
                        )
                        continue

                    rank = dist.get_rank()
                    dataloader_pkl_path = os.path.join(cur_key_ckpt_full_path, f"rank_{rank}.pkl")
                    if not easy_io.exists(dataloader_pkl_path, backend_key=self.load_s3_backend_key):
                        log.info(f"No dataloader checkpoint found at {dataloader_pkl_path}", rank0_only=False)
                        continue

                    log.info(f"- Loading the dataloader {cur_key_ckpt_full_path}...", rank0_only=False)
                    _state_dict = easy_io.load(
                        dataloader_pkl_path,
                        file_format="pkl",
                        backend_key=self.load_s3_backend_key,
                    )
                    dataloader_wrapper = _DataloaderWrapper(self.callbacks)
                    if dataloader_wrapper.has_state():
                        dataloader_wrapper.load_state_dict(_state_dict)

                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")

            if self.callbacks is not None and resume_keys:
                # Note that this callback is never used in the codebase.
                self.callbacks.on_load_checkpoint(model, state_dict={})
            log.info(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")

        else:
            log.info("Training from scratch.")

        torch.cuda.empty_cache()

        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)
        return iteration

    def _checkpoint_async_with_pinned_memory(
        self, checkpoint_file: str, state_dict: Dict[str, Tuple[Any, str]]
    ) -> None:
        assert self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM, "Async mode must be AsyncMode.ASYNC_WITH_PINNED_MEM"

        from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict

        if self.cpu_offload_state_dict is None:
            log.info(f"Preparing the CPU memory for staging")
            self.cpu_offload_state_dict = _create_cpu_state_dict(state_dict, pin_memory=True, share_memory=True)

        log.info(f"Staging the state_dict in CPU memory")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging_ckpt_file = checkpoint_file

        self.staging_stream.synchronize()
        log.info(f"Staging the state_dict in CPU memory completed")

        self.mp_queue_send.put_nowait((self.cpu_offload_state_dict, self.staging_ckpt_file))
        self.checkpoint_in_progress = True
        log.info(f"Submitted checkpoint to background process")

    def _wait_for_previous_async_checkpoint(self) -> None:
        """
        Gets the results of previously submitted checkpoints.
        Pass them to callbacks if checkpoint succeeded.
        """
        assert self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM, "Async mode must be AsyncMode.ASYNC_WITH_PINNED_MEM"

        if not self.checkpoint_in_progress:
            return

        success = False
        try:
            log.info(f"Waiting for checkpoint save result")

            # Note that we set a timeout of 1 hour to avoid blocking the main process
            # indefinitely. Gloo and NCCL timeouts are ~30 minutes, so this timeout
            # should typically be sufficient.
            save_done: SaveDone = self.mp_queue_recv.get(timeout=3600)

            log.info(f"Received checkpoint save result: {save_done}")

            if self.callbacks is not None and save_done.succeeded:
                self.callbacks.on_save_checkpoint_success(
                    iteration=save_done.iteration, elapsed_time=save_done.elapsed_time
                )
            self.checkpoint_in_progress = False
            success = save_done.succeeded

        except Exception as e:
            log.error(f"Error waiting for checkpoint save result: {e}")

        if not success:
            # Terminate training execution upon a failed checkpoint save attempt.
            # A failure at this stage typically indicates a non-recoverable system error.
            # Continuing execution would result in subsequent persistent failures and
            # unnecessary waste of GPU resources.
            raise RuntimeError("Previous checkpoint save failed. Exiting...")

    def get_storage_writer(self, checkpoint_path: str) -> Union[S3StorageWriter, FileSystemWriter]:
        if self.save_to_object_store:
            return S3StorageWriter(
                credential_path=self.config_checkpoint.save_to_object_store.credentials,
                path=checkpoint_path,
                enable_gcs_patch_in_boto3=self.config_checkpoint.enable_gcs_patch_in_boto3,
            )
        return FileSystemWriter(path=checkpoint_path)

    def get_storage_reader(self, checkpoint_path: str) -> Union[S3StorageReader, FileSystemReader]:
        if self.load_from_object_store:
            return S3StorageReader(
                credential_path=self.config_checkpoint.load_from_object_store.credentials,
                path=checkpoint_path,
                enable_gcs_patch_in_boto3=self.config_checkpoint.enable_gcs_patch_in_boto3,
            )
        return FileSystemReader(checkpoint_path)

    def _save_as_pkl(self, obj: Any, output_dir: str) -> None:
        """Save per-rank Python checkpoint state such as no-replace dataloader progress."""
        rank = dist.get_rank()
        path = os.path.join(output_dir, f"rank_{rank}.pkl")
        easy_io.dump(
            obj,
            path,
            file_format="pkl",
            backend_key=self.save_s3_backend_key,
        )
        log.info(f"Saved state to {path}")

    def save_state_dict_worker(self, to_save_dict: Dict[str, Tuple[Any, str]], checkpoint_file: str) -> None:
        for key, (v, full_checkpoint_path) in to_save_dict.items():
            if key == "dataloader":
                self._save_as_pkl(v, full_checkpoint_path)
            else:
                storage_writer = self.get_storage_writer(full_checkpoint_path)
                # Note that it is ok to create a new CustomSavePlanner object
                # for each checkpoint save since the save plans are cached in a
                # class dictionary.
                save_planner = CustomSavePlanner(
                    dedup_save_to_lowest_rank=True,
                    enable_plan_caching=True,
                    cache_plans_key=f"custom_planner_{key}",
                )
                dcp.save(
                    v,
                    storage_writer=storage_writer,
                    planner=save_planner,
                )

        if distributed.is_rank0():
            log.info(f"Saving last checkpoint file {checkpoint_file}")
            self._write_latest_checkpoint_file(checkpoint_file)

        log.info(f"Saved checkpoint to {os.path.join(self.save_dirname, checkpoint_file)}")

    def save(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._wait_for_previous_async_checkpoint()

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_start(model, iteration)

        checkpoint_file = f"iter_{iteration:09}"

        # Use rank-specific key for RNG state to ensure each rank saves its own state
        rng_key = f"rng_state_{dist.get_rank()}"

        to_save_dict = {
            "model": ModelWrapper(model).state_dict(),
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "trainer": {
                "grad_scaler": grad_scaler.state_dict(),
                "iteration": iteration,
                rng_key: get_rand_state_dict(),
            },
        }
        dataloader_wrapper = _DataloaderWrapper(self.callbacks)
        if dataloader_wrapper.has_state():
            to_save_dict["dataloader"] = dataloader_wrapper.state_dict()

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint(model, state_dict=to_save_dict)

        for k in to_save_dict.keys():
            output_dirname = os.path.join(self.save_dirname, f"iter_{iteration:09}/{k}")
            to_save_dict[k] = (to_save_dict[k], output_dirname)

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            dataloader_entry = to_save_dict.pop("dataloader", None)
            if dataloader_entry is not None:
                dataloader_state, dataloader_save_dir = dataloader_entry
                self._save_as_pkl(dataloader_state, dataloader_save_dir)
            self._checkpoint_async_with_pinned_memory(checkpoint_file, to_save_dict)
        else:
            start_time = time.monotonic()
            self.save_state_dict_worker(to_save_dict, checkpoint_file)
            elapsed_time = time.monotonic() - start_time
            log.info(f"Checkpoint save completed: Time taken: {elapsed_time:.2f} seconds")

            if self.callbacks is not None:
                self.callbacks.on_save_checkpoint_success(iteration=iteration, elapsed_time=elapsed_time)

        # This measures exposed (synchronous) checkpoint time, on_save_checkpoint_success()
        # is instead called to measure the entire duration for asynchronous checkpoint for the async case too.
        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    def finalize(self) -> None:
        super().finalize()
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.mp and self.mp.is_alive():
                # Wait for the previous checkpoint to complete.
                self._wait_for_previous_async_checkpoint()

                self.mp_queue_send.put(Terminate())
                self.mp.join()
