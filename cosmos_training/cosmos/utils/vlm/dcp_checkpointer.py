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
│   └── dataloader/                   # Per-rank dataloader state (e.g. no-replace sampler)
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
import time
from multiprocessing import get_context
from typing import Any, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner

from cosmos.checkpoint.base import AbstractCheckpointer
from cosmos.checkpoint.s3_filesystem import S3StorageReader, S3StorageWriter
from cosmos.utils.config import CheckpointConfig, JobConfig
from cosmos.utils import callback, distributed, log, misc
from cosmos.utils.easy_io import easy_io
from cosmos.utils.vlm.model_wrapper import ModelWrapper
from cosmos.utils.vlm.optimizer import OptimizersContainer
from cosmos.utils.vlm.planner import RenameLoadPlanner

# (qsh 2025-01-01) the design is from https://github.com/pytorch/torchtitan/blob/1060feacc1b51cb6b339a04e53a5243b8466552b/torchtitan/checkpoint.py
# we recreate wrapper when needed instead of creating one from the beginning.
# to people who find it difficult to digest the code, official tutorial for torch dcp may be helpful


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class Terminate:
    pass


class SaveDone:
    pass


def save_checkpoint(model_parts: list[nn.Module], ckpt_dir, checkpoint_cred="./credentials/s3_training.secret"):
    log.info(f"Saving (re-)sharded checkpoint to {ckpt_dir}")

    _model_wrapper = ModelWrapper(model_parts)
    state_dict = _model_wrapper.state_dict()

    if ckpt_dir.startswith("s3://"):
        storage_writer = S3StorageWriter(
            credential_path=checkpoint_cred,
            path=ckpt_dir,
        )
        dist.checkpoint.save(state_dict=state_dict, storage_writer=storage_writer)
    else:
        fs_storage_writer = dist.checkpoint.FileSystemWriter(ckpt_dir)
        dist.checkpoint.save(state_dict=state_dict, storage_writer=fs_storage_writer)

    log.info(f"Saved (re-)sharded checkpoint to {ckpt_dir}")


def load_checkpoint(model_parts: list[nn.Module], ckpt_dir, checkpoint_cred="./credentials/s3_training.secret"):
    log.info(f"Loading checkpoint from {ckpt_dir}.")

    _model_wrapper = ModelWrapper(model_parts)
    state_dict = _model_wrapper.state_dict()

    if ckpt_dir.startswith("s3://"):
        storage_reader = S3StorageReader(
            credential_path=checkpoint_cred,
            path=ckpt_dir,
        )
        dist.checkpoint.load(
            state_dict=state_dict,
            storage_reader=storage_reader,
            planner=RenameLoadPlanner(allow_partial_load=False),
        )
    else:
        fs_storage_reader = dist.checkpoint.FileSystemReader(ckpt_dir)
        dist.checkpoint.load(
            state_dict=state_dict,
            storage_reader=fs_storage_reader,
            planner=RenameLoadPlanner(allow_partial_load=False),
        )
    _model_wrapper.load_state_dict(state_dict)

    log.info(f"Finished loading checkpoint from {ckpt_dir}.")


def save_checkpoint_in_background(
    receiver_queue: multiprocessing.Queue,
    sender_queue: multiprocessing.Queue,
    checkpoint_config: CheckpointConfig,
    job_config: JobConfig,
) -> None:
    """
    Handles model checkpoint saving in a separate background process using PyTorch's distributed functionality.
    This function runs in a dedicated process to avoid blocking the main training loop.

    Args:
        receiver_queue: Queue to receive state dictionaries and commands from the main process
        sender_queue: Queue to send completion signals back to the main process
        checkpoint_config: Configuration settings for checkpoint saving behavior
        job_config: Configuration settings for the training job

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
    distributed.init()

    # Initialize checkpointing mechanism
    checkpoint_handler = DistributedCheckpointer(checkpoint_config, job_config, None, disable_async=True)

    try:
        while True:
            log.debug("Checkpoint background process is ready for next task")
            sender_queue.put(SaveDone())

            log.debug("Waiting to receive new state_dict")
            received_data = receiver_queue.get()
            log.debug("Received new state_dict")

            if isinstance(received_data, Terminate):
                log.info("Received termination signal for checkpoint background process")
                return

            assert isinstance(received_data, tuple), "Received data must be a tuple of (state_dict, checkpoint_path)"
            state_dict, checkpoint_path = received_data

            # Save checkpoint and measure time taken
            start_time = time.monotonic()
            checkpoint_handler.save_state_dict_worker(state_dict, checkpoint_path)

            elapsed_time = time.monotonic() - start_time
            log.info(f"Checkpoint saved successfully in background process. Time taken: {elapsed_time:.2f} seconds")

    finally:
        log.info("Cleaning up: destroying distributed process group")
        dist.destroy_process_group()


class DistributedCheckpointer(AbstractCheckpointer):
    KEYS_TO_SAVE = ["model", "optim", "scheduler", "trainer", "dataloader"]

    def __init__(
        self,
        config_checkpoint: CheckpointConfig,
        config_job: JobConfig,
        callbacks: callback.CallBackGroup,
        disable_async: bool = False,
    ):
        super().__init__(config_checkpoint, config_job, callbacks)
        self.config_checkpoint = config_checkpoint
        if config_checkpoint.dcp_async_mode_enabled:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            self.async_mode = AsyncMode.DISABLED

        if disable_async:
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
            self.staging = False
            self.staging_ckpt_file = None
            self.staging_stream = torch.cuda.Stream()

    def keys_to_resume_during_load(self) -> tuple[list[str], str | None]:
        latest_checkpoint_file = self._read_latest_checkpoint_file()

        resume_keys = []

        if latest_checkpoint_file is not None:
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_path = os.path.join(self.load_dirname, latest_checkpoint_file)
            resume_keys.extend(self.KEYS_TO_SAVE)
        else:
            if self.load_path:
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                if self.load_s3_backend_key:
                    checkpoint_path = f"s3://{self.config_checkpoint.load_from_object_store.bucket}/{checkpoint_path}"
                if self.load_training_state:
                    resume_keys.extend(self.KEYS_TO_SAVE)
                else:
                    resume_keys.append("model")
                    if self.only_load_scheduler_state:
                        resume_keys.append("scheduler")
            else:
                checkpoint_path = None
        if len(self.keys_not_to_resume) > 0:
            for key in self.keys_not_to_resume:
                assert key in self.KEYS_TO_SAVE, f"Invalid key to resume: {key} not in {self.KEYS_TO_SAVE}"
            resume_keys = [key for key in resume_keys if key not in self.keys_not_to_resume]

        # Ensure that resume_keys does not have duplicates.
        assert len(set(resume_keys)) == len(resume_keys)
        return resume_keys, checkpoint_path

    @misc.timer("checkpoint loading")
    def load(
        self,
        model_parts: list[nn.Module],
        optimizer: OptimizersContainer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        del grad_scaler

        self.callbacks.on_load_checkpoint_start(model_parts)

        resume_keys, checkpoint_path = self.keys_to_resume_during_load()
        log.info(f"Resume keys: {resume_keys}, checkpoint path: {checkpoint_path}", rank0_only=False)

        iteration = 0


        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            all_state_dicts = {}
            for key in resume_keys:
                cur_key_ckpt_full_path = os.path.join(checkpoint_path, key)
                storage_reader = self.get_stroage_reader(cur_key_ckpt_full_path)
                strict = self.config_checkpoint.strict_resume
                if key == "model":
                    log.info(f"- Loading the model {cur_key_ckpt_full_path}...", rank0_only=False)
                    _model_wrapper = ModelWrapper(model_parts)
                    _state_dict = _model_wrapper.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=RenameLoadPlanner(allow_partial_load=not strict),
                    )
                    log.info("dcp.load done", rank0_only=False)
                    _model_wrapper.load_state_dict(_state_dict)
                    log.info("model.load_state_dict done", rank0_only=False)
                elif key == "optim":
                    if not easy_io.exists(cur_key_ckpt_full_path, backend_key=self.load_s3_backend_key):
                        log.info(
                            f"Checkpoint {cur_key_ckpt_full_path} does not exist, skip loading optimizer.",
                            rank0_only=False,
                        )
                        continue

                    log.info(f"- Loading the optimizer {cur_key_ckpt_full_path}...", rank0_only=False)
                    _optim_wrapper = optimizer
                    _state_dict = _optim_wrapper.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=RenameLoadPlanner(allow_partial_load=not strict),
                    )
                    log.info("dcp.load done", rank0_only=False)
                    _optim_wrapper.load_state_dict(_state_dict)
                    log.info("optim.load_state_dict done", rank0_only=False)
                elif key == "scheduler":
                    log.info(f"- Loading the scheduler {cur_key_ckpt_full_path}...", rank0_only=False)
                    _state_dict = scheduler.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=RenameLoadPlanner(allow_partial_load=True),
                    )
                    scheduler.load_state_dict(_state_dict)
                    log.info("scheduler.load_state_dict done", rank0_only=False)
                elif key == "trainer":
                    log.info(f"- Loading the trainer {cur_key_ckpt_full_path}...", rank0_only=False)
                    # Here we skip loading the trainer, since 1) we only need the iteration which could be parsed from the name 2) we dont use grad_scaler
                    iteration = cur_key_ckpt_full_path.split("iter_")[-1].split("/")[0]
                    iteration = int(iteration)
                elif key == "dataloader":
                    rank = dist.get_rank()
                    dataloader_pkl_path = os.path.join(cur_key_ckpt_full_path, f"rank_{rank}.pkl")
                    if easy_io.exists(dataloader_pkl_path, backend_key=self.load_s3_backend_key):
                        log.info(f"- Loading the dataloader {cur_key_ckpt_full_path}...", rank0_only=False)
                        _state_dict = easy_io.load(
                            dataloader_pkl_path,
                            file_format="pkl",
                            backend_key=self.load_s3_backend_key,
                        )
                        all_state_dicts[key] = _state_dict
                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")

            self.callbacks.on_load_checkpoint(model_parts, state_dict=all_state_dicts)
            log.critical(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")
        else:
            log.info("Training from scratch.")
        torch.cuda.empty_cache()

        self.callbacks.on_load_checkpoint_end(model_parts, iteration=iteration, checkpoint_path=checkpoint_path)
        return iteration

    def _async_with_pinned_memory(self, checkpoint_file: str, state_dict: dict[str, tuple[Any, str]]) -> None:
        try:
            # pyrefly: ignore  # import-error
            from dist._state_dict_utils import _copy_state_dict, _create_cpu_state_dict
        except ImportError as e:
            raise ImportError(
                "Please install the latest PyTorch nightly to use async checkpointing with pinned memory."
            ) from e
        if self.cpu_offload_state_dict is None:
            log.debug(f"Preparing the CPU memory, {time.monotonic()=}.:.2f")
            self.cpu_offload_state_dict = _create_cpu_state_dict(state_dict, pin_memory=True, share_memory=True)

        log.debug(f"Staging the state_dict, {time.monotonic()=}.:.2f")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging = True
            self.staging_ckpt_file = checkpoint_file


        self.maybe_wait_for_staging()

    def maybe_wait_for_staging(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM and self.staging:
            if not self.staging_stream.query():
                self.staging_stream.synchronize()

            def sync_func():
                self.mp_queue_send.put_nowait((self.cpu_offload_state_dict, self.staging_ckpt_file))

            sync_func()
            self.staging = False

    def get_storage_writer(self, checkpoint_path: str) -> Union[S3StorageWriter, FileSystemWriter]:
        if self.save_to_object_store:
            return S3StorageWriter(
                credential_path=self.config_checkpoint.save_to_object_store.credentials,
                path=checkpoint_path,
            )
        return FileSystemWriter(path=checkpoint_path)

    def get_stroage_reader(self, checkpoint_path: str) -> Union[S3StorageReader, FileSystemReader]:
        if self.load_from_object_store:
            return S3StorageReader(
                credential_path=self.config_checkpoint.load_from_object_store.credentials,
                path=checkpoint_path,
            )
        return FileSystemReader(checkpoint_path)

    def _save_as_pkl(self, obj: Any, output_dir: str) -> None:
        """Save a picklable object as rank_{rank}.pkl under output_dir (e.g. for dataloader state)."""
        rank = dist.get_rank()
        path = os.path.join(output_dir, f"rank_{rank}.pkl")
        easy_io.dump(
            obj,
            path,
            file_format="pkl",
            backend_key=self.save_s3_backend_key,
        )
        log.info(f"Saved state to {path}")

    def save_state_dict_worker(
        self,
        to_save_dict: dict[str, tuple[Any, str]],
        checkpoint_file: str,
    ) -> None:
        for k, (v, full_checkpoint_path) in to_save_dict.items():
            if k == "dataloader":
                self._save_as_pkl(v, full_checkpoint_path)
            else:
                log.info(f"Saving {k} checkpoint to {full_checkpoint_path}")
                storage_writer = self.get_storage_writer(full_checkpoint_path)
                # Note that we always save replicated tensors to the lowest rank to
                # minimize the number of files each rank opens when loading the
                # checkpoint. For object stores like S3 without partial read
                # capability, this is the only way to ensure that we do not open
                # several files when loading the checkpoint.
                dcp.save(
                    v,
                    storage_writer=storage_writer,
                    planner=DefaultSavePlanner(dedup_save_to_lowest_rank=True),
                )
                log.info(f"Saved {k} checkpoint to {full_checkpoint_path}")

        if distributed.is_rank0():
            self._write_latest_checkpoint_file(checkpoint_file)

    def save(
        self,
        model_parts: list[nn.Module],
        optimizer: OptimizersContainer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model_parts (list[nn.Module]): The PyTorch model.
            optimizer (OptimizersContainer): Container with the model optimizers.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        del grad_scaler
        self.callbacks.on_save_checkpoint_start(model_parts, iteration)

        checkpoint_file = f"iter_{iteration:09}"
        to_save_dict = {
            "model": ModelWrapper(model_parts).state_dict(),
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            # Dont save trainer, since we dont use grad_scaler
        }

        self.callbacks.on_save_checkpoint(model_parts, state_dict=to_save_dict)

        for k in to_save_dict.keys():
            output_dirname = os.path.join(self.save_dirname, f"iter_{iteration:09}/{k}")
            to_save_dict[k] = (to_save_dict[k], output_dirname)

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            # Dataloader state is a plain dict, not tensors; save in main process and exclude from async staging.
            dataloader_entry = to_save_dict.pop("dataloader", None)
            if dataloader_entry is not None:
                dataloader_state, dataloader_save_dir = dataloader_entry
                self._save_as_pkl(dataloader_state, dataloader_save_dir)
            self._async_with_pinned_memory(checkpoint_file, to_save_dict)
        else:
            self.save_state_dict_worker(to_save_dict, checkpoint_file)

    def finalize(self) -> None:
        super().finalize()
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.mp and self.mp.is_alive():
                self.mp_queue_send.put(Terminate())
                self.mp.join()
