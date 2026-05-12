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

import os
import threading
from typing import TYPE_CHECKING, List, NamedTuple, Tuple

import torch

from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.utils import callback, distributed, log, misc, object_store

if TYPE_CHECKING:
    from cosmos3._src.imaginaire.config import CheckpointConfig, JobConfig

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 11):
    from torch.ao import quantization
    from torch.ao.quantization import FakeQuantizeBase, ObserverBase
elif (
    TORCH_VERSION >= (1, 8)
    and hasattr(torch.quantization, "FakeQuantizeBase")
    and hasattr(torch.quantization, "ObserverBase")
):
    from torch import quantization
    from torch.quantization import FakeQuantizeBase, ObserverBase


class Checkpointer:
    """The checkpointer class. Supports checkpoint saving/loading to both local disk or object store."""

    def __init__(self, config_checkpoint: CheckpointConfig, config_job: JobConfig, callbacks: callback.CallBackGroup):
        """Constructor of the checkpointer.

        Args:
            config_checkpoint (CheckpointConfig): The config object for the checkpointer.
        """
        # Set the callback functions.
        self.callbacks = callbacks



        self.checkpoint_dir_local = f"{config_job.path_local}/checkpoints"
        self.checkpoint_dir_object_store = f"{config_job.path}/checkpoints"
        self.save_to_object_store = config_checkpoint.save_to_object_store.enabled
        self.load_from_object_store = config_checkpoint.load_from_object_store.enabled
        self.strict_resume = config_checkpoint.strict_resume
        self.load_path = config_checkpoint.load_path or None
        self.load_training_state = config_checkpoint.load_training_state
        self.only_load_scheduler_state = config_checkpoint.only_load_scheduler_state
        self.save_thread = None
        # Create the object store client interface.
        if self.save_to_object_store:
            self.object_store_saver = object_store.ObjectStore(config_checkpoint.save_to_object_store)
        if self.load_from_object_store:
            self.object_store_loader = object_store.ObjectStore(config_checkpoint.load_from_object_store)

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
        self.callbacks.on_save_checkpoint_start(model, iteration)

        checkpoint_file = f"iter_{iteration:09}.pt"

        if distributed.get_rank() == 0:
            state_dict = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                grad_scaler=grad_scaler.state_dict(),
                iteration=iteration,
            )
            state_dict = misc.to(state_dict, device="cpu")
            self.callbacks.on_save_checkpoint(model, state_dict=state_dict)
            # Wait for previous saver thread to end.
            if self.save_thread:
                self.save_thread.join()
            # Run the checkpoint saver in a separate thread.
            self.save_thread = threading.Thread(
                target=self._save_worker_object_store if self.save_to_object_store else self._save_worker_local,
                daemon=False,
                args=(state_dict, checkpoint_file, distributed.get_rank()),
            )
            self.save_thread.start()

        # Note: Checkpoints are saved on a separate thread and this callback is not accurate.
        # Please check logs from on_save_checkpoint_success() for better accuracy
        self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    @misc.timer("checkpoint saving (local)")
    def _save_worker_local(self, state_dict: dict[str, torch.Tensor], checkpoint_file: str, rank: int = 0) -> None:
        """Worker to save checkpoint to local disk, spawned with a child thread (runs in parallel with the training).

        Args:
            state_dict (dict[str, torch.Tensor]): The state dict of the model/optimizer/scheduler.
            checkpoint_file (str): The file name of the model checkpoint.
            rank (int): GPU device (default: 0).
        """
        checkpoint_path = os.path.join(self.checkpoint_dir_local, checkpoint_file)
        os.makedirs(self.checkpoint_dir_local, exist_ok=True)
        try:
            torch.save(state_dict, checkpoint_path)
            if rank == 0:
                self._write_latest_checkpoint_file(checkpoint_file)
            log.success(f"Saved checkpoint (local): {checkpoint_path}")
            iteration = int(checkpoint_file.replace("iter_", "").replace(".pt", ""))
            self.callbacks.on_save_checkpoint_success(iteration=iteration)
        except Exception as e:  # noqa: BLE001
            log.exception(f"Checkpoint failed to save (local): {e}")

    @misc.timer("checkpoint saving (object store)")
    def _save_worker_object_store(
        self, state_dict: dict[str, torch.Tensor], checkpoint_file: str, rank: int = 0
    ) -> None:
        """Worker to upload checkpoint to object store, spawned with a child thread (in parallel with the training).

        Args:
            state_dict (dict[str, torch.Tensor]): The state dict of the model/optimizer/scheduler.
            checkpoint_file (str): The file name of the model checkpoint.
            rank (int): GPU device (default: 0).
        """
        checkpoint_path = os.path.join(self.checkpoint_dir_object_store, checkpoint_file)
        try:
            self.object_store_saver.save_object(state_dict, key=checkpoint_path, type="torch")
            if rank == 0:
                self._write_latest_checkpoint_file(checkpoint_file)
            log.success(f"Saved checkpoint (object store): {checkpoint_path}")
            iteration = int(checkpoint_file.replace("iter_", "").replace(".pt", ""))
            self.callbacks.on_save_checkpoint_success(iteration=iteration)
        except Exception as e:  # noqa: BLE001
            log.exception(f"Checkpoint failed to upload (object store): {e}")

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        """Load network weights and optimizer states from a checkpoint in a single process.

        The priority of the checkpoint loading logic is:
        1. Attempt to resume training if possible by looking for latest_checkpoint.txt under the same name.
        2. If no latest checkpoint were found, it loads the model weights specified by config_checkpoint.path.
           - This is typically used for inference mode.
           - If config_checkpoint.load_optimizer_state is True, then also load the optimizer and scheduler states.
        3. If none of the above, randomly initialize the model parameters and train from scratch.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer | None): The model optimizer (default: None).
            scheduler (torch.optim.lr_scheduler.LRScheduler | None): The optimization scheduler (default: None).
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training).

        Returns:
            iteration (int): the iteration number to start/resume from.
        """
        self.callbacks.on_load_checkpoint_start(model)

        latest_checkpoint_file = self._read_latest_checkpoint_file()
        if latest_checkpoint_file is not None:
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_dir = (
                self.checkpoint_dir_object_store if self.load_from_object_store else self.checkpoint_dir_local
            )
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
            resume = True
            only_resume_scheduler = True
        else:
            if self.load_path:
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                resume = self.load_training_state
                only_resume_scheduler = self.only_load_scheduler_state
            else:
                # 3. Randomly initialize the model parameters and train from scratch.
                checkpoint_path = None
                resume = False
                only_resume_scheduler = False
        # Load checkpoint.
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            if self.load_from_object_store:
                log.info(f"Loading checkpoint (object store): {checkpoint_path}")
                state_dict = self.object_store_loader.load_object(key=checkpoint_path, type="torch")
                log.success(f"Complete loading checkpoint (object store): {checkpoint_path}")
            else:
                log.info(f"Loading checkpoint (local): {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
                log.success(f"Complete loading checkpoint (local): {checkpoint_path}")
            self.callbacks.on_load_checkpoint(model, state_dict=state_dict)
            # Load the state dicts.
            log.info("- Loading the model...")
            model.load_state_dict(state_dict["model"], strict=self.strict_resume)
            if resume or only_resume_scheduler:
                iteration = state_dict["iteration"]
                assert scheduler
                log.info("- Loading the scheduler...")
                scheduler.load_state_dict(state_dict["scheduler"])
                scheduler.last_epoch = iteration
            else:
                iteration = 0
            if resume:
                assert optimizer
                log.info("- Loading the optimizer...")
                optimizer.load_state_dict(state_dict["optimizer"])
                log.info("- Loading the gradient scaler...")
                grad_scaler.load_state_dict(state_dict["grad_scaler"])
                log.success(f"Done with loading the checkpoint (iteration {iteration}).")
            else:
                log.success("Done with loading the checkpoint.")
        else:
            # Checkpoint not found and not specified. We will train everything from scratch.
            iteration = 0
            log.info("Training from scratch.")
        torch.cuda.empty_cache()

        self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)

        return iteration

    def _read_latest_checkpoint_file(self) -> str | None:
        """Get the file name of the latest saved checkpoint. If it doesn't exist, return None.

        Returns:
            checkpoint_file (str | None): file name of the latest saved checkpoint.
        """
        checkpoint_file = None
        if self.load_from_object_store:
            latest_path = os.path.join(self.checkpoint_dir_object_store, "latest_checkpoint.txt")
            if self.object_store_loader.object_exists(key=latest_path):
                checkpoint_file = self.object_store_loader.load_object(key=latest_path, type="text").strip()
        else:
            latest_path = os.path.join(self.checkpoint_dir_local, "latest_checkpoint.txt")
            if os.path.isfile(latest_path):
                checkpoint_file = open(latest_path).read().strip()
        return checkpoint_file

    def _write_latest_checkpoint_file(self, checkpoint_file: str) -> None:
        """Track the file name of the latest saved checkpoint.

        Args:
            checkpoint_file (str): file name of the latest saved checkpoint.
        """
        content = f"{checkpoint_file}\n"
        if self.save_to_object_store:
            latest_path = os.path.join(self.checkpoint_dir_object_store, "latest_checkpoint.txt")
            self.object_store_saver.save_object(content, key=latest_path, type="text")
        else:
            latest_path = os.path.join(self.checkpoint_dir_local, "latest_checkpoint.txt")
            with open(latest_path, "w") as file:
                file.write(content)

    def _check_checkpoint_exists(self, checkpoint_path: str) -> None:
        """If the file checkpoint_path does not exist, raise an error.

        Args:
            checkpoint_path (str): full path to the checkpoint.
        """
        if self.load_from_object_store:
            if not self.object_store_loader.object_exists(key=checkpoint_path):
                raise FileNotFoundError(f"File not found (object store): {checkpoint_path}")
        else:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"File not found (local): {checkpoint_path}")

    def finalize(self) -> None:
        """Finalize the checkpointer."""
        if self.save_thread:
            self.save_thread.join()


class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple[str, Tuple[int], Tuple[int]]]),
        ],
    )
):
    pass


class MultiRankCheckpointer(Checkpointer):
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
        # checkpoint_file = f"iter_{iteration:09}.pt"
        postfix, _, total_ema_num = model.get_ckpt_postfix()
        checkpoint_file = f"iter_{iteration:09}{postfix}.pt"
        save_ranks = list(range(total_ema_num))
        for _rank in save_ranks:
            if distributed.get_rank() == _rank:
                state_dict = dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    grad_scaler=grad_scaler.state_dict(),
                    iteration=iteration,
                )
                state_dict = misc.to(state_dict, device="cpu")
                self.callbacks.on_save_checkpoint(model, state_dict=state_dict)
                # Wait for previous saver thread to end.
                if self.save_thread:
                    self.save_thread.join()
                # Run the checkpoint saver in a separate thread.
                self.save_thread = threading.Thread(
                    target=self._save_worker_object_store if self.save_to_object_store else self._save_worker_local,
                    daemon=False,
                    args=(state_dict, checkpoint_file, distributed.get_rank()),
                )
                self.save_thread.start()

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        """Load network weights and optimizer states from a checkpoint in a single process.

        The priority of the checkpoint loading logic is:
        1. Attempt to resume training if possible by looking for latest_checkpoint.txt under the same name.
        2. If no latest checkpoint were found, it loads the model weights specified by config_checkpoint.path.
           - This is typically used for inference mode.
           - If config_checkpoint.load_optimizer_state is True, then also load the optimizer and scheduler states.
        3. If none of the above, randomly initialize the model parameters and train from scratch.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer | None): The model optimizer (default: None).
            scheduler (torch.optim.lr_scheduler.LRScheduler | None): The optimization scheduler (default: None).
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training).

        Returns:
            iteration (int): the iteration number to start/resume from.
        """
        latest_checkpoint_file = self._read_latest_checkpoint_file()
        if latest_checkpoint_file is not None:
            # different from base checkpointer, this support multi-EMA
            postfix, _, total_ema_num = model.get_ckpt_postfix()
            latest_checkpoint_file = latest_checkpoint_file.replace(".pt", f"{postfix}.pt")
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_dir = (
                self.checkpoint_dir_object_store if self.load_from_object_store else self.checkpoint_dir_local
            )
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
            resume = True
        else:
            if self.load_path:
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                # different from base checkpointer, this support multi-EMA
                postfix, _, total_ema_num = model.get_ckpt_postfix()
                checkpoint_path = checkpoint_path.replace(".pt", f"{postfix}.pt")
                resume = self.load_training_state
            else:
                # 3. Randomly initialize the model parameters and train from scratch.
                checkpoint_path = None
                resume = False
        # Load checkpoint.
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            if self.load_from_object_store:
                log.info(f"Loading checkpoint (object store): {checkpoint_path}")
                state_dict = self.object_store_loader.load_object(key=checkpoint_path, type="torch")
                log.success(f"Complete loading checkpoint (object store): {checkpoint_path}")
            else:
                log.info(f"Loading checkpoint (local): {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                log.success(f"Complete loading checkpoint (local): {checkpoint_path}")
            self.callbacks.on_load_checkpoint(model, state_dict=state_dict)
            # Load the state dicts.
            log.info("- Loading the model...")
            log.critical(model.load_state_dict(state_dict["model"], strict=self.strict_resume))
            if resume:
                iteration = state_dict["iteration"]
                assert optimizer and scheduler
                log.info("- Loading the optimizer...")
                optimizer.load_state_dict(state_dict["optimizer"])
                log.info("- Loading the scheduler...")
                scheduler.load_state_dict(state_dict["scheduler"])
                scheduler.last_epoch = iteration
                log.info("- Loading the gradient scaler...")
                grad_scaler.load_state_dict(state_dict["grad_scaler"])
                log.success(f"Done with loading the checkpoint (iteration {iteration}).")
            else:
                iteration = 0
                log.success("Done with loading the checkpoint.")
        else:
            # Checkpoint not found and not specified. We will train everything from scratch.
            iteration = 0
            log.info("Training from scratch.")
        torch.cuda.empty_cache()
        return iteration


# https://github.com/facebookresearch/fvcore/blob/9d683aae73fb899dd35d6cf6720e5ef567761c57/fvcore/common/checkpoint.py
def non_strict_load_model(model: torch.nn.Module, checkpoint_state_dict: dict) -> _IncompatibleKeys:
    # workaround https://github.com/pytorch/pytorch/issues/24139
    model_state_dict = model.state_dict()
    incorrect_shapes = []
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            if "_extra_state" in k:  # Key introduced by TransformerEngine for FP8
                log.warning(f"Skipping key {k} introduced by TransformerEngine for FP8 in the checkpoint.")
                continue
            model_param = model_state_dict[k]
            # Allow mismatch for uninitialized parameters
            if TORCH_VERSION >= (1, 8) and isinstance(model_param, torch.nn.parameter.UninitializedParameter):
                continue
            if not isinstance(model_param, torch.Tensor):
                raise ValueError(
                    f"Find non-tensor parameter {k} in the model. type: {type(model_param)} {type(checkpoint_state_dict[k])}, please check if this key is safe to skip or not."
                )

            shape_model = tuple(model_param.shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                has_observer_base_classes = (
                    TORCH_VERSION >= (1, 8)
                    and hasattr(quantization, "ObserverBase")
                    and hasattr(quantization, "FakeQuantizeBase")
                )
                if has_observer_base_classes:
                    # Handle the special case of quantization per channel observers,
                    # where buffer shape mismatches are expected.
                    def _get_module_for_key(model: torch.nn.Module, key: str) -> torch.nn.Module:
                        # foo.bar.param_or_buffer_name -> [foo, bar]
                        key_parts = key.split(".")[:-1]
                        cur_module = model
                        for key_part in key_parts:
                            cur_module = getattr(cur_module, key_part)
                        return cur_module

                    cls_to_skip = (
                        ObserverBase,
                        FakeQuantizeBase,
                    )
                    target_module = _get_module_for_key(model, k)
                    if isinstance(target_module, cls_to_skip):
                        # Do not remove modules with expected shape mismatches
                        # them from the state_dict loading. They have special logic
                        # in _load_from_state_dict to handle the mismatches.
                        continue

                incorrect_shapes.append((k, shape_checkpoint, shape_model))
                checkpoint_state_dict.pop(k)
    incompatible = model.load_state_dict(checkpoint_state_dict, strict=False)
    # Remove keys with "_extra_state" suffix, which are non-parameter items introduced by TransformerEngine for FP8 handling
    missing_keys = [k for k in incompatible.missing_keys if "_extra_state" not in k]
    unexpected_keys = [k for k in incompatible.unexpected_keys if "_extra_state" not in k]
    return _IncompatibleKeys(
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        incorrect_shapes=incorrect_shapes,
    )
