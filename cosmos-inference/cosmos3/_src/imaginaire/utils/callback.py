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

import sys
import time
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional

import omegaconf
import torch
import torch.distributed as dist
import torch.utils.data
import tqdm
import wandb

from cosmos3._src.imaginaire.lazy_config import instantiate
from cosmos3._src.imaginaire.utils import distributed, log, misc, wandb_util
from cosmos3._src.imaginaire.utils.misc import get_local_tensor_if_DTensor


try:
    from megatron.core import parallel_state
except ImportError:
    parallel_state = None


if TYPE_CHECKING:
    from cosmos3._src.imaginaire.config import Config
    from cosmos3._src.imaginaire.model import ImaginaireModel
    from cosmos3._src.imaginaire.trainer import ImaginaireTrainer


class CallBackGroup:
    """A class for hosting a collection of callback objects.

    It is used to execute callback functions of multiple callback objects with the same method name.
    When callbackgroup.func(args) is executed, internally it loops through the objects in self._callbacks and runs
    self._callbacks[0].func(args), self._callbacks[1].func(args), etc. The method name and arguments should match.

    Attributes:
        _callbacks (list[Callback]): List of callback objects.
    """

    def __init__(self, config: Config, trainer: ImaginaireTrainer) -> None:
        """Initializes the list of callback objects.

        Args:
            config (Config): The config object for the Imaginaire codebase.
            trainer (ImaginaireTrainer): The main trainer.
        """
        self._callbacks = []
        callback_configs = config.trainer.callbacks
        if callback_configs:
            if isinstance(callback_configs, list) or isinstance(callback_configs, omegaconf.listconfig.ListConfig):
                warnings.warn(
                    "The 'config.trainer.callbacks' parameter should be a dict instead of a list. "
                    "Please update your code",
                    DeprecationWarning,
                    stacklevel=2,
                )
                callback_configs = {f"callback_{i}": v for i, v in enumerate(callback_configs)}
            for callback_name, current_callback_cfg in callback_configs.items():
                if "_target_" not in current_callback_cfg:
                    log.critical(
                        f"Callback {callback_name} is missing the '_target_' field. \n SKip {current_callback_cfg}"
                    )
                    continue
                log.critical(f"Instantiating callback {callback_name}: {current_callback_cfg}")
                _callback = instantiate(current_callback_cfg)
                assert isinstance(_callback, Callback), f"{current_callback_cfg} is not a valid callback."
                _callback.config = config
                _callback.trainer = trainer
                self._callbacks.append(_callback)

    def __getattr__(self, method_name: str) -> Callable:
        """Loops through the callback objects to call the corresponding callback function.

        Args:
            method_name (str): Callback method name.
        """

        def multi_callback_wrapper(*args, **kwargs) -> None:
            for callback in self._callbacks:
                assert hasattr(callback, method_name)
                method = getattr(callback, method_name)
                assert callable(method)
                _ = method(*args, **kwargs)

        return multi_callback_wrapper


class Callback:
    """The base class for all callbacks.

    All callbacks should inherit from this class and adhere to the established method names and signatures.
    """

    def __init__(self, config: Optional["Config"] = None, trainer: Optional["ImaginaireTrainer"] = None):
        """Initializes a Callback object.

        Args:
            config (Optional[Config]): The configuration object for the Imaginaire codebase, if available.
            trainer (Optional[ImaginaireTrainer]): The main trainer handling the training loop, if available.

        Notes:
            The config and trainer parameters are optional to maintain backward compatibility.
            In future releases, these parameters will be removed. Upon using these parameters, a deprecation
            warning will be issued.

        """
        if config is not None or trainer is not None:
            warnings.warn(
                "The 'config' and 'trainer' parameters are deprecated and will be removed in a future release. "
                "Please update your code to create Callback instances without these parameters.",
                DeprecationWarning,
                stacklevel=2,
            )
        del config, trainer

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        pass

    def on_training_step_start(self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        """
        Called before the training step, for each batch. This is paired with on_training_step_end() but note that
        when using gradient accumulation, while on_training_step_end() is only called when the optimizer is updated,
        this function is called for every batch.
        Use on_training_step_batch_start and on_training_step_batch_end if you need callbacks that are called
        for every batch, albeit with the same iteration number.
        FIXME - should this either be deprecated, or called only when a new training step is started after having updated
        the optimizer?
        """
        pass

    def on_training_step_batch_start(
        self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        """
        Called before the training step, for each batch, similarly to on_training_step_start(). This function is paired with
        on_training_step_batch_end(), and both functions are called for every batch even when using gradient accumulation.
        Note that the iteration is only updated when the optimizer is updated, and therefore it may be the same for multiple invocations.
        """
        pass

    def on_before_forward(self, iteration: int = 0) -> None:
        pass

    def on_after_forward(self, iteration: int = 0) -> None:
        pass

    def on_before_backward(
        self, model_ddp: distributed.DistributedDataParallel, loss: torch.Tensor, iteration: int = 0
    ) -> None:
        pass

    def on_after_backward(self, model_ddp: distributed.DistributedDataParallel, iteration: int = 0) -> None:
        pass

    def on_before_dataloading(self, iteration: int = 0) -> None:
        pass

    def on_after_dataloading(self, iteration: int = 0) -> None:
        pass

    def on_optimizer_init_start(self) -> None:
        pass

    def on_optimizer_init_end(self) -> None:
        pass

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        pass

    def on_before_zero_grad(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None:
        pass

    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        """
        Called at the end of a training step for every batch even when using gradient accumulation.
        This is paired with on_training_step_batch_start(). Note that the iteration is only updated when the optimizer is updated,
        and therefore it may be the same for multiple batches.
        """
        pass

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        """
        Called at the end of a training step, but note that when using gradient accumulation, this is only called
        when the optimizer is updated, and the iteration incremented, whereas on_training_step_start is called every time.
        Use on_training_step_batch_start and on_training_step_batch_end if you need callbacks that are called
        for every batch.
        """
        pass

    def on_validation_start(
        self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        pass

    def on_validation_step_start(
        self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        pass

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        pass

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        pass

    def on_load_checkpoint_start(self, model: ImaginaireModel) -> None:
        pass

    def on_load_checkpoint_end(
        self, model: ImaginaireModel, iteration: int = 0, checkpoint_path: Optional[str] = None
    ) -> None:
        pass

    def on_load_checkpoint(self, model: ImaginaireModel, state_dict: dict[Any]) -> None:
        """
        Called when checkpoint loading is about to start, but after on_save_checkpoint_start().
        FIXME - why do we need this callback, can't we just use on_save_checkpoint_start()?
        """
        pass

    def on_save_checkpoint_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        """
        Called when checkpoint saving is about to start.
        """
        pass

    def on_save_checkpoint_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        """
        Called when the synchronous part of checkpointing is finished, this function can be used
        along with on_save_checkpoint_start() to measure the exposed (synchronous) checkpoint time.
        Note that for asynchronous checkpoint, the checkpoint may still be ongoing, so this function
        does not mean the checkpoint is finished for the asynchronous case, use on_save_checkpoint_success()
        for that.
        """
        pass

    def on_save_checkpoint_success(self, iteration: int = 0, elapsed_time: float = 0) -> None:
        """
        Called when checkpoint saving is fully finished, and succeeded. Not called if checkpoint failed.
        For synchronous checkpoint, it is called at the same time as on_save_checkpoint_end(), but for asynchronous
        checkpoint, it is called after the asynchronous part has also finished. For checkpointers with out-of-process
        checkpointing, this function is called as soon as the notification is received from the checkpointer process,
        which may not be immediately after the checkpoint has completed but later on. Therefore, if you need to measure
        the full checkpoint duration for the asynchronous part, use the elapsed_time parameter, do not measure it directly
        as this would be a significant overestimate.
        """
        pass

    def on_save_checkpoint(self, model: ImaginaireModel, state_dict: dict[Any]) -> None:
        pass

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        pass

    def on_app_end(self) -> None:
        pass


class EMAModelCallback(Callback):
    """The callback class for tracking EMA model weights."""

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        # Set up the EMA model weight tracker.
        if model.config.ema.enabled:
            assert hasattr(model, "ema"), "EMA should be initialized from ImaginaireModel"
            # EMA model must be kept in FP32 precision.
            model.ema = model.ema.to(dtype=torch.float32)
        else:
            assert not hasattr(model, "ema"), "There should be no EMA initialized."

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # Update the EMA model with the new regular weights.
        if model.config.ema.enabled:
            model.ema.update_average(model, iteration)


class ProgressBarCallback(Callback):
    """The callback class for visualizing the training/validation progress bar in the console."""

    @distributed.rank0_only
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        self.train_pbar = tqdm.trange(self.config.trainer.max_iter, initial=iteration, desc="Training")

    @distributed.rank0_only
    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self.train_pbar.update()

    @distributed.rank0_only
    def on_validation_start(
        self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        if self.config.trainer.max_val_iter is not None:
            num_iter = self.config.trainer.max_val_iter
        else:
            num_iter = len(dataloader_val)
        assert num_iter is not None and num_iter > 0, f"Invalid number of validation iterations: {num_iter}"
        self.val_pbar = tqdm.trange(num_iter, desc="Validating", position=1, leave=False)

    @distributed.rank0_only
    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self.val_pbar.update()

    @distributed.rank0_only
    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        self.val_pbar.close()

    @distributed.rank0_only
    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        self.trainer.checkpointer.finalize()
        self.train_pbar.close()


class IterationLoggerCallback(Callback):
    """The callback class for visualizing the training/validation progress bar in the console."""

    @distributed.rank0_only
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        # self.train_pbar = tqdm.trange(self.config.trainer.max_iter, initial=iteration, desc="Training")
        self.start_iteration_time = time.time()
        self.elapsed_iteration_time = 0

    @distributed.rank0_only
    def on_training_step_start(self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        self.start_iteration_time = time.time()

    @distributed.rank0_only
    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:

        # but this is only called when the optimizer is updated, so it's only the time for the last batch.
        self.elapsed_iteration_time += time.time() - self.start_iteration_time

        if iteration % self.config.trainer.logging_iter == 0:
            avg_time = self.elapsed_iteration_time / self.config.trainer.logging_iter
            log.info(f"Iteration: {iteration}, average iter time: {avg_time:2f}, total loss {loss.item():4f}")

            self.elapsed_iteration_time = 0


class WandBCallback(Callback):
    """The callback class for logging to Weights and Biases (W&B).

    By default, WandBCallback logs the following training stats to W&B every config.trainer.logging_iter:
    - iteration: The current iteration number (useful for visualizing the training progress over time).
    - train/loss: The computed overall loss in the training batch.
    - optim/lr: The current learning rate.
    - timer/*: The averaged timing results of each code block recorded by trainer.training_timer.
    For validation, WandBCallback logs:
    - val/loss: The computed overall loss in the validation dataset.
    """

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        wandb_util.init_wandb(self.config, model=model)

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:  # Log the curent learning rate.
        if iteration % self.config.trainer.logging_iter == 0 and distributed.is_rank0():
            wandb.log({"optim/lr": scheduler.get_last_lr()[0]}, step=iteration)
            wandb.log({"optim/grad_scale": grad_scaler.get_scale()}, step=iteration)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Log the timing results (over a number of iterations) and the training loss.
        if iteration % self.config.trainer.logging_iter == 0:
            timer_results = self.trainer.training_timer.compute_average_results()

            # reduce loss
            sample_size = torch.tensor(misc.get_data_batch_size(data_batch), device="cuda")
            loss_sum = loss * sample_size
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_size, op=dist.ReduceOp.SUM)
            avg_loss = loss_sum.item() / sample_size.item()

            if distributed.is_rank0():
                wandb.log({f"timer/{key}": value for key, value in timer_results.items()}, step=iteration)
                wandb.log({"train/loss": avg_loss}, step=iteration)
                wandb.log({"iteration": iteration}, step=iteration)
            self.trainer.training_timer.reset()

    def on_validation_start(
        self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        # Cache for collecting data/output batches.
        self._val_cache: dict[str, Any] = dict(
            data_batches=[],
            output_batches=[],
            loss=torch.tensor(0.0, device="cuda"),
            sample_size=torch.tensor(0, device="cuda"),
        )

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Collect the validation batch and aggregate the overall loss.
        # Collect the validation batch and aggregate the overall loss.
        batch_size = misc.get_data_batch_size(data_batch)
        self._val_cache["loss"] += loss * batch_size
        self._val_cache["sample_size"] += batch_size

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        # Compute the average validation loss across all devices.
        dist.all_reduce(self._val_cache["loss"], op=dist.ReduceOp.SUM)
        dist.all_reduce(self._val_cache["sample_size"], op=dist.ReduceOp.SUM)
        loss = self._val_cache["loss"].item() / self._val_cache["sample_size"]
        # Log data/stats of validation set to W&B.
        if distributed.is_rank0():
            log.info(f"Validation loss (iteration {iteration}): {loss:4f}")
            wandb.log({"val/loss": loss}, step=iteration)

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        wandb.finish()


class LowPrecisionCallback(Callback):
    """The callback class handling low precision training"""

    def __init__(self, config: Config, trainer: ImaginaireTrainer, update_iter: int):
        self.update_iter = update_iter

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if model.precision == torch.float32:
            log.critical("Using fp32. We should disable master weights update.")
            self.update_iter = sys.maxsize
        else:
            assert model.precision in [
                torch.bfloat16,
                torch.float16,
                torch.half,
            ], "LowPrecisionCallback must use a low precision dtype."
        self.precision_type = model.precision

    def on_training_step_start(self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data[k]):
                data[k] = v.to(dtype=self.precision_type)

    def on_validation_step_start(
        self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data[k]):
                data[k] = v.to(dtype=self.precision_type)

    def on_before_zero_grad(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None:
        if iteration % self.update_iter == 0:
            if getattr(optimizer, "master_weights", False):
                params, master_params = [], []
                for group, group_master in zip(optimizer.param_groups, optimizer.param_groups_master):
                    for p, p_master in zip(group["params"], group_master["params"]):
                        params.append(get_local_tensor_if_DTensor(p).data)
                        master_params.append(get_local_tensor_if_DTensor(p_master).data)
                torch._foreach_copy_(params, master_params)


class NVTXCallback(Callback):
    """The callback for creating NVTX ranges"""

    def __init__(
        self,
        synchronize: bool = False,
        config: Optional["Config"] = None,
        trainer: Optional["ImaginaireTrainer"] = None,
    ):
        super().__init__(config, trainer)
        self.synchronize = synchronize

    def on_before_forward(self, iteration: int = 0) -> None:
        if self.synchronize:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("forward")

    def on_after_forward(self, iteration: int = 0) -> None:
        if self.synchronize:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    def on_before_backward(
        self, model_ddp: distributed.DistributedDataParallel, loss: torch.Tensor, iteration: int = 0
    ) -> None:
        if self.synchronize:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("backward")

    def on_after_backward(self, model_ddp: distributed.DistributedDataParallel, iteration: int = 0) -> None:
        if self.synchronize:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        if self.synchronize:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("optimizer_step")

    def on_before_zero_grad(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None:
        if self.synchronize:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    def on_before_dataloading(self, iteration: int = 0) -> None:
        torch.cuda.nvtx.range_push("dataloading")

    def on_after_dataloading(self, iteration: int = 0) -> None:
        torch.cuda.nvtx.range_pop()


