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
import inspect
import os
import signal

import torch
import torch.distributed as dist
import torch.utils.data

from cosmos.utils.flags import INTERNAL
from cosmos.utils.context_managers import distributed_init
from cosmos.utils.profiling import maybe_enable_memory_snapshot, maybe_enable_nsys_profiling, maybe_enable_profiling

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False


from cosmos.utils.lazy_config import LazyConfig, instantiate
from cosmos.model._base import ImaginaireModel
from cosmos.utils import callback, distributed, ema, log, misc
from cosmos.utils.checkpointer import Checkpointer
from cosmos.utils.misc import StragglerDetectorV2



class ImaginaireTrainer:
    """The base trainer class of Imaginaire.

    All trainers in Imaginaire should inherit ImaginaireTrainer. It contains the basic functionality for model training
    (particularly suited for large-scale training), including data parallel (DDP/FSDP), model weight average (EMA),
    mixed-precision training (fp16/bf16).

    Attributes:
        checkpointer (Checkpointer): checkpointer object to save/load model weights and optimizer states.
        training_timer (misc.Timer): Timer object to time code blocks and functions.
    """

    def __init__(self, config):
        """Constructor of the trainer.

        Args:
            config (Config): The config object for the Imaginaire codebase.
        """
        super().__init__()
        self.config = config
        # Set up the distributed computing environment.
        with distributed_init():
            distributed.init()
            # Set up parallel states.
            if hasattr(config.model, "context_parallel_size"):
                if config.model_parallel.context_parallel_size > 1:
                    raise ValueError(
                        "Both config.model.context_parallel_size and config.model_parallel.context_parallel_size are set. "
                        "config.model.context_parallel_size is deprecated. Please only set config.model_parallel.context_parallel_size."
                    )
                else:
                    log.critical(
                        "Using deprecated config.model.context_parallel_size. Please use config.model_parallel.context_parallel_size instead."
                    )
                    config.model_parallel.context_parallel_size = config.model.context_parallel_size
            if USE_MEGATRON:
                if (
                    "create_gloo_process_groups"
                    in inspect.signature(parallel_state.initialize_model_parallel).parameters
                ):
                    parallel_state.initialize_model_parallel(
                        pipeline_model_parallel_size=config.model_parallel.pipeline_model_parallel_size,
                        tensor_model_parallel_size=config.model_parallel.tensor_model_parallel_size,
                        context_parallel_size=config.model_parallel.context_parallel_size,
                        create_gloo_process_groups=False,
                    )
                else:
                    parallel_state.initialize_model_parallel(
                        pipeline_model_parallel_size=config.model_parallel.pipeline_model_parallel_size,
                        tensor_model_parallel_size=config.model_parallel.tensor_model_parallel_size,
                        context_parallel_size=config.model_parallel.context_parallel_size,
                    )
                # `config.model_parallel.sequence_parallel` is a bool that indicates whether to use sequence parallelism.
                # It is not part of the original `parallel_state` API, so we need to set it manually.
                parallel_state.sequence_parallel = config.model_parallel.sequence_parallel
                if parallel_state.sequence_parallel:
                    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        # Create the local job directory, save the config file, and pipe to a local log.
        if distributed.is_rank0():
            os.makedirs(config.job.path_local, exist_ok=True)
            # Save the config as .pkl for reproducibility.
            LazyConfig.save_pkl(config, f"{config.job.path_local}/config.pkl")
            # Save the config as .yaml for reading or parsing experiment hyperparameters.
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        dist.barrier()
        if INTERNAL:
            log.init_loguru_file(f"{config.job.path_local}/stdout.log")
            if distributed.is_rank0():
                # Print important environment variables and the effective config.
                log.info("Config:\n" + config.pretty_print(use_color=True))
            misc.print_environ_variables(["TORCH_HOME", "IMAGINAIRE_OUTPUT_ROOT", "ENABLE_ONELOGGER"])
        else:
            misc.print_environ_variables(["HF_HOME", "IMAGINAIRE_OUTPUT_ROOT"])
        # Set the random seed. If multi-GPU, different ranks are set with different seeds.
        misc.set_random_seed(seed=config.trainer.seed, by_rank=True)
        # Initialize cuDNN.
        torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
        torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
        # Initialize the callback functions.
        self.callbacks = callback.CallBackGroup(config=config, trainer=self)
        # Initialize the model checkpointer.
        if config.checkpoint.type is None:
            self.checkpointer = Checkpointer(config.checkpoint, config.job, callbacks=self.callbacks)
        else:
            self.checkpointer: Checkpointer = instantiate(
                config.checkpoint.type, config.checkpoint, config.job, callbacks=self.callbacks
            )
        # Initialize the timer for speed benchmarking.
        self.training_timer = misc.TrainingTimer()
        # Initialize Straggler Detection
        self.straggler_detector = StragglerDetectorV2(
            enabled=self.config.trainer.straggler_detection.enabled,
            report_freq=self.config.trainer.straggler_detection.report_freq,
            profile_freq=self.config.trainer.straggler_detection.profile_freq,
            max_diff=self.config.trainer.straggler_detection.max_diff,
            raise_error=self.config.trainer.straggler_detection.raise_error,
            save_s3=self.config.trainer.straggler_detection.save_s3,
        )
        misc.set_torch_compile_options(
            self.config.trainer.compile_config.recompile_limit, self.config.trainer.compile_config.use_duck_shape
        )
        self.straggler_detector.initialize()
        # Send a TimeoutError if a training step takes over timeout_period seconds.
        signal.signal(signal.SIGALRM, functools.partial(misc.timeout_handler, config.trainer.timeout_period))  # type: ignore

    def _fetch_and_broadcast_data(
        self,
        model: ImaginaireModel,
        dataloader_iter,
        iteration: int,
    ):
        """
        Fetches data from the dataloader on the batch owner rank and broadcasts it to all other ranks in the Context Parallel group if CP is enabled.
        When CP is disabled, data is fetched from the dataloader on the current rank and no broadcasting is needed.

        Args:
            model (ImaginaireModel): The model containing parallel dimensions info.
            dataloader_iter: Iterator for the dataloader.
            iteration (int): Current iteration number to determine the batch owner.

        Returns:
            tuple: (data_batch, stop_signal)
                - data_batch: The fetched data batch (or None if stopped/not owner).
                - stop_signal (bool): True if StopIteration was encountered.
        """
        parallel_dims = getattr(model, "parallel_dims", None)
        if parallel_dims is None or not parallel_dims.cp_enabled:
            try:
                return next(dataloader_iter), False
            except StopIteration:
                return None, True

        # To prevent redundant data loading among the Context Parallel ranks,
        # one of the Context Parallel ranks (round-robin) broadcasts the data to all other cp ranks.
        batch_owner_rank = iteration % parallel_dims.cp_mesh.size()
        stop_signal = False
        data_batch = None

        if parallel_dims.cp_rank == batch_owner_rank:
            try:
                data_batch = next(dataloader_iter)
            except StopIteration:
                stop_signal = True
                data_batch = None

        objs = [data_batch, stop_signal]

        # Calculate the global rank of the batch owner within the CP group
        global_src_rank = dist.get_global_rank(parallel_dims.cp_mesh.get_group(), batch_owner_rank)

        dist.broadcast_object_list(
            objs,
            src=global_src_rank,
            group=parallel_dims.cp_mesh.get_group(),
        )

        return objs[0], objs[1]

    def train(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:
        """The training function.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_train (torch.utils.data.DataLoader): The training data loader.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
        """
        # Leaving this for backward compability for now, but we can think about moving this to model.on_train_start for all models.
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)  # type: ignore
        model.on_train_start(self.config.trainer.memory_format)

        # Initialize the optimizer, scheduler, and grad_scaler.
        self.callbacks.on_optimizer_init_start()
        optimizer, scheduler = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()
        # Load the model checkpoint and get the starting iteration number.
        iteration = self.checkpointer.load(model, optimizer, scheduler, grad_scaler)
        if hasattr(dataloader_train, "set_start_iteration"):
            dataloader_train.set_start_iteration(iteration * self.config.trainer.grad_accum_iter)
        grad_accum_iter = 0
        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        if self.config.trainer.distributed_parallelism == "ddp":
            # Create a DDP model wrapper.
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")

        log.info("Starting training...")
        sm_carveout = int(os.environ.get("GROUPED_MM_SM_CARVEOUT", "0"))
        if sm_carveout:
            torch._C._set_sm_carveout_experimental(sm_carveout)
            log.info(f"Set SM carveout to {sm_carveout}")
        self.callbacks.on_train_start(model, iteration=iteration)
        # Initial validation.
        if self.config.trainer.run_validation and iteration == 0 and self.config.trainer.run_validation_on_start:
            self.validate(model, dataloader_val, iteration=iteration)

        if self.config.trainer.save_zero_checkpoint and iteration == 0:
            self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=0)

        _end_training = False
        if torch.are_deterministic_algorithms_enabled():
            # Re-seed all global RNGs after init (model load, checkpoint load, compile warmup,
            # callbacks) so data-augmentation randomness starts from a deterministic state
            # regardless of how much RNG state init consumed.
            misc.set_random_seed(seed=self.config.trainer.seed, by_rank=True)
        with (
            maybe_enable_profiling(self.config, global_step=iteration) as torch_profiler,
            maybe_enable_memory_snapshot(self.config, global_step=iteration) as memory_profiler,
            maybe_enable_nsys_profiling(self.config, global_step=iteration) as nsys_profiler,
        ):
            while True:
                dataloader_train_iter = iter(dataloader_train)
                while True:
                    self.callbacks.on_before_dataloading(iteration)
                    try:
                        with (
                            self.training_timer("dataloader_train"),
                            self.straggler_detector.profile_section(
                                "dataloading",
                                self.config.trainer.straggler_detection.analyze_dataloading,
                                profile_cuda=False,
                            ),
                        ):
                            data_batch, stop_signal = self._fetch_and_broadcast_data(
                                model,
                                dataloader_train_iter,
                                iteration,
                            )
                            if stop_signal:
                                raise StopIteration
                    except StopIteration:
                        break
                    finally:
                        self.callbacks.on_after_dataloading(iteration)
                    # If max_iter is reached, exit the training loop.
                    if iteration >= self.config.trainer.max_iter:
                        _end_training = True
                        break
                    # Move all tensors in the data batch to GPU device.
                    data_batch = misc.to(data_batch, device="cuda")
                    # The actual training step.
                    self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
                    self.callbacks.on_training_step_batch_start(model, data_batch, iteration=iteration)
                    if not model.training:
                        model_ddp.train()
                    assert model_ddp.training, "model_ddp is not in training mode."
                    assert model.training, "model is not in training mode."
                    output_batch, loss, grad_accum_iter = self.training_step(
                        model_ddp,
                        optimizer,
                        scheduler,
                        grad_scaler,
                        data_batch,
                        iteration=iteration,
                        grad_accum_iter=grad_accum_iter,
                    )
                    self.callbacks.on_training_step_batch_end(
                        model, data_batch, output_batch, loss, iteration=iteration
                    )
                    # If the gradients are still being accumulated, continue to load the next training batch.
                    if grad_accum_iter != 0:
                        continue
                    # Do the following when an actual optimizer (update) step has been made.
                    iteration += 1
                    # Save checkpoint.
                    if iteration % self.config.checkpoint.save_iter == 0:
                        self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
                    self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)
                    # Validation.
                    if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
                        self.validate(model, dataloader_val, iteration=iteration)
                    # This iteration is successful; reset the timeout signal.
                    signal.alarm(self.config.trainer.timeout_period)
                    self.straggler_detector.generate_report(iteration)
                    if torch_profiler:
                        torch_profiler.step()
                    if memory_profiler:
                        memory_profiler.step()
                    if nsys_profiler:
                        nsys_profiler.step()
                if _end_training:
                    break
        log.success("Done with training.")
        if sm_carveout:
            torch._C._set_sm_carveout_experimental(None)
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def training_step(
        self,
        model_ddp: torch.nn.Module | distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        data: dict[str, torch.Tensor],
        iteration: int = 0,
        grad_accum_iter: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        """The training step.

        Args:
            model_ddp (torch.nn.Module | distributed.DistributedDataParallel): The model with a DDP wrapper or, the bare
              module, depending on whether distributed training is enabled or not.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            data (dict[str, torch.Tensor]): Data batch (dictionary of tensors).
            iteration (int): Current iteration number.
            grad_accum_iter (int): Number of gradient accumulation iterations.

        Returns:
            output (dict[str, torch.Tensor]): The model output from the training data batch (dictionary of tensors).
            loss (torch.Tensor): The total loss of the training data batch.
        """
        # Only let DDP sync gradient at the last iteration of the gradient accumulation window
        with distributed.ddp_sync_grad(model_ddp, grad_accum_iter == self.config.trainer.grad_accum_iter - 1):
            self.callbacks.on_before_forward(iteration=iteration)
            with self.training_timer("forward"):
                with self.straggler_detector.profile_section(
                    "fwd", self.config.trainer.straggler_detection.analyze_forward
                ):
                    output_batch, loss = model_ddp.training_step(data, iteration)
            self.callbacks.on_after_forward(iteration=iteration)
            self.callbacks.on_before_backward(model_ddp, loss, iteration=iteration)
            with self.training_timer("backward"):
                with self.straggler_detector.profile_section(
                    "bwd", self.config.trainer.straggler_detection.analyze_backward
                ):
                    loss_scaled = grad_scaler.scale(loss / self.config.trainer.grad_accum_iter)
                    loss_scaled.backward()
                    if self.config.trainer.distributed_parallelism == "ddp":
                        model_ddp.module.on_after_backward()
                    else:
                        model_ddp.on_after_backward()
            self.callbacks.on_after_backward(model_ddp, iteration=iteration)
        grad_accum_iter += 1
        if grad_accum_iter == self.config.trainer.grad_accum_iter:
            with self.training_timer("optimizer_step"):
                with self.straggler_detector.profile_section(
                    "opt", self.config.trainer.straggler_detection.analyze_optimizer
                ):
                    self.callbacks.on_before_optimizer_step(
                        model_ddp, optimizer, scheduler, grad_scaler, iteration=iteration
                    )
                    model = model_ddp.module if self.config.trainer.distributed_parallelism == "ddp" else model_ddp
                    self._optimizer_step(model, optimizer, scheduler, grad_scaler, iteration=iteration)
                    self.callbacks.on_before_zero_grad(model_ddp, optimizer, scheduler, iteration=iteration)
                    if self.config.trainer.distributed_parallelism == "ddp":
                        model_ddp.module.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                    else:
                        model_ddp.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                    self._zero_grad(model, optimizer, iteration)
            grad_accum_iter = 0
        return output_batch, loss, grad_accum_iter

    def _optimizer_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Execute the optimizer step. Override to customise (e.g. PhaseOptimizer)."""
        grad_scaler.step(optimizer)
        grad_scaler.update()
        scheduler.step()

    def _zero_grad(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int) -> None:
        """Zero gradients. Override to customise (e.g. PhaseOptimizer)."""
        optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def validate(self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0) -> None:
        """Validate on the full validation dataset.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
            iteration (int): Current iteration number.
        """
        self.callbacks.on_validation_start(model, dataloader_val, iteration=iteration)
        model.eval()
        # Evaluate on the full validation set.
        with ema.ema_scope(model, enabled=model.config.ema.enabled):
            for val_iter, data_batch in enumerate(dataloader_val):
                if self.config.trainer.max_val_iter is not None and val_iter >= self.config.trainer.max_val_iter:
                    break
                data_batch = misc.to(data_batch, device="cuda")
                self.callbacks.on_validation_step_start(model, data_batch, iteration=iteration)
                output_batch, loss = model.validation_step(data_batch, iteration)
                self.callbacks.on_validation_step_end(model, data_batch, output_batch, loss, iteration=iteration)
        self.callbacks.on_validation_end(model, iteration=iteration)
