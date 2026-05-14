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

from typing import Any, Optional

import torch

from cosmos.model._base import ImaginaireModel
from cosmos.utils import distributed
from cosmos.utils.callback import Callback
from cosmos.utils.log import logger
from cosmos.utils.misc import get_data_batch_size
from cosmos.utils.training_telemetry.utils import (
    get_checkpoint_strategy,
    get_telemetry_recorder,
    import_training_telemetry,
)


class TelemetryCallback(Callback):
    """Callback for Telemetry"""

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.training_telemetry = import_training_telemetry()
        self.recorder = get_telemetry_recorder()
        self.spans: dict[self.training_telemetry.events.EventName, self.training_telemetry.Span] = {}
        self.iteration_elapsed = 0.0
        self.forward_elapsed = 0.0
        self.backward_elapsed = 0.0
        self.dataloader_elapsed = 0.0
        self.validation_elapsed = 0.0
        self.validation_loss = 0.0
        self.validation_iter = 0
        self.checkpoint_strategy = "async"
        self.checkpoint_interval = 0
        self.validation_interval = 0

    def _start_span(self, span_name: Any, metrics: Any = None, verbosity: Any = None, color: Any = None) -> None:
        """Start a span with the given event name and optional metrics, color, and verbosity"""
        if span_name in self.spans:
            logger.warning(f"Span {span_name} already started, stopping it but this is unexpected")
            self.recorder.stop(self.spans[span_name])
            del self.spans[span_name]
        if verbosity is None:
            verbosity = self.training_telemetry.Verbosity.INFO
        self.spans[span_name] = self.recorder.start(
            name=span_name,
            color=color,
            verbosity=verbosity,
            metrics=metrics,
        )

    def _stop_span(self, span_name: Any, metrics: Any = None) -> float:
        """Stop a span and return the elapsed time"""
        span = self.spans.get(span_name)
        if span is None:
            logger.warning(f"Span {span_name} was not started, this is unexpected")
            return 0.0
        if metrics is not None:
            span.add_metrics(metrics)
        self.recorder.stop(span)
        del self.spans[span_name]
        return span.duration.elapsed

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        span_name = self.training_telemetry.SpanName.TRAINING_LOOP
        self._start_span(span_name)
        self.iteration_elapsed = 0
        self.forward_elapsed = 0
        self.backward_elapsed = 0
        self.dataloader_elapsed = 0

        try:
            self.checkpoint_strategy = get_checkpoint_strategy(self.config.checkpoint)
        except Exception as e:
            logger.warning(f"Failed to get checkpoint strategy using default {self.checkpoint_strategy}: {e}")

        try:
            self.checkpoint_interval = self.config.checkpoint.save_iter
            self.validation_interval = self.config.trainer.validation_iter
        except Exception as e:
            logger.warning(
                f"Failed to get intervals using default {self.checkpoint_interval} and {self.validation_interval}: {e}"
            )

    def on_training_step_batch_start(
        self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        span_name = self.training_telemetry.SpanName.ITERATION
        self._start_span(
            span_name,
            color=self.training_telemetry.SpanColor.BLUE,
            verbosity=self.training_telemetry.Verbosity.PROFILING,
        )

    def on_before_forward(self, iteration: int = 0) -> None:
        span_name = self.training_telemetry.SpanName.MODEL_FORWARD
        self._start_span(
            span_name,
            color=self.training_telemetry.SpanColor.GREEN,
            verbosity=self.training_telemetry.Verbosity.PROFILING,
        )

    def on_after_forward(self, iteration: int = 0) -> None:
        span_name = self.training_telemetry.SpanName.MODEL_FORWARD
        self.forward_elapsed += self._stop_span(span_name)

    def on_before_backward(
        self, model_ddp: distributed.DistributedDataParallel, loss: torch.Tensor, iteration: int = 0
    ) -> None:
        span_name = self.training_telemetry.SpanName.MODEL_BACKWARD
        self._start_span(
            span_name,
            color=self.training_telemetry.SpanColor.YELLOW,
            verbosity=self.training_telemetry.Verbosity.PROFILING,
        )

    def on_after_backward(self, model_ddp: distributed.DistributedDataParallel, iteration: int = 0) -> None:
        span_name = self.training_telemetry.SpanName.MODEL_BACKWARD
        self.backward_elapsed += self._stop_span(span_name)

    def on_before_dataloading(self, iteration: int = 0) -> None:
        span_name = self.training_telemetry.SpanName.DATA_LOADING
        self._start_span(
            span_name,
            color=self.training_telemetry.SpanColor.GREEN,
            verbosity=self.training_telemetry.Verbosity.PROFILING,
        )

    def on_after_dataloading(self, iteration: int = 0) -> None:
        span_name = self.training_telemetry.SpanName.DATA_LOADING
        self.dataloader_elapsed += self._stop_span(span_name)

    def on_optimizer_init_start(self) -> None:
        span_name = self.training_telemetry.SpanName.OPTIMIZER_INIT
        self._start_span(span_name)

    def on_optimizer_init_end(self) -> None:
        span_name = self.training_telemetry.SpanName.OPTIMIZER_INIT
        self._stop_span(span_name)

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
        span_name = self.training_telemetry.SpanName.ITERATION
        self.iteration_elapsed += self._stop_span(span_name)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        """Unlike on_training_step_batch_end, this function is called when the optimizer is updated, and the iteration incremented."""
        if iteration % self.config.trainer.logging_iter == 0:
            event_name = self.training_telemetry.events.EventName.TRAINING_ITERATIONS
            avg_iteration_time = self.iteration_elapsed / self.config.trainer.logging_iter
            avg_forward_time = self.forward_elapsed / self.config.trainer.logging_iter
            avg_backward_time = self.backward_elapsed / self.config.trainer.logging_iter
            avg_dataloader_time = self.dataloader_elapsed / self.config.trainer.logging_iter
            batch_size = get_data_batch_size(data_batch)

            # throughput = num_floating_point_operations(batch_size) / (
            #    avg_iteration_time * 10**12 * distributed.get_world_size)

            metrics = self.training_telemetry.IterationMetrics.create(
                current_iteration=iteration,
                num_iterations=self.config.trainer.logging_iter,
                interval=self.config.trainer.logging_iter,
                average_iteration_time=avg_iteration_time,
                average_forward_time=avg_forward_time,
                average_backward_time=avg_backward_time,
                average_dataloader_time=avg_dataloader_time,
                tflops=0.0,
                tokens_per_second=0.0,
                loss=loss.item(),
                batch_size=batch_size,
            )
            self.recorder.event(self.training_telemetry.events.Event.create(event_name, metrics))
            self.iteration_elapsed = 0
            self.forward_elapsed = 0
            self.backward_elapsed = 0
            self.dataloader_elapsed = 0

    def on_validation_start(
        self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        span_name = self.training_telemetry.SpanName.VALIDATION_LOOP
        self._start_span(span_name)
        self.validation_loss = 0.0
        self.validation_iter = 0
        self.validation_elapsed = 0.0

    def on_validation_step_start(
        self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        span_name = self.training_telemetry.SpanName.ITERATION
        self._start_span(
            span_name,
            color=self.training_telemetry.SpanColor.RED,
            verbosity=self.training_telemetry.Verbosity.PROFILING,
        )

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self.validation_loss += loss.item()
        self.validation_iter += 1
        span_name = self.training_telemetry.SpanName.ITERATION
        self.validation_elapsed += self._stop_span(span_name)

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if self.validation_iter == 0:
            return
        span_name = self.training_telemetry.SpanName.VALIDATION_LOOP
        avg_validation_time = self.validation_elapsed / self.validation_iter
        metrics = self.training_telemetry.IterationMetrics.create(
            current_iteration=iteration,
            num_iterations=self.validation_iter,
            interval=self.validation_interval,
            average_iteration_time=avg_validation_time,
            loss=self.validation_loss / self.validation_iter,
        )
        self._stop_span(span_name, metrics=metrics)
        self.validation_elapsed = 0.0
        self.validation_loss = 0.0
        self.validation_iter = 0

    def on_load_checkpoint_start(self, model: ImaginaireModel) -> None:
        span_name = self.training_telemetry.SpanName.CHECKPOINT_LOAD
        self._start_span(span_name)

    def on_load_checkpoint_end(
        self, model: ImaginaireModel, iteration: int = 0, checkpoint_path: Optional[str] = None
    ) -> None:
        span_name = self.training_telemetry.SpanName.CHECKPOINT_LOAD
        metrics = self.training_telemetry.CheckpointMetrics.create(
            checkpoint_type=self.training_telemetry.CheckPointType.GLOBAL,
            current_iteration=iteration,
            checkpoint_directory=checkpoint_path,
        )
        self._stop_span(span_name, metrics=metrics)

    def on_load_checkpoint(self, model: ImaginaireModel, state_dict: dict[Any]) -> None:
        pass

    def on_save_checkpoint_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        span_name = (
            self.training_telemetry.SpanName.CHECKPOINT_SAVE_SYNC
            if self.checkpoint_strategy == "sync"
            else self.training_telemetry.SpanName.CHECKPOINT_SAVE_ASYNC
        )
        checkpoint_metrics = self.training_telemetry.CheckpointMetrics.create(
            checkpoint_type=self.training_telemetry.CheckPointType.GLOBAL,
            current_iteration=iteration,
            interval=self.checkpoint_interval,
        )
        self._start_span(span_name, metrics=checkpoint_metrics)

    def on_save_checkpoint_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        span_name = (
            self.training_telemetry.SpanName.CHECKPOINT_SAVE_SYNC
            if self.checkpoint_strategy == "sync"
            else self.training_telemetry.SpanName.CHECKPOINT_SAVE_ASYNC
        )
        self._stop_span(span_name)

    def on_save_checkpoint_success(self, iteration: int = 0, elapsed_time: float = 0) -> None:
        pass

    def on_save_checkpoint(self, model: ImaginaireModel, state_dict: dict[Any]) -> None:
        pass

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        span_name = self.training_telemetry.SpanName.TRAINING_LOOP
        self._stop_span(span_name)

    def on_app_end(self) -> None:
        pass
