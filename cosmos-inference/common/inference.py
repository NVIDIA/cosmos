# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import dataclasses
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ContextManager, Self, Sequence, final

import torch
import torch.profiler

from cosmos3.common.args import GuardrailArgs, SampleArgs, SampleOutputs, SetupArgs
from cosmos3.common.init import is_rank0
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.misc import TrainingTimer

if TYPE_CHECKING:
    from cosmos3._src.imaginaire.auxiliary.guardrail.common.core import GuardrailRunner


@contextlib.contextmanager
def sync_distributed_errors():
    """Catches local exceptions and synchronizes the error state across all distributed ranks.

    Raises a DistributedError on all ranks if ANY rank encountered an exception.
    """
    error_flag = torch.zeros(1, dtype=torch.int32, device="cuda")  # [1]
    local_error: Exception | None = None

    try:
        yield
    except Exception as e:
        error_flag += 1
        local_error = e

    if torch.distributed.is_initialized():
        # Sync the error count across all GPUs
        torch.distributed.all_reduce(error_flag, op=torch.distributed.ReduceOp.SUM)

    if error_flag.item() > 0:
        # If we got here, somebody failed.
        # Ranks that failed will raise their actual error.
        # Ranks that succeeded will raise a generic error so they gracefully abort too.
        err_to_raise = local_error if local_error else RuntimeError("A different GPU rank failed.")
        raise err_to_raise


@dataclass
class GuardrailRunners:
    text: "GuardrailRunner"
    video: "GuardrailRunner"

    @classmethod
    def create(cls, args: GuardrailArgs, /) -> Self:
        from cosmos3._src.imaginaire.auxiliary.guardrail.common import presets

        return cls(
            text=presets.create_text_guardrail_runner(offload_model_to_cpu=args.offload_guardrail_models),
            video=presets.create_video_guardrail_runner(offload_model_to_cpu=args.offload_guardrail_models),
        )


@dataclass(kw_only=True)
class Inference(ABC):
    """Inference pipeline base class."""

    setup_args: SetupArgs
    model: torch.nn.Module
    guardrails: GuardrailRunners | None

    _timer: TrainingTimer | None
    _timer_context: list[str] = dataclasses.field(default_factory=list)

    @property
    @abstractmethod
    def model_config(self) -> Any:
        """Get model config."""

    @classmethod
    @abstractmethod
    def _create(cls, setup_args: SetupArgs, /, **kwargs: Any) -> Self:
        """Create instance."""

    @abstractmethod
    def create_batches(
        self, sample_args_list: Sequence[SampleArgs]
    ) -> Iterator[tuple[list[SampleArgs], dict[str, Any]]]:
        """Create batches of sample data."""

    @abstractmethod
    def generate_batch(
        self, sample_args_list: Sequence[SampleArgs], data_batch: dict[str, Any], *, warmup: bool = False
    ) -> list[SampleOutputs]:
        """Generate a batch of samples."""

    @final
    @classmethod
    def create(cls, setup_args: SetupArgs, /) -> Self:
        """Create instance."""
        timer = TrainingTimer() if setup_args.benchmark else None
        guardrails = GuardrailRunners.create(setup_args) if setup_args.guardrails else None
        return cls._create(setup_args, guardrails=guardrails, _timer=timer)

    @torch.no_grad()
    @final
    def generate(self, sample_args_list: list[SampleArgs]) -> list[SampleOutputs]:
        """Generate a list of samples."""
        # Create batches
        try:
            with sync_distributed_errors():
                batches = self.create_batches(sample_args_list)
        except Exception as e:
            return [self._handle_sample_exception(sample_args, e) for sample_args in sample_args_list]

        # Generate batches
        sample_outputs: list[SampleOutputs] = []
        for i_batch, (sample_args_batch, data_batch) in enumerate(batches):
            log.debug(f"[{i_batch + 1}] Processing batch", rank0_only=False)

            if self.setup_args.profile:
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                )
            else:
                profiler = contextlib.nullcontext()

            with self._get_timer_context("warmup"):
                for _ in range(self.setup_args.warmup):
                    with self._get_timer(f"{self.__class__.__name__}.generate_batch"):
                        self.generate_batch(sample_args_batch, data_batch, warmup=True)
            with self._get_timer(f"{self.__class__.__name__}.generate_batch"), profiler:
                sample_outputs.extend(self.generate_batch(sample_args_batch, data_batch))

            if self.setup_args.profile and is_rank0():
                assert isinstance(profiler, torch.profiler.profile)
                sample_args = sample_args_batch[0]
                profile_file = sample_args.output_dir / "profile.json.gz"
                profiler.export_chrome_trace(str(profile_file))
                log.success(f"Saved profile to '{profile_file}'")

        return sample_outputs

    def _get_timer(self, func_name: str) -> ContextManager:
        if self._timer is None:
            return nullcontext()
        if self._timer_context:
            context = ".".join(self._timer_context)
            func_name = f"[{context}] {func_name}"
        return self._timer(func_name)

    @contextmanager
    def _get_timer_context(self, func_name: str):
        self._timer_context.append(func_name)
        try:
            yield
        finally:
            self._timer_context.pop()

    def get_timer_results(self) -> dict | None:
        if self._timer is None:
            return None
        return {
            "all": self._timer.results,
            "average": self._timer.compute_average_results(),
        }

    def _handle_sample_exception(self, sample_args: SampleArgs, e: Exception) -> SampleOutputs:
        msg = f"Error generating sample '{sample_args.name}': {e}"
        if not self.setup_args.keep_going:
            raise ValueError(msg) from e
        log.error(msg)
        return SampleOutputs(
            args=sample_args.model_dump(mode="json"), status="error", message=msg, stack_trace=traceback.format_exc()
        )

    @final
    def _run_text_guardrail(self, name: str, prompt: str) -> None:
        """Run guardrail checks on the prompt."""
        if self.guardrails is None:
            return

        from cosmos3._src.imaginaire.auxiliary.guardrail.common import presets

        if not presets.run_text_guardrail(prompt, self.guardrails.text):
            raise ValueError(f"Guardrail blocked prompt '{name}': '{prompt}'")

    @final
    def _run_video_guardrail(self, name: str, video_cthw: torch.Tensor) -> torch.Tensor:
        """Run guardrail checks on the video and apply face blur."""
        if self.guardrails is None:
            return video_cthw
        processed_video_cthw, message = _run_video_guardrail(self.guardrails.video, video_cthw)
        if processed_video_cthw is None:
            raise ValueError(f"Guardrail blocked video '{name}': {message}")
        return processed_video_cthw


def _run_video_guardrail(
    video_guardrail_runner: "GuardrailRunner", video_cthw: torch.Tensor
) -> tuple[torch.Tensor | None, str]:
    """Run video guardrail and apply face blur.

    Returns a ``(video_or_none, message)`` tuple. When the guardrail blocks
    the video, ``video_or_none`` is ``None`` and ``message`` contains the
    underlying reason (unsafe frame ratio, categories, etc.) as produced by
    :class:`GuardrailRunner.run_safety_check`.
    """
    if video_cthw.ndim != 4:
        raise ValueError(f"Video tensor must have 4 dimensions, got {video_cthw.shape}")
    frames_thwc = (
        (video_cthw * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 2, 3, 0).detach().cpu().numpy()
    )  # [T,H,W,C]

    # Inline of presets.run_video_guardrail so we can forward `message` (the helper drops it).
    is_safe, message = video_guardrail_runner.run_safety_check(frames_thwc)
    if not is_safe:
        log.critical(f"GUARDRAIL BLOCKED: {message}")
        return None, message

    frames_thwc = video_guardrail_runner.postprocess(frames_thwc)
    video_cthw = (torch.from_numpy(frames_thwc).float().permute(3, 0, 1, 2) / 255.0).to(  # [C,T,H,W]
        video_cthw.device, dtype=video_cthw.dtype
    )
    return video_cthw, message
