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

from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Generator, List, Optional, Union

import numpy as np
import torch

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False

from cosmos.utils import distributed, log

if TYPE_CHECKING:
    from cosmos.model._base import ImaginaireModel


class FastEmaModelUpdater:
    """
    This class is used to update target model~(EMA) given source model~(regular model) and beta.
    The method interaface mimic :class:`EMAModelTracker` and :class:`PowerEMATracker`.
    Different from two classes, this class does not maintain the EMA model weights as buffers. It expects the user to have two module with same architecture and weights shape.
    The class is proposed to work with FSDP model where above two classes are not working as expected. Besides, it is strange to claim model weights as buffers and do unnecessary name changing in :class:`EMAModelTracker` and :class:`PowerEMATracker`. Moeving forward, we should use this class instead of above two classes.
    """

    def __init__(self):
        # Flag to indicate whether the cache is taken or not. Useful to avoid cache overwrite
        self.is_cached = False

    def update_average(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module, beta: float = 0.9999) -> None:
        target_list = []
        source_list = []
        for tgt_params, src_params in zip(tgt_model.parameters(), src_model.parameters()):
            assert tgt_params.dtype == torch.float32, (
                f"EMA model only works in FP32 dtype, got {tgt_params.dtype} instead."
            )
            target_list.append(tgt_params)
            source_list.append(src_params.data)
        torch._foreach_mul_(target_list, beta)
        torch._foreach_add_(target_list, source_list, alpha=1.0 - beta)

    def copy_to(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module) -> None:
        for tgt_params, src_params in zip(tgt_model.parameters(), src_model.parameters()):
            tgt_params.data.copy_(src_params.data)

    def cache(self, parameters: Any, is_cpu: bool = False) -> None:
        """Save the current parameters for restoring later.

        Args:
            parameters (iterable): Iterable of torch.nn.Parameter to be temporarily stored.
        """
        assert self.is_cached is False, "EMA cache is already taken. Did you forget to restore it?"
        device = "cpu" if is_cpu else "cuda"
        self.collected_params = [param.clone().to(device) for param in parameters]
        self.is_cached = True

    def restore(self, parameters: Any) -> None:
        """Restore the parameters in self.collected_params.

        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before copy_to().
        After validation (or model saving), use this to restore the former parameters.

        Args:
            parameters (iterable): Iterable of torch.nn.Parameter to be updated with the stored parameters.
        """
        assert self.is_cached, "EMA cache is not taken yet."
        for c_param, param in zip(self.collected_params, parameters, strict=False):
            param.data.copy_(c_param.data.type_as(param.data))
        self.collected_params = []
        # Release the cache after we call restore
        self.is_cached = False


def get_buffer_name(param_name: str, torch_compile_buffer_renaming: bool = False) -> str:
    """
    This function creates buffer name used by EMA from parameter's name

    Args:
        param_name (str): Model's parameter name
    Returns:
        buffer_name (str): buffer name to be used for given parameter name
    """

    buffer_name = param_name.replace(".", "-")

    if torch_compile_buffer_renaming:
        # torch.compile() adds _orig_mod to state dict names, this way we get original name
        buffer_name = buffer_name.replace("_orig_mod-", "")

    return buffer_name


class EMAModelTracker(torch.nn.Module):
    """This is a class to track the EMA model weights.

    The EMA weights are registered as buffers, which are extractable as state dicts. The names follow those of the
    regular weights, except all "." are replaced with "-" (limitation of register_buffer()). This is similar to SDXL's
    implementation of EMA. There are no optimizable parameters.
    TODO(snah): multi-EMA weights.

    Attributes:
        collected_params (list): temporarily stores the regular weights while in EMA mode.
        beta (float): EMA decay rate. (default: 0.9999).
        torch_compile_buffer_renaming (bool): whether to remove '_orig_mod-' from buffer names when torch.compile is used
    """

    def __init__(self, model: ImaginaireModel, beta: float = 0.9999, torch_compile_buffer_renaming: bool = False):
        """Constructor of the EMA model weight tracker.

        Args:
            model (ImaginaireModel): The PyTorch model.
            beta (float): EMA decay rate. (default: 0.9999).
        """
        super().__init__()
        self.torch_compile_buffer_renaming: bool = torch_compile_buffer_renaming
        if not 0.0 <= beta <= 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.beta = beta
        for name, param in model.named_parameters():
            if param.requires_grad:
                buffer_name = get_buffer_name(name, self.torch_compile_buffer_renaming)
                self.register_buffer(buffer_name, param.clone().detach().data)
        self.collected_params = []
        # Flag to indicate whether the cache is taken or not. Useful to avoid cache overwrite
        self.is_cached = False

    @torch.no_grad()
    def update_average(self, model: ImaginaireModel, iteration: Optional[int] = None) -> None:
        del iteration
        target_list = []
        source_list = []
        ema_buffers = self.state_dict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                buffer_name = get_buffer_name(name, self.torch_compile_buffer_renaming)
                buffer = ema_buffers[buffer_name]
                assert buffer.dtype == torch.float32, f"EMA model only works in FP32 dtype, got {buffer.dtype} instead."
                target_list.append(buffer)
                source_list.append(param.data)
        torch._foreach_mul_(target_list, self.beta)
        torch._foreach_add_(target_list, source_list, alpha=1.0 - self.beta)

    def copy_to(self, model: ImaginaireModel) -> None:
        ema_buffers = self.state_dict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                buffer_name = get_buffer_name(name, self.torch_compile_buffer_renaming)
                buffer = ema_buffers[buffer_name]
                param.data.copy_(buffer.data)

    def cache(self, parameters: Any, is_cpu: bool = False) -> None:
        """Save the current parameters for restoring later.

        Args:
            parameters (iterable): Iterable of torch.nn.Parameter to be temporarily stored.
        """
        assert self.is_cached is False, "EMA cache is already taken. Did you forget to restore it?"
        device = "cpu" if is_cpu else "cuda"
        self.collected_params = [param.clone().to(device) for param in parameters]
        self.is_cached = True

    def restore(self, parameters: Any) -> None:
        """Restore the parameters in self.collected_params.

        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before copy_to().
        After validation (or model saving), use this to restore the former parameters.

        Args:
            parameters (iterable): Iterable of torch.nn.Parameter to be updated with the stored parameters.
        """
        assert self.is_cached, "EMA cache is not taken yet."
        for c_param, param in zip(self.collected_params, parameters, strict=False):
            param.data.copy_(c_param.data.type_as(param.data))
        self.collected_params = []
        # Release the cache after we call restore
        self.is_cached = False

    @classmethod
    def initialize_multi_rank_ema(
        cls, model: torch.nn.Module, rate: Union[float, List[float]], num: int = 1, enabled: bool = True
    ) -> Optional[EMAModelTracker]:
        """
        Class method to initialize per rank EMA Model Tracker with different rate.
        Each rank will have a different rate based on the given configuration, resulting in different EMA weights.

        Args:
            model (torch.nn.Module): The neural network model to be tracked.
            rate (Union[float, List[float]]): The decay rate(s) for the EMA. If a list is provided,
                                              it corresponds to rates for different ranks.
            num (int, optional): The number of leading ranks to consider for different rates.
                                 Defaults to 1.
            enabled (bool, optional): Flag to enable or disable the creation of the tracker.
                                      If False, returns None. Defaults to True.

        Returns:
            Optional[EMAModelTracker]: An instance of EMAModelTracker if enabled, otherwise None.

        Example:
            >>> model = torch.nn.Linear(10, 2)
            >>> tracker = EMAModelTracker.initialize_ema_from_settings(model, rate=[0.1, 0.2], num=2)
            >>> print(tracker)

        Notes:
            If `rate` is a list and the current rank is less than `num`, the rate for the current rank
            is used. If the current rank exceeds `num`, the first rate in the list is used by default.
        """
        if not enabled:
            return None
        if USE_MEGATRON and parallel_state.is_initialized():
            cur_dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
            log.critical(f"using MCore parallel_state for EMA initialization. DP RANK: {cur_dp_rank}", rank0_only=False)
            log.warning("It should not used together with FSDP!")
        else:
            cur_dp_rank = distributed.get_rank()
            log.critical(f"using torch.distributed for EMA initialization. DP RANK: {cur_dp_rank}", rank0_only=False)
        rate = rate if isinstance(rate, list) else [rate]
        num = min(num, len(rate))
        rate = rate[cur_dp_rank] if cur_dp_rank < num else rate[0]
        if cur_dp_rank < num:
            print(f"EMAModelTracker: rank {cur_dp_rank}, rate {rate}")
        return cls(model, rate)


class PowerEMATracker(EMAModelTracker):
    def __init__(self, model: ImaginaireModel, s: float = 0.1, torch_compile_buffer_renaming: bool = False):
        """Constructor of the EMA model weight tracker.

        Args:
            model (ImaginaireModel): The PyTorch model.
            s (float): EMA decay rate. See EDM2 paper
            torch_compile_buffer_renaming (bool): whether to remove '_orig_mod-' from buffer names when torch.compile is used
        """
        super().__init__(model=model, beta=0.0, torch_compile_buffer_renaming=torch_compile_buffer_renaming)
        self.exp = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()

    @torch.no_grad()
    def update_average(self, model: ImaginaireModel, iteration: Optional[int] = None) -> None:
        if iteration == 0:
            beta = 0.0
        else:
            i = iteration + 1
            beta = (1 - 1 / i) ** (self.exp + 1)
        self.beta = beta

        super().update_average(model, iteration)

    @classmethod
    def initialize_multi_rank_ema(
        cls, model: torch.nn.Module, rate: float, num: int, enabled: bool = True
    ) -> Optional[PowerEMATracker]:
        """
        Class method to initialize per rank EMA Model Tracker with different rate.
        Each rank will have a different rate based on the given configuration, resulting in different EMA weights.

        Args:
            model (torch.nn.Module): The neural network model for which the EMA tracker is being set up.
            num (int): The number of ranks for which the rate adjustment is applied. Beyond this, the rate remains unchanged.
            rate (float): The base decay rate for the EMA calculation.
            enabled (bool, optional): Flag to enable or disable the initialization of the tracker. If False, returns None.
                                      Defaults to True.

        Returns:
            Optional[PowerEMATracker]: An instance of PowerEMATracker with adjusted rate if enabled, otherwise None.

        Raises:
            None

        Example:
            >>> model = torch.nn.Linear(10, 2)
            >>> tracker = PowerEMATracker.initialize_multi_rank_ema(model, num=3, rate=0.99)
            >>> print(tracker)

        Notes:
            The decay rate is modified by dividing it by 2 raised to the power of the rank for each rank less than `num`.
            If the rank is greater than or equal to `num`, the base rate is used without modification. This approach
            allows higher ranked processes to have a less aggressive decay, potentially reflecting their delayed synchronization
            in a distributed training scenario.
        """
        if not enabled:
            return None
        if USE_MEGATRON and parallel_state.is_initialized():
            cur_dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
            log.critical(f"using MCore parallel_state for EMA initialization. DP RANK: {cur_dp_rank}", rank0_only=False)
            log.warning("It should not used together with FSDP!")
        else:
            cur_dp_rank = distributed.get_rank()
            log.critical(f"using torch.distributed for EMA initialization. DP RANK: {cur_dp_rank}", rank0_only=False)

        divider = 2**cur_dp_rank if cur_dp_rank < num else 1
        if cur_dp_rank < num:
            print(f"PowerEMATracker: rank {cur_dp_rank}, rate {rate / divider}")
        return cls(model, rate / divider)


@contextmanager
def ema_scope(model: ImaginaireModel, enabled: bool = False, context: str | None = None) -> Generator[None, None, None]:
    """Context manager for switching between regular and EMA model weights.

    This function is a dispatcher that handles two main cases:
    1.  If the model has its own `ema_scope` method, it will be used.
        This allows models to define custom EMA logic (e.g., for FSDP).
    2.  If not, it falls back to a generic mechanism that expects the model
        to have a `.ema` attribute containing an EMA tracker object.

    Args:
        model (ImaginaireModel): The PyTorch model.
        enabled (bool): Whether switching to EMA weights is enabled (default: False).
        context (str | None): A logging context string, passed to the model's ema_scope if used.
    """

    def scope_function():
        if enabled:
            has_custom_scope = hasattr(model, "ema_scope") and callable(model.ema_scope)
            has_generic_ema = hasattr(model, "ema") and isinstance(
                model.ema, (FastEmaModelUpdater, EMAModelTracker, PowerEMATracker)
            )
            assert has_custom_scope or has_generic_ema

            if has_custom_scope:
                return model.ema_scope(context=context)
            else:
                return ema_scope_generic(model)
        else:
            return nullcontext()

    with scope_function():
        yield


@contextmanager
def ema_scope_generic(model: ImaginaireModel) -> Generator[None, None, None]:
    """Generic context manager for switching between regular and EMA model weights.

    Args:
        model (ImaginaireModel): The PyTorch model, which must have a `.ema` attribute.
    """
    model.ema.cache(model.parameters())
    model.ema.copy_to(model)

    log.info("EMA: switched to EMA weights.")
    try:
        yield
    finally:
        model.ema.restore(model.parameters())
        log.info("EMA: restored regular weights.")
