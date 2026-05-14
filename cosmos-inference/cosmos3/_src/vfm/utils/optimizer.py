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

import collections
import functools
import itertools
import math
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LambdaLR

from cosmos3._src.imaginaire.utils import log


def _convert_omegaconf_to_python(obj: Any) -> Any:
    """Convert OmegaConf types to plain Python types.

    This is needed because PyTorch's checkpoint utilities don't handle
    OmegaConf types like ListConfig and DictConfig.
    """
    if isinstance(obj, (ListConfig, DictConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def _optimizer_cls(
    params: list[nn.Parameter] | list[dict[str, Any]],
    optimizer_type: str,
    **optimizer_kwargs: Any,
) -> torch.optim.Optimizer:
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(params, **optimizer_kwargs)
    elif optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(params, **optimizer_kwargs)
    elif optimizer_type.lower() == "fusedadam":
        from cosmos3._src.vfm.utils.fused_adam import FusedAdam

        optimizer = FusedAdam(params, capturable=True, master_weights=True, **optimizer_kwargs)
    else:
        raise NotImplementedError(f"Optimizer {optimizer_type} not found.")
    return optimizer


def _filter_params(net: nn.Module, keys_to_select: list[str] = []) -> list[nn.Parameter]:
    """
    Filter the parameters of the network based on the keys to select.
    For the parameters that are not in the keys_to_select, we set requires_grad to False.
    """
    total_params = sum(1 for _, _ in net.named_parameters())
    param_dict = {pn: p for pn, p in net.named_parameters() if p.requires_grad}
    params_filtered = []

    if len(keys_to_select) > 0:
        for pn, p in param_dict.items():
            if any([key_to_select in pn for key_to_select in keys_to_select]):
                params_filtered.append(p)
            else:
                p.requires_grad = False
    else:
        params_filtered = list(param_dict.values())

    log.info(
        f"Total parameters: {total_params}, "
        f"trainable parameters: {len(params_filtered)}, "
        f"frozen parameters: {total_params - len(params_filtered)}, "
        f"unselected parameters: {len(param_dict) - len(params_filtered)}"
    )
    return params_filtered


def _build_optimizer_param_groups(
    net: nn.Module,
    keys_to_select: list[str],
    lr_multipliers: dict[str, float],
    base_lr: float,
    disable_weight_decay_for_1d_params: bool,
) -> list[dict[str, Any]]:
    """Build optimizer parameter groups after applying selection and grouping rules.

    Parameters not matching ``keys_to_select`` are frozen by setting
    ``requires_grad=False``. Selected parameters are grouped by matching
    ``lr_multipliers`` pattern and each group receives ``base_lr * multiplier``.
    If ``disable_weight_decay_for_1d_params`` is true, each LR group is split
    into weight-decay and no-weight-decay groups; one-dimensional parameters,
    such as norm weights and biases, are assigned ``weight_decay=0.0``.
    """
    total_params = sum(1 for _ in net.parameters())
    param_dict = {pn: p for pn, p in net.named_parameters() if p.requires_grad}

    multiplier_groups: dict[float, list[nn.Parameter]] = collections.defaultdict(list)

    for pn, p in param_dict.items():
        if len(keys_to_select) > 0 and not any(key in pn for key in keys_to_select):
            p.requires_grad = False
            continue
        matched_mult = 1.0
        for pattern, mult in lr_multipliers.items():
            if pattern in pn:
                matched_mult = mult
                break
        multiplier_groups[matched_mult].append(p)

    trainable = sum(len(g) for g in multiplier_groups.values())
    log.info(
        f"Total parameters: {total_params}, "
        f"trainable parameters: {trainable}, "
        f"frozen parameters: {total_params - trainable}, "
        f"unselected parameters: {len(param_dict) - trainable}"
    )

    optimizer_param_groups: list[dict[str, Any]] = []
    for mult, params in sorted(multiplier_groups.items()):
        if not params:
            continue

        lr = base_lr * mult
        if disable_weight_decay_for_1d_params:
            decay_params = [p for p in params if p.dim() >= 2]
            no_decay_params = [p for p in params if p.dim() < 2]
            param_groups: list[dict[str, Any]] = []
            if decay_params:
                param_groups.append({"params": decay_params, "lr": lr})
            if no_decay_params:
                param_groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
        else:
            param_groups = [{"params": params, "lr": lr}]

        optimizer_param_groups.extend(param_groups)
        num_tensors_with_weight_decay = sum(
            len(group["params"]) for group in param_groups if group.get("weight_decay") != 0.0
        )
        num_tensors_without_weight_decay = sum(
            len(group["params"]) for group in param_groups if group.get("weight_decay") == 0.0
        )
        log.info(
            f"Param group (lr_mult={mult}x): "
            f"{len(params)} tensors, "
            f"{sum(p.numel() for p in params)} parameters, "
            f"lr={lr}, "
            f"num_tensors_with_weight_decay={num_tensors_with_weight_decay}, "
            f"num_tensors_without_weight_decay={num_tensors_without_weight_decay}, "
            f"disable_weight_decay_for_1d_params={disable_weight_decay_for_1d_params}"
        )

    return optimizer_param_groups


def build_optimizer(
    model: nn.Module,
    optimizer_type: str,
    **optimizer_kwargs: Any,
) -> torch.optim.Optimizer:
    """Build an optimizer for a model.

    Args:
        model: The model to build an optimizer for.
        optimizer_type: The type of optimizer to build.
        **optimizer_kwargs: Additional keyword arguments to pass to the optimizer.
            lr_multipliers: Optional dict mapping parameter name patterns to LR
                multipliers. E.g. ``{"sound2llm": 5.0, "llm2sound": 5.0}`` gives those
                params 5x the base LR. Unmatched selected params use multiplier 1.0.
            disable_weight_decay_for_1d_params: If true, one-dimensional parameters
                such as norm weights and biases use weight_decay=0.0. Defaults to
                false to preserve the historical optimizer behavior.

    Returns:
        A torch.optim.Optimizer.
    """
    # Convert OmegaConf types to plain Python types to avoid issues with checkpoint saving.
    # PyTorch's _copy_state_dict doesn't handle OmegaConf types like ListConfig.
    optimizer_kwargs = {k: _convert_omegaconf_to_python(v) for k, v in optimizer_kwargs.items()}

    fused = optimizer_kwargs.pop("fused", False)
    assert fused, "Optimizers with fused=False are not supported."
    keys_to_select = optimizer_kwargs.pop("keys_to_select", [])
    lr_multipliers: dict[str, float] = optimizer_kwargs.pop("lr_multipliers", {})
    disable_weight_decay_for_1d_params = optimizer_kwargs.pop("disable_weight_decay_for_1d_params", False)

    base_lr = optimizer_kwargs["lr"]
    optimizer_param_groups = _build_optimizer_param_groups(
        model,
        keys_to_select,
        lr_multipliers,
        base_lr,
        disable_weight_decay_for_1d_params,
    )
    return _optimizer_cls(optimizer_param_groups, optimizer_type, **optimizer_kwargs)


class OptimizersContainer(Stateful):
    """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages
    and saving/loading optimizer state_dict at checkpoint.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_type: str,
        **optimizer_kwargs: Any,
    ) -> None:
        self.model_parts = model_parts
        self.optimizers = [[] for _ in self.model_parts]

        fused = optimizer_kwargs.pop("fused", False)
        keys_to_select = optimizer_kwargs.pop("keys_to_select", [])

        for model_id, model in enumerate(self.model_parts):
            filtered_params = _filter_params(model, keys_to_select)

            if fused:
                # Group the parameters by device mesh to do optimizer fusion.
                parameters_by_mesh = collections.defaultdict(list)
                for p in filtered_params:
                    device_mesh = p.device_mesh if hasattr(p, "device_mesh") else "default"
                    parameters_by_mesh[device_mesh].append(p)
                for params in parameters_by_mesh.values():
                    optimizer = _optimizer_cls(params, optimizer_type, **optimizer_kwargs)
                    self.optimizers[model_id].append(optimizer)
            else:
                for p in filtered_params:
                    optimizer = _optimizer_cls([p], optimizer_type, **optimizer_kwargs)
                    self.optimizers[model_id].append(optimizer)

    def __iter__(self) -> torch.optim.Optimizer:
        return iter(itertools.chain(*self.optimizers))

    def step(self) -> None:
        for optimizer in itertools.chain(*self.optimizers):
            optimizer.step()

    def zero_grad(self, set_to_none: bool = False) -> None:
        for optimizer in itertools.chain(*self.optimizers):
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        sd = {}
        for model, optimizers in zip(self.model_parts, self.optimizers):
            sd.update(
                get_optimizer_state_dict(
                    model=model,
                    optimizers=optimizers,
                    options=StateDictOptions(flatten_optimizer_state_dict=True),
                )
            )
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for model, optimizers in zip(self.model_parts, self.optimizers):
            set_optimizer_state_dict(
                model=model,
                optimizers=optimizers,
                optim_state_dict=state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )


# consider split between PP and non-PP
def build_optimizers(
    model_parts: list[nn.Module],
    optimizer_type: str,
    **optimizer_kwargs: dict[str, Any],
) -> OptimizersContainer:
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """
    return OptimizersContainer(model_parts, optimizer_type, **optimizer_kwargs)


class SchedulersContainer(Stateful):
    """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

    def __init__(self, optimizers: OptimizersContainer, lr_lambda) -> None:
        self.schedulers = []
        for optimizer in optimizers:
            self.schedulers.append(LambdaLR(optimizer, lr_lambda=lr_lambda))

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        # Currently, we have one scheduler per optimizer. However, when using MultiSchedule PP or optimizer-in-backward,
        # there are multiple optimizers and schedulers, but the scheduler state_dict remains the same for all.
        # Therefore, we only save the first one and later load it for all.
        assert len(self.schedulers) > 0, "Must have at least one scheduler to save state_dict"
        return self.schedulers[0].state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Load the same state_dict for all schedulers. The key value we're concerned with in scheduler.state_dict() is `last_epoch`,
        # which is an integer that will be automatically copied. As long as `training.steps` and `training.warmup_iters` remain
        # unchanged when resuming from a checkpoint, this approach is safe. We call `.copy()` here to ensure extra safety.
        last_epoch = state_dict["last_epoch"]  # Extract last known epoch
        _step_count = state_dict["_step_count"]
        log.info(f"Resuming schedulers by stepping them to last_epoch: {last_epoch}; _step_count: {_step_count}")

        # Manually step all schedulers to match the saved state -- this is a workaround for the inherited issue in the state dict saving (only saved the first scheduler)
        # But we have different learning rate for each scheduler, so we need to step them separately instead of loading the state dict
        # The benefit of this approach is that we can resume from a checkpoint even if the learning rate is changed
        for idx, scheduler in enumerate(self.schedulers):
            for step in range(_step_count):
                scheduler.step()  # Step forward to match previous training state
            log.info(f"Scheduler {idx + 1}/{len(self.schedulers)} stepped {_step_count} times.")
            log.info(f"Updated learning rate: {scheduler.get_last_lr()}")

    def get_last_lr(self) -> list[float]:
        return [scheduler.get_last_lr() for scheduler in self.schedulers]


def linear_warmup_linear_decay(warmup_iters: int, decay_steps: int, current_step: int) -> float:
    """Computes linear warmup followed by linear decay.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    if current_step < warmup_iters:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (warmup_iters + 1))

    else:
        # linear decay
        normalized_step = decay_steps - (current_step - warmup_iters)
        curr_adjustment = 1 - (decay_steps - normalized_step) / decay_steps

    return curr_adjustment


def linear_warmup(warmup_iters: int, current_step: int) -> float:
    """Computes linear warmup only
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    if current_step < warmup_iters:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (warmup_iters + 1))
    else:
        curr_adjustment = 1

    return curr_adjustment


def linear_warmup_cosine_cooldown(
    warmup_iters: int, cooldown_steps: int, current_step: int, base_lr: float, init_lr: float, end_lr: float
) -> float:
    """This scheduler will warmup the learning rate from init_lr to base_lr for warmup_iters,
    then decay the learning rate from base_lr to end_lr for cooldown_steps. After cooldown_steps + warmup_iters,
    the learning rate will be set to end_lr.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.

    Args:
        warmup_iters (int): The number of steps to warmup the learning rate.
        cooldown_steps (int): The number of steps to decay the learning rate.
        current_step (int): The current step.
        base_lr (float): The base learning rate.
        init_lr (float): The initial learning rate before warmup.
        end_lr (float): The final learning rate after cooldown.

    Returns:
        float: The multiplicative factor to adjust the learning rate.
    """
    total_steps = warmup_iters + cooldown_steps

    # Normalize
    init_multiplier = init_lr / base_lr
    end_multiplier = end_lr / base_lr
    if current_step <= warmup_iters:
        progress = float(current_step / warmup_iters)
        return init_multiplier + (1.0 - init_multiplier) * progress
    elif current_step <= total_steps:
        progress = (current_step - warmup_iters) / cooldown_steps
        return end_multiplier + 0.5 * (1.0 - end_multiplier) * (1 + math.cos(math.pi * progress))
    else:
        return end_multiplier


def build_lr_schedulers(
    optimizers: OptimizersContainer,
    name: str,
    lr: float,
    warmup_iters: int,
    lr_decay_iters: Optional[int] = None,
    init_lr: Optional[float] = None,
    end_lr: Optional[float] = None,
) -> SchedulersContainer:
    decay_steps = float(max(1, lr_decay_iters - warmup_iters)) if lr_decay_iters is not None else None
    if name == "warmup_cosine_lr":
        assert init_lr is not None and end_lr is not None, "init_lr and end_lr must be provided for warmup_cosine_lr"
        assert lr_decay_iters is not None, "lr_decay_iters must be provided for warmup_cosine_lr"
        lr_lambda = functools.partial(
            linear_warmup_cosine_cooldown,
            warmup_iters,
            decay_steps,
            base_lr=lr,
            init_lr=init_lr,
            end_lr=end_lr,
        )
    elif name == "lambdalinear":
        assert lr_decay_iters is not None, "lr_decay_iters must be provided for lambdalinear"
        lr_lambda = functools.partial(linear_warmup_linear_decay, warmup_iters, decay_steps)
    else:
        lr_lambda = functools.partial(linear_warmup, warmup_iters)

    return SchedulersContainer(optimizers, lr_lambda)
