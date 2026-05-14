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
from copy import deepcopy
from typing import Any, Optional

import attrs
import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LambdaLR

from cosmos.utils.config import make_freezable
from cosmos.utils.lazy_config import LazyDict
from cosmos.utils import log
from cosmos.utils.vlm.fused_adam import FusedAdam


@make_freezable
@attrs.define(slots=False)
class OptimizerConfig:
    name: str = "FusedAdam"
    lr: float = 2e-6
    init_lr: float = 1e-7
    end_lr: float = 1e-6
    fused: bool = False
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    lr_multiplier: LazyDict = LazyDict(dict(vision_encoder=0.1, mm_projector=1.0, llm=1.0))

    # model freeze config
    freeze_vision_encoder: bool = False
    freeze_mm_projector: bool = False
    freeze_llm: bool = False
    freeze_llm_moe_gates: bool = False


def _optimizer_cls(params: list[nn.Parameter], optimizer_kwargs: dict[str, Any], name: str):
    if name.lower() == "adam":

        optimizer = torch.optim.Adam(params, **optimizer_kwargs)
    elif name.lower() == "adamw":
        optimizer = torch.optim.AdamW(params, **optimizer_kwargs)
    elif name.lower() == "fusedadam":
        optimizer = FusedAdam(
            params,
            lr=optimizer_kwargs["lr"],
            weight_decay=optimizer_kwargs["weight_decay"],
            betas=optimizer_kwargs["betas"],
            capturable=True,
            master_weights=True,
        )
    else:
        raise NotImplementedError(f"Optimizer {name} not added.")
    return optimizer


class OptimizersContainer(Stateful):
    """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages
    and saving/loading optimizer state_dict at checkpoint.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_kwargs: dict[str, Any],
        name: str,
        lr_multiplier: dict[str, float],
        model_part_names: list[str],
    ) -> None:
        assert len(model_parts) == len(model_part_names), "model_parts and model_part_names must have the same length"
        self.model_parts = model_parts
        self.optimizers = [[] for _ in self.model_parts]
        self.model_part_names = model_part_names
        for model_id, model in enumerate(self.model_parts):
            optimizer_kwargs_copy = deepcopy(optimizer_kwargs)
            optimizer_kwargs_copy["lr"] *= lr_multiplier[model_part_names[model_id]]
            log.info(
                f"model_id: {model_id} | model_part_names: {model_part_names[model_id]} | lr_multiplier: {lr_multiplier[model_part_names[model_id]]} | lr: {optimizer_kwargs_copy['lr']}"
            )

            if optimizer_kwargs_copy["fused"]:
                # Group the parameters by device mesh to do optimizer fusion.
                parameters_by_mesh = collections.defaultdict(list)
                for p in model.parameters():
                    if p.requires_grad:
                        device_mesh = p.device_mesh if hasattr(p, "device_mesh") else "default"
                        parameters_by_mesh[device_mesh].append(p)
                for params in parameters_by_mesh.values():
                    optimizer = _optimizer_cls(params, optimizer_kwargs_copy, name)
                    self.optimizers[model_id].append(optimizer)
            else:
                for p in model.parameters():
                    if p.requires_grad:
                        optimizer = _optimizer_cls([p], optimizer_kwargs_copy, name)
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
        result = {}
        for idx, (model_part, optimizer) in enumerate(zip(self.model_parts, self.optimizers)):
            sd = get_optimizer_state_dict(
                model=model_part,
                optimizers=optimizer,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )
            prefix = f"optimizer_{idx}/"
            result.update({f"{prefix}{k}": v for k, v in sd.items()})
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for idx, (model, optimizers) in enumerate(zip(self.model_parts, self.optimizers)):
            prefix = f"optimizer_{idx}/"
            optimizer_state = {
                k[len(prefix) :]: v  # Remove prefix
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }

            set_optimizer_state_dict(
                model=model,
                optimizers=optimizers,
                optim_state_dict=optimizer_state,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )


# consider split between PP and non-PP
def build_optimizers(
    model_parts: list[nn.Module],
    config: OptimizerConfig,
    model_part_names: list[str],
) -> OptimizersContainer:
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """
    lr_multiplier = config.lr_multiplier
    for part_name in model_part_names:
        assert part_name in lr_multiplier, f"lr_multiplier must have the key {part_name}"

    name = config.name
    lr = config.lr
    fused = config.fused
    optimizer_kwargs = {
        "lr": lr,
        "betas": config.betas,
        "weight_decay": config.weight_decay,
        "fused": fused,
        "foreach": not fused,
    }

    return OptimizersContainer(model_parts, optimizer_kwargs, name, lr_multiplier, model_part_names)


class SchedulersContainer(Stateful):
    """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

    def __init__(self, optimizers: OptimizersContainer, lr_lambda) -> None:
        self.schedulers = []
        for optimizer in optimizers:
            self.schedulers.append(LambdaLR(optimizer, lr_lambda=lr_lambda))

    def step(self) -> None:
        for id, scheduler in enumerate(self.schedulers):
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
