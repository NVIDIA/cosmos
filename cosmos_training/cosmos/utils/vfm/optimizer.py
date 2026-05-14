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
import copy
from typing import Any, Iterator, NamedTuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from cosmos.utils.functional.lr_scheduler import LambdaLinearScheduler, LambdaWarmUpCosineScheduler
from cosmos.utils import log


class ParamMetadata(NamedTuple):
    lr: float
    enable_weight_decay: bool


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
    """Dispatch to the requested optimizer class with the right kwargs.

    Supported ``optimizer_type`` values (case-insensitive):

    - ``"adam"`` / ``"adamw"``: forwarded to ``torch.optim.Adam`` /
      ``torch.optim.AdamW``.  ``fused`` (if present in ``optimizer_kwargs``)
      flows through and selects the fused CUDA kernel.
    - ``"fusedadam"``: NVIDIA's :class:`projects.cosmos3.vfm.utils.fused_adam.FusedAdam`.
      It is fused by construction and rejects a ``fused`` kwarg, so any
      ``fused`` entry is popped before instantiation.  We also force
      ``capturable=True`` and ``master_weights=True`` because those are the
      only settings exercised in our distributed training stack.

    Raises ``NotImplementedError`` for any other ``optimizer_type``.
    """
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(params, **optimizer_kwargs)
    elif optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(params, **optimizer_kwargs)
    elif optimizer_type.lower() == "fusedadam":
        from cosmos.utils.vfm.fused_adam import FusedAdam

        # FusedAdam is fused by construction and does not accept a ``fused`` kwarg.
        optimizer_kwargs.pop("fused", None)
        optimizer = FusedAdam(params, capturable=True, master_weights=True, **optimizer_kwargs)
    else:
        raise NotImplementedError(f"Optimizer {optimizer_type} not found.")
    return optimizer


def _build_params_with_metadata(
    net: nn.Module,
    keys_to_select: list[str],
    lr_multipliers: dict[str, float],
    base_lr: float,
    disable_weight_decay_for_1d_params: bool,
) -> list[tuple[nn.Parameter, ParamMetadata]]:
    """Filter trainable parameters and tag each with its effective LR and weight-decay flag.

    Walks ``net.named_parameters()`` once and produces one
    ``(param, ParamMetadata)`` entry per parameter that survives the
    ``keys_to_select`` filter.  Has the side effect of freezing
    (``requires_grad=False``) every parameter that does NOT match
    ``keys_to_select`` when that list is non-empty.

    Selection rule:
        - If ``keys_to_select`` is empty, every parameter is kept.
        - Otherwise, a parameter is kept iff its dotted name contains at least
          one entry of ``keys_to_select`` as a substring.

    Effective LR:
        ``ParamMetadata.lr = base_lr * matched_multiplier`` where
        ``matched_multiplier`` is the value associated with the first
        ``lr_multipliers`` pattern whose key occurs as a substring in the
        parameter name (iteration order of the dict is significant); defaults
        to ``1.0`` if no pattern matches.

    Weight-decay flag:
        ``ParamMetadata.enable_weight_decay`` is ``False`` exactly when
        ``disable_weight_decay_for_1d_params`` is True AND the parameter has
        fewer than two dimensions (the standard heuristic for norm weights,
        biases, and other 1-D tensors).  Otherwise it is ``True``.  The flag is
        consumed downstream by :func:`_build_optimizer_internal`, which
        materializes the ``weight_decay=0.0`` override on the corresponding
        param group.

    Args:
        net: Module whose ``named_parameters()`` to scan.
        keys_to_select: Substrings used to allowlist parameter names.  Empty
            list = train everything.
        lr_multipliers: Ordered mapping from name-substring to LR multiplier.
            First match wins per parameter.
        base_lr: Base learning rate; each parameter's effective LR is
            ``base_lr * matched_multiplier``.
        disable_weight_decay_for_1d_params: When ``True``, 1-D parameters are
            tagged ``enable_weight_decay=False``.

    Returns:
        List of ``(nn.Parameter, ParamMetadata)`` pairs covering every kept
        parameter.  Frozen parameters are not included.
    """
    param_dict = {pn: p for pn, p in net.named_parameters() if p.requires_grad}
    total_tensors = sum(1 for _ in net.parameters())
    total_elements = sum(p.numel() for p in net.parameters())
    trainable_elements = sum(p.numel() for p in param_dict.values())
    log.info(
        f"Parameters: {total_elements:,} ({total_tensors} tensors), "
        f"trainable: {trainable_elements:,} ({len(param_dict)} tensors)"
    )

    params_with_metadata: list[tuple[nn.Parameter, ParamMetadata]] = []

    for pn, p in param_dict.items():
        if len(keys_to_select) > 0 and not any(key in pn for key in keys_to_select):
            p.requires_grad = False
            continue

        # Find the matching multiplier for the parameter.
        matched_mult = 1.0
        for pattern, mult in lr_multipliers.items():
            if pattern in pn:
                matched_mult = mult
                break

        if disable_weight_decay_for_1d_params and p.dim() < 2:
            enable_weight_decay = False
        else:
            enable_weight_decay = True

        params_with_metadata.append(
            (
                p,
                ParamMetadata(
                    lr=base_lr * matched_mult,
                    enable_weight_decay=enable_weight_decay,
                ),
            )
        )
    return params_with_metadata


def _build_optimizer_internal(
    params_with_metadata: list[tuple[nn.Parameter, ParamMetadata]],
    optimizer_type: str,
    **optimizer_kwargs: Any,
) -> torch.optim.Optimizer:
    """Bucket per-parameter metadata into PyTorch param groups and instantiate the optimizer.

    Parameters sharing the same ``ParamMetadata`` (i.e. identical effective LR
    *and* identical weight-decay enablement) are collapsed into a single
    ``torch.optim`` param group.  Each resulting group carries an explicit
    ``"lr"``; groups whose metadata has ``enable_weight_decay=False`` also
    receive an explicit ``"weight_decay": 0.0`` override that suppresses the
    optimizer-wide ``weight_decay`` kwarg for those parameters only.

    Args:
        params_with_metadata: Output of :func:`_build_params_with_metadata` —
            one ``(param, ParamMetadata)`` entry per trainable parameter.
        optimizer_type: Optimizer kind string accepted by :func:`_optimizer_cls`
            (``"adam"``, ``"adamw"``, or ``"fusedadam"``).
        **optimizer_kwargs: Forwarded to the optimizer constructor.  ``"lr"`` is
            ignored at the group level (each group sets its own LR) but must
            still be present because PyTorch optimizers require a default.

    Returns:
        A ``torch.optim.Optimizer`` configured with one param group per distinct
        ``ParamMetadata`` value found in ``params_with_metadata``.
    """
    params_by_metadata: dict[ParamMetadata, list[nn.Parameter]] = collections.defaultdict(list)
    for param, metadata in params_with_metadata:
        params_by_metadata[metadata].append(param)

    for metadata, params in params_by_metadata.items():
        log.info(
            f"Param group (lr={metadata.lr}, WD={metadata.enable_weight_decay}): "
            f"{len(params):,} tensors, {sum(p.numel() for p in params):,} elements"
        )

    param_groups: list[dict[str, Any]] = []
    for metadata, params in params_by_metadata.items():
        param_group: dict[str, Any] = {"params": params, "lr": metadata.lr}
        if not metadata.enable_weight_decay:
            param_group["weight_decay"] = 0.0
        param_groups.append(param_group)

    return _optimizer_cls(param_groups, optimizer_type, **optimizer_kwargs)


class OptimizersContainer(Stateful):
    """
    Utility for calling step/zero_grad on multiple optimizers, and
    saving/loading optimizer state_dict at checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_type: str,
        **optimizer_kwargs: Any,
    ) -> None:
        """Build per-device-mesh optimizers and expose them as a single Stateful container.

        Parameters are first filtered + tagged with metadata via
        :func:`_build_params_with_metadata` and then sub-grouped by
        ``param.device_mesh.mesh_dim_names`` (or ``"default"`` when no mesh is
        attached).  One ``torch.optim`` instance is created per resulting mesh
        bucket so fused kernels see only same-mesh DTensors — required for
        FSDP2-aware optimizer state-dict resharding.

        Args:
            model: Model to build the optimizer(s) for.  ``model.parameters()``
                provides the parameter universe and is also passed through to
                ``get_optimizer_state_dict`` / ``set_optimizer_state_dict`` for
                checkpointing.
            optimizer_type: Optimizer kind string accepted by
                :func:`_optimizer_cls` (``"adam"``, ``"adamw"``, ``"fusedadam"``).
            **optimizer_kwargs: Forwarded to the optimizer constructor, with the
                following container-level kwargs intercepted first:

                - ``fused`` (required): Must be truthy. Forwarded to
                  ``torch.optim.AdamW`` / ``Adam`` so they use the fused CUDA
                  kernel; ignored for ``"fusedadam"`` which is fused by
                  construction.  Eager optimizers are not supported.
                - ``lr`` (required): Base learning rate.  Each param group's
                  effective LR is ``lr * matched_multiplier``; ``lr`` is still
                  required because PyTorch optimizers expect a default.
                - ``keys_to_select`` (optional, default ``[]``): List of
                  substrings; only params whose names contain at least one are
                  trained.  Empty list = train every parameter.
                - ``lr_multipliers`` (optional, default ``{}``): Ordered dict
                  mapping param-name substrings to LR multipliers.  First match
                  wins; unmatched selected params use multiplier ``1.0``.
                - ``disable_weight_decay_for_1d_params`` (optional, default
                  ``False``): When true, parameters with ``dim() < 2`` (norm
                  weights, biases, etc.) get ``weight_decay=0.0`` in their
                  param group regardless of the optimizer-wide ``weight_decay``.
        """
        self.model = model
        self.optimizers: list[torch.optim.Optimizer] = []

        # Pop only the container-level kwargs.  ``fused`` and ``lr`` are
        # intentionally NOT popped: ``fused`` must flow through to
        # ``torch.optim.AdamW`` / ``Adam`` so the fused CUDA kernel is actually
        # selected, and ``lr`` is needed both here (as the base for per-group
        # multipliers) and by the underlying optimizer (as the group-level
        # default).
        keys_to_select = optimizer_kwargs.pop("keys_to_select", [])
        lr_multipliers: dict[str, float] = optimizer_kwargs.pop("lr_multipliers", {})
        disable_weight_decay_for_1d_params = optimizer_kwargs.pop("disable_weight_decay_for_1d_params", False)

        if not optimizer_kwargs.get("fused", False):
            raise ValueError("Optimizers with fused=False are not supported; pass fused=True in optimizer_kwargs.")
        if "lr" not in optimizer_kwargs:
            raise ValueError("`lr` is required in optimizer_kwargs (used as the base for per-group LR multipliers).")

        base_lr = optimizer_kwargs["lr"]
        params_with_metadata = _build_params_with_metadata(
            model,
            keys_to_select=keys_to_select,
            lr_multipliers=lr_multipliers,
            base_lr=base_lr,
            disable_weight_decay_for_1d_params=disable_weight_decay_for_1d_params,
        )

        # Sub-group by device mesh so fused optimizers operate on same-mesh params.
        mesh_groups: dict[str, list[tuple[nn.Parameter, ParamMetadata]]] = collections.defaultdict(list)
        for param, metadata in params_with_metadata:
            if hasattr(param, "device_mesh"):
                # ``mesh_dim_names`` is ``tuple[str, ...] | None`` on DeviceMesh —
                # fall back to ``default`` when names weren't assigned.
                names = param.device_mesh.mesh_dim_names
                mesh_key = "-".join(names) if names else "default"
            else:
                mesh_key = "default"
            mesh_groups[mesh_key].append((param, metadata))

        # Create one optimizer per mesh, each with per-LR,weight-decay param groups.
        for mesh_key, mesh_params in mesh_groups.items():
            log.info(f"Building optimizer for mesh '{mesh_key}'")
            optimizer = _build_optimizer_internal(
                mesh_params,
                optimizer_type,
                **optimizer_kwargs,
            )
            self.optimizers.append(optimizer)

        log.info(f"Created {len(self.optimizers)} optimizers")

    def __iter__(self) -> Iterator[torch.optim.Optimizer]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        # Default matches ``torch.optim.Optimizer.zero_grad`` (set_to_none=True
        # since PyTorch 1.7); zero-in-place is strictly slower and uses extra
        # memory, so callers must opt-in explicitly.
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return get_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizers,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizers,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )


def build_optimizer(
    model: nn.Module,
    optimizer_type: str,
    **optimizer_kwargs: Any,
) -> "OptimizersContainer":
    """Build an :class:`OptimizersContainer` for a model.

    Returns the container — NOT a raw ``torch.optim.Optimizer`` — because the
    underlying implementation creates one optimizer per device mesh.  Callers
    that need PyTorch ``param_groups`` must iterate ``container.optimizers``.

    Before instantiation, every kwarg is normalized via
    :func:`_convert_omegaconf_to_python` so that OmegaConf containers
    (``ListConfig`` / ``DictConfig``) become plain Python types — PyTorch's
    checkpoint utilities ``_copy_state_dict`` does not handle OmegaConf types.

    Args:
        model: The model to build an optimizer for.  Trainable-parameter
            selection happens inside the container; parameters not matched by
            ``keys_to_select`` (when non-empty) are frozen as a side effect.
        optimizer_type: Optimizer kind string accepted by :func:`_optimizer_cls`
            (``"adam"``, ``"adamw"``, or ``"fusedadam"``).
        **optimizer_kwargs: Forwarded to :class:`OptimizersContainer`, which
            intercepts a few container-level kwargs and forwards the rest to
            the underlying ``torch.optim`` constructor.  Container-level kwargs:

            - ``fused`` (required, truthy): Use the fused CUDA kernel.  Eager
              optimizers are intentionally not supported.
            - ``lr`` (required): Base learning rate.  Each param group's
              effective LR is ``lr * matched_multiplier``.
            - ``keys_to_select`` (optional, default ``[]``): Substrings used to
              allowlist parameter names; empty list = train everything.
            - ``lr_multipliers`` (optional, default ``{}``): Ordered dict
              mapping parameter-name substrings to LR multipliers.  E.g.
              ``{"sound2llm": 5.0, "llm2sound": 5.0}`` gives those params 5x
              the base LR.  First match wins; unmatched selected params use
              multiplier ``1.0``.
            - ``disable_weight_decay_for_1d_params`` (optional, default
              ``False``): If true, one-dimensional parameters such as norm
              weights and biases get ``weight_decay=0.0`` in their param group
              regardless of the optimizer-wide ``weight_decay``.

    Returns:
        An :class:`OptimizersContainer` wrapping one ``torch.optim.Optimizer``
        per device mesh discovered on the model's parameters.
    """
    # Convert OmegaConf types to plain Python types to avoid issues with checkpoint saving.
    # PyTorch's _copy_state_dict doesn't handle OmegaConf types like ListConfig.
    optimizer_kwargs = {k: _convert_omegaconf_to_python(v) for k, v in optimizer_kwargs.items()}
    return OptimizersContainer(model, optimizer_type, **optimizer_kwargs)


def _lr_scheduler_cls(
    lr_scheduler_type: str,
    **lr_scheduler_kwargs: Any,
) -> LambdaLinearScheduler | LambdaWarmUpCosineScheduler:
    """Instantiate a lambda-style scheduler whose ``.schedule(step)`` returns an LR multiplier.

    Both returned classes expose a ``schedule(step) -> float`` callable that
    :class:`LRSchedulersContainer` wraps with ``torch.optim.lr_scheduler.LambdaLR``
    to drive each optimizer's param-group LRs.  ``lr_scheduler_type`` matching is
    case-insensitive; valid values are ``"lambdalinear"`` (linear decay) and
    ``"lambdacosine"`` (warmup + cosine decay).  Any other value raises
    ``NotImplementedError``.  All remaining ``**lr_scheduler_kwargs`` are
    forwarded verbatim to the underlying scheduler constructor (e.g.
    ``warm_up_steps``, ``cycle_lengths``, ``f_start``, ``f_max``, ``f_min``,
    ``verbosity_interval``).
    """
    if lr_scheduler_type.lower() == "lambdalinear":
        lr_scheduler = LambdaLinearScheduler(**lr_scheduler_kwargs)
    elif lr_scheduler_type.lower() == "lambdacosine":
        lr_scheduler = LambdaWarmUpCosineScheduler(**lr_scheduler_kwargs)
    else:
        raise NotImplementedError(f"LR Scheduler {lr_scheduler_type} not found.")
    return lr_scheduler


class LRSchedulersContainer(Stateful):
    """Wraps one ``torch.optim.lr_scheduler.LambdaLR`` per inner optimizer.

    Mirrors the structure of :class:`OptimizersContainer`: one scheduler is
    created per element of an ``OptimizersContainer`` (i.e. one per device
    mesh), and ``step()`` / ``state_dict()`` / ``load_state_dict()`` fan out
    over them.  All schedulers share the same underlying lambda
    (``lr_scheduler.schedule``), so they advance in lock-step but multiply each
    of their optimizer's per-group ``base_lr`` values independently.

    Note:
        Subclasses must keep the ``step``, ``state_dict``, and
        ``load_state_dict`` signatures compatible with
        ``torch.optim.lr_scheduler.LRScheduler`` so that downstream training
        and checkpoint code can treat the container as a drop-in scheduler.

    Checkpoint format:
        ``LambdaLR.state_dict`` contains both scalar fields (``last_epoch``,
        ``_step_count``, ``verbose``, ``_get_lr_called_within_step`` — all
        identical across schedulers because they step in lock-step) and
        per-param-group lists (``base_lrs``, ``_last_lr``, ``lr_lambdas`` —
        whose length and values depend on the underlying optimizer's
        ``param_groups``).

        When ``len(self.schedulers) == 1`` (the common single-mesh case),
        :meth:`state_dict` returns scheduler-0's dict verbatim.  This matches
        the historical on-disk shape, so single-scheduler checkpoints written
        by earlier versions of this class continue to round-trip.

        When ``len(self.schedulers) > 1`` (heterogeneous mesh case), the
        per-scheduler list fields differ in length and content across
        schedulers.  :meth:`state_dict` augments scheduler-0's dict with a
        ``"schedulers"`` entry — a ``list[dict]`` of per-scheduler
        ``LambdaLR.state_dict`` values — so each scheduler's
        ``base_lrs`` / ``_last_lr`` / ``lr_lambdas`` are persisted and
        restored without corruption.

    Args:
        optimizers: The optimizer container to drive.  Each inner optimizer
            gets its own ``LambdaLR`` instance sharing the same schedule
            lambda.
    """

    schedulers: list[LRScheduler]

    def __init__(
        self,
        optimizers: OptimizersContainer,
        lr_scheduler_type: str,
        **lr_scheduler_kwargs: Any,
    ) -> None:
        if len(optimizers) == 0:
            raise ValueError("Must have at least one optimizer to create LRScheduler")
        lr_scheduler = _lr_scheduler_cls(lr_scheduler_type, **lr_scheduler_kwargs)

        self.schedulers = [LambdaLR(optimizer, lr_scheduler.schedule) for optimizer in optimizers]
        log.info(f"Created {len(self.schedulers)} schedulers")

    def __iter__(self) -> Iterator[LRScheduler]:
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        # Single-scheduler: return the legacy shape verbatim so existing
        # on-disk checkpoints continue to round-trip with strict_resume=True.
        # Multi-scheduler: also persist per-scheduler state so heterogeneous
        # param-group structures (different base_lrs / _last_lr lengths) are
        # captured without corrupting one scheduler with another's data.
        state_dict = self.schedulers[0].state_dict()
        if len(self.schedulers) > 1:
            state_dict["schedulers"] = [scheduler.state_dict() for scheduler in self.schedulers]
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Inverse of ``state_dict``: presence of ``"schedulers"`` implies the
        # multi-scheduler shape and *exactly* matches ``len(self.schedulers)``;
        # absence implies the single-scheduler shape.  ``state_dict`` is always
        # produced by ``self.state_dict()`` on either the same or a compatible
        # container, so a mismatch is a real config drift, not a recoverable
        # path — raise with an actionable message.
        per_scheduler_states = state_dict.get("schedulers")
        if per_scheduler_states is None:
            if len(self.schedulers) != 1:
                raise ValueError(
                    f"LRSchedulersContainer.load_state_dict: checkpoint holds a single "
                    f"scheduler's state, but this container has {len(self.schedulers)}. "
                    "Resume from a checkpoint saved with a matching number of schedulers, "
                    "or add the scheduler to `checkpoint.keys_not_to_resume` to start "
                    "scheduler state fresh."
                )
            self.schedulers[0].load_state_dict(copy.deepcopy(state_dict))
            return

        if len(per_scheduler_states) != len(self.schedulers):
            raise ValueError(
                f"LRSchedulersContainer.load_state_dict: checkpoint has state for "
                f"{len(per_scheduler_states)} schedulers, but this container has "
                f"{len(self.schedulers)}. Resume from a checkpoint with a matching number "
                "of schedulers, or add the scheduler to `checkpoint.keys_not_to_resume` to "
                "start scheduler state fresh."
            )
        for scheduler, sub_state in zip(self.schedulers, per_scheduler_states):
            # Deepcopy so nested mutable values (lists) are not aliased across
            # schedulers after ``LambdaLR.__dict__.update``.
            scheduler.load_state_dict(copy.deepcopy(sub_state))

    def get_last_lr(self) -> list[float | torch.Tensor]:
        # Concatenate the per-optimizer ``get_last_lr()`` lists.  Each
        # underlying scheduler reports one entry per param group, and
        # different optimizers (one per device mesh) may carry distinct
        # ``base_lr`` values via ``lr_multipliers``, so the full snapshot
        # requires aggregating across schedulers.
        return [lr for scheduler in self.schedulers for lr in scheduler.get_last_lr()]

    def get_lr(self) -> list[float | torch.Tensor]:
        # Same aggregation as ``get_last_lr``; see comment there.
        return [lr for scheduler in self.schedulers for lr in scheduler.get_lr()]


def build_lr_scheduler(
    optimizer: OptimizersContainer,
    lr_scheduler_type: str,
    **lr_scheduler_kwargs: Any,
) -> LRSchedulersContainer:
    """Create an :class:`LRSchedulersContainer` for the given optimizer container.

    Thin factory used by Hydra/LazyConfig scheduler registrations (see
    ``configs/base/defaults/optimizer.py``).  At
    instantiation time, ``optimizer`` is supplied by the model's
    ``init_optimizer_scheduler`` hook via
    ``lazy_instantiate(scheduler_config, optimizer=optimizer)``.

    Note:
        Callers who need different scheduler behavior can subclass
        :class:`LRSchedulersContainer` and write their own factory; pointing
        the Hydra ``_target_`` at that factory swaps it in.

    Args:
        optimizer: The optimizer container the scheduler should drive.  One
            :class:`torch.optim.lr_scheduler.LambdaLR` will be built per
            element of this container.
        lr_scheduler_type: Scheduler kind accepted by :func:`_lr_scheduler_cls`
            — ``"lambdalinear"`` or ``"lambdacosine"`` (case-insensitive).
        **lr_scheduler_kwargs: Forwarded verbatim to the underlying lambda
            scheduler constructor (e.g. ``warm_up_steps``, ``cycle_lengths``,
            ``f_start``, ``f_max``, ``f_min``, ``verbosity_interval``).

    Returns:
        An :class:`LRSchedulersContainer` wrapping one ``LambdaLR`` per inner
        optimizer.
    """
    return LRSchedulersContainer(optimizer, lr_scheduler_type, **lr_scheduler_kwargs)
