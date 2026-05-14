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

import math
from collections import defaultdict

import torch
import wandb
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel

from cosmos.utils.callback import Callback


@torch.compile
def _fused_nan_to_num(grads: list[torch.Tensor]) -> None:
    """Replace NaN/Inf entries with 0.0 in every floating-point grad in-place.

    The Python-level loop over ``grads`` is wrapped in ``@torch.compile`` so
    Inductor can fuse the per-tensor ``nan_to_num`` ops into a single CUDA
    kernel.  This is NOT the ``torch._foreach_*`` API; it is fusion-via-compile
    and depends on the grad list structure (length, dtypes, shapes) staying
    stable across iterations so Dynamo can reuse its specialized graph.
    """
    grads = [g for g in grads if torch.is_floating_point(g)]
    for g in grads:
        torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0, out=g)


class _MagnitudeRecord:
    def __init__(self) -> None:
        self.state: torch.Tensor | None = None
        self.iter_count: int = 0

    def reset(self) -> None:
        self.state = None
        self.iter_count = 0

    def update(self, cur_state: torch.Tensor) -> None:
        if self.state is None:
            self.state = cur_state.detach().clone()
        else:
            self.state.add_(cur_state)
        self.iter_count += 1

    def get_stat(self) -> float:
        if self.state is not None and self.iter_count > 0:
            avg_state = (self.state / self.iter_count).item()
        else:
            avg_state = 0.0
        self.reset()
        return avg_state


@torch.no_grad()
def _clip_grad(
    parameters: list[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    return_norm_only: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Params are grouped by their ``device_mesh`` (by mesh-dim-names string —
    plain (non-DTensor) params map to ``"default"``). A scalar L2 norm is
    computed per mesh group, DTensor results are reduced to local scalars
    via ``.full_tensor()``, the per-mesh scalars are combined into one
    global norm, and (unless ``return_norm_only=True``) every mesh group
    is rescaled with that single global scalar.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        return_norm_only: if True, skip in-place rescaling of grads and only
            return the computed norms.

    Returns:
        ``(total_norm, per_mesh_norms)`` where ``total_norm`` is the global
        scalar norm used for the rescale, and ``per_mesh_norms`` maps each
        mesh-dim-names key (or ``"default"`` for plain params) to its
        pre-clip per-mesh L2 norm.

    """
    param_set: set[torch.Tensor] = set()
    # Group the parameters by their device meshes.
    parameters_by_mesh: dict[str, list[torch.Tensor]] = defaultdict(list)
    for param in parameters:
        if param in param_set:
            continue
        param_set.add(param)
        if param.grad is None:
            raise ValueError(
                f"_clip_grad received a parameter with no gradient "
                f"(shape={tuple(param.shape)}, dtype={param.dtype}); "
                "callers are expected to pre-filter."
            )

        # If one parameter belongs to multiple meshes, use a flattened mesh name
        # by concatenating all the mesh-dim names together.  ``mesh_dim_names``
        # is ``tuple[str, ...] | None`` on DeviceMesh — fall back to ``default``
        # when names weren't assigned.
        if hasattr(param, "device_mesh"):
            names = param.device_mesh.mesh_dim_names
            device_mesh_str = "-".join(names) if names else "default"
        else:
            device_mesh_str = "default"
        parameters_by_mesh[device_mesh_str].append(param)

    # Compute the norm for each mesh group
    per_mesh_norms: dict[str, torch.Tensor] = {}
    per_mesh_norm_list = []
    for mesh, params in parameters_by_mesh.items():
        grads = [p.grad for p in params if p.grad is not None]
        assert len(grads) > 0, "No gradients to compute norm"
        mesh_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

        # If mesh_norm is a DTensor, the placements must be
        # `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this
        # tensor's process group and then convert it to a local tensor.

        # 1. to make sure the total norm is computed correctly when PP is used (see below)
        # 2. to return a reduced mesh_norm tensor whose .item() would return the correct value
        if isinstance(mesh_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, mesh_norm will be a local tensor.

            # Remove FT replicate dimension if it exists.
            mesh_norm = mesh_norm.full_tensor()
        # Expose the (rank-replicated) per-mesh scalar for diagnostic logging.
        per_mesh_norms[mesh] = mesh_norm

        # Make the norm to be a 1D tensor so we can call cat() later.
        if mesh_norm.ndim == 0:
            mesh_norm = mesh_norm.reshape(1)
        per_mesh_norm_list.append(mesh_norm)

    # Compute the total norm among all meshes.
    if len(per_mesh_norm_list) > 1:
        per_mesh_norm_tensor = torch.cat(per_mesh_norm_list)
        if math.isinf(norm_type):
            total_norm = torch.max(per_mesh_norm_tensor)
        else:
            per_mesh_norm_tensor **= norm_type
            total_norm = torch.sum(per_mesh_norm_tensor)
            total_norm **= 1.0 / norm_type
    else:
        assert per_mesh_norm_list[0].numel() == 1, "total_norm should be a scalar"
        total_norm = per_mesh_norm_list[0].view(-1)[0]

    if not return_norm_only:
        # Perform clipping on each mesh group
        for mesh, params in parameters_by_mesh.items():
            torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, foreach)

    return total_norm, per_mesh_norms


class GradClip(Callback):
    """Unified gradient-clipping callback for both VFM (diffusion) and VLM training.

    The heavy lifting is delegated to ``_clip_grad``: it groups
    params by their ``device_mesh`` (using mesh-dim-names as the key),
    computes a scalar L2 norm per mesh group (reducing any DTensor result
    via ``.full_tensor()``), combines the per-mesh scalars into ONE global
    norm via ``sqrt(sum(per_mesh_norm**2))``, and applies
    ``torch.nn.utils.clip_grads_with_norm_`` per mesh group with the SAME
    global scalar — a SINGLE GLOBAL rescale across every parameter.

    This is necessary for correctness when parameters live on multiple device
    meshes (e.g. dense FSDP-shard + EP-shard MoE experts): clipping each
    mesh independently with stock ``torch.nn.utils.clip_grad_norm_`` would
    assign a different rescale factor per mesh and distort the relative
    magnitudes of dense vs MoE updates. Under VFM's current FSDP-only
    setup the math reduces to a single mesh group and is identical to
    stock ``clip_grad_norm_``; this implementation is forward-correct
    once EP is enabled.

    For diagnostics, the callback ALSO records pre-clip per-mesh sub-norms
    alongside the actual global norm. When ``track_per_modality=True`` (VFM),
    samples are bucketed by image/video via ``model.is_image_batch(data_batch)``,
    producing wandb keys ``clip_grad_norm/{image|video}/{mesh_key}`` plus a
    ``.../global`` synthetic key carrying the actual rescale norm. When False
    (VLM), keys are ``clip_grad_norm/{mesh_key}`` plus ``clip_grad_norm/global``.

    Param-source semantics:
      * ``track_per_modality=True`` (VFM): caller passes the ``OmniMoTModel``;
        only ``model_ddp.net.parameters()`` is iterated, matching legacy VFM
        behavior (the optimizer is built from ``self.net``).
      * ``track_per_modality=False`` (VLM): caller passes a single
        ``nn.Module`` or a list of model parts; each is unwrapped from
        ``DistributedDataParallel`` if needed, then ``parameters()`` is
        iterated and filtered by grad-presence.

    Args:
      clip_norm: max norm to clip to.
      force_finite: if True, NaN/Inf in any grad is zeroed in-place before
        the norm computation.
      track_per_modality: if True, route stats into image/video buckets via
        ``model.is_image_batch(data_batch)``. If False, accumulate into a
        single un-bucketed log group.
    """

    def __init__(
        self,
        clip_norm: float = 1.0,
        force_finite: bool = True,
        track_per_modality: bool = False,
    ):
        self.clip_norm = clip_norm
        self.force_finite = force_finite
        self.track_per_modality = track_per_modality

        # Outer key: modality bucket name. For VLM we use a single bucket "" so
        # wandb keys are short (`clip_grad_norm/{mesh}`); for VFM the bucket is
        # "image" or "video" (`clip_grad_norm/image/{mesh}`).
        # Inner key: mesh string, plus the synthetic "global" key for the
        # actual rescale norm returned by _clip_grad.
        self._states: dict[str, dict[str, _MagnitudeRecord]] = defaultdict(lambda: defaultdict(_MagnitudeRecord))
        self._state_key: str = ""

    def on_training_step_start(
        self,
        model: torch.nn.Module,
        data_batch: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        if not self.track_per_modality:
            return
        self._state_key = "image" if model.is_image_batch(data_batch) else "video"

    def on_before_optimizer_step(
        self,
        model_ddp: torch.nn.Module | list[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del optimizer, scheduler, grad_scaler

        # 1. Resolve which parameters to clip.
        if self.track_per_modality:
            # VFM: only clip `.net` params, matching legacy semantics + the
            # optimizer's actual param set.
            assert not isinstance(model_ddp, list), "track_per_modality=True expects a single OmniMoTModel, not a list"
            model_parts = [model_ddp.module if isinstance(model_ddp, DistributedDataParallel) else model_ddp]
        else:
            # VLM: list of model parts (or single module). Unwrap DDP per part.
            model_parts = model_ddp if isinstance(model_ddp, list) else [model_ddp]
            model_parts = [m.module if isinstance(m, DistributedDataParallel) else m for m in model_parts]

        # 2. Collect params with grads.
        all_params: list[torch.Tensor] = []
        for part in model_parts:
            for p in part.parameters():
                if p.grad is not None:
                    all_params.append(p)

        # 3. No-grad / all-frozen step → skip. _clip_grad's empty
        #    fallback uses torch.cuda.current_device() and would crash on CPU.
        if not all_params:
            return

        # 4. Optionally zero NaN/Inf in grads.
        if self.force_finite:
            _fused_nan_to_num([p.grad for p in all_params])

        # 5. Compute per-mesh norms, the global rescale norm, and clip in
        #    one call. ``_clip_grad`` groups params by mesh,
        #    computes per-mesh L2 norms (reducing DTensor results to local
        #    scalars), combines them into a single global norm, and
        #    rescales every mesh group with that scalar.
        #
        #    When ``force_finite`` is False we did NOT sanitize the grads, so
        #    ask ``get_total_norm`` to raise rather than silently producing a
        #    NaN ``total_norm`` that would taint every parameter on rescale.
        global_norm, per_mesh_norms = _clip_grad(
            all_params,
            self.clip_norm,
            error_if_nonfinite=False,
            foreach=True,
        )

        # 6. Record diagnostic stats: pre-clip per-mesh sub-norms plus the
        #    actual global rescale norm.
        cur_state = self._states[self._state_key]
        for mesh_str, mesh_norm in per_mesh_norms.items():
            cur_state[mesh_str].update(mesh_norm)
        cur_state["global"].update(global_norm)

        # 7. Log every logging_iter.
        if iteration % self.config.trainer.logging_iter == 0 and wandb.run:
            log_dict: dict[str, float | int] = {"iteration": iteration}
            for modality, state in self._states.items():
                for mesh_str, record in state.items():
                    avg = record.get_stat()
                    if self.track_per_modality:
                        log_dict[f"clip_grad_norm/{modality}/{mesh_str}"] = avg
                    else:
                        log_dict[f"clip_grad_norm/{mesh_str}"] = avg
            wandb.log(log_dict, step=iteration)
