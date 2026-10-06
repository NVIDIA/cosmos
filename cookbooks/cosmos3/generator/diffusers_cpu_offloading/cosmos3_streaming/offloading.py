"""Lifecycle operations for the Diffusers group-offload hooks we install."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def offload_groups(module: torch.nn.Module) -> None:
    """Restore every group's CPU weights, including interrupted/prefetched groups."""
    from diffusers.hooks.group_offloading import GroupOffloadingHook

    groups = {}
    for child in module.modules():
        registry = getattr(child, "_diffusers_hook", None)
        if registry is not None:
            for hook in registry.hooks.values():
                if isinstance(hook, GroupOffloadingHook):
                    groups[id(hook.group)] = hook.group
    for group in groups.values():
        if group.stream is not None:
            group.stream.synchronize()
        device = torch.device(group.onload_device)
        context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with context:
            group.offload_()
        # The pinned streamed hook restores module parameters, but not module
        # buffers. Preserve tensor identities used by its CPU backing store.
        for child in group.modules:
            for buffer in child.buffers():
                if buffer.device != group.offload_device:
                    backing = group.cpu_param_dict.get(buffer)
                    buffer.data = backing if backing is not None else buffer.data.to(group.offload_device)


def remove_group_hooks(module: torch.nn.Module) -> None:
    """Remove only offload/prefetch hooks; preserve other hooks and FP8 wrappers."""
    from diffusers.hooks.group_offloading import (
        GroupOffloadingHook,
        LayerExecutionTrackerHook,
        LazyPrefetchGroupOffloadingHook,
    )

    hook_types = (GroupOffloadingHook, LayerExecutionTrackerHook, LazyPrefetchGroupOffloadingHook)
    for child in module.modules():
        registry = getattr(child, "_diffusers_hook", None)
        if registry is not None:
            for name, hook in reversed(list(registry.hooks.items())):
                if isinstance(hook, hook_types):
                    registry.remove_hook(name, recurse=False)
