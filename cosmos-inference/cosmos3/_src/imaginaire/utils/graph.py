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

"""A rework of make_graphed_callabled function from TransformerEngine so that it works with inference-only."""

from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

import torch
from torch._C import _graph_pool_handle
from torch.utils._pytree import tree_flatten as _tree_flatten
from torch.utils._pytree import tree_unflatten as _tree_unflatten
from transformer_engine.pytorch.distributed import get_all_rng_states, graph_safe_rng_available
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

from cosmos3._src.imaginaire.utils import log

__all__ = ["create_cuda_graph"]


_IS_GRAPH_CAPTURING = False

_T = TypeVar("_T")
SingleOrTuple = Union[_T, Tuple[_T, ...]]


def set_capture_start() -> None:
    """Record beginning of `make_graphed_callables`."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = True


def set_capture_end() -> None:
    """Record end of `make_graphed_callables`."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = False


def is_graph_capturing() -> None:
    """Return whether within `make_graphed_callables`."""
    return _IS_GRAPH_CAPTURING


def graph_pool_handle():
    """
    Returns an opaque token representing the id of a graph memory pool.
    """
    return _graph_pool_handle()


def _make_graphed_callables(
    callables: SingleOrTuple[Callable],
    sample_args: SingleOrTuple[Tuple[torch.Tensor, ...]],
    num_warmup_iters: int = 3,
    sample_kwargs: Optional[SingleOrTuple[Dict[str, Any]]] = None,
    pool: Optional[Tuple[int, ...]] = None,
) -> SingleOrTuple[Callable]:
    """
    Helper method for `make_graphed_callables`
    """

    if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        raise RuntimeError(
            "make_graphed_callables does not support the autocast caching. Please set `cache_enabled=False`."
        )

    # Default is to pass no kwargs to callables
    if sample_kwargs is None:
        if isinstance(callables, tuple):
            sample_kwargs = tuple({} for _ in range(len(sample_args)))
        else:
            sample_kwargs = {}

    # Canonicalize args as tuples
    just_one_callable = False
    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        sample_args = (sample_args,)
        sample_kwargs = (sample_kwargs,)

    # Check sizes of args
    assert len(sample_args) == len(callables)
    assert len(sample_kwargs) == len(callables)

    # Check callables
    for c in callables:
        if isinstance(c, torch.nn.Module):
            assert len(c._backward_hooks) == 0 and len(c._forward_hooks) == 0 and len(c._forward_pre_hooks) == 0, (
                "Modules must not have hooks registered at the time they are passed. "
                + "However, registering hooks on modules after passing them "
                + "through make_graphed_callables is allowed."
            )
            assert all(b.requires_grad is False for b in c.buffers()), (
                "In any :class:`~torch.nn.Module` passed to "
                + ":func:`~make_graphed_callables`, only parameters may be trainable. "
                + "All buffers must have ``requires_grad=False``."
            )

    # Flatten callable arguments
    per_callable_kwargs_keys = [list(kwargs.keys()) for kwargs in sample_kwargs]
    flatten_sample_args = []
    for args, kwargs, kwargs_keys in zip(sample_args, sample_kwargs, per_callable_kwargs_keys):
        flatten_arg, _ = _tree_flatten(args)
        flatten_kwarg, _ = _tree_flatten([kwargs[key] for key in kwargs_keys])
        flatten_sample_args.append(tuple(flatten_arg + flatten_kwarg))
        assert all(isinstance(arg, torch.Tensor) for arg in flatten_arg), (
            "In the beta API, sample_args "
            + "for each callable must contain only Tensors. Other types are not allowed."
        )

    # If a callable is an nn.Module, its graph's full input surface is the args the user explicitly
    # passes to forward (ie, its sample_args) AND the module's parameter attributes.
    per_callable_len_user_args = [len(args) for args in flatten_sample_args]
    per_callable_module_params = [tuple(c.parameters()) if isinstance(c, torch.nn.Module) else () for c in callables]
    per_callable_static_input_surfaces = [
        flatten_sample_args[i] + per_callable_module_params[i] for i in range(len(callables))
    ]

    fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(flatten_sample_args))]
    graph_callables = [None for _ in range(len(flatten_sample_args))]

    # For cases with multiple active RNG states, e.g. TP.
    if graph_safe_rng_available():
        for _, state in get_all_rng_states().items():
            for fwd_graph in fwd_graphs:
                fwd_graph.register_generator_state(state)

    mempool = graph_pool_handle() if pool is None else pool

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    torch.cuda.synchronize()

    # Get warmup func and func_idx.
    warmup_func_idx = []
    warmup_func = []
    for func_idx, func in enumerate(callables):
        warmup_func_idx.append(func_idx)
        warmup_func.append(func)
    assert len(warmup_func) == len(sample_args), f"Warmup runs {len(warmup_func)} don't match args {len(sample_args)}."
    assert len(warmup_func_idx) == len(set(warmup_func_idx)), (
        f"Warmup runs {len(warmup_func)} but only {len(set(warmup_func_idx))} are unique."
    )

    # Filter the TE modules that cudagraph can access.
    visited_te_modules = set()

    def hook_fn(module, inputs, outputs):  # pylint: disable=unused-argument
        if isinstance(module, TransformerEngineBaseModule):
            visited_te_modules.add(module)

    # Run warmup and do the above filtering.
    with torch.cuda.stream(torch.cuda.Stream()):
        for func_idx, func in zip(warmup_func_idx, warmup_func):
            args = sample_args[func_idx]
            kwargs = sample_kwargs[func_idx]
            for _ in range(num_warmup_iters):
                hooks = []
                for module in func.modules():
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)
                outputs, _ = _tree_flatten(func(*args, **kwargs))
                for hook in hooks:
                    hook.remove()
                del outputs
            # The following code is added specifically for MCore's special requirements,
            # aimed at preventing warmup from altering the control flow.
            for module in func.modules():
                if hasattr(module, "is_first_microbatch"):
                    module.is_first_microbatch = True
    torch.cuda.synchronize()

    # All captures here share a mempool. To avoid replays corrupting each other's memory,
    # the safest approach is to capture all passes in the same order they'll run:
    # Capture forward graphs
    per_callable_static_outputs = []
    per_callable_output_unflatten_spec = []
    graph_id = 0
    for func, args, kwargs, fwd_graph in zip(callables, sample_args, sample_kwargs, fwd_graphs):
        with torch.cuda.graph(fwd_graph, pool=mempool):
            outputs = func(*args, **kwargs)
        graph_callables[graph_id] = func
        graph_id += 1

        flatten_outputs, spec = _tree_flatten(outputs)
        per_callable_static_outputs.append(tuple(flatten_outputs))
        per_callable_output_unflatten_spec.append(spec)

    def make_graphed_autograd_function(
        fwd_graph,
        module_params,
        kwargs_keys,
        len_user_args,
        output_unflatten_spec,
        static_input_surface,
        static_outputs,
    ):
        class Graphed(torch.autograd.Function):
            """Autograd function for graph replay."""

            @staticmethod
            def forward(ctx, *inputs):
                # pylint: disable=missing-function-docstring

                # Copy values from new tensors into static tensors
                for i in range(len_user_args):
                    if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                        static_input_surface[i].copy_(inputs[i])

                # Replay forward graph
                fwd_graph.replay()
                assert isinstance(static_outputs, tuple)
                return tuple(o.detach() for o in static_outputs)

        def functionalized(*user_args, **user_kwargs):
            # Check that required kwargs are provided
            for key in kwargs_keys:
                if key not in user_kwargs:
                    raise TypeError(
                        f"Graphed callable was initialized with kwarg {key} ,but it was not provided in graph replay"
                    )

            # Runs the autograd function with inputs == all inputs to
            # the graph that might require grad (explicit user args +
            # module parameters)
            # Assumes module params didn't change since capture.
            flatten_user_args, _ = _tree_flatten(user_args)
            flatten_user_kwargs, _ = _tree_flatten([user_kwargs[key] for key in kwargs_keys])
            func_args = tuple(flatten_user_args) + tuple(flatten_user_kwargs) + module_params
            out = Graphed.apply(*func_args)
            return _tree_unflatten(out, output_unflatten_spec)

        return functionalized

    # Put together the final graphed callables
    ret = []
    for i in range(len(sample_args)):
        graphed = make_graphed_autograd_function(
            fwd_graphs[i],
            per_callable_module_params[i],
            per_callable_kwargs_keys[i],
            per_callable_len_user_args[i],
            per_callable_output_unflatten_spec[i],
            per_callable_static_input_surfaces[i],
            per_callable_static_outputs[i],
        )

        func = graph_callables[i]
        if isinstance(func, torch.nn.Module):

            def make_graphed_forward(func, graph_training_state, graphed, orig_fwd):
                def new_fwd(*user_args, **user_kwargs):
                    # If the module's training-or-eval state matches what we graphed,
                    # run the graph, otherwise run the original forward method
                    if func.training == graph_training_state:
                        return graphed(*user_args, **user_kwargs)
                    return orig_fwd(*user_args, **user_kwargs)

                return new_fwd

            forward = make_graphed_forward(func, func.training, graphed, func.forward)
            ret.append(forward)
        else:
            ret.append(graphed)

    if just_one_callable:
        return ret[0]

    return tuple(ret)


def make_graphed_callables_forward(
    modules: SingleOrTuple[Callable],
    sample_args: SingleOrTuple[Tuple[torch.Tensor, ...]],
    num_warmup_iters: int = 3,
    sample_kwargs: Optional[SingleOrTuple[Dict[str, Any]]] = None,
    pool: Optional[Tuple[int, ...]] = None,
) -> Union[Callable, Tuple[Callable, ...]]:
    """
    Make CUDA graph version of Transformer Engine modules
    A variation of PyTorch's `make_graphed_callables` utility function.
    `original PyTorch implementation <https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html>`_
    for more documentation.
    Graphing parameters
    -------------------
    modules: (tuple of) callable
             Callable or callables to graph.
    sample_args: (tuple of) tuple of torch.Tensor
                 Positional arguments to callable(s).
    num_warmup_iters: int, default = 3
                      Number of warmup iterations.
    sample_kwargs: (tuple of) dict, optional
                   Keyword arguments to callable(s)
    pool: (tuple of) int, default = `None`, optional
          An instance returned from function `torch.cuda.graph_pool_handle` that hints
          this graph may share memory with the indicated pool.
    """
    set_capture_start()

    # Handle single module.
    just_one_callable = False
    if not isinstance(modules, tuple):
        just_one_callable = True
        modules = (modules,)

    forward_funcs = []
    for module in modules:
        assert isinstance(module, torch.nn.Module), f"Graphing for {type(module)} is not supported."
        forward_funcs.append(module)

    if just_one_callable:
        forward_funcs = forward_funcs[0]
    else:
        forward_funcs = tuple(forward_funcs)

    # Save RNG state.
    if graph_safe_rng_available():
        generators = [
            torch.cuda.default_generators[torch.cuda.current_device()],
            *get_all_rng_states().values(),
        ]
        original_rng_states = [state.get_state() for state in generators]
    else:
        original_rng_states = torch.cuda.get_rng_state()

    graphed_callables = _make_graphed_callables(
        forward_funcs,
        sample_args,
        num_warmup_iters=num_warmup_iters,
        sample_kwargs=sample_kwargs,
        pool=pool,
    )

    # Ensures warmup does not affect numerics for ops such as dropout.
    if graph_safe_rng_available():
        for gen, state in zip(generators, original_rng_states):
            gen.set_state(state)
    else:
        torch.cuda.set_rng_state(original_rng_states)
    set_capture_end()
    return graphed_callables


def create_cuda_graph(
    cuda_graphs_storage: dict,
    blocks: torch.nn.ModuleList,
    tensor_args: list[Any],
    tensor_kwargs: dict[str, Any],
    extra_key: Optional[str] = None,
) -> str:
    def _make_dummy_tensor_like(t: torch.Tensor) -> torch.Tensor:
        if t.dtype.is_floating_point:
            return torch.randn(t.shape, device=t.device, dtype=t.dtype)
        if t.dtype == torch.bool:
            return torch.zeros(t.shape, device=t.device, dtype=t.dtype)
        if t.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            if t.numel() > 0:
                low = int(t.min().item())
                high = int(t.max().item())
                if high == low:
                    high = low + 1
                else:
                    high = high + 1
            else:
                low, high = 0, 1
            return torch.randint(low, high, t.shape, device=t.device, dtype=t.dtype)
        # Fallback: use zeros for uncommon dtypes (e.g., complex) to avoid dtype/range pitfalls.
        return torch.zeros(t.shape, device=t.device, dtype=t.dtype)

    def _make_dummy_tree(x: Any) -> Any:
        flat, spec = _tree_flatten(x)
        dummy_flat: list[torch.Tensor] = []
        for leaf in flat:
            if not isinstance(leaf, torch.Tensor):
                raise TypeError(
                    f"create_cuda_graph only supports pytrees of torch.Tensor leaves; got leaf type {type(leaf)}"
                )
            dummy = _make_dummy_tensor_like(leaf)
            dummy.requires_grad = leaf.requires_grad
            dummy_flat.append(dummy)
        return _tree_unflatten(dummy_flat, spec)

    real_args = [arg for arg in tensor_args if arg is not None]
    real_kwargs = {k: v for k, v in tensor_kwargs.items() if v is not None}

    # Shapes key must reflect all tensor leaves (supports tuple/list/dict structures).
    flat_tensors: list[torch.Tensor] = []
    for arg in real_args:
        flat, _ = _tree_flatten(arg)
        for leaf in flat:
            if not isinstance(leaf, torch.Tensor):
                raise TypeError(
                    f"create_cuda_graph only supports pytrees of torch.Tensor leaves; got leaf type {type(leaf)}"
                )
            flat_tensors.append(leaf)
    for _, kwarg in real_kwargs.items():
        flat, _ = _tree_flatten(kwarg)
        for leaf in flat:
            if not isinstance(leaf, torch.Tensor):
                raise TypeError(
                    f"create_cuda_graph only supports pytrees of torch.Tensor leaves; got leaf type {type(leaf)}"
                )
            flat_tensors.append(leaf)

    shapes_key = "_".join(str(shape_component) for t in flat_tensors for shape_component in t.shape)
    if extra_key:
        shapes_key = f"{shapes_key}_{extra_key}"
    if shapes_key not in cuda_graphs_storage:
        callables = []
        sample_args = []
        sample_kwargs = []
        for block in blocks:
            callables.append(block)
            args = []
            kwargs = {}
            for arg in real_args:
                args.append(_make_dummy_tree(arg))
            for name, kwarg in real_kwargs.items():
                kwargs[name] = _make_dummy_tree(kwarg)
            sample_args.append(tuple(args))
            sample_kwargs.append(kwargs)

        log.critical(f"Creating graph for shape {shapes_key}")
        cuda_graphs_storage[shapes_key] = make_graphed_callables_forward(
            tuple(callables),
            tuple(sample_args),
            sample_kwargs=tuple(sample_kwargs),
            num_warmup_iters=11,
        )
        log.critical(f"Created graph for shape {shapes_key}")
    return shapes_key
