from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import CosmosAttentionBackend


def load_attention_backend(spec: str, **kwargs: Any) -> CosmosAttentionBackend:
    """Instantiate an attention backend from ``module.path:attribute``."""

    try:
        module_name, attribute_name = spec.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError(f"invalid attention backend {spec!r}; expected 'module.path:ClassName'") from exc

    module = import_module(module_name)
    factory = getattr(module, attribute_name)
    backend = factory(**kwargs)
    if not isinstance(backend, CosmosAttentionBackend):
        raise TypeError(f"{spec!r} did not produce a CosmosAttentionBackend-compatible object")
    return backend
