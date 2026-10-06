"""Cosmos 3 Nano streaming execution for Diffusers modular pipelines."""

from .config import StreamingConfig
from .metrics import RuntimeStats

__all__ = ["RuntimeStats", "StreamingConfig"]
__version__ = "0.1.0"


def build_pipeline(*args, **kwargs):
    """Lazily import the unified Diffusers-dependent pipeline builder."""

    from .modular import build_pipeline as _build

    return _build(*args, **kwargs)


def build_text_to_video_pipeline(*args, **kwargs):
    """Compatibility alias for the unified pipeline builder."""

    from .modular import build_text_to_video_pipeline as _build

    return _build(*args, **kwargs)


__all__.extend(["build_pipeline", "build_text_to_video_pipeline"])
