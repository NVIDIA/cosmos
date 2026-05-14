# Backward-compatibility shim: this module was renamed to unified_mot.py.
# Existing serialized configs / checkpoints may reference the old module path,
# so we re-export everything from the new location.
from cosmos.model.vfm.mot.unified_mot import *  # noqa: F401, F403
from cosmos.model.vfm.mot.unified_mot import (  # noqa: F401  # explicit re-exports for type checkers
    LayerTypes,
    MoTDecoderLayer,
    Nemotron3DenseVLTextConfig,
    Nemotron3DenseVLTextForCausalLM,
    Nemotron3DenseVLTextModel,
    PackedAttentionMoT,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeTextForCausalLM,
    Qwen3VLMoeTextModel,
    Qwen3VLTextConfig,
    Qwen3VLTextForCausalLM,
    Qwen3VLTextModel,
    Qwen3VLTextMoTDecoderLayer,
)
