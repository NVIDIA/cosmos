from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

DEFAULT_FP8_MODEL_ID = "nvidia/Cosmos3-Nano-FP8"


@dataclass(frozen=True, slots=True)
class FP8LoadReport:
    """Evidence that ModelOpt restored a usable pre-quantized transformer."""

    fp8_parameters: int
    quantizer_modules: int
    meta_tensors: int


def materialize_fp8_checkpoint(
    model_id: str,
    *,
    revision: str | None,
    local_files_only: bool,
) -> Path:
    """Resolve an FP8 checkpoint to a local directory.

    ModelOpt 0.44 restores ``modelopt_state.pth`` relative to the model path and
    does not reliably honor the modular loader's separate ``subfolder`` value.
    Materializing the snapshot lets the adapter load ``transformer/`` through
    its complete path while keeping all other components on the normal
    Diffusers modular path.
    """

    local_path = Path(model_id).expanduser()
    if local_path.is_dir():
        checkpoint_dir = local_path.resolve()
    else:
        from huggingface_hub import snapshot_download

        checkpoint_dir = Path(
            snapshot_download(
                model_id,
                revision=revision,
                local_files_only=local_files_only,
            )
        ).resolve()

    modelopt_state = checkpoint_dir / "transformer" / "modelopt_state.pth"
    if not modelopt_state.is_file():
        raise RuntimeError(
            f"FP8 was requested, but the checkpoint has no ModelOpt state at {modelopt_state}. "
            f"Use {DEFAULT_FP8_MODEL_ID} with published Diffusers checkpoint files, "
            "or a local Diffusers FP8 snapshot."
        )
    return checkpoint_dir


def activate_modelopt_fp8() -> None:
    """Register ModelOpt's Hugging Face restore path before model loading."""

    try:
        # Importing the backend registers the FP8 GEMM implementation used by
        # the restored quantized Linear modules.
        import modelopt.torch.quantization.backends.fp8_per_tensor_gemm  # noqa: F401
        from modelopt.torch.opt import enable_huggingface_checkpointing
    except ImportError as error:
        raise ImportError(
            "FP8 checkpoint loading requires NVIDIA ModelOpt. In your active Diffusers environment, run "
            "`uv pip install nvidia-modelopt==0.44.0` (or `python -m pip install nvidia-modelopt==0.44.0`). "
            "Alternatively install this package with its [fp8] extra."
        ) from error

    enable_huggingface_checkpointing()


def load_fp8_transformer(
    checkpoint_dir: Path,
    *,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Restore the ModelOpt transformer from its complete local component path."""

    activate_modelopt_fp8()

    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer

    transformer = Cosmos3OmniTransformer.from_pretrained(
        str(checkpoint_dir / "transformer"),
        dtype=dtype,
        local_files_only=True,
    )
    verify_fp8_transformer(transformer)
    return transformer


def verify_fp8_transformer(transformer: torch.nn.Module) -> FP8LoadReport:
    """Reject partial restores that contain FP8 tensors but no quantizer graph."""

    named_tensors: list[tuple[str, Any]] = list(transformer.named_parameters()) + list(transformer.named_buffers())
    fp8_parameters = sum(parameter.dtype == torch.float8_e4m3fn for parameter in transformer.parameters())
    quantizer_modules = sum("quantizer" in name.lower() for name, _ in transformer.named_modules())
    meta_tensors = sum(tensor.is_meta for _, tensor in named_tensors)
    report = FP8LoadReport(
        fp8_parameters=fp8_parameters,
        quantizer_modules=quantizer_modules,
        meta_tensors=meta_tensors,
    )
    if not fp8_parameters or not quantizer_modules or meta_tensors:
        raise RuntimeError(
            "invalid Cosmos 3 FP8 restore: "
            f"fp8_parameters={fp8_parameters}, quantizer_modules={quantizer_modules}, "
            f"meta_tensors={meta_tensors}"
        )
    return report
