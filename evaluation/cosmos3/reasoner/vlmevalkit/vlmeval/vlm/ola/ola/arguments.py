from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_speech_projector: bool = field(default=False)
    tune_speech_encoder: bool = field(default=False)
    tune_speech_generator_only: bool = field(default=False)
    speech_encoder_type: Optional[str] = field(default=None)
    speech_encoder: Optional[str] = field(default=None)
    pretrain_speech_projector: Optional[str] = field(default=None)
    speech_projector_type: Optional[str] = field(default='linear')
    speech_encoder_ds_rate: int = 5
    speech_encoder_hidden_size: int = 1280


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    is_multimodal: bool = False
    input_type: str = field(default="mel")
    speech_normalize: bool = False
    mel_size: int = 128
    has_tgt_units: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    freeze_speech_projector: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    speech_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
