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

import json
import os
import sys
from types import SimpleNamespace
from typing import Optional

from transformers import PreTrainedTokenizerFast

from cosmos.data.vfm.processors.base import BaseVLMProcessor
from cosmos.data.vfm.processors.nemotron3densevl_processor import Nemotron3DenseVLProcessor
from cosmos.data.vfm.processors.nemotronvl_processor import NemotronVLProcessor
from cosmos.data.vfm.processors.qwen3vl_processor import Qwen3VLProcessor
from cosmos.model.vfm.tokenizers.tokenization_qwen2 import Qwen2Tokenizer
from cosmos.utils.vfm.vlm.pretrained_models_downloader import maybe_download_hf_model_from_s3

_VARIANT_TO_CREDENTIALS = {
    "s3": ("credentials/s3_training.secret", "checkpoints-us-east-1"),
    "gcp": ("credentials/gcp_checkpoint.secret", "nv-00-10206-checkpoint"),
    # "hf" => no S3 backing store: pass empty credentials/bucket so the downloader
    # falls back to a direct HuggingFace Hub download (matches the legacy
    # ``download_tokenizer_files(model_name, "hf")`` behavior on origin/main, which
    # simply returned the model name and let from_pretrained pull from HF).
    "hf": ("", ""),
}

# S3 prefix under which HuggingFace model files are stored in the checkpoint buckets.
_LLM_S3_PREFIX = "cosmos3/pretrained/huggingface"


class LLMTokenizerProcessor(BaseVLMProcessor):
    """Wrapper that adapts a bare LLM tokenizer to the ``BaseVLMProcessor`` API.

    Used by LLM-only (no-vision) tokenizer configs so that all augmentors and
    model code can treat LLM-only and full VLM configs uniformly through the
    same ``proc.tokenizer`` / ``proc.tokenize_text`` surface. The base class
    handles ``tokenize_text`` / ``encode`` / ``decode``; we only need to wire
    up ``self.processor`` so ``.tokenizer`` resolves.
    """

    def __init__(self, tokenizer):
        self.processor = SimpleNamespace(tokenizer=tokenizer)


def _patch_nemotron_llm_tokenizer_vision_tokens(destination_dir: str) -> None:
    """Remap reserved placeholder tokens to vision special tokens in-place.

    The Nemotron LLM tokenizer reserves ``<SPECIAL_20>`` / ``<SPECIAL_21>``
    at IDs 20/21 -- the same slots the VLM tokenizer uses for
    ``<|vision_start|>`` / ``<|vision_end|>``.  Renaming them here keeps
    every vision-token ID inside the original vocab_size (131072) so no
    embedding-layer resize is needed during FSDP training.  The function is
    idempotent: re-applying it after the tokens are already renamed is a no-op.
    """
    remap = {"<SPECIAL_20>": "<|vision_start|>", "<SPECIAL_21>": "<|vision_end|>"}

    tokenizer_json_path = os.path.join(destination_dir, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path) as f:
            data = json.load(f)
        for entry in data.get("added_tokens", []):
            if entry["content"] in remap:
                entry["content"] = remap[entry["content"]]
        vocab = data.get("model", {}).get("vocab", {})
        for old_name, new_name in remap.items():
            if old_name in vocab:
                vocab[new_name] = vocab.pop(old_name)
        with open(tokenizer_json_path, "w") as f:
            json.dump(data, f)

    tokenizer_config_path = os.path.join(destination_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path) as f:
            tc_data = json.load(f)
        for entry in tc_data.get("added_tokens_decoder", {}).values():
            if entry.get("content") in remap:
                entry["content"] = remap[entry["content"]]
        with open(tokenizer_config_path, "w") as f:
            json.dump(tc_data, f)


def _download_llm_tokenizer(
    tokenizer_type: str,
    credentials: str,
    bucket: str,
    cache_dir: Optional[str] = None,
) -> str:
    return maybe_download_hf_model_from_s3(
        tokenizer_type,
        credentials=credentials,
        bucket=bucket,
        include_model_weights=False,
        cache_dir=cache_dir,
        s3_prefix=_LLM_S3_PREFIX,
    )


def build_processor(
    tokenizer_type: str,
    config_variant: Optional[str] = None,
    credentials: Optional[str] = None,
    bucket: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    if credentials is None or bucket is None:
        if config_variant is None:
            config_variant = "s3"
        if config_variant not in _VARIANT_TO_CREDENTIALS:
            raise ValueError(f"config_variant must be one of {list(_VARIANT_TO_CREDENTIALS)}, got {config_variant!r}")
        variant_credentials, variant_bucket = _VARIANT_TO_CREDENTIALS[config_variant]
        credentials = credentials if credentials is not None else variant_credentials
        bucket = bucket if bucket is not None else variant_bucket
    elif config_variant is not None:
        raise ValueError("Provide either config_variant or (credentials, bucket), not both")
    if "Qwen/Qwen3-VL" in tokenizer_type or "Siglip2-Qwen3-1.7B" in tokenizer_type:
        return Qwen3VLProcessor(tokenizer_type, credentials=credentials, bucket=bucket, cache_dir=cache_dir)
    elif "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16" in tokenizer_type:
        return NemotronVLProcessor(tokenizer_type, credentials=credentials, bucket=bucket, cache_dir=cache_dir)
    elif "NVIDIA-Nemotron-3-Dense-VL" in tokenizer_type or "Qwen3-2B-ViT" in tokenizer_type:
        return Nemotron3DenseVLProcessor(tokenizer_type, credentials=credentials, bucket=bucket, cache_dir=cache_dir)
    elif "Qwen/Qwen3-0.6B" in tokenizer_type:
        local_path = _download_llm_tokenizer(tokenizer_type, credentials, bucket, cache_dir)
        return LLMTokenizerProcessor(Qwen2Tokenizer.from_pretrained(local_path))
    elif "Nemotron/NVIDIA-Nemotron-3-2B-BF16" in tokenizer_type:
        local_path = _download_llm_tokenizer(tokenizer_type, credentials, bucket, cache_dir)
        _patch_nemotron_llm_tokenizer_vision_tokens(local_path)
        return LLMTokenizerProcessor(PreTrainedTokenizerFast.from_pretrained(local_path, trust_remote_code=True))
    else:
        raise ValueError(f"Tokenizer type {tokenizer_type} not supported")


def build_processor_lazy(*args, **kwargs):
    """LazyCall wrapper that resolves ``build_processor`` on this module at call time.

    LazyCall captures its target at config-construction time, so a direct
    ``L(build_processor)`` would freeze the original function reference and
    bypass any later ``monkeypatch.setattr`` on this module's
    ``build_processor`` attribute. This wrapper performs a fresh module-level
    lookup on every call, so test fixtures patching ``build_processor`` are
    honored when the config is instantiated.
    """
    return sys.modules[__name__].build_processor(*args, **kwargs)
