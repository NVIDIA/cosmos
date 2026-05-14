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

# Configs for VLM / LLM models

import os

import attrs
import torch.distributed as dist

from cosmos.utils.flags import INTERNAL
from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.lazy_config import LazyDict
from cosmos.utils.lazy_config import instantiate as lazy_instantiate
from cosmos.utils import log
from cosmos.utils.config_helper import ConfigStore
from cosmos.utils.easy_io import easy_io
from cosmos.model.vfm.mot.unified_mot import (
    Nemotron3DenseVLTextConfig,
    Nemotron3DenseVLTextForCausalLM,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeTextForCausalLM,
    Qwen3VLTextConfig,
    Qwen3VLTextForCausalLM,
)
from cosmos.data.vfm.processors import build_processor_lazy
from cosmos.model.vfm.tokenizers.tokenization_qwen2 import Qwen2Tokenizer


def create_vlm_config(base_config: LazyDict, **overrides):
    vlm_config = lazy_instantiate(base_config)
    for key, value in overrides.items():
        setattr(vlm_config, key, value)
    return vlm_config


def get_rank_safe() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0  # default to rank 0 when not in distributed mode


################################################################################
# Download tokenizer files from s3
# Download to ~/.cache/imaginaire4/tokenizer_files/{model_name} and then load from there.
def download_tokenizer_files(model_name: str, config_variant: str) -> str:
    if config_variant == "hf":
        return model_name

    if config_variant == "s3":
        ckpt_bucket = "checkpoints-us-east-1"
        credentials = "credentials/s3_checkpoint.secret"
    elif config_variant == "gcp":
        ckpt_bucket = "nv-00-10206-checkpoint"
        credentials = "credentials/gcp_checkpoint.secret"
    else:
        raise ValueError(f"Invalid config variant: {config_variant}")

    model_path = f"s3://{ckpt_bucket}/cosmos3/pretrained/huggingface/{model_name}"
    if not INTERNAL:
        from cosmos.utils.checkpoint_db import download_checkpoint_v2

        model_path = download_checkpoint_v2(model_path)
        if "://" not in model_path:
            return model_path

    imaginaire_cache_dir = os.environ.get("IMAGINAIRE_CACHE_DIR", os.path.expanduser("~/.cache/imaginaire4"))
    destination_dir = os.path.join(imaginaire_cache_dir, f"tokenizer_files/{model_name}/rank_{get_rank_safe()}")
    s3_backend_args = {
        "backend": "s3",
        "s3_credential_path": credentials,
    }

    extensions = ["json", "txt", "jinja"]
    for extension in extensions:
        for file_path in easy_io.list_dir_or_file(
            model_path,
            list_dir=False,
            list_file=True,
            suffix=extension,
            recursive=False,
            backend_args=s3_backend_args,
        ):
            full_path = easy_io.join_path(model_path, file_path, backend_args=s3_backend_args)
            local_path = f"{destination_dir}/{file_path}"
            if os.path.exists(local_path):
                log.debug(f"Skipping already downloaded tokenizer file: {local_path}")
                continue
            log.info(f"Downloading tokenizer file: {full_path} to {local_path}, cwd: {os.getcwd()}")
            # Download the file
            file_data = easy_io.get(full_path, backend_args=s3_backend_args)
            easy_io.put(file_data, local_path)
    return destination_dir


def create_qwen2_tokenizer_with_download(pretrained_model_name: str, config_variant: str, **_unused_kwargs):
    # **_unused_kwargs absorbs extras (e.g. tokenizer_type) that OmegaConf
    # merges in from a vlm_config preset's tokenizer block when an experiment
    # overrides the tokenizer with this function but doesn't fully replace
    # the preset's kwarg dict.
    destination_dir = download_tokenizer_files(pretrained_model_name, config_variant)
    return Qwen2Tokenizer.from_pretrained(destination_dir)


@attrs.define(slots=False)
class VLMConfig:
    # Name of the huggingface model
    model_name: str = ""

    # Langugage model class to instantiate
    model_instance: LazyDict | None = None

    # Tokenizer / processor used by data augmentors.
    # For VLM configs this is a unified processor (build_processor); for LLM-only configs a raw
    # tokenizer factory.  Code that needs the raw tokenizer should check is_unified_processor.
    tokenizer: LazyDict | None = None

    # Path to the checkpoint
    checkpoint_path: str = ""

    # Path to the credential file
    credential_path: str = ""  # Path to the credential file

    # Whether to enable GCS patch in boto3 for DCP loading from GCS
    enable_gcs_patch_in_boto3: bool = False

    # Whether to load the pretrained LLM / VLM
    load_pretrained: bool = True

    # Layer module to use. We override the decoder layer in huggingface model with this class.
    # This is needed as we need to initialize MoT layers.
    layer_module: str = "Qwen2MoTDecoderLayer"

    # Whether to use QK normalization for text expert
    qk_norm_for_text: bool = False

    # Whether to use QK normalization for diffusion expert
    qk_norm_for_diffusion: bool = True  # Whether to use QK normalization for diffusion expert

    # If True, use the same word embedding matrices for input and outut embedding layers.
    tie_word_embeddings: bool = False

    # Whether to prepend a system prompt during text tokenization.
    # Checkpoints trained with system prompt enabled require this to be True at inference time.
    use_system_prompt: bool = False

    # If set, forces safetensors weight remapping ("qwen3" vs "nemotron_3_dense_vl"/"nemotron_3_llm"). None = auto-detect.
    vlm_checkpoint_format: str | None = None


# Configs for LLM models
Qwen3MoT_LLM_0p6b_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-0.6B",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/llm/qwen3/configs/Qwen3-0.6B.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-0.6B",
        config_variant="hf",
    ),
    checkpoint_path="s3://checkpoints-us-east-1/cosmos3/pretrained/huggingface/Qwen/Qwen3-0.6B/",
    credential_path="credentials/s3_training.secret",
    load_pretrained=True,
)

Qwen3MoT_LLM_0p6b_GCP_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-0.6B",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/llm/qwen3/configs/Qwen3-0.6B.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-0.6B",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Qwen/Qwen3-0.6B/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
)

Nemotron3_LLM_2b_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/NVIDIA-Nemotron-3-2B-BF16",
    model_instance=L(Nemotron3DenseVLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Nemotron3DenseVLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=False,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=False,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Nemotron/NVIDIA-Nemotron-3-2B-BF16",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Nemotron/NVIDIA-Nemotron-3-2B-BF16/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
    vlm_checkpoint_format="nemotron_3_llm",
)

# Configs for VL instruct models

# Config for Qwen3VL 30B A3B Instruct model
# Qwen3VLMoE uses Qwen2Tokenizer
Qwen3VLMoT_VLM_30b_a3b_Instruct_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
    model_instance=L(Qwen3VLMoeTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoeTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl_moe/configs/Qwen3-VL-30B-A3B-Instruct.json"
            ),
            layer_module="Qwen3VLMoeTextMoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-30B-A3B-Instruct",
        config_variant="s3",
    ),
    checkpoint_path="s3://checkpoints-us-east-1/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-30B-A3B-Instruct/",
    credential_path="credentials/s3_training.secret",
    load_pretrained=True,
)


Qwen3VLMoT_VLM_30b_a3b_Instruct_GCP_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
    model_instance=L(Qwen3VLMoeTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoeTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl_moe/configs/Qwen3-VL-30B-A3B-Instruct.json"
            ),
            layer_module="Qwen3VLMoeTextMoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-30B-A3B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-30B-A3B-Instruct/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)

CosmosReason2_VLM_30b_a3b_Private_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos-Reason2-30B-A3B-Private",
    model_instance=L(Qwen3VLMoeTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoeTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl_moe/configs/Qwen3-VL-30B-A3B-Instruct.json"
            ),
            layer_module="Qwen3VLMoeTextMoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-30B-A3B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos-Reason2-30B-A3B-Private/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)

# Config for Qwen3VL 235B A22B Instruct model
# Qwen3VLMoE uses Qwen2Tokenizer
Qwen3VLMoT_VLM_235b_a22b_Instruct_GCP_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-235B-A22B-Instruct",
    model_instance=L(Qwen3VLMoeTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoeTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl_moe/configs/Qwen3-VL-235B-A22B-Instruct.json"
            ),
            layer_module="Qwen3VLMoeTextMoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-235B-A22B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-235B-A22B-Instruct/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)


# Config for Qwen3VL 2B Instruct model
# Qwen3VL uses Qwen2Tokenizer
Qwen3VLMoT_VLM_2b_Instruct_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-2B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-2B-Instruct",
        config_variant="s3",
    ),
    checkpoint_path="s3://checkpoints-us-east-1/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-2B-Instruct/",
    credential_path="credentials/s3_training.secret",
    load_pretrained=True,
)

Qwen3VLMoT_VLM_2b_Instruct_GCP_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-2B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-2B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-2B-Instruct/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)

Qwen3VLMoT_VLM_2b_Instruct_HF_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-2B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-2B-Instruct",
        config_variant="hf",
    ),
    load_pretrained=True,
)

Nemotron3DenseVL_VLM_2b_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Nemotron-3-Dense-VL-2B-BF16-Alignment",
    model_instance=L(Nemotron3DenseVLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Nemotron3DenseVLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=False,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=False,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Nemotron/NVIDIA-Nemotron-3-Dense-VL-2B-BF16-Alignment",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Nemotron/NVIDIA-Nemotron-3-Dense-VL-2B-BF16-Alignment/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
    vlm_checkpoint_format="nemotron_3_dense_vl",
)

Cosmos3Reasoner_Nemotron_VLM_2b_Private_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Reasoner-2B-Private",
    model_instance=L(Nemotron3DenseVLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Nemotron3DenseVLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=False,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=False,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Nemotron/NVIDIA-Nemotron-3-Dense-VL-2B-BF16-Alignment",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/nvidia/Cosmos3-Reasoner-2B-Private/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
    vlm_checkpoint_format="nemotron_3_dense_vl",
)

CosmosReason2_VLM_2b_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos-Reason2-2B",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-2B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-2B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos-Reason2-2B/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)

CosmosReason2_VLM_2b_Private_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos-Reason2-2B-Private",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-2B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-2B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos-Reason2-2B-Private/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)

Cosmos3Reasoner_VLM_2b_Private_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Reasoner-2B-Private",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-2B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-2B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Reasoner-2B-Private/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)

# Config for Qwen3VL 4B Instruct model
# Qwen3VL uses Qwen2Tokenizer
Qwen3VLMoT_VLM_4b_Instruct_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-4B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-4B-Instruct",
        config_variant="s3",
    ),
    checkpoint_path="s3://checkpoints-us-east-1/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-4B-Instruct/",
    credential_path="credentials/s3_training.secret",
    load_pretrained=True,
)

Qwen3VLMoT_VLM_4b_Instruct_GCP_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-4B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-4B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-4B-Instruct/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)

# Config for Qwen3VL 8B Instruct model
# Qwen3VL uses Qwen2Tokenizer
Qwen3VLMoT_VLM_8b_Instruct_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-8B-Instruct",
        config_variant="s3",
    ),
    checkpoint_path="s3://checkpoints-us-east-1/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-8B-Instruct/",
    credential_path="credentials/s3_training.secret",
    load_pretrained=True,
)

Qwen3VLMoT_VLM_8b_Instruct_GCP_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-8B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-8B-Instruct/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)

CosmosReason2_VLM_8b_Private_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos-Reason2-8B-Private",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-8B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos-Reason2-8B-Private/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)

Cosmos3Reasoner_VLM_8b_Private_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Reasoner-8B-Private",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-8B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Reasoner-8B-Private/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)

Cosmos3NanoReasoner_VLM_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Nano-Reasoner",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(create_qwen2_tokenizer_with_download)(
        pretrained_model_name="Qwen/Qwen3-VL-8B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Nano-Reasoner/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)

# Config for Qwen3VL 32B Instruct model
# Qwen3VL uses Qwen2Tokenizer
Qwen3VLMoT_VLM_32b_Instruct_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-32B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-32B-Instruct",
        config_variant="s3",
    ),
    checkpoint_path="s3://checkpoints-us-east-1/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-32B-Instruct/",
    credential_path="credentials/s3_training.secret",
    load_pretrained=True,
)

Qwen3VLMoT_VLM_32b_Instruct_GCP_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-32B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-32B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-32B-Instruct/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)

CosmosReason2_VLM_32b_Private_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos-Reason2-32B-Private",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-32B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos-Reason2-32B-Private/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)

Cosmos3Reasoner_VLM_32b_Private_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Reasoner-32B-Private",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-32B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Reasoner-32B-Private/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)

Cosmos3SuperReasoner_VLM_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Super-Reasoner",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLTextConfig.from_json_file)(
                json_file="cosmos/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
            ),
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            tie_word_embeddings=True,
            freeze_und=False,
        ),
    ),
    tokenizer=L(create_qwen2_tokenizer_with_download)(
        pretrained_model_name="Qwen/Qwen3-VL-32B-Instruct",
        config_variant="gcp",
    ),
    checkpoint_path="s3://nv-00-10206-checkpoint/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Super-Reasoner/",
    credential_path="credentials/gcp_checkpoint.secret",
    load_pretrained=True,
    enable_gcs_patch_in_boto3=True,
)


def register_vlm():
    cs = ConfigStore.instance()
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_mot_0p6b",
        node=Qwen3MoT_LLM_0p6b_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_mot_0p6b_gcp",
        node=Qwen3MoT_LLM_0p6b_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="nemotron_3_llm_2b_gcp",
        node=Nemotron3_LLM_2b_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_30b_a3b_instruct",
        node=Qwen3VLMoT_VLM_30b_a3b_Instruct_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_30b_a3b_instruct_gcp",
        node=Qwen3VLMoT_VLM_30b_a3b_Instruct_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_235b_a22b_instruct_gcp",
        node=Qwen3VLMoT_VLM_235b_a22b_Instruct_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_2b_instruct",
        node=Qwen3VLMoT_VLM_2b_Instruct_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_2b_instruct_gcp",
        node=Qwen3VLMoT_VLM_2b_Instruct_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_2b_instruct_hf",
        node=Qwen3VLMoT_VLM_2b_Instruct_HF_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="nemotron_3_dense_vl_2b_gcp",
        node=Nemotron3DenseVL_VLM_2b_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_reasoner_nemotron_vlm_2b_private_gcp",
        node=Cosmos3Reasoner_Nemotron_VLM_2b_Private_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos_reason2_vlm_2b_gcp",
        node=CosmosReason2_VLM_2b_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos_reason2_vlm_2b_private_gcp",
        node=CosmosReason2_VLM_2b_Private_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_reasoner_vlm_2b_private_gcp",
        node=Cosmos3Reasoner_VLM_2b_Private_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos_reason2_vlm_8b_private_gcp",
        node=CosmosReason2_VLM_8b_Private_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_reasoner_vlm_8b_private_gcp",
        node=Cosmos3Reasoner_VLM_8b_Private_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_nano_reasoner_vlm_gcp",
        node=Cosmos3NanoReasoner_VLM_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos_reason2_vlm_32b_private_gcp",
        node=CosmosReason2_VLM_32b_Private_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_reasoner_vlm_32b_private_gcp",
        node=Cosmos3Reasoner_VLM_32b_Private_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_super_reasoner_vlm_gcp",
        node=Cosmos3SuperReasoner_VLM_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos_reason2_vlm_30b_a3b_private_gcp",
        node=CosmosReason2_VLM_30b_a3b_Private_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_4b_instruct",
        node=Qwen3VLMoT_VLM_4b_Instruct_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_4b_instruct_gcp",
        node=Qwen3VLMoT_VLM_4b_Instruct_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_8b_instruct",
        node=Qwen3VLMoT_VLM_8b_Instruct_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_8b_instruct_gcp",
        node=Qwen3VLMoT_VLM_8b_Instruct_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_32b_instruct",
        node=Qwen3VLMoT_VLM_32b_Instruct_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_32b_instruct_gcp",
        node=Qwen3VLMoT_VLM_32b_Instruct_GCP_Config,
    )
