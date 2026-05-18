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

import functools
from uuid import uuid4

import pydantic

from cosmos3.flags import EARLY_ACCESS
from cosmos3._src.imaginaire.flags import TRAINING
from cosmos3._src.imaginaire.utils.checkpoint_db import (
    CheckpointConfig,
    CheckpointDirHf,
    CheckpointDirS3,
    CheckpointFileHf,
    CheckpointFileS3,
    RepositoryType,
    register_checkpoint,
)


@functools.cache
def register_checkpoints():
    for repository, revision in [
        ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
        ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"),
        ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
        ("Qwen/Qwen3-VL-32B-Instruct", "0cfaf48183f594c314753d30a4c4974bc75f3ccb"),
    ]:
        # Used by 'cosmos3/_src/vfm/configs/base/defaults/vlm.py::download_tokenizer_files'.
        register_checkpoint(
            CheckpointConfig(
                uuid=uuid4().hex,
                name=repository,
                s3=CheckpointDirS3(
                    uri=f"s3://bucket/cosmos3/pretrained/huggingface/{repository}",
                ),
                hf=CheckpointDirHf(
                    repository=repository,
                    revision=revision,
                    include=() if TRAINING else ("*.json", "*.txt"),
                ),
            ),
        )

    register_checkpoint(
        CheckpointConfig(
            uuid=uuid4().hex,
            name="Cosmos3-Nano-Reasoner",
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Reasoner-8B-Private",
            ),
            hf=CheckpointDirHf(
                repository="nvidia/Cosmos3-Nano-Reasoner",
                revision="6406357cdc32fbf8db5f51ff7992343803b06961",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid=uuid4().hex,
            name="Cosmos3-Super-Reasoner",
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Reasoner-32B-Private",
            ),
            hf=CheckpointDirHf(
                repository="nvidia/Cosmos3-Super-Reasoner",
                revision="b9b716f3508dfa442e0c8ba32fb5d0c9adf2a32c",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid=uuid4().hex,
            name="Wan2.1/vae",
            s3=CheckpointFileS3(
                uri="s3://bucket/pretrained/tokenizers/video/wan2pt1/Wan2.1_VAE.pth",
            ),
            hf=CheckpointFileHf(
                repository="Wan-AI/Wan2.1-T2V-14B",
                revision="a064a6c71f5be440641209c07bf2a5ce7a2ff5e4",
                filename="Wan2.1_VAE.pth",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid=uuid4().hex,
            name="Wan2.2/vae",
            s3=CheckpointFileS3(
                uri="s3://bucket/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
            ),
            hf=CheckpointFileHf(
                repository="Wan-AI/Wan2.2-TI2V-5B",
                revision="921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
                filename="Wan2.2_VAE.pth",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid=uuid4().hex,
            name="AVAE",
            s3=CheckpointDirS3(
                uri="s3://bucket/pretrained/tokenizers/audio/avae",
            ),
            hf=CheckpointDirHf(
                repository="nvidia/Cosmos3-Experimental",
                revision="c243efd72b3c9138196ba903deb4a0ad26f2bf20",
                subdirectory="avae",
            ),
        ),
    )


class DatasetConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    hf: CheckpointDirHf
    """Config for dataset on Hugging Face."""


_DATASETS_EXPERIMENTAL = {
    "nvidia/bridge-v2-subset-synthetic-captions": DatasetConfig(
        hf=CheckpointDirHf(
            repository_type=RepositoryType.DATASET,
            repository="nvidia/bridge-v2-subset-synthetic-captions",
            revision="46468e12ac0dd36901e9e3240d4fc7620942b5d7",
        ),
    ),
    "nvidia/LIBERO_LeRobot_v3": DatasetConfig(
        hf=CheckpointDirHf(
            repository_type=RepositoryType.DATASET,
            repository="nvidia/LIBERO_LeRobot_v3",
            revision="ddc1edeb6e51e2b7d4d2ba7a1433daaecd37aa64",
        ),
    ),
    "nvidia/bridge_lerobot_v3": DatasetConfig(
        hf=CheckpointDirHf(
            repository_type=RepositoryType.DATASET,
            repository="nvidia/bridge_lerobot_v3",
            revision="b887e193b141f2fe5b6e3d567577aa51c475693b",
        ),
    ),
}

_DATASETS_EA = {}
for name in [
    "nvidia/bridge-v2-subset-synthetic-captions",
    "nvidia/LIBERO_LeRobot_v3",
    "nvidia/bridge_lerobot_v3",
]:
    dataset = _DATASETS_EXPERIMENTAL[name]
    _DATASETS_EA[name] = dataset.model_copy(
        update=dict(
            hf=dataset.hf.model_copy(
                update=dict(
                    repository=name.replace("nvidia/", "nvidia-cosmos-ea/"),
                )
            ),
        )
    )

DATASETS = _DATASETS_EXPERIMENTAL.copy()
if EARLY_ACCESS:
    DATASETS.update(_DATASETS_EA)
