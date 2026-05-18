# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path

import omegaconf
import pytest
from typing_extensions import TYPE_CHECKING

from cosmos3.args import (
    _CHECKPOINTS_EA,
    _CHECKPOINTS_EXPERIMENTAL,
    DEFAULT_CHECKPOINT,
    DEFAULT_CHECKPOINT_NAME,
    MODEL_MEMORY_BYTES_BY_SIZE,
    ModelMode,
    OmniSampleOverrides,
    OmniSetupOverrides,
    _get_dp_shard_size,
)
from cosmos3.common.args import ParallelismOverrides
from cosmos3.common.config import structure_config
from cosmos3.model import Cosmos3OmniConfig

if TYPE_CHECKING:
    from cosmos3._src.vfm.models.omni_mot_model import OmniMoTModel

_H100_MEMORY_BYTES = 80 * 1024**3
_GB200_MEMORY_BYTES = 192 * 1024**3


def test_dp_shard_size():
    device_memory_utilization = ParallelismOverrides().device_memory_utilization
    assert (
        _get_dp_shard_size(
            model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["8B"],
            device_memory_bytes=_H100_MEMORY_BYTES,
            device_memory_utilization=device_memory_utilization,
        )
        == 1
    )
    assert (
        _get_dp_shard_size(
            model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
            device_memory_bytes=_H100_MEMORY_BYTES,
            device_memory_utilization=device_memory_utilization,
        )
        == 2
    )
    assert (
        _get_dp_shard_size(
            model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["8B"],
            device_memory_bytes=_GB200_MEMORY_BYTES,
            device_memory_utilization=device_memory_utilization,
        )
        == 1
    )
    assert (
        _get_dp_shard_size(
            model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
            device_memory_bytes=_GB200_MEMORY_BYTES,
            device_memory_utilization=device_memory_utilization,
        )
        == 1
    )


def test_build_parallelism(monkeypatch: pytest.MonkeyPatch):
    parallelism_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir="outputs",
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["8B"],
        parallelism_preset="latency",
    ).build_parallelism(world_size=16, device_memory_bytes=_H100_MEMORY_BYTES)
    assert parallelism_args.dp_shard_size == 1
    assert parallelism_args.dp_replicate_size == 16
    assert parallelism_args.cp_size == 8
    assert parallelism_args.cfgp_size == 2

    parallelism_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir="outputs",
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["8B"],
        parallelism_preset="throughput",
    ).build_parallelism(world_size=16, device_memory_bytes=_H100_MEMORY_BYTES)
    assert parallelism_args.dp_shard_size == 1
    assert parallelism_args.dp_replicate_size == 16
    assert parallelism_args.cp_size == 1
    assert parallelism_args.cfgp_size == 1

    parallelism_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir="outputs",
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
        parallelism_preset="latency",
    ).build_parallelism(world_size=16, device_memory_bytes=_H100_MEMORY_BYTES)
    assert parallelism_args.dp_shard_size == 2
    assert parallelism_args.dp_replicate_size == 8
    assert parallelism_args.cp_size == 8
    assert parallelism_args.cfgp_size == 2

    parallelism_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir="outputs",
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
        parallelism_preset="throughput",
    ).build_parallelism(world_size=16, device_memory_bytes=_H100_MEMORY_BYTES)
    assert parallelism_args.dp_shard_size == 2
    assert parallelism_args.dp_replicate_size == 8
    assert parallelism_args.cp_size == 1
    assert parallelism_args.cfgp_size == 1

    parallelism_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir="outputs",
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
        parallelism_preset="latency",
    ).build_parallelism(world_size=0, device_memory_bytes=_H100_MEMORY_BYTES)
    assert parallelism_args.dp_shard_size == 2
    assert parallelism_args.dp_replicate_size == 1
    assert parallelism_args.cp_size == 1
    assert parallelism_args.cfgp_size == 1

    parallelism_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir="outputs",
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["8B"],
        parallelism_preset="latency",
        compile_dynamic=False,
    ).build_parallelism(world_size=16, device_memory_bytes=_H100_MEMORY_BYTES)
    assert parallelism_args.compile_dynamic is False


def _normalize_s3_uri(uri: str) -> str:
    # Format '{project}/{group}/{name}/checkpoints/iter_{iter}/model'
    uri = uri.rstrip("/").removesuffix("/model")
    parts = Path(uri).parts
    assert len(parts) >= 5
    return "/".join(parts[-5:])


def test_checkpoints():
    assert set(_CHECKPOINTS_EA).issubset(_CHECKPOINTS_EXPERIMENTAL)
    for name, ckpt in _CHECKPOINTS_EXPERIMENTAL.items():

        if name in ["Cosmos3-2B-Action-Libero"]:
            continue

        assert ckpt.hf.repository.split("/")[0] == "nvidia"

        # Download a file to ensure that the repository/revision is valid
        ckpt_hf = ckpt.hf.model_copy(update=dict(include=("checkpoint.json",)))
        cfg = json.loads((Path(ckpt_hf.download()) / "checkpoint.json").read_text())
        s3_uri = cfg["checkpoint_path"]
        assert _normalize_s3_uri(ckpt.s3_uri) == _normalize_s3_uri(s3_uri)

        if name in _CHECKPOINTS_EA:
            ckpt_ea = _CHECKPOINTS_EA[name]
            assert _normalize_s3_uri(ckpt_ea.s3_uri) == _normalize_s3_uri(s3_uri)
            assert ckpt_ea.hf.repository.split("/")[0] == "nvidia-cosmos-ea"

            # Download a file to ensure that the repository/revision is valid
            ckpt_ea_hf = ckpt_ea.hf.model_copy(update=dict(include=("checkpoint.json",)))
            # Check that checkpoints are identical
            cfg_ea = json.loads((Path(ckpt_ea_hf.download()) / "checkpoint.json").read_text())
            assert cfg_ea == cfg


def test_setup_args(tmp_path: Path):
    overrides = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=tmp_path / "outputs",
    )
    args = overrides.build_setup()

    # Check idempotent
    assert overrides.build_setup() == args
    assert OmniSetupOverrides.model_validate(args.model_dump()).build_setup() == args


def test_sample_args(tmp_path: Path):
    hf_config = Cosmos3OmniConfig.from_pretrained(
        **DEFAULT_CHECKPOINT.pretrained_kwargs,
    )
    model_dict: "OmniMoTModel" = structure_config(hf_config.model, omegaconf.DictConfig)

    # Check that all fields are optional
    for name, field in OmniSampleOverrides.model_fields.items():
        assert field.default is None, name

    overrides = OmniSampleOverrides(
        name="test",
    )
    overrides.output_dir = tmp_path / "inputs"
    args = overrides.build_sample(model_config=model_dict.config)

    # Check idempotent
    assert overrides.build_sample(model_config=model_dict.config) == args
    overrides_dump = {k: v for k, v in args.model_dump().items() if k in OmniSampleOverrides.model_fields}
    assert OmniSampleOverrides.model_validate(overrides_dump).build_sample(model_config=model_dict.config) == args

    text2image_args = OmniSampleOverrides(
        name="text2image",
        output_dir=tmp_path / "text2image",
        model_mode=ModelMode.TEXT2IMAGE,
    ).build_sample(model_config=model_dict.config)
    assert text2image_args.aspect_ratio == "1,1"
    assert text2image_args.num_steps == 50
    assert text2image_args.guidance == 4.0
    assert text2image_args.shift == 3.0
