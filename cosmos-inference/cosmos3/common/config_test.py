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

import importlib
from pathlib import Path
from typing import Type

import attrs
import omegaconf
import pytest
import torch

from cosmos3.common.args import DEFAULT_CONFIG_FILE
from cosmos3.common.config import (
    _is_type_cls,
    apply_config_replacements,
    config_converter,
    deserialize_config,
    load_config,
    serialize_config,
    structure_config,
    undo_config_replacements,
    unstructure_config,
)
from cosmos3._src.imaginaire.flags import TRAINING
from cosmos3._src.imaginaire.lazy_config import LazyCall as L
from cosmos3._src.imaginaire.lazy_config.registry import convert_target_to_string


def test_is_type():
    assert not _is_type_cls(int)
    assert _is_type_cls(type[int])
    assert _is_type_cls(Type[int])


@attrs.define
class Config:
    tp: type[int]

    list_config: omegaconf.ListConfig

    x: int

    device: torch.device
    dtype: torch.dtype
    layout: torch.layout
    memory_format: torch.memory_format


@attrs.define
class Cls:
    x: int = 5


def test_config_converter():
    def round_trip(obj):
        return config_converter.structure(config_converter.unstructure(obj), type(obj))

    tensor = torch.Tensor([1, 2, 3])
    assert torch.equal(round_trip(tensor), tensor)

    config = Config(
        tp=int,
        list_config=omegaconf.ListConfig(
            [
                omegaconf.OmegaConf.structured(Cls(x=1)),
                L(Cls)(x=2),
            ]
        ),
        x=1,
        device=torch.device("cuda"),
        dtype=torch.float32,
        layout=torch.strided,
        memory_format=torch.preserve_format,
    )

    config_dict = unstructure_config(config)
    assert config_dict == {
        "_type": convert_target_to_string(Config),
        "tp": convert_target_to_string(int),
        "list_config": [
            {
                "_type": convert_target_to_string(Cls),
                "x": 1,
            },
            {
                "_target_": convert_target_to_string(Cls),
                "x": 2,
            },
        ],
        "x": 1,
        "device": "cuda",
        "dtype": "float32",
        "layout": "strided",
        "memory_format": "preserve_format",
    }

    structured_config = attrs.evolve(
        config,
        list_config=omegaconf.ListConfig(
            [
                dict(_type=convert_target_to_string(Cls), x=1),
                dict(_target_=convert_target_to_string(Cls), x=2),
            ]
        ),
    )
    assert structure_config(config_dict, Config) == structured_config

    # Test missing fields are populated with defaults
    for i in range(2):
        del config_dict["list_config"][i]["x"]
        structured_config.list_config[i].x = 5
    assert structure_config(config_dict, Config) == structured_config



if TRAINING:

    @pytest.mark.parametrize("config_file", sorted(set([DEFAULT_CONFIG_FILE])))
    def test_make_config(config_file: str):
        from cosmos3._src.imaginaire.utils import config_helper

        config_module = importlib.import_module(config_helper.get_config_module(config_file))
        config_module.make_config()

    def test_serialize_config(tmp_path: Path):
        config = load_config(
            config_file="cosmos3/_src/vfm/configs/base/config.py",
            experiment="t2w_mot_dryrun_exp100_006_qwen3_0p6b_256res_resume_from_t2i",
        )

        for suffix in [".yaml", ".json"]:
            config_file = tmp_path / f"config{suffix}"
            serialize_config(config, config_file)
            deserialize_config(config_file, type(config))
