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

from cosmos3.common.init import init_script

init_script(
    training=True,
    env={"COSMOS_TRAINING": "1"},
    default_env={"COSMOS_VERBOSE": "1"},
)

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import hydra
import omegaconf
import pydantic
import tyro

from cosmos3.common.args import ResolvedFilePath, ResolvedPath, tyro_cli
from cosmos3.common.checkpoints import register_checkpoints
from cosmos3.common.config import (
    ROOT_DIR,
    TYPE_KEY,
    deserialize_config_dict,
    serialize_config,
    structure_config,
)
from cosmos3.common.init import init_output_dir, is_rank0
from cosmos3._src.imaginaire.flags import SMOKE
from cosmos3._src.imaginaire.trainer import ImaginaireTrainer
from cosmos3._src.imaginaire.utils import log

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from cosmos3._src.imaginaire.config import Config
    from cosmos3._src.vfm.models.omni_mot_model import OmniMoTModel


def _validate_config_file(v: Path) -> Path:
    if v.suffix != ".yaml":
        raise ValueError(f"Config file must be a YAML file: {v}")
    return v


ConfigFilePath = Annotated[ResolvedFilePath, pydantic.AfterValidator(_validate_config_file)]


class Args(pydantic.BaseModel):
    output_dir: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]

    config_file: ConfigFilePath
    """Hydra config yaml file."""
    config_overrides: list[str] = pydantic.Field(default_factory=list)
    """Hydra config overrides."""

    dry_run: bool = False
    """Dry run (no training)."""


def _get_config_overrides(args: Args, config_dict: dict) -> list[str]:
    model_name = config_dict["model"]["config"]["vlm_config"]["model_name"]
    overrides = [
        *args.config_overrides,
    ]
    if SMOKE:
        overrides.extend(
            [
                "trainer.max_iter=2",
                "trainer.logging_iter=1",
            ]
        )
        if model_name.startswith("Qwen/Qwen3-VL-"):
            overrides.extend(
                [
                    "model.config.vlm_config.model_instance.config.num_hidden_layers=2",
                    "model.config.vlm_config.model_instance.config.num_window_layers=2",
                ]
            )
    return overrides


def _get_job_dir(config: "Config") -> Path:
    output_root = Path(os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output"))
    return output_root / config.job.project / config.job.group / config.job.name


def train(args: Args) -> None:
    init_output_dir(args.output_dir)

    # Create config
    config_dict = deserialize_config_dict(args.config_file)
    overrides = _get_config_overrides(args, config_dict)
    log.debug(f"Config overrides: {overrides}")
    overrides_omegaconf = omegaconf.OmegaConf.from_dotlist(overrides)
    config_omegaconf = omegaconf.OmegaConf.merge(config_dict, overrides_omegaconf)

    # Finalize config
    omegaconf.OmegaConf.save(config_omegaconf, args.output_dir / "config_raw.yaml")
    omegaconf.OmegaConf.resolve(config_omegaconf)
    config: "Config" = structure_config(config_omegaconf, config_omegaconf[TYPE_KEY])
    config.validate()
    config.freeze()  # type: ignore
    serialize_config(config, args.output_dir / "config.yaml")

    # Instantiate
    register_checkpoints()
    with contextlib.chdir(ROOT_DIR):
        # Trainer init sets the rank-local CUDA device before tokenizers allocate weights.
        trainer: "ImaginaireTrainer" = config.trainer.type(config)
        model: "OmniMoTModel" = hydra.utils.instantiate(config.model)
        dataloader_train: "DataLoader" = hydra.utils.instantiate(config.dataloader_train)
        dataloader_val: "DataLoader" = hydra.utils.instantiate(config.dataloader_val)

    if is_rank0():
        # Symlink job directory
        job_dir = _get_job_dir(config)
        job_dir.mkdir(parents=True, exist_ok=True)
        if (args.output_dir / "job").exists():
            os.remove(args.output_dir / "job")
        os.symlink(job_dir, args.output_dir / "job")
        log.info(f"Job directory: {job_dir}")

    if args.dry_run:
        return

    # Start training
    trainer.train(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
    )


def main() -> None:
    args = tyro_cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    train(args)


if __name__ == "__main__":
    main()
