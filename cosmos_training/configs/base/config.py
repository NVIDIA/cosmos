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

from typing import Any, List

import attrs
from omegaconf import OmegaConf

from cosmos.utils import config
from cosmos.trainer import ImaginaireTrainer as Trainer
from cosmos.utils.config_helper import import_all_modules_from_package
from configs.base.defaults.model_config import ModelConfig


@attrs.define(slots=False)
class DataSetting:
    """Configuration for data.

    Attributes:
        qwen_max_video_token_length: Maximum video token length.
        qwen_target_fps: Target fps for video sampling.
        text_chat_order: Order of text items in user messages.
    """

    qwen_max_video_token_length: int = 8192


@attrs.define(slots=False)
class Config(config.Config):
    data_setting: DataSetting = attrs.field(factory=DataSetting)
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": "mot_fsdp"},
            {"data_train": None},
            {"data_val": None},
            {"optimizer": "adamw"},
            {"scheduler": "warmup_cosine_lr"},
            {"checkpoint": "s3"},
            {"callbacks": ["basic", "optimization", "job_monitor", "generation"]},
            {"ema": "power"},
            {"tokenizer": "wan2pt2_tokenizer"},
            {"sound_tokenizer": None},  # Optional: for audio-video generation
            {"cluster": "gcp_iad_gb200"},
            {"vlm_config": None},
            {"ckpt_type": "dcp"},
            {"experiment": None},
        ]
    )

    def validate(self) -> None:
        super().validate()
        self._dispatch_model_config_validate()

    def _dispatch_model_config_validate(self) -> None:
        """Run model-family validation on the composed model.config.

        validate() runs before instantiate(), so self.model.config is a
        DictConfig wrapping the structured schema rather than the attrs class.
        DictConfig surfaces fields but not methods, so to drive the typed
        isinstance dispatch the schema must first be materialized via
        OmegaConf.to_object.
        """
        materialized = OmegaConf.to_object(self.model.config)
        if isinstance(materialized, ModelConfig):
            materialized.validate(self)


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    c.job.project = "cosmos3_vfm"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 20
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    c.upload_reproducible_setup = False

    from configs.base.defaults.callbacks import register_callbacks
    from configs.base.defaults.checkpointer import register_checkpoint, register_ckpt_type
    from configs.base.defaults.cluster import register_cluster
    from configs.base.defaults.ema import register_ema

    # from configs.base.defaults.data import register_data
    from configs.base.defaults.model import register_model
    from configs.base.defaults.optimizer import register_optimizer, register_scheduler
    from configs.base.defaults.tokenizer import register_sound_tokenizer, register_tokenizer
    from configs.base.defaults.vlm import register_vlm
    from configs.base.defaults.open_source_dataloader import register_open_source_dataloaders

    # Call this function to register config groups for advanced overriding. the order follows the default config groups
    # register_data()
    register_model()
    register_checkpoint()
    register_ckpt_type()
    register_optimizer()
    register_scheduler()
    register_callbacks()
    register_tokenizer()
    register_sound_tokenizer()
    register_ema()
    register_cluster()
    register_vlm()
    register_open_source_dataloaders()

    # Register only the mixed_modality_sft_8b experiment.
    # Importing the module triggers its top-level cs.store(...) call.
    # To register more experiments later, add explicit imports here
    # (or switch to import_all_modules_from_package("experiments", reload=True)
    # for auto-discovery of every .py under experiments/).
    import experiments.sft.mixed_modality_sft_8b  # noqa: F401
    return c
