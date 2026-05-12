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

from cosmos3._src.imaginaire.trainer import ImaginaireTrainer
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.config_helper import import_all_modules_from_package
from cosmos3._src.vfm.configs.base.vlm.defaults.callbacks import register_callbacks
from cosmos3._src.vfm.configs.base.vlm.defaults.checkpointer import register_checkpoint, register_ckpt_type
from cosmos3._src.vfm.configs.base.vlm.defaults.config import Config
from cosmos3._src.vfm.configs.base.vlm.defaults.dataloader import register_data_debug
from cosmos3._src.vfm.configs.base.vlm.defaults.dataloader_weighted_url import (
    register_data_recipe,
    register_data_weighted_url,
    register_data_weighted_url_with_text,
)
from cosmos3._src.vfm.configs.base.vlm.defaults.model import register_model
from cosmos3._src.vfm.configs.base.vlm.defaults.optimizer import register_optimizer, register_scheduler
from cosmos3._src.vfm.configs.base.vlm.defaults.vlm_policy import register_vlm_policy


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    c.job.project = "cosmos_reason2"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    # Unified path: ImaginaireTrainer drives both VLM and VFM.
    c.trainer.type = ImaginaireTrainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 20
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None
    c.trainer.cudnn.benchmark = False
    c.upload_reproducible_setup = True

    # Call this function to register config groups for advanced overriding. the order follows the default config groups
    register_model()
    register_vlm_policy()
    # Register dataloader configs
    register_data_weighted_url()
    register_data_recipe()
    register_data_weighted_url_with_text()
    register_data_debug()
    log.info("Registering optimizer, scheduler, checkpoint, ckpt type, and callbacks")
    register_optimizer()
    register_scheduler()
    register_checkpoint()
    register_ckpt_type()
    register_callbacks()
    import_all_modules_from_package("cosmos3._src.vfm.configs.base.vlm.experiment", reload=True)
    return c
