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

"""Dataloader config options.
Based on projects/cosmos/ar/v1/configs/registry.py
"""

from hydra.core.config_store import ConfigStore

from cosmos3._src.imaginaire.callbacks.manual_gc import ManualGarbageCollection
from cosmos3._src.imaginaire.lazy_config import PLACEHOLDER
from cosmos3._src.imaginaire.lazy_config import LazyCall as L
from cosmos3._src.imaginaire.utils.callback import WandBCallback
from cosmos3._src.vfm.callbacks.dataloader_state import DataLoaderStateCallback
from cosmos3._src.vfm.callbacks.dataloading_monitor import DetailedDataLoadingSpeedMonitor
from cosmos3._src.vfm.callbacks.hf_export import HFExportCallback
from cosmos3._src.vfm.callbacks.low_precision import LowPrecisionCallback
from cosmos3._src.vfm.callbacks.vlm.grad_clip import GradClip
from cosmos3._src.vfm.configs.base.defaults.callbacks import JOB_MONITOR_CALLBACKS
from projects.cosmos3.vlm.callbacks.data_stats import DataStatsCallback
from projects.cosmos3.vlm.callbacks.iter_speed import IterSpeed
from projects.cosmos3.vlm.callbacks.log_tensor_shape import LogTensorShapeCallback
from projects.cosmos3.vlm.callbacks.param_count import ParamCount
from projects.cosmos3.vlm.callbacks.wandb_log import WandbCallback as WandBCallbackMultiplier
from projects.cosmos3.vlm.callbacks.wandb_log_eval import WandbCallback as WandBCallbackEval
from projects.cosmos3.vlm.callbacks.wandb_log_simgple import WandbCallback as WandBCallbackMultiplierSimple
from projects.cosmos3.vlm.callbacks.wandb_vis import VisualizationLoggingCallback

# from cosmos3._src.imaginaire.utils.callback import NVTXCallback


def register_callbacks():
    cs = ConfigStore.instance()
    BASIC_CALLBACKS = dict(
        iter_speed=L(IterSpeed)(  # does not use model or optimizer
            every_n="${trainer.logging_iter}",
            save_s3="${upload_reproducible_setup}",
            save_s3_every_log_n=500,
            hit_thres=50,
        ),
        manual_gc=L(ManualGarbageCollection)(every_n=5),  # does not use model or optimizer
        wandb=L(WandBCallback)(),
        param_count=L(ParamCount)(  # use model
            save_s3="${upload_reproducible_setup}",
        ),
        dataloader_speed=L(DetailedDataLoadingSpeedMonitor)(
            every_n=100,
            save_s3="${upload_reproducible_setup}",
        ),
        grad_clip=L(GradClip)(clip_norm=1.0),  # use model
        low_precision=L(LowPrecisionCallback)(
            update_iter=1,
            config=PLACEHOLDER,
            trainer=PLACEHOLDER,
        ),  # reads model.precision; no extra kwarg needed

        # nvtx=L(NVTXCallback)(synchronize=True),
    )

    PER_DATASET_PERN_CALLBACKS = dict(
        wandb_10x=L(WandBCallbackMultiplier)(
            logging_iter_multipler=10,
            save_logging_iter_multipler=1,
            save_s3="${upload_reproducible_setup}",
        ),
        wandb_2x=L(WandBCallbackMultiplier)(
            logging_iter_multipler=2,
            save_logging_iter_multipler=1,
            save_s3="${upload_reproducible_setup}",
        ),
        data_stats=L(DataStatsCallback)(
            logging_iter_multipler=1,
            save_s3="${upload_reproducible_setup}",
        ),
        wandb_val=L(WandBCallbackEval)(
            save_s3="${upload_reproducible_setup}",
        ),
    )

    SIMPLE_LOG_CALLBACKS = dict(
        wandb_10x=L(WandBCallbackMultiplierSimple)(
            logging_iter_multipler=10,
            save_logging_iter_multipler=1,
            save_s3="${upload_reproducible_setup}",
        ),
        wandb_2x=L(WandBCallbackMultiplierSimple)(
            logging_iter_multipler=2,
            save_logging_iter_multipler=1,
            save_s3="${upload_reproducible_setup}",
        ),
        log_tensor_shape=L(LogTensorShapeCallback)(num_log=10),
        dataloader_state=L(DataLoaderStateCallback)(
            distributor_type="${data_setting.distributor_type}",
        ),
    )
    cs.store(group="callbacks", package="trainer.callbacks", name="basic_vlm", node=BASIC_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="per_dataset", node=PER_DATASET_PERN_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="simple_log", node=SIMPLE_LOG_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="job_monitor", node=JOB_MONITOR_CALLBACKS)

    DATA_VIS_CALLBACKS_QWEN = dict(
        wandb_vis=L(VisualizationLoggingCallback)(
            every_n=500,
        ),
    )
    cs.store(group="callbacks", package="trainer.callbacks", name="data_vis_qwen", node=DATA_VIS_CALLBACKS_QWEN)

    HF_EXPORT_CALLBACKS = dict(
        hf_export=L(HFExportCallback)(
            dtype="${model.config.policy.parallelism.precision}",
        ),
    )
    cs.store(group="callbacks", package="trainer.callbacks", name="hf_export", node=HF_EXPORT_CALLBACKS)
