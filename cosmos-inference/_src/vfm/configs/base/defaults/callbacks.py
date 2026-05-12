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

"""Dataloader config options."""

from hydra.core.config_store import ConfigStore

from cosmos3._src.imaginaire.callbacks.manual_gc import ManualGarbageCollection
from cosmos3._src.imaginaire.lazy_config import PLACEHOLDER
from cosmos3._src.imaginaire.lazy_config import LazyCall as L
from cosmos3._src.imaginaire.utils.callback import WandBCallback
from cosmos3._src.vfm.callbacks.compile_tokenizer import CompileTokenizer
from cosmos3._src.vfm.callbacks.dataloading_monitor import DetailedDataLoadingSpeedMonitor
from cosmos3._src.vfm.callbacks.device_monitor import DeviceMonitor
from cosmos3._src.vfm.callbacks.every_n_draw_sample import EveryNDrawSample
from cosmos3._src.vfm.callbacks.expert_heatmap import ExpertHeatmap
from cosmos3._src.vfm.callbacks.grad_clip import GradClip
from cosmos3._src.vfm.callbacks.heart_beat import HeartBeat
from cosmos3._src.vfm.callbacks.iter_speed import IterSpeed
from cosmos3._src.vfm.callbacks.load_pretrained import LoadPretrained
from cosmos3._src.vfm.callbacks.low_precision import LowPrecisionCallback
from cosmos3._src.vfm.callbacks.mfu import MFUCallback
from cosmos3._src.vfm.callbacks.moe_specialization_callback import MoESpecializationCallback
from cosmos3._src.vfm.callbacks.moe_stability_callback import MoEStabilityCallback
from cosmos3._src.vfm.callbacks.norm_monitor import NormMonitor
from cosmos3._src.vfm.callbacks.ofu import OFUCallback
from cosmos3._src.vfm.callbacks.param_count import ParamCount
from cosmos3._src.vfm.callbacks.sequence_packing_padding import SequencePackingPadding
from cosmos3._src.vfm.callbacks.sigma_loss_analysis import SigmaLossAnalysis
from cosmos3._src.vfm.callbacks.skip_nan_step import SkipNaNStep
from cosmos3._src.vfm.callbacks.training_stats import TrainingStatsCallback
from cosmos3._src.vfm.callbacks.wandb_log import WandbCallback as WandBCallbackMultiplier
from cosmos3._src.vfm.callbacks.wandb_log_eval import WandbCallback as WandBCallbackEval

BASIC_CALLBACKS = dict(
    iter_speed=L(IterSpeed)(  # does not use model or optimizer
        every_n="${trainer.logging_iter}",
        save_s3="${upload_reproducible_setup}",
        save_s3_every_log_n=500,
        hit_thres=50,
    ),
    manual_gc=L(ManualGarbageCollection)(every_n=5),  # does not use model or optimizer
    wandb=L(WandBCallback)(),
    wandb_2x=L(WandBCallbackMultiplier)(
        logging_iter_multipler=2,
        save_logging_iter_multipler=1,
        save_s3="${upload_reproducible_setup}",
    ),
    param_count=L(ParamCount)(  # use model
        save_s3="${upload_reproducible_setup}",
    ),
    dataloader_speed=L(DetailedDataLoadingSpeedMonitor)(
        every_n=100,
        save_s3="${upload_reproducible_setup}",
    ),
    wandb_val=L(WandBCallbackEval)(
        save_s3="${upload_reproducible_setup}",
    ),
    moe_stability=L(MoEStabilityCallback)(every_n=250),
    moe_specialization=L(MoESpecializationCallback)(every_n=250),
    expert_heatmap=L(ExpertHeatmap)(),
    load_pretrained=L(LoadPretrained)(),
    compile_tokenizer=L(CompileTokenizer)(enabled=False, compile_after_iterations=3),
    norm_monitor=L(NormMonitor)(
        every_n=5000,
        log_stat_wandb=True,
        save_s3="${upload_reproducible_setup}",
        track_activations=True,
    ),
    sigma_loss_analysis=L(SigmaLossAnalysis)(
        every_n=5000,
        every_n_viz=5000,
        save_s3="${upload_reproducible_setup}",
    ),
    sequence_packing_padding=L(SequencePackingPadding)(every_n="${trainer.logging_iter}"),
    mfu=L(MFUCallback)(every_n="${trainer.logging_iter}", grad_accum_iter="${trainer.grad_accum_iter}"),
    ofu=L(OFUCallback)(every_n="${trainer.logging_iter}"),
)

JOB_MONITOR_CALLBACKS = dict(
    heart_beat=L(HeartBeat)(
        every_n=200,
        update_interval_in_minute=20,
        save_s3="${upload_reproducible_setup}",
    ),
    device_monitor=L(DeviceMonitor)(
        every_n=200,
        save_s3="${upload_reproducible_setup}",
        upload_every_n_mul=5,
    ),
)

OPTIMIZATION_CALLBACKS = dict(
    skip_nan_step=L(SkipNaNStep)(max_consecutive_nan=100),
    grad_clip=L(GradClip)(clip_norm=1.0),  # use model, not supported yet
    low_precision=L(LowPrecisionCallback)(update_iter=1, config=PLACEHOLDER, trainer=PLACEHOLDER),  # use model
)

VIZ_ONLINE_SAMPLING_CALLBACKS = dict(
    every_n_sample_reg=L(EveryNDrawSample)(
        every_n=5000,
        save_s3=True,
        do_x0_prediction=False,
    ),
    every_n_sample_ema=L(EveryNDrawSample)(
        every_n=5000,
        is_ema=True,
        save_s3=True,
        do_x0_prediction=False,
    ),
)


def register_callbacks():
    cs = ConfigStore.instance()
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=BASIC_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="job_monitor", node=JOB_MONITOR_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="optimization", node=OPTIMIZATION_CALLBACKS)
    # Online sampling generation callback
    cs.store(
        group="callbacks", package="trainer.callbacks", name="viz_online_sampling", node=VIZ_ONLINE_SAMPLING_CALLBACKS
    )
    # Register "generation" as alias for "viz_online_sampling" (expected by base config.py defaults)
    cs.store(group="callbacks", package="trainer.callbacks", name="generation", node=VIZ_ONLINE_SAMPLING_CALLBACKS)

    TRAINING_STATS_CALLBACKS = dict(
        training_stats=L(TrainingStatsCallback)(
            log_freq=100,
        )
    )
    cs.store(group="callbacks", package="trainer.callbacks", name="training_stats", node=TRAINING_STATS_CALLBACKS)
