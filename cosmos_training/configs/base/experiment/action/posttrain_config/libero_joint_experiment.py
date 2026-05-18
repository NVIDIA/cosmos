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

# Joint Training Experiment for LIBERO - randomly samples from all modes (FD, ID, policy, video)

from datetime import datetime

from hydra.core.config_store import ConfigStore

from configs.base.experiment.action._experiment_helpers import (
    LIBERO_BASELINE_BATCH_SIZE,
    LIBERO_BASELINE_NUM_WORKERS,
    make_libero_dataset,
)
from configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment

cs = ConfigStore.instance()

# Joint mode trains 2x longer than the libero baseline (4_000) — keep this local.
TRAINING_ITERATIONS = 8000


def _build_libero_joint_base(
    exp_name: str,
    *,
    training_iterations: int = TRAINING_ITERATIONS,
    batch_size: int = LIBERO_BASELINE_BATCH_SIZE,
    num_workers: int = LIBERO_BASELINE_NUM_WORKERS,
    job_group: str = "action_libero",
    job_project: str | None = None,
    **dataset_kwargs,
) -> dict:
    # mode="joint" is the defining trait of this experiment family.
    dataset_kwargs.setdefault("mode", "joint")
    exp = make_2b_experiment(
        exp_name=exp_name,
        datasets=make_libero_dataset(**dataset_kwargs),
        batch_size=batch_size,
        num_workers=num_workers,
        training_iterations=training_iterations,
    )

    # Match libero_experiment.libero_exp post-build mutations.
    for i, d in enumerate(exp["defaults"]):
        if isinstance(d, dict) and "override /callbacks" in d:
            exp["defaults"][i] = {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                    "training_stats",
                ]
            }
            break
    exp["scheduler"]["f_max"] = [1.0]
    exp["scheduler"]["warm_up_steps"] = [training_iterations // 20]

    exp["job"]["group"] = job_group
    if job_project is not None:
        exp["job"]["project"] = job_project
    return exp


# ---------------------------------------------------------------------------
# libero_exp_joint — base joint training (8k iters, lr=2e-4)
# ---------------------------------------------------------------------------
libero_exp_joint = _build_libero_joint_base(
    exp_name=f"libero_exp_joint_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
libero_exp_joint["optimizer"]["lr"] = 2e-4

cs.store(
    group="experiment",
    package="_global_",
    name="libero_exp_joint",
    node=libero_exp_joint,
)


# ---------------------------------------------------------------------------
# Chunk size sweep (lr=2e-4, mode=joint, varying chunk_length)
# ---------------------------------------------------------------------------
def chunk_size_sweep():
    for chunk_length in [8, 16, 24, 32, 40, 48]:
        name = f"libero_exp_lr2e4_joint_chunk{chunk_length}"
        exp = _build_libero_joint_base(
            exp_name=name,
            job_project="cosmos3_action_libero",
            chunk_length=chunk_length,
        )
        exp["optimizer"]["lr"] = 2e-4
        cs.store(group="experiment", package="_global_", name=name, node=exp)


# ---------------------------------------------------------------------------
# Sweep camera_mode × action_space × rotation_space (lr=2e-4, mode=joint)
# ---------------------------------------------------------------------------
def action_camera_rotation_sweep():
    camera_modes = ["image", "wrist_image"]
    action_spaces = ["relative", "frame_wise_relative"]
    rotation_spaces = ["9d", "6d", "3d"]

    for camera_mode in camera_modes:
        for action_space in action_spaces:
            for rotation_space in rotation_spaces:
                cam_short = "img" if camera_mode == "image" else "wrist"
                act_short = "rel" if action_space == "relative" else "fwrel"
                rot_short = rotation_space
                name = f"libero_exp_joint_{cam_short}_{act_short}_{rot_short}_varlen"

                exp = _build_libero_joint_base(
                    exp_name=name,
                    camera_mode=camera_mode,
                    action_space=action_space,
                    rotation_space=rotation_space,
                )
                exp["optimizer"]["lr"] = 2e-4
                cs.store(group="experiment", package="_global_", name=name, node=exp)


# ---------------------------------------------------------------------------
# Single-view 60k iter joint experiment (and chunk8 variant)
# ---------------------------------------------------------------------------
def single_view_joint_exp():
    """
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 scripts/train.py --config=configs/base/config.py -- experiment=libero_exp_joint_single_view_lr2e4_60k
    """
    name = "libero_exp_joint_single_view_lr2e4_60k"
    exp = _build_libero_joint_base(
        exp_name=name,
        training_iterations=60_000,
        batch_size=256,
        job_project="cosmos3_action_libero",
    )
    exp["optimizer"]["lr"] = 2e-4
    exp["scheduler"]["warm_up_steps"] = [100]
    exp["scheduler"]["cycle_lengths"] = [60_000]
    exp["checkpoint"]["load_from_object_store"] = dict(bucket="nv-00-10206-checkpoint")

    cs.store(group="experiment", package="_global_", name=name, node=exp)

    chunk8_name = "libero_exp_joint_single_view_lr2e4_60k_chunk8"
    exp_chunk8 = _build_libero_joint_base(
        exp_name=chunk8_name,
        training_iterations=60_000,
        batch_size=256,
        job_project="cosmos3_action_libero",
        chunk_length=8,
    )
    exp_chunk8["optimizer"]["lr"] = 2e-4
    exp_chunk8["scheduler"]["warm_up_steps"] = [100]
    exp_chunk8["scheduler"]["cycle_lengths"] = [60_000]
    exp_chunk8["checkpoint"]["load_from_object_store"] = dict(bucket="nv-00-10206-checkpoint")
    cs.store(group="experiment", package="_global_", name=chunk8_name, node=exp_chunk8)


# ---------------------------------------------------------------------------
# Wrist-view 60k iter joint experiment
# ---------------------------------------------------------------------------
def wrist_view_joint_exp():
    """
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 scripts/train.py --config=configs/base/config.py -- experiment=libero_exp_joint_wrist_view_lr2e4_60k
    """
    name = "libero_exp_joint_wrist_view_lr2e4_60k"
    exp = _build_libero_joint_base(
        exp_name=name,
        training_iterations=60_000,
        batch_size=256,
        job_project="cosmos3_action_libero",
        camera_mode="wrist_image",
    )
    exp["optimizer"]["lr"] = 2e-4
    exp["scheduler"]["warm_up_steps"] = [100]
    exp["scheduler"]["cycle_lengths"] = [60_000]
    exp["trainer"]["run_validation"] = False
    exp["checkpoint"]["load_from_object_store"] = dict(bucket="nv-00-10206-checkpoint")
    cs.store(group="experiment", package="_global_", name=name, node=exp)


chunk_size_sweep()
single_view_joint_exp()
wrist_view_joint_exp()
action_camera_rotation_sweep()
