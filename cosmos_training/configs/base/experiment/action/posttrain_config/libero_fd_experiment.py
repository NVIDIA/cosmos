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

# Forward Dynamics Experiment for LIBERO

from datetime import datetime

from hydra.core.config_store import ConfigStore

from configs.base.experiment.action._experiment_helpers import (
    LIBERO_BASELINE_BATCH_SIZE,
    LIBERO_BASELINE_NUM_WORKERS,
    LIBERO_BASELINE_TRAINING_ITERATIONS,
    make_libero_dataset,
)
from configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment

cs = ConfigStore.instance()


def _build_libero_fd_base(
    exp_name: str,
    *,
    training_iterations: int = LIBERO_BASELINE_TRAINING_ITERATIONS,
    batch_size: int = LIBERO_BASELINE_BATCH_SIZE,
    num_workers: int = LIBERO_BASELINE_NUM_WORKERS,
    job_group: str = "debugging",
    job_project: str | None = None,
    **dataset_kwargs,
) -> dict:
    dataset_kwargs.setdefault("mode", "forward_dynamics")
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
# libero_exp_fd — base FD experiment
# ---------------------------------------------------------------------------
libero_exp_fd = _build_libero_fd_base(
    exp_name=f"libero_exp_fd_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
cs.store(
    group="experiment",
    package="_global_",
    name="libero_exp_fd",
    node=libero_exp_fd,
)


# ---------------------------------------------------------------------------
# Sweep camera_mode × action_space × rotation_space (lr=2e-4, mode=forward_dynamics)
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
                name = f"libero_exp_fd_{cam_short}_{act_short}_{rot_short}_varlen"

                exp = _build_libero_fd_base(
                    exp_name=name,
                    job_group="action_libero",
                    camera_mode=camera_mode,
                    action_space=action_space,
                    rotation_space=rotation_space,
                )
                exp["optimizer"]["lr"] = 2e-4
                cs.store(group="experiment", package="_global_", name=name, node=exp)


action_camera_rotation_sweep()
