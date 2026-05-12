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

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import attrs
import wandb
import wandb.util
from omegaconf import DictConfig

from cosmos3._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos3._src.imaginaire.utils import distributed, log, object_store
from cosmos3._src.imaginaire.utils.easy_io import easy_io

if TYPE_CHECKING:
    from cosmos3._src.imaginaire.config import CheckpointConfig, Config, JobConfig
    from cosmos3._src.imaginaire.model import ImaginaireModel

JOB_INFO = {}


def set_wandb_job_info(job_info: dict) -> None:
    """Set the job info for the W&B logger.

    Args:
        job_info (dict): The job info.
    """
    JOB_INFO.update(job_info)


@distributed.rank0_only
def init_wandb(config: Config, model: ImaginaireModel) -> None:
    """Initialize Weights & Biases (wandb) logger.

    Args:
        config (Config): The config object for the Imaginaire codebase.
        model (ImaginaireModel): The PyTorch model.
    """
    if isinstance(config.job, DictConfig):
        from cosmos3._src.imaginaire.config import JobConfig

        config_job = JobConfig(**config.job)
    else:
        config_job = config.job
    config_checkpoint = config.checkpoint
    # Try to fetch the W&B job ID for resuming training.
    wandb_id = _read_wandb_id(config_job, config_checkpoint)
    if wandb_id is None:
        # Generate a new W&B job ID.
        wandb_id = wandb.util.generate_id()
        _write_wandb_id(config_job, config_checkpoint, wandb_id=wandb_id)
        log.info(f"Generating new wandb ID: {wandb_id}")
    else:
        log.info(f"Resuming with existing wandb ID: {wandb_id}")
    # refactor config so that wandb better understands it
    local_safe_yaml_fp = LazyConfig.save_yaml(config, os.path.join(config_job.path_local, "config.yaml"))
    if os.path.exists(local_safe_yaml_fp):
        config_resolved = easy_io.load(local_safe_yaml_fp)
    else:
        config_resolved = attrs.asdict(config)
    # Initialize the wandb library. If we attempt to resume an existing run
    # but the current user does not have permission to update that run
    # (common when re-using an ID created by someone else), fall back to
    # creating a fresh run ID and re-initializing.
    try:
        wandb.init(
            force=True,
            id=wandb_id,
            project=config_job.project,
            group=config_job.group,
            name=config_job.name,
            config=config_resolved,
            dir=config_job.path_local,
            resume="allow",
            mode=config_job.wandb_mode,
        )
    except Exception as e:
        # Detect common permission / upload errors from wandb and recover
        msg = str(e)
        if (
            "member role does not have Update Run permission" in msg
            or "Error uploading run" in msg
            or "returned error 403" in msg
        ):
            log.warning("W&B run exists but current user lacks update permission; starting a new run instead.")
            # Generate and persist a new wandb id, then create a fresh run.
            wandb_id = wandb.util.generate_id()
            _write_wandb_id(config_job, config_checkpoint, wandb_id=wandb_id)
            wandb.init(
                force=True,
                id=wandb_id,
                project=config_job.project,
                group=config_job.group,
                name=config_job.name,
                config=config_resolved,
                dir=config_job.path_local,
                mode=config_job.wandb_mode,
            )
        elif "returned error 401" in msg or "user is not logged in" in msg:
            log.warning("W&B authentication failed (401); falling back to offline mode. Error: %s", msg)
            wandb.init(
                force=True,
                id=wandb_id,
                project=config_job.project,
                group=config_job.group,
                name=config_job.name,
                config=config_resolved,
                dir=config_job.path_local,
                mode="offline",
            )
        else:
            raise

    if wandb.run:
        wandb.run.config.update({f"JOB_INFO/{k}": v for k, v in JOB_INFO.items()}, allow_val_change=True)


def _read_wandb_id(config_job: JobConfig, config_checkpoint: CheckpointConfig) -> str | None:
    """Read the W&B job ID. If it doesn't exist, return None.

    Args:
        config_wandb (JobConfig): The config object for the W&B logger.
        config_checkpoint (CheckpointConfig): The config object for the checkpointer.

    Returns:
        wandb_id (str | None): W&B job ID.
    """
    wandb_id = None
    if config_checkpoint.load_from_object_store.enabled:
        object_store_loader = object_store.ObjectStore(config_checkpoint.load_from_object_store)
        wandb_id_path = f"{config_job.path}/wandb_id.txt"
        if object_store_loader.object_exists(key=wandb_id_path):
            wandb_id = object_store_loader.load_object(key=wandb_id_path, type="text").strip()
    else:
        wandb_id_path = f"{config_job.path_local}/wandb_id.txt"
        if os.path.isfile(wandb_id_path):
            wandb_id = open(wandb_id_path).read().strip()
    return wandb_id


def _write_wandb_id(config_job: JobConfig, config_checkpoint: CheckpointConfig, wandb_id: str) -> None:
    """Write the generated W&B job ID.

    Args:
        config_wandb (JobConfig): The config object for the W&B logger.
        config_checkpoint (CheckpointConfig): The config object for the checkpointer.
        wandb_id (str): The W&B job ID.
    """
    content = f"{wandb_id}\n"
    if config_checkpoint.save_to_object_store.enabled:
        object_store_saver = object_store.ObjectStore(config_checkpoint.save_to_object_store)
        wandb_id_path = f"{config_job.path}/wandb_id.txt"
        object_store_saver.save_object(content, key=wandb_id_path, type="text")
    else:
        wandb_id_path = f"{config_job.path_local}/wandb_id.txt"
        with open(wandb_id_path, "w") as file:
            file.write(content)
