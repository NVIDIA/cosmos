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
import os
import os.path as osp
import time
from typing import Any, Optional

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter

from cosmos3._src.imaginaire.checkpointer.s3_filesystem import S3StorageReader
from cosmos3._src.imaginaire.lazy_config import instantiate
from cosmos3._src.imaginaire.utils import log, misc
from cosmos3._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos3._src.vfm.checkpointer.dcp import CustomLoadPlanner, CustomSavePlanner, ModelWrapper

###################################################
# below are the load_model function for inference #
###################################################
# Thus these load_model functions are designed with less dependency.


def checkpoint_path_to_cached_path(path: str, cache_rootdir: Optional[str] = None) -> str:
    if cache_rootdir is None:
        homedir = os.getenv("HOME") or ""
        cache_rootdir = osp.join(homedir, ".cache/imaginaire4/checkpoints/")

    if path.startswith("s3://"):
        return osp.join(cache_rootdir, path.removeprefix("s3://").split("/", maxsplit=1)[1])
    else:
        return path


def _load_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    credential_path: str | None,
    enable_gcs_patch_in_boto3: bool = False,
    load_ema_to_reg: bool = False,
    keys_to_skip_loading: list[str] | None = None,
) -> None:
    """
    Args:
        model: The model to load weights into
        checkpoint_path: Path to checkpoint (can be s3 or local path)
        credential_path: Path to S3 credentials (can be none if load local)
        enable_gcs_patch_in_boto3: Whether to enable GCS patch in boto3 for DCP loading from GCS
        load_ema_to_reg: Whether to load EMA weights into the regular (non-EMA) model parameters.
        keys_to_skip_loading: List of key substrings to skip when loading from checkpoint.
            Useful for loading pretrained checkpoints that are missing certain keys (e.g. action heads).
    """

    log.info(f"Loading model from {checkpoint_path}")
    start_time = time.time()

    state_dict = ModelWrapper(model).state_dict()

    if checkpoint_path.startswith("s3://"):
        storage_reader = S3StorageReader(
            credential_path=credential_path or "",
            path=checkpoint_path,
            enable_gcs_patch_in_boto3=enable_gcs_patch_in_boto3,
        )
    else:
        storage_reader = FileSystemReader(checkpoint_path)

    load_planner = CustomLoadPlanner(
        load_ema_to_reg=load_ema_to_reg,
        keys_to_skip_loading=keys_to_skip_loading or [],
    )

    dcp.load(
        state_dict=state_dict,
        storage_reader=storage_reader,
        planner=load_planner,
    )

    log.info(f"Successfully loaded model from {checkpoint_path}")
    log.info(f"Time taken to load model: {time.time() - start_time:.2f} seconds")


def _save_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    save_reg_to_ema: bool = False,
) -> None:
    """
    Args:
        model: The model to load weights into
        checkpoint_path: Path to cached checkpoint (can be s3 or local path)
        save_reg_to_ema: Whether to save regular (non-EMA) model parameters to EMA model parameters.
    """

    log.info(f"Saving model to {checkpoint_path}")
    start_time = time.time()

    state_dict = ModelWrapper(model).state_dict()

    assert not checkpoint_path.startswith("s3://"), "Cached checkpoint path must be local path"
    storage_writer = FileSystemWriter(checkpoint_path)

    save_planner = CustomSavePlanner(save_reg_to_ema=save_reg_to_ema, dedup_save_to_lowest_rank=True)

    dcp.save(
        state_dict=state_dict,
        storage_writer=storage_writer,
        planner=save_planner,
    )

    log.info(f"Successfully saved model to {checkpoint_path}")
    log.info(f"Time taken to save model: {time.time() - start_time:.2f} seconds")


def load_model_from_checkpoint(
    experiment_name: str,
    checkpoint_path: Optional[str] = None,
    credential_path: Optional[str] = None,
    enable_gcs_patch_in_boto3: bool = False,
    config_file: str = "cosmos3/_src/vfm/configs/base/config.py",
    load_ema_to_reg: bool = False,
    parallelism_config: dict[str, Any] = {},
    seed: int = 0,
    experiment_opts: list[str] = [],
    use_cache_checkpoint: bool = False,
    cache_checkpoint_rootdir: Optional[str] = None,
    disable_torch_compile: bool = False,
    keys_to_skip_loading: list[str] | None = None,
) -> tuple[torch.nn.Module, Any]:
    """
    Args:
        experiment_name: Experiment name.
        checkpoint_path: Path to the checkpoint (local path or s3 URI).
        credential_path: Path to credentials file (if required for remote storage). Optional.
        enable_gcs_patch_in_boto3: Whether to enable the boto3 patch for GCS S3-compatibility.
        config_file: Path to the config file used to construct the experiment/model.
        load_ema_to_reg: If True, load EMA weights into the regular (non-EMA) model parameters.
        parallelism_config: Dictionary of parallelism configuration options.
        seed: Random seed used for initialization (if applicable).
        experiment_opts: Extra experiment/config override options.
        use_cache_checkpoint: If True, locally save & read remote checkpoints to speed up repeated loads.
            Be aware, the default cache path is $HOME/.cache/imaginaire4/checkpoints/<same s3 path>
        cache_checkpoint_rootdir: Customizable root directory for checkpoint cache. Optional.
        disable_torch_compile: If True, do not use torch.compile even if the experiment enables it.
        keys_to_skip_loading: List of key substrings to skip when loading from checkpoint.
            Useful for loading pretrained checkpoints that are missing certain keys (e.g. action heads).

    Returns:
        The loaded model and config
    """

    # Ensure checkpoint_path is provided
    if checkpoint_path is None:
        raise ValueError("'checkpoint_path' must be provided.")

    if not checkpoint_path.strip("/").endswith("model"):
        checkpoint_path = os.path.join(checkpoint_path, "model")

    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={experiment_name}"] + experiment_opts)

    if parallelism_config is not None:
        for key, value in parallelism_config.items():
            if hasattr(config.model.config.parallelism, key):
                setattr(config.model.config.parallelism, key, value)
            else:
                raise ValueError(f"Key {key} not found in config.model.config.parallelism")

    if disable_torch_compile:
        config.model.config.parallelism.use_torch_compile = False

    config.model.config.ema.enabled = False

    config.validate()
    config.freeze()  # type: ignore

    misc.set_random_seed(seed=seed, by_rank=True)

    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark

    with misc.timer("instantiate model"):
        model = instantiate(config.model).cuda()  # type: ignore
        model.on_train_start()

    checkpoint_cache_path = None
    if use_cache_checkpoint:
        checkpoint_cache_path = checkpoint_path_to_cached_path(checkpoint_path, cache_checkpoint_rootdir)

    if checkpoint_cache_path is not None and osp.exists(checkpoint_cache_path):
        checkpoint_load_path = checkpoint_cache_path
    else:
        checkpoint_load_path = checkpoint_path

    _load_model(
        model,
        checkpoint_path=checkpoint_load_path,
        credential_path=credential_path,
        enable_gcs_patch_in_boto3=enable_gcs_patch_in_boto3,
        load_ema_to_reg=load_ema_to_reg,
        keys_to_skip_loading=keys_to_skip_loading,
    )

    if checkpoint_cache_path is not None and not osp.exists(checkpoint_cache_path):
        _save_model(
            model,
            checkpoint_path=checkpoint_cache_path,
            save_reg_to_ema=load_ema_to_reg,
        )

    return model, config
