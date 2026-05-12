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

"""
Copied from https://gitlab-master.nvidia.com/dir/imaginaire4/-/blob/d0921eb675d1251e73c4b19acdd78e6ad936ae3b/projects/cosmos/reason2/configs/base/defaults/checkpointer.py without changes
"""

from typing import Dict

from hydra.core.config_store import ConfigStore

from cosmos3._src.imaginaire import config
from cosmos3._src.imaginaire.checkpointer.dummy import Checkpointer as DummyCheckpointer
from cosmos3._src.imaginaire.config import CheckpointConfig
from cosmos3._src.imaginaire.lazy_config import LazyCall as L
from cosmos3._src.vfm.checkpointer.dcp import DistributedCheckpointer

local_object_store = config.ObjectStoreConfig(
    enabled=False,
)

pdx_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/pdx_vfm_checkpoint.secret",
    bucket="checkpoints",
)

s3_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/s3_training.secret",
    bucket="bucket",
)

s3_eu_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/s3_training_eu.secret",
    bucket="bucket",
)

gcp_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/gcp_checkpoint.secret",
    bucket="bucket",
)

neb_eu_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/neb_eu.secret",
    bucket="nv-01-10206-checkpoint-experiments",
)

CHECKPOINT_LOCAL = CheckpointConfig(
    save_to_object_store=local_object_store,
    load_from_object_store=local_object_store,
    save_iter=5000,
    broadcast_via_filesystem=True,
    dcp_async_mode_enabled=True,
)

CHECKPOINT_PDX = CheckpointConfig(
    save_to_object_store=pdx_object_store,
    load_from_object_store=pdx_object_store,
    save_iter=5000,
    broadcast_via_filesystem=True,
    dcp_async_mode_enabled=True,
)

CHECKPOINT_S3 = CheckpointConfig(
    save_to_object_store=s3_object_store,
    load_from_object_store=s3_object_store,
    save_iter=5000,
    broadcast_via_filesystem=True,
    dcp_async_mode_enabled=True,
)

CHECKPOINT_S3_EU = CheckpointConfig(
    save_to_object_store=s3_eu_object_store,
    load_from_object_store=s3_eu_object_store,
    save_iter=5000,
    broadcast_via_filesystem=True,
    dcp_async_mode_enabled=True,
)

CHECKPOINT_GCP = CheckpointConfig(
    save_to_object_store=gcp_object_store,
    save_iter=1000,
    load_from_object_store=gcp_object_store,
    load_path="",
    load_training_state=False,
    strict_resume=True,
    enable_gcs_patch_in_boto3=True,
    dcp_async_mode_enabled=True,
)

CHECKPOINT_NEB_EU = CheckpointConfig(
    save_to_object_store=neb_eu_object_store,
    load_from_object_store=neb_eu_object_store,
    save_iter=2000,
    broadcast_via_filesystem=True,
)


def register_checkpoint():
    cs = ConfigStore.instance()
    cs.store(group="checkpoint", package="checkpoint", name="local", node=CHECKPOINT_LOCAL)
    cs.store(group="checkpoint", package="checkpoint", name="pdx", node=CHECKPOINT_PDX)
    cs.store(group="checkpoint", package="checkpoint", name="s3", node=CHECKPOINT_S3)
    cs.store(group="checkpoint", package="checkpoint", name="s3_eu", node=CHECKPOINT_S3_EU)
    cs.store(group="checkpoint", package="checkpoint", name="gcp", node=CHECKPOINT_GCP)
    cs.store(group="checkpoint", package="checkpoint", name="neb_eu", node=CHECKPOINT_NEB_EU)


DUMMY_CHECKPOINTER: Dict[str, str] = L(DummyCheckpointer)()
DISTRIBUTED_CHECKPOINTER: Dict[str, str] = L(DistributedCheckpointer)()


def register_ckpt_type():
    cs = ConfigStore.instance()
    cs.store(group="ckpt_type", package="checkpoint.type", name="dummy", node=DUMMY_CHECKPOINTER)
    cs.store(group="ckpt_type", package="checkpoint.type", name="dcp", node=DISTRIBUTED_CHECKPOINTER)
