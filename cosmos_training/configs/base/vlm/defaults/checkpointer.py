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

from typing import Dict

from hydra.core.config_store import ConfigStore

from cosmos.utils import config
from cosmos.checkpoint.dummy import Checkpointer as DummyCheckpointer
from cosmos.utils.config import CheckpointConfig
from cosmos.utils.lazy_config import LazyCall as L
from cosmos.checkpoint.dcp import DistributedCheckpointer

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
    bucket="checkpoints-us-east-1",
)

s3_eu_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/s3_training_eu.secret",
    bucket="checkpoints-eu-west-3",
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


def register_checkpoint() -> None:
    cs = ConfigStore.instance()
    cs.store(group="checkpoint", package="checkpoint", name="local", node=CHECKPOINT_LOCAL)
    cs.store(group="checkpoint", package="checkpoint", name="pdx", node=CHECKPOINT_PDX)
    cs.store(group="checkpoint", package="checkpoint", name="s3", node=CHECKPOINT_S3)


DUMMY_CHECKPOINTER: Dict[str, str] = L(DummyCheckpointer)()
DISTRIBUTED_CHECKPOINTER: Dict[str, str] = L(DistributedCheckpointer)()


def register_ckpt_type() -> None:
    cs = ConfigStore.instance()
    cs.store(group="ckpt_type", package="checkpoint.type", name="dummy", node=DUMMY_CHECKPOINTER)
    cs.store(group="ckpt_type", package="checkpoint.type", name="dcp", node=DISTRIBUTED_CHECKPOINTER)
