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
from cosmos.utils.vlm.dcp_checkpointer import DistributedCheckpointer

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

# Permanent store for initial HF model weights on AWS (different bucket from training checkpoints).
# Used by train.py to download pretrained weights; NOT used by the checkpointer for auto-resume.
aws_load_from_object_store_permanent = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/s3_training.secret",
    bucket="nv-cosmos-vlm",
)

s3_eu_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/s3_training_eu.secret",
    bucket="checkpoints-eu-west-3",
)

neb_eu_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/neb_eu.secret",
    bucket="nv-00-10583-checkpoints",
)

gcp_object_store = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/gcp_training.secret",
    bucket="nv-00-10206-checkpoint-experiments",
)

# Permanent store for initial HF model weights on GCP (different bucket from training checkpoints).
# Used by train.py to download pretrained weights; NOT used by the checkpointer for auto-resume.
gcp_load_from_object_store_permanent = config.ObjectStoreConfig(
    enabled=True,
    credentials="credentials/gcp_training.secret",
    bucket="nv-00-10206-checkpoint",
)

CHECKPOINT_PDX = CheckpointConfig(
    save_to_object_store=pdx_object_store,
    load_from_object_store=pdx_object_store,
    save_iter=5000,
    broadcast_via_filesystem=True,
)

CHECKPOINT_S3 = CheckpointConfig(
    save_to_object_store=s3_object_store,
    load_from_object_store=s3_object_store,
    save_iter=5000,
    broadcast_via_filesystem=True,
)

CHECKPOINT_NEB_EU = CheckpointConfig(
    save_to_object_store=neb_eu_object_store,
    load_from_object_store=neb_eu_object_store,
    save_iter=2000,
    broadcast_via_filesystem=True,
)

CHECKPOINT_GCP = CheckpointConfig(
    save_to_object_store=gcp_object_store,
    load_from_object_store=gcp_object_store,
    save_iter=2000,
    broadcast_via_filesystem=True,
)


def register_checkpoint():
    cs = ConfigStore.instance()
    cs.store(group="checkpoint", package="checkpoint", name="pdx", node=CHECKPOINT_PDX)
    cs.store(group="checkpoint", package="checkpoint", name="s3", node=CHECKPOINT_S3)
    cs.store(group="checkpoint", package="checkpoint", name="neb_eu", node=CHECKPOINT_NEB_EU)
    cs.store(group="checkpoint", package="checkpoint", name="gcp", node=CHECKPOINT_GCP)


DUMMY_CHECKPOINTER: Dict[str, str] = L(DummyCheckpointer)()
DISTRIBUTED_CHECKPOINTER: Dict[str, str] = L(DistributedCheckpointer)()


def register_ckpt_type():
    cs = ConfigStore.instance()
    cs.store(group="ckpt_type", package="checkpoint.type", name="dummy", node=DUMMY_CHECKPOINTER)
    cs.store(group="ckpt_type", package="checkpoint.type", name="dcp", node=DISTRIBUTED_CHECKPOINTER)
