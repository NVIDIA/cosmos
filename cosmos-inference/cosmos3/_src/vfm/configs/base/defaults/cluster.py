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

import attrs
from hydra.core.config_store import ConfigStore


@attrs.define(slots=False)
class ClusterConfig:
    """
    Config for the cluster specific information.
    Everything cluster specific should be here.
    """

    object_store_bucket_data: str
    object_store_bucket_checkpoint: str
    object_store_bucket_pretrained: str

    object_store_credential_data: str
    object_store_credential_checkpoint: str
    object_store_credential_pretrained: str


AWSIADH100Config: ClusterConfig = ClusterConfig(
    object_store_bucket_data="",
    object_store_bucket_checkpoint="bucket",
    object_store_bucket_pretrained="bucket",
    object_store_credential_data="credentials/s3_training.secret",
    object_store_credential_checkpoint="credentials/s3_checkpoint.secret",
    object_store_credential_pretrained="credentials/s3_checkpoint.secret",
)

GCPIADGB200Config: ClusterConfig = ClusterConfig(
    object_store_bucket_data="",
    object_store_bucket_checkpoint="bucket",
    object_store_bucket_pretrained="bucket",
    object_store_credential_data="credentials/gcp_checkpoint.secret",
    object_store_credential_checkpoint="credentials/gcp_training.secret",
    object_store_credential_pretrained="credentials/gcp_training.secret",
)


def register_cluster():
    cs = ConfigStore.instance()
    cs.store(group="cluster", package="job.cluster", name="aws_iad_h100", node=AWSIADH100Config)
    cs.store(group="cluster", package="job.cluster", name="gcp_iad_gb200", node=GCPIADGB200Config)
