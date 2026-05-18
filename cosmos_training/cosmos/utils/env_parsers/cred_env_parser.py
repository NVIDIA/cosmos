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

from cosmos.utils.env_parsers.env_parser import EnvParser
from cosmos.utils.validator import String


class CredentialEnvParser(EnvParser):
    APP_ENV = String(default="")
    PROD_FT_AWS_CREDS_ACCESS_KEY_ID = String(default="")
    PROD_FT_AWS_CREDS_SECRET_ACCESS_KEY = String(default="")
    PROD_FT_AWS_CREDS_ENDPOINT_URL = String(default="https://s3.us-west-2.amazonaws.com")
    PROD_FT_AWS_CREDS_REGION_NAME = String(default="us-west-2")

    PROD_S3_CHECKPOINT_ACCESS_KEY_ID = String(default="")
    PROD_S3_CHECKPOINT_SECRET_ACCESS_KEY = String(default="")
    PROD_S3_CHECKPOINT_ENDPOINT_URL = String(default="")
    PROD_S3_CHECKPOINT_REGION_NAME = String(default="")

    PROD_GCP_CHECKPOINT_ACCESS_KEY_ID = String(default="")
    PROD_GCP_CHECKPOINT_SECRET_ACCESS_KEY = String(default="")
    PROD_GCP_CHECKPOINT_ENDPOINT_URL = String(default="")
    PROD_GCP_CHECKPOINT_REGION_NAME = String(default="")

    PROD_PDX_BENCHMARK_ACCESS_KEY_ID = String(default="")
    PROD_PDX_BENCHMARK_SECRET_ACCESS_KEY = String(default="")
    PROD_PDX_BENCHMARK_ENDPOINT_URL = String(default="")
    PROD_PDX_BENCHMARK_REGION_NAME = String(default="")

    PROD_TEAM_DIR_ACCESS_KEY_ID = String(default="")
    PROD_TEAM_DIR_SECRET_ACCESS_KEY = String(default="")
    PROD_TEAM_DIR_ENDPOINT_URL = String(default="")
    PROD_TEAM_DIR_REGION_NAME = String(default="")

    PICASSO_AUTH_MODEL_REGISTRY_API_KEY = String(default="")
    PICASSO_API_ENDPOINT_URL = String(default="https://invalid")


CRED_ENVS = CredentialEnvParser()
CRED_ENVS_DICT = {
    "PROD_FT_AWS_CREDS": {
        "aws_access_key_id": CRED_ENVS.PROD_FT_AWS_CREDS_ACCESS_KEY_ID,
        "aws_secret_access_key": CRED_ENVS.PROD_FT_AWS_CREDS_SECRET_ACCESS_KEY,
        "endpoint_url": CRED_ENVS.PROD_FT_AWS_CREDS_ENDPOINT_URL,
        "region_name": CRED_ENVS.PROD_FT_AWS_CREDS_REGION_NAME,
    },
    "PROD_S3_CHECKPOINT": {
        "aws_access_key_id": CRED_ENVS.PROD_S3_CHECKPOINT_ACCESS_KEY_ID,
        "aws_secret_access_key": CRED_ENVS.PROD_S3_CHECKPOINT_SECRET_ACCESS_KEY,
        "endpoint_url": CRED_ENVS.PROD_S3_CHECKPOINT_ENDPOINT_URL,
        "region_name": CRED_ENVS.PROD_S3_CHECKPOINT_REGION_NAME,
    },
    "PROD_GCP_CHECKPOINT": {
        "aws_access_key_id": CRED_ENVS.PROD_GCP_CHECKPOINT_ACCESS_KEY_ID,
        "aws_secret_access_key": CRED_ENVS.PROD_GCP_CHECKPOINT_SECRET_ACCESS_KEY,
        "endpoint_url": CRED_ENVS.PROD_GCP_CHECKPOINT_ENDPOINT_URL,
        "region_name": CRED_ENVS.PROD_GCP_CHECKPOINT_REGION_NAME,
    },
    "PROD_PDX_BENCHMARK": {
        "aws_access_key_id": CRED_ENVS.PROD_PDX_BENCHMARK_ACCESS_KEY_ID,
        "aws_secret_access_key": CRED_ENVS.PROD_PDX_BENCHMARK_SECRET_ACCESS_KEY,
        "endpoint_url": CRED_ENVS.PROD_PDX_BENCHMARK_ENDPOINT_URL,
        "region_name": CRED_ENVS.PROD_PDX_BENCHMARK_REGION_NAME,
    },
    "PROD_TEAM_DIR": {
        "aws_access_key_id": CRED_ENVS.PROD_TEAM_DIR_ACCESS_KEY_ID,
        "aws_secret_access_key": CRED_ENVS.PROD_TEAM_DIR_SECRET_ACCESS_KEY,
        "endpoint_url": CRED_ENVS.PROD_TEAM_DIR_ENDPOINT_URL,
        "region_name": CRED_ENVS.PROD_TEAM_DIR_REGION_NAME,
    },
}
