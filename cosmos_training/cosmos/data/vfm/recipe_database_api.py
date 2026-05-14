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

from functools import lru_cache
from pathlib import Path

import psycopg2
import yaml
from psycopg2.extras import NamedTupleCursor


@lru_cache(maxsize=1)
def get_recipe_cursor(is_production_database: bool = False) -> NamedTupleCursor:
    config_path = Path("credentials/config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Credential file for Recipe System is not found at {config_path}")

    dbname = None
    user = None
    password = None
    endpoint = None
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        postgres_profiles = config.get("postgres", {}).get("profiles", {})
        if is_production_database:
            dbname = "dirdb_prd"
            profile_name = "dirdb_viewer_prd"
        else:
            dbname = "dirdb_dev"
            profile_name = "dirdb_viewer_dev"
        profile = postgres_profiles.get(profile_name, {})
        user = profile.get("user")
        password = profile.get("password")
        endpoint = profile.get("endpoint")

    if not all([user, password, endpoint]):
        raise ValueError(f"Missing database credentials for profile: {profile_name}")

    try:
        conn = psycopg2.connect(
            database=dbname,
            user=user,
            password=password,
            host=endpoint,
            port=5432,
        )
        postgres_cursor = conn.cursor(cursor_factory=NamedTupleCursor)
        return postgres_cursor
    except psycopg2.Error as e:
        raise ValueError(f"Failed to connect to the database for Recipe System: {e}") from e


def get_datasource_sql(datacollection_name: str, storage_type: str, data_type: str) -> str:
    return f"""
    SELECT collection.datasource_name,collection.ratio,source.name,source.sensitivity,source.text_only
    FROM datacollection_to_datasource as collection
    INNER JOIN datasource as source
    ON collection.datasource_name=source.name
    WHERE datacollection_name='{datacollection_name}' AND collection.data_type='{data_type}' AND collection.storage_type='{storage_type}' AND source.storage_type='{storage_type}'
    """


def get_wdinfo_sql(datasource: str, storage_type: str, data_type: str, split_type: str = "train") -> str:
    """Get wdinfo SQL query for a datasource.

    Args:
        datasource: Name of the datasource
        storage_type: Storage type (s3, pdx, neb_eu)
        data_type: Data type (vlm, vfm, etc)
        split_type: Dataset split (train, val, test etc.), defaults to train

    Returns:
        SQL query string
    """
    return f"""
    SELECT datasource_name, wdinfo
    FROM datasource_to_wdinfo
    WHERE datasource_name='{datasource}' AND data_type='{data_type}' AND storage_type='{storage_type}' AND split_type='{split_type}'
    """
