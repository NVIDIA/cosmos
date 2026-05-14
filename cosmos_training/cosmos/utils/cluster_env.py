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

import os
from enum import Enum
from functools import lru_cache
from typing import Dict


class ClusterType(Enum):
    LOCAL = "local"
    NGC = "ngc"
    SLURM = "slurm"


class ClusterEnvInfo(Enum):
    BASIC = "basic"
    DETAILED = "detailed"
    ALL = "all"


NGC_ENV_BASIC_VARS = [
    "NGC_JOB_ID",
    "NGC_ARRAY_SIZE",
    "NGC_GPUS_PER_NODE",
]

SLURM_ENV_BASIC_VARS = [
    "SLURM_JOB_USER",
    "SLURM_JOB_PARTITION",
    "SLURM_LOG_DIR",
    "SLURM_JOBID",
    "SLURM_NNODES",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURMD_NODENAME",
]


@lru_cache()
def is_local() -> bool:
    """
    Check if the code is running on a local machine.
    """
    return not is_ngc() and not is_slurm()


@lru_cache()
def is_ngc() -> bool:
    """
    Check if the code is running on NGC.
    """
    return "NGC_ARRAY_SIZE" in os.environ


@lru_cache()
def is_slurm() -> bool:
    """
    Check if the code is running on SLURM.
    """
    return "SLURM_JOB_ID" in os.environ


def get_ngc_env(level: ClusterEnvInfo = ClusterEnvInfo.BASIC) -> Dict[str, str]:
    """
    Retrieves NVIDIA GPU Cloud (NGC) environment variables based on the specified detail level.
    The function filters environment variables to include only those relevant to NGC,
    differentiated by the detail level specified.

    Parameters:
        level (ClusterInfoLevel): The level of detail for the information returned.
                                  Defaults to ClusterInfoLevel.BASIC.

    Returns:
        dict: A dictionary containing the environment variables. If the level is BASIC,
              it includes only predefined key variables that are considered basic.
              If the level is DETAILED, it includes all environment variables that start
              with "NGC_".

    Raises:
        ValueError: If an unknown level is specified, an exception is raised indicating that the
                    level is not recognized.
    """
    if level == ClusterEnvInfo.BASIC:
        return {k: os.environ[k] for k in NGC_ENV_BASIC_VARS if k in os.environ}
    elif level == ClusterEnvInfo.DETAILED:
        return {k: os.environ[k] for k in os.environ if k.startswith("NGC_")}
    elif level == ClusterEnvInfo.ALL:
        return {k: v for k, v in os.environ}
    else:
        raise ValueError(f"Unknown level {level}")


def get_slurm_env(level: ClusterEnvInfo = ClusterEnvInfo.BASIC) -> Dict[str, str]:
    """
    Retrieves SLURM environment variables based on the specified detail level.
    This function filters the environment variables related to the SLURM job scheduler
    environment based on the provided detail level of the cluster information.

    Parameters:
        level (ClusterEnvInfo): The detail level of the environment variables to retrieve.
                                This can be BASIC, DETAILED, or ALL. Defaults to BASIC.

    Returns:
        Dict[str, str]: A dictionary containing the SLURM environment variables. The contents of
                        the dictionary vary based on the level:
                        - BASIC: Returns predefined key variables important for basic SLURM variables.
                        - DETAILED: Includes all variables that start with "SLURM_".
                        - ALL: Returns all environment variables available in the current session.

    Raises:
        ValueError: If an unknown level is specified, it raises an exception indicating
                    that the level is not recognized.
    """
    if level == ClusterEnvInfo.BASIC:
        return {k: os.environ[k] for k in SLURM_ENV_BASIC_VARS if k in os.environ}
    elif level == ClusterEnvInfo.DETAILED:
        return {k: os.environ[k] for k in os.environ if k.startswith("SLURM_")}
    elif level == ClusterEnvInfo.ALL:
        return {k: v for k, v in os.environ.items()}
    else:
        raise ValueError(f"Unknown level {level}")


def get_cluster_env(level: ClusterEnvInfo = ClusterEnvInfo.BASIC) -> Dict[str, str]:
    """
    Retrieves a combination of environment variables from the cluster, merging information from
    both NVIDIA GPU Cloud (NGC) and SLURM environments based on the specified detail level.
    This function provides a unified dictionary of environment settings that are crucial for
    applications running in clustered computing environments.

    Parameters:
        level (ClusterEnvInfo): The level of detail for the environment variables to be retrieved.
                                The level can be BASIC, DETAILED, or ALL. Defaults to BASIC.
                                - BASIC: Gathers basic environment variables from both NGC and SLURM.
                                - DETAILED: Includes more detailed information from both NGC and SLURM.
                                - ALL: Combines all available environment variables from the system
                                       with NGC and SLURM specific ones.

    Returns:
        Dict[str, str]: A dictionary containing key-value pairs of environment variables.
                        Initially includes the current working directory under the key 'PWD'.
    """
    env_info = {
        "PWD": os.getcwd(),  # Always include the present working directory.
    }
    if level == ClusterEnvInfo.ALL:
        env_info.update(os.environ)  # Adds all system environment variables.
        return env_info

    # For BASIC and DETAILED levels, merge environment variables from NGC and SLURM:
    env_info.update(get_ngc_env(level))
    env_info.update(get_slurm_env(level))
    return env_info
