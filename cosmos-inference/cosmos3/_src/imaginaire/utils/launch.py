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

import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf

from cosmos3._src.imaginaire.config import Config
from cosmos3._src.imaginaire.utils import distributed, log
from cosmos3._src.imaginaire.utils.cluster_env import get_cluster_env
from cosmos3._src.imaginaire.utils.easy_io import easy_io
from cosmos3._src.imaginaire.utils.env_parsers.cred_env_parser import CRED_ENVS
from cosmos3._src.imaginaire.utils.wandb_util import set_wandb_job_info

# Global variable to track S3 readiness
S3_READY = False


def log_reproducible_setup(config: Config, args: argparse.Namespace) -> None:
    """
    Configures the environment for reproducibility of experiments by setting up
    S3 backends for storage, logging important job details, and saving configuration and
    environment details both locally and on S3.
    This function is crucial for ensuring that all aspects of the computational environment are captured and can be
    replicated for future runs or analysis.

    Parameters:
        config (Config): A configuration object containing all the settings necessary
                         for the job, including paths and credentials.
        args (argparse.Namespace): An argparse namespace containing the command line
                                   arguments passed to the script. This includes configurations
                                   and any overrides specified at runtime.

    Actions:
        - Sets up S3 backend for storing user data and other outputs.
        - Logs job paths and critical information regarding job execution.
        - Saves the job configuration locally only for the main node in a distributed setting.
        - Captures and logs command-line execution details.
        - Optionally reads git commit and branch information if available and logs them.
        - Saves both job environment information and launch details locally and syncs these to S3.
        - Supports conditional integration with Weights & Biases (wandb) for experiment tracking.

    Notes:
        - The function is designed to run within a distributed environment where certain actions
          (like saving configurations) are restricted to the main node (rank 0).
        - It uses the 'easy_io' module for interacting with S3, ensuring files are written and
          read correctly from the object store.
        - It leverages OmegaConf for saving YAML configurations
        - git information is read from 'git_commit.txt' and 'git_branch.txt' files if they exist.
        - snapshot codebase is saved as 'codebase.zip' if it exists in the current directory.

    Raises:
        FileNotFoundError: If specific files like 'git_commit.txt' or 'codebase.zip' are expected
                           but not found.
        IOError: If there are issues in file handling operations, particularly with file
                 reading/writing.
    """

    run_timestamp = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    time_tensor = torch.ByteTensor(bytearray(run_timestamp, "utf-8")).cuda()
    distributed.broadcast(time_tensor, 0)
    run_timestamp = time_tensor.cpu().numpy().tobytes().decode("utf-8")

    global S3_READY
    if os.path.exists(config.checkpoint.save_to_object_store.credentials) or CRED_ENVS.APP_ENV in [
        "prod",
        "dev",
        "stg",
    ]:
        easy_io.set_s3_backend(
            backend_args={
                "backend": "s3",
                "path_mapping": {
                    "s3://timestamps_rundir/": f"s3://{config.checkpoint.save_to_object_store.bucket}/{config.job.path}/job_runs/{run_timestamp}/",
                    "s3://rundir/": f"s3://{config.checkpoint.save_to_object_store.bucket}/{config.job.path}/",
                },
                "s3_credential_path": config.checkpoint.save_to_object_store.credentials,
            }
        )
        S3_READY = True
    else:
        log.warning("S3 credentials not found. Skipping easy_io S3 setup.")

    log.warning(f"Job path: {config.job.path}")
    job_info = get_cluster_env()
    # save cfg to local
    if distributed.get_rank() == 0:
        job_local_path = config.job.path_local
        log.critical(f"Job local path: {job_local_path}")
        os.makedirs(config.job.path_local, exist_ok=True)
        launch_info = {
            "cmd": " ".join(sys.argv),
            "args_cfg_path": args.config,
            "args_override": args.opts,
        }

        job_info["job_local_path"] = str(job_local_path)
        job_info["s3"] = f"s3://{config.checkpoint.save_to_object_store.bucket}/{config.job.path}/"
        # optional read git_commit.txt and save git commit id
        if os.path.exists("git_commit.txt"):
            with open("git_commit.txt", "r") as f:
                job_info["commit_id"] = f.read().strip()
                log.critical(f"Commit id: {job_info['commit_id']}")
        if os.path.exists("git_branch.txt"):
            with open("git_branch.txt", "r") as f:
                job_info["git_branch"] = f.read().strip()
                log.critical(f"git branch: {job_info['git_branch']}")
        if os.path.exists("git_diff.txt"):
            with open("git_diff.txt", "r") as f:
                job_info["git_diff"] = f.read().strip()
                log.critical(f"git diff: {job_info['git_diff']}")

        with open(f"{job_local_path}/job_env.yaml", "w") as f:
            OmegaConf.save(job_info, f)
        with open(f"{job_local_path}/launch_info.yaml", "w") as f:
            OmegaConf.save(launch_info, f)
        set_wandb_job_info(job_info)

        # by default, we upload run in ngc and slurm
        if config.upload_reproducible_setup:
            # sync to s3
            if S3_READY:
                log.critical(
                    f"Uploading reproducible setup to s3://{config.checkpoint.save_to_object_store.bucket}/{config.job.path}/job_runs/{run_timestamp}/"
                )

                config_pkl_save_fp = f"{config.job.path_local}/config.pkl"
                easy_io.copyfile_from_local(
                    config_pkl_save_fp, f"s3://timestamps_rundir/{config_pkl_save_fp.split('/')[-1]}"
                )
                config_yaml_save_fp = config_pkl_save_fp.replace(".pkl", ".yaml")
                easy_io.copyfile_from_local(
                    config_yaml_save_fp, f"s3://timestamps_rundir/{config_yaml_save_fp.split('/')[-1]}"
                )
                easy_io.copyfile_from_local(f"{job_local_path}/job_env.yaml", "s3://timestamps_rundir/job_env.yaml")
                easy_io.copyfile_from_local(
                    f"{job_local_path}/launch_info.yaml",
                    "s3://timestamps_rundir/launch_info.yaml",
                )
                if os.path.exists("codebase.zip"):
                    easy_io.copyfile_from_local("codebase.zip", "s3://timestamps_rundir/codebase.zip")
                if os.path.exists("code.tar.gz"):
                    easy_io.copyfile_from_local("code.tar.gz", "s3://timestamps_rundir/code.tar.gz")
                if os.path.exists("git_diff.txt"):
                    easy_io.copyfile_from_local("git_diff.txt", "s3://timestamps_rundir/git_diff.txt")
                if easy_io.exists("s3://rundir/job_history.yaml"):
                    job_history = easy_io.load("s3://rundir/job_history.yaml")
                else:
                    job_history = {}
                job_history[len(job_history)] = {
                    "timestamp": run_timestamp,
                    "reproduce_dir": f"s3://{config.checkpoint.save_to_object_store.bucket}/{config.job.path}/job_runs/{run_timestamp}/",
                    **launch_info,
                }
                print(job_history)
                easy_io.dump(job_history, "s3://rundir/job_history.yaml")
            else:
                log.warning("S3 credentials not found. Skipping upload of reproducible setup.")

    # save per rank cluster information to s3
    if config.upload_reproducible_setup:
        if S3_READY:
            easy_io.dump(job_info, f"s3://timestamps_rundir/cluster_env/RANK_{distributed.get_rank():06d}.yaml")
