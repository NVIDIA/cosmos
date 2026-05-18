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
from abc import ABC, abstractmethod
from typing import Optional

import torch

from cosmos.utils.config import CheckpointConfig, JobConfig
from cosmos.utils.flags import INTERNAL
from cosmos.model._base import ImaginaireModel
from cosmos.utils import callback
from cosmos.utils.easy_io import easy_io


class AbstractCheckpointer(ABC):
    """The checkpointer class. Supports checkpoint saving/loading to both local disk or object store."""

    def __init__(
        self,
        config_checkpoint: CheckpointConfig,
        config_job: JobConfig,
        callbacks: Optional[callback.CallBackGroup] = None,
    ):
        """Constructor of the checkpointer.

        Args:
            config_checkpoint (CheckpointConfig): The config object for the checkpointer.
        """
        self.config_checkpoint = config_checkpoint
        # Set the callback functions.
        self.callbacks = callbacks
        self.save_to_object_store = config_checkpoint.save_to_object_store.enabled
        self.load_from_object_store = config_checkpoint.load_from_object_store.enabled

        # Set checkpoint directories for local and object store paths
        self._local_dirname = os.path.join(config_job.path_local, "checkpoints")
        self._object_store_dirname = os.path.join(config_job.path, "checkpoints")

        self.strict_resume = config_checkpoint.strict_resume
        load_path = config_checkpoint.load_path or None
        if not INTERNAL:
            from cosmos.utils.checkpoint_db import download_checkpoint_v2

            if load_path:
                load_path = download_checkpoint_v2(load_path)
        self.load_path = load_path
        self.load_training_state = config_checkpoint.load_training_state
        self.only_load_scheduler_state = config_checkpoint.only_load_scheduler_state
        self.save_thread = None
        self.verbose = config_checkpoint.verbose
        self.keys_not_to_resume = config_checkpoint.keys_not_to_resume
        self.keys_to_skip_loading = getattr(config_checkpoint, "keys_to_skip_loading", [])
        self.broadcast_via_filesystem = config_checkpoint.broadcast_via_filesystem
        # Create the object store client interface.
        if config_checkpoint.load_from_object_store.enabled:
            self.load_s3_backend_key = "_ckpt_s3_loader"
            easy_io.set_s3_backend(
                key="_ckpt_s3_loader",
                backend_args={
                    "backend": "s3",
                    "path_mapping": {
                        "s3://ckpt/": f"s3://{config_checkpoint.load_from_object_store.bucket}/",
                    },
                    "s3_credential_path": config_checkpoint.load_from_object_store.credentials,
                },
            )
        else:
            self.load_s3_backend_key = None

        if config_checkpoint.save_to_object_store.enabled:
            self.save_s3_backend_key = "_ckpt_s3_saver"
            easy_io.set_s3_backend(
                key="_ckpt_s3_saver",
                backend_args={
                    "backend": "s3",
                    "path_mapping": {
                        "s3://ckpt/": f"s3://{config_checkpoint.save_to_object_store.bucket}/",
                    },
                    "s3_credential_path": config_checkpoint.save_to_object_store.credentials,
                },
            )
        else:
            self.save_s3_backend_key = None

    @abstractmethod
    def save(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        pass

    @abstractmethod
    def load(
        self,
        model: ImaginaireModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        grad_scaler: Optional[torch.amp.GradScaler] = None,
    ) -> int:
        pass

    @property
    def save_bucket(self):
        """Get the bucket name for saving checkpoints."""
        return self.config_checkpoint.save_to_object_store.bucket if self.save_to_object_store else None

    @property
    def load_bucket(self):
        """Get the bucket name for loading checkpoints."""
        return self.config_checkpoint.load_from_object_store.bucket if self.load_from_object_store else None

    @property
    def save_dirname(self):
        return (
            f"s3://{self.save_bucket}/{self._object_store_dirname}"
            if self.save_to_object_store
            else self._local_dirname
        )

    @property
    def load_dirname(self):
        return (
            f"s3://{self.load_bucket}/{self._object_store_dirname}"
            if self.load_from_object_store
            else self._local_dirname
        )

    def finalize(self) -> None:
        """Finalize the checkpointer."""
        if self.save_thread:
            self.save_thread.join()

    def _read_latest_checkpoint_file(self) -> str | None:
        """Get the file name of the latest saved checkpoint. If it doesn't exist, return None.

        Returns:
            checkpoint_file (str | None): file name of the latest saved checkpoint.
        """
        checkpoint_file = None
        checkpoint_path = os.path.join(self.load_dirname, "latest_checkpoint.txt")
        if easy_io.exists(f"{checkpoint_path}", backend_key=self.load_s3_backend_key):
            checkpoint_file = easy_io.load(f"{checkpoint_path}", backend_key=self.load_s3_backend_key).strip()

        return checkpoint_file

    def _write_latest_checkpoint_file(self, checkpoint_file: str) -> None:
        """Track the file name of the latest saved checkpoint.

        Args:
            checkpoint_file (str): file name of the latest saved checkpoint.
        """
        content = f"{checkpoint_file}\n"
        checkpoint_path = os.path.join(self.save_dirname, "latest_checkpoint.txt")
        easy_io.dump(
            content,
            checkpoint_path,
            backend_key=self.save_s3_backend_key,
        )

    def _check_checkpoint_exists(self, checkpoint_path: str) -> None:
        """If the file checkpoint_path does not exist, raise an error.

        Args:
            checkpoint_path (str): full path to the checkpoint.
        """
        if not easy_io.exists(f"{checkpoint_path}", backend_key=self.load_s3_backend_key):
            raise FileNotFoundError(f"File not found (object store): {checkpoint_path}")
