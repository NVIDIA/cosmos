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

"""Database of released checkpoints.

The database maps checkpoint internal URIs to public URIs and associates metadata (e.g.
experiment name).

## Usage

Register a checkpoint:

```python
CheckpointConfig(
    uuid="0e8177cc-0db5-4cfd-a8a4-b820c772f4fc",
    s3=CheckpointDirS3(
        uri="s3://bucket/path/to/checkpoint",
    ),
    hf=CheckpointDirHf(
        repository="org/repo",
        revision="revision",
        subdirectory="path/to/checkpoint",
    ),
).register()
```

Checkpoints can be referenced by UUID, S3 URI, or local path. Optionally, use `get_checkpoint_uri` to validate and normalize the URI.

```python
# S3 URI
checkpoint_uri = get_checkpoint_uri("s3://bucket/path/to/checkpoint")
# UUID
checkpoint_uri = get_checkpoint_uri("0e8177cc-0db5-4cfd-a8a4-b820c772f4fc")
# Local path
checkpoint_uri = get_checkpoint_uri("/path/to/checkpoint", check_exists=True)
```

When the checkpoint is loaded, call 'download_checkpoint':

```python
from cosmos3._src.imaginaire.flags import INTERNAL

if not INTERNAL:
    from cosmos3._src.imaginaire.utils.checkpoint_db import download_checkpoint

    checkpoint_uri = download_checkpoint(checkpoint_uri)

load_checkpoint(checkpoint_uri)
```
"""

import functools
import json
import os
import shlex
import subprocess
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, TypeAlias

import pydantic
from typing_extensions import Self, override

from cosmos3._src.imaginaire.flags import EXPERIMENTAL_CHECKPOINTS, INTERNAL
from cosmos3._src.imaginaire.utils import log

HF_VERSION = "1.13.0"


def _is_uuid(checkpoint_uri: str) -> bool:
    """Return True if the URI is a UUID."""
    try:
        uuid.UUID(str(checkpoint_uri))
        return True
    except ValueError:
        return False


def _is_path(checkpoint_uri: str) -> bool:
    """Return True if the URI is a local path."""
    return not ("://" in checkpoint_uri or _is_uuid(checkpoint_uri))


def normalize_uri(checkpoint_uri: str) -> str:
    """Normalize checkpoint URI."""
    checkpoint_uri = checkpoint_uri.rstrip("/")
    if checkpoint_uri.startswith("s3://"):
        checkpoint_uri = checkpoint_uri.removesuffix("/model")
    return checkpoint_uri


def sanitize_uri(checkpoint_uri: str) -> str:
    """Sanitize checkpoint URI."""
    checkpoint_uri = normalize_uri(checkpoint_uri)
    if checkpoint_uri.startswith("s3://"):
        checkpoint_uri = checkpoint_uri.removeprefix("s3://").split("/", 1)[1]
        checkpoint_uri = f"s3://bucket/{checkpoint_uri}"
    return checkpoint_uri


class _CheckpointUri(pydantic.BaseModel):
    """Config for checkpoint file/directory."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    metadata: dict = pydantic.Field(default_factory=dict)
    """File metadata.

    Only used for debugging.
    """


def _validate_s3_uri(uri: str) -> str:
    """Validate and normalize S3 URI."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}. Must start with 's3://'")
    return normalize_uri(uri)


S3Uri = Annotated[str, pydantic.AfterValidator(_validate_s3_uri)]


class _CheckpointS3(_CheckpointUri):
    """Config for checkpoint on S3."""

    uri: S3Uri
    """S3 URI."""


class CheckpointFileS3(_CheckpointS3):
    """Config for checkpoint file on S3."""


class CheckpointDirS3(_CheckpointS3):
    """Config for checkpoint directory on S3."""


CheckpointS3: TypeAlias = CheckpointFileS3 | CheckpointDirS3


def _hf_download(cmd_args: list[str]) -> str:
    """Run Hugging Face CLI download command and return the local path.

    Uses a newer Hugging Face CLI version to download checkpoint. The dependency
    version is very old and not robust.
    """
    is_rank0 = os.environ.get("RANK", "0") == "0"
    cmd = [
        "uvx",
        f"hf@{HF_VERSION}",
        "download",
        "--format=json",
        *cmd_args,
    ]
    log.info(f"{shlex.join(cmd)}")
    output = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None if is_rank0 else subprocess.PIPE,
        text=True,
        check=True,
    )
    return json.loads(output.stdout)["path"]


class _CheckpointHf(_CheckpointUri, ABC):
    """Config for checkpoint on Hugging Face."""

    repository: str
    """Repository id (organization/repository)."""
    revision: str
    """Git revision id which can be a branch name, a tag, or a commit hash."""

    _path: str | None = None
    """Local path."""

    @abstractmethod
    def _download(self) -> str: ...

    def download(self) -> str:
        """Download checkpoint and return the local path."""
        if self._path is None:
            self._path = self._download()
        return self._path


class CheckpointFileHf(_CheckpointHf):
    """Config for checkpoint file on Hugging Face."""

    filename: str
    """File name."""

    @override
    def _download(self) -> str:
        """Download checkpoint and return the local path."""
        cmd_args = [
            self.repository,
            "--repo-type",
            "model",
            "--revision",
            self.revision,
            self.filename,
        ]
        path = _hf_download(cmd_args)
        assert os.path.exists(path), path
        return path


class CheckpointDirHf(_CheckpointHf):
    """Config for checkpoint directory on Hugging Face."""

    subdirectory: str = ""
    """Repository subdirectory."""
    include: tuple[str, ...] = ()
    """Include patterns.

    See https://huggingface.co/docs/huggingface_hub/en/guides/download#filter-files-to-download
    """
    exclude: tuple[str, ...] = ()
    """Exclude patterns.

    See https://huggingface.co/docs/huggingface_hub/en/guides/download#filter-files-to-download
    """

    @override
    def _download(self) -> str:
        """Download checkpoint and return the local path."""
        include = list(self.include) or ["*"]
        exclude = list(self.exclude)
        if self.subdirectory:
            for patterns in [include, exclude]:
                for i, pattern in enumerate(patterns):
                    patterns[i] = os.path.join(self.subdirectory, pattern)

        cmd_args = [
            self.repository,
            "--repo-type",
            "model",
            "--revision",
            self.revision,
        ]
        for pattern in include:
            cmd_args.extend(["--include", pattern])
        for pattern in exclude:
            cmd_args.extend(["--exclude", pattern])
        path = _hf_download(cmd_args)
        if self.subdirectory:
            path = os.path.join(path, self.subdirectory)
        assert os.path.exists(path), path
        return path


CheckpointHf: TypeAlias = CheckpointFileHf | CheckpointDirHf


class CheckpointConfig(pydantic.BaseModel):
    """Config for checkpoint."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    uuid: str
    """Checkpoint UUID."""
    name: str
    """Checkpoint name.

    Only used for debugging.
    """
    metadata: dict = pydantic.Field(default_factory=dict)
    """Checkpoint metadata.

    Only used for debugging.
    """
    experiment: str | None = None
    """Hydra experiment name."""
    config_file: str | None = None
    """Hydra config file."""

    s3: CheckpointS3
    """Config for checkpoint on S3."""
    hf: CheckpointHf
    """Config for checkpoint on Hugging Face."""

    @property
    def full_name(self) -> str:
        """Return full name for debugging."""
        return f"{self.name}({self.uuid})"

    def download(self) -> str:
        """Download checkpoint and return the local path."""
        if INTERNAL:
            return self.s3.uri

        log.info(f"Downloading checkpoint {self.full_name}")
        return self.hf.download()

    @classmethod
    def maybe_from_uri(cls, uri: str) -> Self | None:
        """Return checkpoint config for URI if found, otherwise None."""
        uri = normalize_uri(uri)
        return _CHECKPOINTS.get(uri, None)

    @classmethod
    def from_uri(cls, uri: str) -> Self:
        """Return checkpoint config for URI if found, otherwise raise an error."""
        self = cls.maybe_from_uri(uri)
        if self is None:
            raise ValueError(
                f"Checkpoint '{uri}' not found. Set 'export COSMOS_EXPERIMENTAL_CHECKPOINTS=1' to include experimental checkpoints."
            )
        return self

    def register(self):
        """Register checkpoint config."""
        register_checkpoint(self)


_CHECKPOINTS: dict[str, CheckpointConfig] = {}
"""Mapping from checkpoint URI to checkpoint config."""


def register_checkpoint(checkpoint_config: CheckpointConfig):
    """Register checkpoint config.

    DEPRECATED: Use 'CheckpointConfig.register' instead.
    """
    if not EXPERIMENTAL_CHECKPOINTS:
        if checkpoint_config.hf.repository in ["nvidia/Cosmos-Experimental", "nvidia-cosmos-ea/Cosmos-Experimental"]:
            # Don't register experimental checkpoints. An exception will be
            # raised in CI if the checkpoint is used without
            # EXPERIMENTAL_CHECKPOINTS.
            return
    for uri in [checkpoint_config.uuid, checkpoint_config.s3.uri]:
        if uri in _CHECKPOINTS:
            raise ValueError(f"Checkpoint '{uri}' already registered.")
        _CHECKPOINTS[uri] = checkpoint_config


def get_checkpoint_uri(checkpoint_uri: str, *, check_exists: bool = False) -> str:
    """Validate and normalize checkpoint URI."""
    checkpoint_uri = normalize_uri(checkpoint_uri)
    if (checkpoint := CheckpointConfig.maybe_from_uri(checkpoint_uri)) is not None:
        return checkpoint.s3.uri
    if checkpoint_uri.startswith("hf://"):
        return checkpoint_uri
    if _is_path(checkpoint_uri):
        if check_exists:
            checkpoint_path = Path(checkpoint_uri).expanduser().absolute()
            if not checkpoint_path.exists():
                raise ValueError(f"Checkpoint '{checkpoint_path}' does not exist.")
            checkpoint_uri = str(checkpoint_path)
        return checkpoint_uri
    if INTERNAL:
        return checkpoint_uri
    raise ValueError(
        f"Checkpoint '{checkpoint_uri}' not found. Set 'export COSMOS_EXPERIMENTAL_CHECKPOINTS=1' to include experimental checkpoints."
    )


@functools.lru_cache
def _download_hf_checkpoint(checkpoint_hf: str) -> str:
    # Parse hf://org/repo/path/to/file.pth
    assert checkpoint_hf.startswith("hf://"), f"Not a HuggingFace URI: {checkpoint_hf}"
    hf_path = checkpoint_hf.removeprefix("hf://")
    # Split into repo_id (org/repo) and filename (path/to/file.pth)
    parts = hf_path.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid HuggingFace URI format: {checkpoint_hf}. Expected format: hf://org/repo/path/to/file.pth"
        )
    repo_id = "/".join(parts[:2])  # org/repo
    filename = "/".join(parts[2:])  # path/to/file.pth
    return CheckpointFileHf(
        repository=repo_id,
        revision="main",
        filename=filename,
    ).download()


@functools.lru_cache
def download_checkpoint(checkpoint_uri: str, *, check_exists: bool = True) -> str:
    """Download a checkpoint by URI and return the local path.

    DEPRECATED: Use 'download_checkpoint_v2' instead.

    This should only be used when the checkpoint is loaded. If you just need a
    URI, use 'get_checkpoint_uri' instead.

    Downloaded checkpoints are cached, so calling this multiple times will
    return the same path.

    Supports:
    - Checkpoint UUID: 0e8177cc-0db5-4cfd-a8a4-b820c772f4fc
    - S3 URI: s3://bucket/path/to/checkpoint
    - HuggingFace URI: hf://org/repo/path/to/file.pth
    - Local path: /path/to/checkpoint
    """
    if INTERNAL:
        return checkpoint_uri
    if (checkpoint := CheckpointConfig.maybe_from_uri(checkpoint_uri)) is not None:
        return checkpoint.download()
    if checkpoint_uri.startswith("hf://"):
        return _download_hf_checkpoint(checkpoint_uri)
    if check_exists and not os.path.exists(checkpoint_uri):
        raise ValueError(f"Checkpoint path {checkpoint_uri} does not exist.")
    return checkpoint_uri


@functools.lru_cache
def download_checkpoint_v2(checkpoint_uri: str, *, check_exists: bool = True) -> str:
    """Maybe download a checkpoint by URI and return the local path.

    Similar to 'download_checkpoint', but unknown S3 URIs are passed through.
    """
    if INTERNAL:
        return checkpoint_uri
    if (checkpoint := CheckpointConfig.maybe_from_uri(sanitize_uri(checkpoint_uri))) is not None:
        return checkpoint.download()
    if checkpoint_uri.startswith("s3://"):
        return checkpoint_uri
    if checkpoint_uri.startswith("hf://"):
        return _download_hf_checkpoint(checkpoint_uri)
    if check_exists and not os.path.exists(checkpoint_uri):
        raise ValueError(f"Checkpoint path {checkpoint_uri} does not exist.")
    return checkpoint_uri


get_checkpoint_path = download_checkpoint
"""DEPRECATED: Use 'download_checkpoint' instead."""
