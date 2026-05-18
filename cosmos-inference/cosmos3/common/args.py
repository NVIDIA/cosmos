# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import glob
import itertools
import json
import os
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Literal,
    NoReturn,
    TypeVar,
    overload,
    override,
)

import pydantic
import tyro
import yaml
from typing_extensions import Self, assert_never
from tyro.conf import Suppress

from cosmos3.common.config import deserialize_config, deserialize_config_dict, unstructure_config
from cosmos3.common.init import is_rank0 as is_rank0
from cosmos3._src.imaginaire.config import Config
from cosmos3._src.imaginaire.flags import TRAINING, StrEnum
from cosmos3._src.imaginaire.utils.checkpoint_db import CheckpointDirHf

if TYPE_CHECKING:
    from cosmos3.common.inference import Inference

if TRAINING or TYPE_CHECKING:
    T = TypeVar("T")
    Training = Annotated[T, None]
else:
    Training = Suppress


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
VIDEO_EXTENSIONS = [".mp4"]
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS


def _download_file_url(url: str, path: Path):
    if "huggingface.co" in url:
        _download_file_hf(url, path)
    else:
        import obstore

        base_url, file_name = url.rsplit("/", 1)
        store = obstore.store.from_url(base_url)
        result = obstore.get(store, file_name)
        with path.open("wb") as f:
            for chunk in iter(result):
                f.write(chunk)


def _download_file_hf(url: str, path: Path):
    """Download from HuggingFace with token auth."""
    import urllib.request

    url = url.replace("/blob/", "/resolve/")
    headers: dict[str, str] = {}
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.is_file():
            token = token_path.read_text().strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, path.open("wb") as f:
        while chunk := resp.read(8192):
            f.write(chunk)


def _download_file(url: str, path: Path):
    if "://" not in url and Path(url).resolve() == path.resolve():
        return
    meta_path = Path(f"{path}.meta")
    if path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text())["url"] == url:
            return

    if "://" in url:
        # Download to a temporary directory and symlink to the final path.
        # This keeps the output directory small.
        local_path = Path(tempfile.TemporaryDirectory(delete=False).name) / path.name
        _download_file_url(url, local_path)
    else:
        local_path = Path(url)

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.symlink_to(local_path)
    meta_path.write_text(json.dumps({"url": url}))


@overload
def download_file(url: str, output_dir: Path, output_name: str) -> str: ...


@overload
def download_file(url: None, output_dir: Path, output_name: str) -> None: ...


def download_file(url, output_dir, output_name):
    """Download a file from a URL to a local path.

    * Skip if the file already exists.
    * Only download on rank 0.
    """
    if not url or "://" not in url:
        return url
    ext = Path(url).suffix.lower()
    download_path = output_dir / f"{output_name}{ext}"
    if is_rank0():
        _download_file(url, download_path)
    return str(download_path)


@overload
def path_to_str(v: Path | str) -> str: ...


@overload
def path_to_str(v: None) -> None: ...


def path_to_str(v):
    """Convert optional path to optional string."""
    if v is None:
        return None
    return str(v)


@overload
def str_to_path(v: Path | str) -> Path: ...


@overload
def str_to_path(v: None) -> None: ...


def str_to_path(v):
    if v is None:
        return None
    return Path(v)


_PydanticModelT = TypeVar("_PydanticModelT", bound=pydantic.BaseModel)


def _get_root_exception(exception: BaseException) -> BaseException:
    if exception.__cause__ is not None:
        return _get_root_exception(exception.__cause__)
    if exception.__context__ is not None:
        return _get_root_exception(exception.__context__)
    return exception


def handle_tyro_exception(exception: Exception) -> NoReturn:
    root_exception = _get_root_exception(exception)
    if isinstance(root_exception, pydantic.ValidationError):
        raise root_exception from None
    raise exception


_T = TypeVar("_T")


def tyro_cli(cls: type[_T], **kwargs) -> _T:
    kwargs.setdefault("console_outputs", is_rank0())
    try:
        return tyro.cli(cls, **kwargs)
    except Exception as e:
        handle_tyro_exception(e)


def _resolve_path(v: Path) -> Path:
    """Resolve path to absolute."""
    return v.expanduser().absolute()


def _resolve_file_or_url(v: str) -> str:
    """Validate a file path or URL. URLs pass through; local paths must exist and are resolved to absolute."""
    if v.startswith(("http://", "https://", "s3://")):
        return v
    p = Path(v).expanduser().absolute()
    if not p.is_file():
        raise ValueError(f"Path does not point to a file: {v}")
    return str(p)


ResolvedPath = Annotated[Path, pydantic.AfterValidator(_resolve_path)]
ResolvedFilePath = Annotated[pydantic.FilePath, pydantic.AfterValidator(_resolve_path)]
ResolvedFilePathOrUrl = Annotated[str, pydantic.AfterValidator(_resolve_file_or_url)]
ResolvedDirectoryPath = Annotated[pydantic.DirectoryPath, pydantic.AfterValidator(_resolve_path)]


class ArgsBase(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    @classmethod
    def from_files(cls, paths: list[Path]) -> list[Self]:
        return from_files(cls, paths)


_ArgsT = TypeVar("_ArgsT", bound=ArgsBase)


class OverridesBase(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    @classmethod
    def from_files(cls, paths: list[Path], *, overrides: pydantic.BaseModel | None = None) -> list[Self]:
        return from_files(cls, paths, overrides=overrides)

    def download(self, output_dir: Path):
        """Download all urls."""
        pass

    def _build(self, target: type[_ArgsT], **kwargs) -> _ArgsT:
        """Build arguments from overrides."""
        return target.model_validate(self.model_dump() | kwargs, extra="ignore")


_OverridesT = TypeVar("_OverridesT", bound=OverridesBase)


class ConfigFileType(StrEnum):
    """Config file type."""

    MODULE = "module"
    """Hydra config store module."""
    YAML = "yaml"
    """Hydra config yaml."""
    JSON = "json"
    """Hugging Face model json."""

    @classmethod
    def from_path(cls, path: str) -> Self:
        suffix = Path(path).suffix.lower()
        if suffix == ".py":
            return cls("module")
        if suffix in [".yaml", ".yml"]:
            return cls("yaml")
        if suffix == ".json":
            return cls("json")
        raise ValueError(f"Invalid config file: {path}")


def _validate_config_file(v: str) -> str:
    config_file_type = ConfigFileType.from_path(v)
    if config_file_type == ConfigFileType.MODULE:
        # Relative module path
        return v

    # Absolute file path
    p = Path(v).expanduser().absolute()
    if not p.is_file():
        raise ValueError(f"Config file does not exist: {v}")
    return str(p)


ConfigFilePath = Annotated[str, pydantic.AfterValidator(_validate_config_file)]


class ConfigArgs(ArgsBase):
    config_file: ConfigFilePath
    config_file_type: ConfigFileType
    experiment: str
    experiment_overrides: list[str]

    def load_config(self) -> Config:
        """Load Hydra config."""
        from cosmos3.common.config import load_config

        match self.config_file_type:
            case ConfigFileType.MODULE:
                return load_config(self.config_file, self.experiment, overrides=self.experiment_overrides)
            case ConfigFileType.YAML | ConfigFileType.JSON:
                config = deserialize_config(Path(self.config_file))
                assert isinstance(config, Config)
                return config
            case _:
                assert_never(self.config_file_type)

    def load_model_config_dict(self) -> dict:
        """Load model config dict."""
        match self.config_file_type:
            case ConfigFileType.MODULE:
                return unstructure_config(self.load_config().model)
            case ConfigFileType.YAML | ConfigFileType.JSON:
                return deserialize_config_dict(Path(self.config_file))["model"]
            case _:
                assert_never(self.config_file_type)


DEFAULT_CONFIG_FILE = "cosmos3/_src/vfm/configs/base/config.py"


class ConfigOverrides(OverridesBase):
    """Hydra config arguments."""

    config_file: Training[ConfigFilePath] = DEFAULT_CONFIG_FILE
    """Hydra config store module, Hydra config yaml file, or Hugging Face model json file."""
    config_file_type: Suppress[ConfigFileType | None] = None
    """Hydra config file type."""
    experiment: Training[str] = ""
    """Hydra experiment name."""
    experiment_overrides: Training[list[str]] = pydantic.Field(default_factory=list)
    """Hydra experiment overrides."""

    def _build_config(self) -> None:
        self.config_file_type = ConfigFileType.from_path(self.config_file)

    def build_config(self) -> ConfigArgs:
        self._build_config()
        return self._build(ConfigArgs)


class CheckpointType(StrEnum):
    """Checkpoint type."""

    HF = "hf"
    """Hugging Face checkpoint."""
    DCP = "dcp"
    """DCP checkpoint."""

    @classmethod
    def from_path(cls, path: Path) -> Self:
        if any(path.glob("*.safetensors")):
            if not (path / "config.json").exists():
                raise ValueError(f"Invalid Hugging Face checkpoint: {path}")
            return cls("hf")
        if any(path.glob("*.distcp")):
            if not (path / ".metadata").exists():
                raise ValueError(f"Invalid DCP checkpoint: {path}")
            return cls("dcp")
        raise ValueError(f"Unknown checkpoint type: {path}")


class CheckpointConfig(pydantic.BaseModel):
    """Checkpoint config."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    model_memory_bytes: int | None = None
    """Approximate model size in bytes.

    Used for automatic sharding.
    """

    s3_uri: str
    """Checkpoint S3 URI."""
    hf: CheckpointDirHf
    """Config for checkpoint on Hugging Face."""

    def download(self) -> str:
        return self.hf.download()

    @property
    def pretrained_kwargs(self) -> dict[str, Any]:
        return dict(
            pretrained_model_name_or_path=self.hf.repository,
            subfolder=self.hf.subdirectory,
            revision=self.hf.revision,
        )


class CheckpointArgs(ConfigArgs):
    checkpoint_path: str

    checkpoint_type: CheckpointType
    model_memory_bytes: int | None

    checkpoint_hf: CheckpointDirHf | None

    credential_path: str
    use_ema_weights: bool
    checkpoint_cache_dir: Path | None

    def download_checkpoint(self) -> Path:
        if self.checkpoint_hf is not None:
            return Path(self.checkpoint_hf.download())
        if "://" in self.checkpoint_path:
            raise ValueError(f"Invalid checkpoint path: {self.checkpoint_path}")
        return Path(self.checkpoint_path)

    @pydantic.model_validator(mode="after")
    def _validate_checkpoint(self) -> Self:
        if self.checkpoint_type == CheckpointType.DCP:
            if not self.config_file:
                raise ValueError("'config_file' is required")
            if self.config_file_type == ConfigFileType.MODULE and not self.experiment:
                raise ValueError("'experiment' is required")
        return self


class CheckpointOverrides(ConfigOverrides):
    """Checkpoint arguments."""

    checkpoint_path: str
    """Model name or path.

    * Model name: Cosmos3-Nano
    * Local path: /path/to/checkpoint
    """

    checkpoint_type: Suppress[CheckpointType | None] = None
    """Checkpoint type."""
    model_memory_bytes: Suppress[int | None] = None
    """Approximate model size in bytes."""

    checkpoint_hf: Suppress[CheckpointDirHf | None] = None
    """Hugging Face checkpoint directory."""

    credential_path: Training[str] = "credentials/gcp_checkpoint.secret"
    """Path to S3 credentials file for remote checkpoint loading."""
    use_ema_weights: Training[bool] = True
    """If True, use EMA weights. Otherwise, use regular weights."""
    checkpoint_cache_dir: Training[Path | None] = None
    """Directory for caching S3 checkpoints."""

    def _build_checkpoint(self, checkpoints: dict[str, CheckpointConfig]):
        # Detect checkpoint type
        if self.checkpoint_path in checkpoints:
            self.checkpoint_type = CheckpointType.HF
            checkpoint = checkpoints[self.checkpoint_path]
            if not self.model_memory_bytes:
                self.model_memory_bytes = checkpoint.model_memory_bytes
            self.checkpoint_hf = checkpoint.hf
        elif self.checkpoint_path.startswith("s3://"):
            self.checkpoint_type = CheckpointType.DCP
            self.checkpoint_path = self.checkpoint_path.rstrip("/")
            # Strip '/model' suffix, since it isn't included in checkpoint_db.
            # Automatically added during checkpoint load by
            # 'cosmos3._src.vfm.utils.model_loader.load_model_from_checkpoint'.
            if not self.checkpoint_path.endswith("/model"):
                self.checkpoint_path = self.checkpoint_path + "/model"
        else:
            checkpoint_dir = Path(self.checkpoint_path).expanduser().absolute()
            if not checkpoint_dir.is_dir():
                raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
            if (checkpoint_dir / "model").is_dir():
                checkpoint_dir = checkpoint_dir / "model"
            self.checkpoint_path = str(checkpoint_dir)
            self.checkpoint_type = CheckpointType.from_path(checkpoint_dir)
            if (checkpoint_dir / "config.json").is_file():
                self.config_file = str(checkpoint_dir / "config.json")
        self.config_file_type = ConfigFileType.from_path(self.config_file)

        if self.checkpoint_type == CheckpointType.DCP and self.config_file_type == ConfigFileType.MODULE:
            # Infer missing values from checkpoint path
            if not self.experiment:
                pattern = r"/(?P<experiment>[\w-]+)/checkpoints/iter_(?P<iter>\d+)/"
                match = re.search(pattern, f"/{self.checkpoint_path}/")
                if match is None:
                    raise ValueError(f"Could not infer experiment from checkpoint path: {self.checkpoint_path}")
                if not self.experiment:
                    self.experiment = match.group("experiment")

            self.experiment_overrides = [
                f"checkpoint.load_from_object_store.enabled={self.checkpoint_path.startswith('s3://')}",
                # Pretrained weights are only needed for training.
                # See 'cosmos3._src.vfm.configs.base.config.Config._set_skip_pretrained_if_checkpoint_exists()'.
                "model.config.vlm_config.load_pretrained=False",
                "model.config.diffusion_expert_config.load_weights_from_pretrained=False",
                *self.experiment_overrides,
            ]

    def build_checkpoint(self, *, checkpoints: dict[str, CheckpointConfig]) -> CheckpointArgs:
        self._build_checkpoint(checkpoints=checkpoints)
        self._build_config()
        return self._build(CheckpointArgs)


ParallelismPreset = Literal["throughput", "latency"]
CfgpSize = Annotated[int, pydantic.Field(ge=1, le=2)]
CompiledRegion = Literal["all", "language"]


class ParallelismArgs(ArgsBase):
    """Parallelism arguments."""

    dp_replicate_size: pydantic.PositiveInt
    dp_shard_size: pydantic.PositiveInt
    tp_size: pydantic.PositiveInt
    cp_size: pydantic.PositiveInt
    cfgp_size: CfgpSize
    use_torch_compile: bool
    use_cuda_graphs: bool
    compiled_region: CompiledRegion
    compile_dynamic: bool
    use_separate_pipeline_vision_decode_gpu: bool

    @property
    def world_size(self) -> int:
        return max(
            self.dp_replicate_size * self.dp_shard_size,
            self.cp_size * self.cfgp_size,
        )


class ParallelismOverrides(OverridesBase):
    parallelism_preset: ParallelismPreset = "latency"
    """Preset for automatic sharding."""
    device_memory_utilization: Training[float] = pydantic.Field(default=0.75, ge=0.0, le=1.0)
    """Fraction of device memory to use for model weights.

    Used for automatic sharding.
    """

    dp_replicate_size: pydantic.NonNegativeInt = 1
    """Data parallel size."""
    dp_shard_size: pydantic.NonNegativeInt = 1
    """FSDP size."""
    tp_size: pydantic.NonNegativeInt = 1
    """Tensor parallel size."""
    cp_size: pydantic.NonNegativeInt = 1
    """Context parallel size."""
    cfgp_size: CfgpSize | Literal[0] = 1
    """CFG (Classifier Free Guidance) parallel size.

    If set to 1, runs conditional and unconditional guidance on the same GPU.
    If set to 2, parallelizes conditional and unconditional guidance onto two GPUs.
    """

    use_torch_compile: bool = True
    """Whether to use torch compile."""
    use_cuda_graphs: bool = True
    """Whether to use CUDA graphs."""
    compiled_region: CompiledRegion = "all"
    """Torch compile region."""
    compile_dynamic: bool = True
    """Compile with symbolic-shape kernels (maps to ``torch.compile(dynamic=...)``).

    Defaults to ``True`` for backward compatibility with training, which can
    see varying shapes across batches.  Setting to ``False`` produces faster
    kernels when the shapes are stable (e.g. single-prompt AR inference), at
    the cost of a recompile on shape change.
    """
    use_separate_pipeline_vision_decode_gpu: bool = False
    """Whether to place pipeline vision decode on a spare local GPU when one is available."""

    def _build_parallelism(self, world_size: int | None, local_world_size: int | None, device_memory_bytes: int | None):
        if not self.dp_replicate_size:
            self.dp_replicate_size = 1
        if not self.dp_shard_size:
            self.dp_shard_size = 1
        if not self.tp_size:
            self.tp_size = 1
        if not self.cp_size:
            self.cp_size = 1
        if not self.cfgp_size:
            self.cfgp_size = 1

    def build_parallelism(
        self, world_size: int | None = None, local_world_size: int | None = None, device_memory_bytes: int | None = None
    ) -> ParallelismArgs:
        self._build_parallelism(
            world_size=world_size, local_world_size=local_world_size, device_memory_bytes=device_memory_bytes
        )
        return self._build(ParallelismArgs)


class GuardrailArgs(ArgsBase):
    """Guardrail arguments."""

    guardrails: bool
    offload_guardrail_models: bool


class GuardrailOverrides(OverridesBase):
    guardrails: bool = False
    """Enable guardrails."""
    offload_guardrail_models: bool = False
    """Offload guardrail models to CPU."""


class SetupArgs(ABC, CheckpointArgs, ParallelismArgs, GuardrailArgs):
    output_dir: ResolvedPath
    keep_going: bool
    skip_invalid_samples: bool
    debug: bool
    profile: bool
    benchmark: bool
    warmup: pydantic.NonNegativeInt
    max_model_len: pydantic.PositiveInt | None
    max_num_seqs: pydantic.PositiveInt | None

    # Subclass must implement these fields/methods
    # ------------------------------------------------------------
    sample_overrides: pydantic.BaseModel

    @classmethod
    @abstractmethod
    def get_sample_overrides_cls(cls) -> type["SampleOverrides"]:
        """Get sample overrides class."""

    @classmethod
    @abstractmethod
    def get_sample_args_cls(cls) -> type["SampleArgs"]:
        """Get sample arguments class."""

    @classmethod
    @abstractmethod
    def get_inference_cls(cls) -> type["Inference"]:
        """Get inference class."""

    @classmethod
    def get_variant(cls) -> str:
        return cls.model_fields["variant"].default


class SetupOverrides(ABC, CheckpointOverrides, ParallelismOverrides, GuardrailOverrides):
    """Inference setup arguments."""

    output_dir: Annotated[ResolvedPath | None, tyro.conf.arg(aliases=("-o",))] = None
    """Output directory."""
    keep_going: bool = False
    """If True, catch and log errors instead of raising them."""
    skip_invalid_samples: bool = False
    """If True, skip samples whose modality (e.g. action, sound) is not supported by the
    loaded model and emit a ``status='skip'`` output instead of raising. Useful for tests
    and examples that exercise multiple modalities against checkpoints with varying support."""
    debug: bool = False
    """If True, enable debug outputs."""
    profile: bool = False
    """Run profiler and save report to output directory."""
    benchmark: bool = False
    """If set, measures and reports inference runtime (disables tqdm)."""
    warmup: pydantic.NonNegativeInt = 0
    """Number of warmup generations before each sample."""
    max_model_len: pydantic.PositiveInt | None = None
    """Maximum total tokens per batch.  When set, samples are packed into
    batches by token count."""
    max_num_seqs: pydantic.PositiveInt | None = 1
    """Maximum number of sequences per batch.  When set, samples are packed into
    batches by number of sequences."""

    def _build_setup(self):
        pass

    @abstractmethod
    def build_setup(self) -> SetupArgs:
        """Build setup arguments."""


class SampleArgs(ArgsBase):
    """Inference sample arguments."""

    output_dir: ResolvedPath
    model: str
    extra: dict

    name: str
    num_outputs: pydantic.PositiveInt
    seed: int | None
    tensors_file: ResolvedFilePath | None
    pickle_file: ResolvedFilePath | None

    def get_data(self, *, device: str | int = "cpu") -> dict[str, Any]:
        import pickle

        import safetensors.torch

        data: dict[str, Any] = {}
        if self.tensors_file is not None:
            data |= safetensors.torch.load_file(self.tensors_file, device=device)
        if self.pickle_file is not None:
            with self.pickle_file.open("rb") as f:
                data |= dict(pickle.load(f))
        return data


class SampleOverrides(OverridesBase):
    """Inference sample arguments."""

    output_dir: Suppress[ResolvedPath | None] = None
    """Output directory."""
    model: str | None = None
    """Model name."""
    extra: Suppress[dict | None] = None
    """Extra arguments."""

    name: Suppress[str | None] = None
    """Name of the sample."""
    num_outputs: Training[Annotated[pydantic.PositiveInt | None, tyro.conf.arg(aliases=("-n",))]] = None
    """Number of outputs to generate per sample."""
    seed: int | None = None
    """Seed for the random number generator."""

    tensors_file: ResolvedFilePath | None = None
    """Path to data tensors file."""
    pickle_file: ResolvedFilePath | None = None
    """Path to data pickle file."""

    @override
    @classmethod
    def from_files(cls, paths: list[Path], *, overrides: pydantic.BaseModel | None = None) -> list[Self]:
        objs_per_file = _from_files(cls, paths, overrides=overrides)

        # Check names
        all_objs: list[Self] = []
        names: set[str] = set()
        for path, objs in objs_per_file.items():
            for line, obj in enumerate(objs):
                if not obj.name:
                    if path.suffix.lower() == ".jsonl":
                        obj.name = f"{path.stem}_{line}"
                    else:
                        obj.name = path.stem
                if obj.name in names:
                    raise ValueError(f"Duplicate name: '{obj.name}'")
                all_objs.append(obj)
        return all_objs

    def _build_sample(self):
        if self.model is None:
            self.model = ""
        if self.extra is None:
            self.extra = {}

        if self.num_outputs is None:
            self.num_outputs = 1

    @abstractmethod
    def build_sample(self, *, model_config: Any) -> SampleArgs:
        """Build sample arguments."""


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base*, merging nested dicts instead of replacing them."""
    merged = base.copy()
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _from_file(cls: type[_PydanticModelT], path: Path, override_data: dict[str, Any]) -> list[_PydanticModelT]:
    """Load arguments from a json/jsonl/yaml file.

    Returns a list of arguments.
    """
    # Load data from file
    if path.suffix in [".json"]:
        data_list = [json.loads(path.read_text())]
    elif path.suffix in [".jsonl"]:
        data_list = [json.loads(line) for line in path.read_text().splitlines() if line]
    elif path.suffix in [".yaml", ".yml"]:
        data_list = [yaml.safe_load(path.read_text())]
    else:
        raise ValueError(f"Unsupported file extension: {path}")

    # Validate data
    # Input paths are relative to the file path
    path = path.expanduser().absolute()
    with contextlib.chdir(path.parent):
        objs: list[_PydanticModelT] = []
        for i, data in enumerate(data_list):
            data = _deep_merge(data, override_data)
            try:
                objs.append(cls.model_validate(data))
            except pydantic.ValidationError as e:
                raise ValueError(
                    f"Error validating parameters from '{path}' at sample {i}\nParameters: {data}\n{e}"
                ) from e

    return objs


def _from_files(
    cls: type[_PydanticModelT], paths: list[Path], *, overrides: pydantic.BaseModel | None = None
) -> dict[Path, list[_PydanticModelT]]:
    """Load arguments from a list of json/jsonl/yaml files.

    Returns a list of arguments per file.
    """
    if not paths:
        raise ValueError("No inference parameter files")

    if overrides is None:
        override_data = {}
    else:
        override_data = overrides.model_dump(exclude_none=True)

    # Expand glob patterns
    expanded_paths: list[Path] = []
    for path in paths:
        pattern = str(path)
        if "*" in pattern:
            expanded_paths.extend(Path(g) for g in glob.glob(pattern, recursive=True))
        else:
            expanded_paths.append(path)
    paths = sorted(set(expanded_paths))

    # Load arguments from files
    objs_per_file: dict[Path, list[_PydanticModelT]] = {}
    for path in paths:
        objs_per_file[path] = _from_file(cls, path, override_data)
    return objs_per_file


def from_files(
    cls: type[_PydanticModelT], paths: list[Path], *, overrides: pydantic.BaseModel | None = None
) -> list[_PydanticModelT]:
    return list(itertools.chain.from_iterable(_from_files(cls, paths, overrides=overrides).values()))


class SampleOutput(ArgsBase):
    content: dict = pydantic.Field(default_factory=dict)
    """Output json."""
    files: list[Path] = pydantic.Field(default_factory=list)
    """List of output file paths."""

    def map_files(self, func: Callable[[Path], Path]) -> Self:
        return self.model_copy(update={"files": [func(p) for p in self.files]})


class SampleOutputs(ArgsBase):
    """Inference sample outputs."""

    args: dict
    """Sample arguments."""

    status: Literal["success", "error", "skip"] = "success"
    """Generation status. ``skip`` indicates the sample was bypassed because the loaded
    model does not support the requested modality (e.g. action/sound)."""
    message: str | None = None
    """Generation error or skip reason message."""
    stack_trace: str | None = None
    """Generation error stack trace."""

    outputs: list[SampleOutput] = pydantic.Field(default_factory=list)
    """List of sample outputs."""

    @pydantic.model_validator(mode="after")
    def _validate_name(self) -> Self:
        if "name" not in self.args:
            raise ValueError("'name' is required")
        return self

    @property
    def name(self) -> str:
        return self.args["name"]

    def map_files(self, func: Callable[[Path], Path]) -> Self:
        return self.model_copy(update={"outputs": [output.map_files(func) for output in self.outputs]})
