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

"""Preprocess dataset for inference."""

from pathlib import Path
from typing import Annotated, Any, Literal, cast, get_args

import hydra
import pydantic
import tyro
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import Dataset, IterableDataset
from typing_extensions import assert_never

from cosmos3.common.args import (
    ConfigFileType,
    ResolvedFilePath,
    ResolvedPath,
    SetupArgs,
)
from cosmos3.common.config import deserialize_config_dict
from cosmos3.dataset_samples import (
    DatasetSamples,
    IterableDatasetSamples,
    MapDatasetSamples,
)
from cosmos3.scripts.dataset_utils import (
    DEFAULT_GCS_CREDENTIALS,
    download_from_gcs_uri,
    override_dataset_root,
)

Split = Literal["train", "val", "full"]
ModelMode = Literal["forward_dynamics", "inverse_dynamics", "policy"]
ALL_MODEL_MODES: list[str] = list(get_args(ModelMode))


def _select_dataloader(
    dataloaders_dict: dict[str, Any],
    dataset_name: str,
) -> tuple[str, dict[str, Any], dict[str, Any], Any]:
    """Pick the right dataloader entry from a multi-dataloader config.

    Selection strategy:
    1. If *dataset_name* exactly matches a key in *dataloaders_dict*, use it.
    2. Otherwise, search each dataloader's ``list_of_datasets`` for an entry
       whose ``name`` matches *dataset_name* and use the first hit.
    3. Fall back to the first dataloader in iteration order.

    Returns ``(dataset_key, dataloader_config, dataset_config, inner_ds)``
    where *dataset_config* is a mutable plain dict (via
    ``OmegaConf.to_container``) and *inner_ds* is the (possibly narrowed)
    ``list_of_datasets`` value.
    """

    def _resolve(key: str) -> tuple[str, dict[str, Any], dict[str, Any], Any]:
        dl_cfg: dict[str, Any] = dataloaders_dict[key]["dataloader"]
        raw_ds = dl_cfg["dataset"]
        if not OmegaConf.is_config(raw_ds):
            raw_ds = OmegaConf.create(raw_ds)
        ds_cfg = cast(dict[str, Any], OmegaConf.to_container(raw_ds, resolve=True))
        return key, dl_cfg, ds_cfg, ds_cfg.get("list_of_datasets")

    if dataset_name and dataset_name in dataloaders_dict:
        key, dl_cfg, ds_cfg, inner = _resolve(dataset_name)
        return key, dl_cfg, ds_cfg, inner

    if dataset_name:
        for candidate_key in dataloaders_dict:
            _, dl_cfg, ds_cfg, inner = _resolve(candidate_key)
            if inner is not None and isinstance(inner, list):
                names = [e.get("name") for e in inner if isinstance(e, dict)]
                if dataset_name in names:
                    matched = [e for e in inner if isinstance(e, dict) and e.get("name") == dataset_name]
                    ds_cfg["list_of_datasets"] = [matched[0]]
                    inner = ds_cfg["list_of_datasets"]
                    logger.info(f"Narrowed list_of_datasets to single entry: {dataset_name!r}")
                    return candidate_key, dl_cfg, ds_cfg, inner

        available: dict[str, list[str | None]] = {}
        for k in dataloaders_dict:
            _, _, ds, lod = _resolve(k)
            if lod is not None and isinstance(lod, list):
                available[k] = [e.get("name") for e in lod if isinstance(e, dict)]
            else:
                available[k] = []
        raise ValueError(
            f"dataset={dataset_name!r} not found in any dataloader. "
            f"Available dataloaders and their datasets: {available}"
        )

    key, dl_cfg, ds_cfg, inner = _resolve(next(iter(dataloaders_dict)))
    return key, dl_cfg, ds_cfg, inner


class DatasetArgs(pydantic.BaseModel):
    """Arguments controlling which dataset to load and how to prepare it.

    The Hydra ``config_file`` and ``experiment`` used to resolve the dataset
    are supplied separately (typically from the setup args) so that CLI
    flag names never collide with the model's setup arguments.
    """

    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    dataset_experiment: str = ""
    """Hydra experiment for the dataset config. If empty, the setup experiment is used."""

    dataset: str = ""
    """Dataset selector (matched against dataloader keys or list_of_datasets names)."""
    sample_name: str = ""
    """Override the dataset name used in sample paths (output directories, ground truth).
    Falls back to ``dataset`` if empty."""
    dataset_split: Split = "val"
    """Dataset split."""
    model_mode: ModelMode | Literal["joint"] = "joint"
    """Model mode(s) to iterate over."""

    num_samples: int | None = None
    """Maximum number of samples to load."""
    sample_stride: int = 1
    """Stride between sample indices."""

    gcs_path_map: dict[str, str] = {}
    """Maps default dataset root paths to S3 URIs.
    When the resolved config root(s) matches a key, the data is downloaded from
    the corresponding S3 URI and the root is overridden with the local cache path."""
    gcs_credentials: str = DEFAULT_GCS_CREDENTIALS
    """Path to GCS credentials JSON file."""
    root_override: str = ""
    """Override the dataset root path without downloading. Takes precedence over gcs_path_map."""
    gcs_root_override: str = ""
    """S3 URI to download and use as the dataset root. Takes precedence over gcs_path_map
    but unlike root_override, triggers a download first."""
    force_download: bool = False
    """Force re-download even if local dataset directory already exists."""
    dataset_name_override: str = ""
    """Override the inner dataset's name field after narrowing."""
    dataset_kwargs: dict[str, Any] = {}
    """Extra keyword arguments applied to the inner dataset config before instantiation.
    Useful for overriding constructor parameters like ``snap_to_subtask`` or ``max_subtasks_per_episode``."""
    cache_dir: Annotated[ResolvedPath | None, tyro.conf.arg(aliases=("-c",))] = None
    """Local cache directory for GCS downloads. Required when ``gcs_path_map`` triggers a download."""

    config_file: ResolvedFilePath | None = None
    """Path to a serialized YAML dataset config file."""

    @property
    def dataset_label(self) -> str:
        """Label used for sample output paths; falls back to the selector when unset."""
        return self.sample_name or self.dataset


def create_dataset(
    args: DatasetArgs,
    config_args: SetupArgs | None = None,
) -> DatasetSamples:
    """Load a dataset and return a lazy ``DatasetSamples`` iterator of ``(sample_args, data_batch)`` pairs.

    Two config sources are supported (mirroring checkpoint handling):

    * **config store:** provide ``config_args`` with
      ``config_file`` + ``experiment`` to resolve the config via Hydra.
    * **YAML:** set ``args.dataset_yaml`` to a serialized YAML config

    Args:
        args: Dataset selection and download parameters.
        config_args: Hydra config arguments (internal config store).
    """

    # --- load config ----------------------------------------------------------
    if args.config_file is not None:
        config = deserialize_config_dict(args.config_file)
    elif config_args is not None:
        if args.dataset_experiment:
            config_args = config_args.model_copy(update={"experiment": args.dataset_experiment})
        # load_config() only handles .py module configs; YAML/JSON use the dict path.
        if config_args.config_file_type == ConfigFileType.MODULE:
            config = config_args.load_config()
        else:
            config = deserialize_config_dict(Path(config_args.config_file))
    else:
        raise ValueError("Provide 'config_args' or set 'dataset_yaml'")

    def _cfg(key: str) -> Any:
        return config[key] if isinstance(config, dict) else getattr(config, key)

    _train = _cfg("dataloader_train")
    _val = _cfg("dataloader_val")

    if args.dataset_split == "train":
        dataloaders_config = _train
    elif args.dataset_split in ("val", "full"):
        dataloaders_config = _val if _val is not None else _train
    else:
        assert_never(args.dataset_split)
    assert dataloaders_config is not None, "Neither dataloader_val nor dataloader_train found in config"

    dataloaders_dict: dict[str, Any] = dataloaders_config["dataloaders"]
    dataset_key, _, dataset_config, inner_ds = _select_dataloader(dataloaders_dict, args.dataset)
    logger.info(f"Selected dataloader {dataset_key!r} for dataset={args.dataset!r}")

    if isinstance(inner_ds, list):
        if len(inner_ds) == 1 and isinstance(inner_ds[0], dict):
            inner_ds = inner_ds[0]
        else:
            raise ValueError("Use --dataset to select a single dataset entry by name.")
    resolution: str | None = inner_ds.get("resolution") if isinstance(inner_ds, dict) else None
    # wrap_dataset also accepts a global resolution that individual entries can override
    if resolution is None:
        resolution: str | None = dataset_config.get("resolution")

    # --- resolve the inner dataset target -------------------------------------
    # Dataset config fields may be nested inside inner_ds["dataset"] when the
    # full dataset_entry wrapper dict is preserved (after _select_dataloader narrowing),
    # or live directly on inner_ds when the raw dataset dict is returned.
    _inner = inner_ds
    if isinstance(inner_ds, dict) and isinstance(inner_ds.get("dataset"), dict):
        _inner = inner_ds["dataset"]

    # --- apply dataset_name override ------------------------------------------
    if args.dataset_name_override and isinstance(_inner, dict):
        for key in ("dataset_name", "name"):
            if key in _inner:
                logger.info(f"Overriding inner dataset {key}={_inner[key]!r} -> {args.dataset_name_override!r}")
                _inner[key] = args.dataset_name_override
                break

    # --- resolve root via overrides / gcs_path_map and download ----------
    root_override: str | list[str] = ""
    config_root = _inner.get("root") if isinstance(_inner, dict) else None
    if args.root_override:
        root_override = args.root_override
    elif args.gcs_root_override:
        if args.cache_dir is None:
            raise ValueError("cache_dir is required when gcs_root_override triggers a download")
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        root_override = str(
            download_from_gcs_uri(
                args.gcs_root_override,
                args.cache_dir,
                gcs_credentials=args.gcs_credentials,
                force_download=args.force_download,
            )
        )
        logger.info(f"Downloaded dataset from {args.gcs_root_override} to {root_override}")
    elif (config_root is not None) and args.gcs_path_map:
        root_is_sequence = False
        if isinstance(config_root, str):
            orig_roots: list[str] = [config_root]
        elif isinstance(config_root, (list, tuple)):
            root_is_sequence = True
            orig_roots = list(config_root)
        else:
            raise ValueError(f"Unexpected root type: {type(config_root)}")

        # Populate the cache directory.
        if args.cache_dir is None:
            raise ValueError("cache_dir is required when gcs_path_map triggers a download")
        args.cache_dir.mkdir(parents=True, exist_ok=True)

        for i, orig_root in enumerate(orig_roots):
            if orig_root.startswith("s3://"):
                # Already an S3 URI — use directly, no mapping needed.
                logger.info(f"Root {orig_root} is already an S3 URI, skipping gcs_path_map")
                continue
            uri = args.gcs_path_map.get(orig_root.rstrip("/"))
            if uri is None:
                raise ValueError(f"No entry found in gcs_path_map for root='{orig_root.rstrip('/')}'")

            orig_roots[i] = str(
                download_from_gcs_uri(
                    uri,
                    args.cache_dir,
                    gcs_credentials=args.gcs_credentials,
                    force_download=args.force_download,
                    dataset_name_override=args.dataset_name_override,
                )
            )

        if orig_roots:
            root_override = orig_roots if root_is_sequence else orig_roots[0]

    if root_override:
        override_dataset_root(dataset_config, root_override)

    # --- override split -------------------------------------------------------
    if isinstance(_inner, dict) and "split" in _inner:
        _inner["split"] = args.dataset_split
        logger.info(f"Overriding dataset split to {args.dataset_split!r}")

    if isinstance(_inner, dict) and "is_val" in _inner:
        is_val = args.dataset_split == "val"
        _inner["is_val"] = is_val
        logger.info(f"Overriding dataset is_val to {is_val!r}")

    # --- disable training-only augmentations for val and 'full' --------------------------
    if args.dataset_split in ("val", "full") and "cfg_dropout_rate" in dataset_config:
        dataset_config["cfg_dropout_rate"] = 0.0
        logger.info("Overriding cfg_dropout_rate to 0.0 for val export")

    # --- disable shuffle for val dataset export -------------------------------
    if args.dataset_split == "val" and isinstance(_inner, dict) and "shuffle" in _inner:
        _inner["shuffle"] = False
        logger.info("Overriding dataset shuffle to False for val export")

    # --- apply dataset_kwargs overrides --------------------------------------
    if args.dataset_kwargs and isinstance(_inner, dict):
        for key, value in args.dataset_kwargs.items():
            logger.info(f"Overriding dataset {key}={_inner.get(key)!r} -> {value!r}")
            _inner[key] = value

    # --- apply gcs credentials override (internal-only)--------------------
    if isinstance(_inner, dict) and "credential_path" in _inner:
        _inner["credential_path"] = args.gcs_credentials
        logger.info(f"Overriding dataset credential_path to {_inner['credential_path']!r}")

    # --- instantiate dataset --------------------------------------------------
    logger.info(f"Instantiating dataset with config: {dataset_config}")
    dataset: Dataset = hydra.utils.instantiate(dataset_config)
    dataset_size = len(dataset)  # type: ignore

    transform = getattr(dataset, "transform", None)
    if isinstance(dataset, IterableDataset):
        inner = getattr(dataset, "dataset", None)
        raw = getattr(inner, "dataset", inner)
        if raw is not None and hasattr(raw, "__getitem__"):
            dataset = raw

    if hasattr(dataset, "_register_sources") and hasattr(dataset, "_all_shard_roots") and dataset_size == 0:
        dataset._register_sources()
        dataset_size = len(dataset)  # type: ignore

    def _create_dataset_samples(ds: Dataset, m: list[str], ids: list[int]) -> DatasetSamples:
        sample_overrides_data = config_args.sample_overrides.model_dump(exclude_none=True) if config_args else {}
        if isinstance(ds, IterableDataset):
            return IterableDatasetSamples(ds, m, ids, transform, resolution, args.dataset_label, sample_overrides_data)
        return MapDatasetSamples(ds, m, ids, transform, resolution, args.dataset_label, sample_overrides_data)

    # --- compute sample indices -----------------------------------------------
    sample_ids = list(range(0, dataset_size, args.sample_stride))
    if args.num_samples:
        sample_ids = sample_ids[: args.num_samples]

    modes = list(ALL_MODEL_MODES) if args.model_mode == "joint" else [args.model_mode]

    # The transform's text tokenizer produces text_token_ids for training only;
    # during inference the model re-tokenizes from the caption string (see
    # OmniMoTModel._get_inference_text_tokens).  Disable it to avoid tokenizer
    # compatibility issues in environments where apply_chat_template returns a dict.
    if transform is not None and hasattr(transform, "text_tokenizer"):
        transform.text_tokenizer = None

    logger.info(f"Created lazy iterator for {len(modes) * len(sample_ids)} samples across modes {modes}")
    return _create_dataset_samples(dataset, modes, sample_ids)
