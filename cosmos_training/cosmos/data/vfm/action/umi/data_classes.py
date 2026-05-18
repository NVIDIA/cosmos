from dataclasses import dataclass
from typing import Any, Union, cast

from omegaconf import DictConfig, OmegaConf


@dataclass
class SourceDataMeta:
    name: str
    """The data name from the source dataset."""
    shape: tuple[int, ...]
    """The shape of a single time step of the data."""
    include_indices: list[int]
    """Indices of the data to include in the dataset relative to the current step (0). Negative indices means the data is from the past."""

    def __post_init__(self):
        if isinstance(self.shape, list):
            self.shape = tuple(self.shape)
        if len(self.include_indices) == 0:
            raise ValueError(f"include_indices must be a non-empty list in {self.name}.")
        for i, index in enumerate(self.include_indices):
            if i < len(self.include_indices) - 1:
                if index > self.include_indices[i + 1]:
                    raise ValueError(
                        f"include_indices must be monotonically increasing, but got {self.include_indices} in {self.name}."
                    )
        if len(self.shape) == 0:
            raise ValueError(f"shape must be a non-empty list in {self.name}.")


@dataclass
class DataMeta:
    name: str
    """The output name to be used for training."""
    shape: tuple[int, ...]
    """The shape of a single time step of the data."""
    data_type: str
    """low_dim or image"""
    length: int
    """The length of the data."""
    normalizer: str
    """identity, range, normal. range: normalize to [-1, 1]; normal: normalize to mean=0, std=1"""
    augmentation: list[dict[str, Any]]
    """The augmentation to apply to the data."""
    source_entry_names: list[str]
    """The source entry names to use for the data."""

    def __post_init__(self):
        if isinstance(self.shape, list):
            self.shape = tuple(self.shape)

        if self.data_type not in ["low_dim", "image"]:
            raise ValueError(f"data_type must be one of ['low_dim', 'image'] in {self.name}.")

        if len(self.source_entry_names) == 0:
            raise ValueError(f"source_entry_names must be a non-empty list in {self.name}.")

        if self.length <= 0:
            raise ValueError(f"length must be greater than 0 in {self.name}.")

        if len(self.shape) == 0:
            raise ValueError(f"shape must be a non-empty list in {self.name}.")

        if self.normalizer not in ["identity", "range", "normal", "clamped_range"]:
            raise ValueError(
                f"normalizer must be one of ['identity', 'range', 'normal', 'clamped_range'] in {self.name}."
            )


def construct_data_meta(
    data_meta: Union[dict[str, dict[str, Any]], DictConfig],
) -> dict[str, DataMeta]:
    if isinstance(data_meta, DictConfig):
        data_meta = cast(dict[str, dict[str, Any]], OmegaConf.to_container(data_meta, resolve=True))
    data_meta_dict = {}
    for name, entry_meta_dict in data_meta.items():
        entry_meta_dict.update({"name": name})
        data_meta_dict[name] = DataMeta(**entry_meta_dict)
    return data_meta_dict


def construct_source_data_meta(
    source_data_meta: Union[dict[str, dict[str, Any]], DictConfig],
) -> dict[str, SourceDataMeta]:
    if isinstance(source_data_meta, DictConfig):
        source_data_meta = cast(
            dict[str, dict[str, Any]],
            OmegaConf.to_container(source_data_meta, resolve=True),
        )
    source_data_meta_dict = {}
    for name, entry_meta_dict in source_data_meta.items():
        entry_meta_dict.update({"name": name})
        source_data_meta_dict[name] = SourceDataMeta(**entry_meta_dict)
    return source_data_meta_dict
