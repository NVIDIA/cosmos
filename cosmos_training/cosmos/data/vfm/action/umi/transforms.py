from typing import Any, Union

import kornia.augmentation as K
import torch
import torch.nn as nn
from einops import rearrange

from cosmos.data.vfm.action.umi.data_classes import DataMeta


class BaseTransforms:
    def __init__(self, data_meta: dict[str, DataMeta], apply_image_augmentation_in_cpu: bool):
        self.transforms: dict[str, Union[K.VideoSequential, nn.Sequential]] = {}
        self.data_meta: dict[str, DataMeta] = data_meta
        self.apply_image_augmentation_in_cpu: bool = apply_image_augmentation_in_cpu
        for entry_meta in data_meta.values():
            transforms_list = []
            for aug_cfg in entry_meta.augmentation:
                aug_name = aug_cfg["name"]
                aug_cfg.pop("name")
                if entry_meta.data_type == "image":
                    if aug_name in K.__dict__:
                        transform_cls = K.__dict__[aug_name]
                    else:
                        raise ValueError(f"Augmentation {aug_name} not found in kornia.augmentation.")
                elif entry_meta.data_type == "low_dim":
                    raise ValueError(
                        f"Augmentation {aug_name} not found in low dim transforms. Please implement your own augmentation method."
                    )

                transforms_list.append(transform_cls(**aug_cfg))
            if len(transforms_list) > 0:
                if entry_meta.data_type == "image":
                    self.transforms[entry_meta.name] = K.VideoSequential(*transforms_list)
                else:
                    self.transforms[entry_meta.name] = nn.Sequential(*transforms_list)

    def to(self, device: Union[torch.device, str]):
        for transform in self.transforms.values():
            transform.to(device)

    def apply(self, data_dict: dict[str, Any], consistent_on_batch: bool = False) -> dict[str, torch.Tensor]:
        for name, data in data_dict.items():
            if not self.apply_image_augmentation_in_cpu and self.data_meta[name].data_type == "image":
                continue
            if isinstance(data, dict):
                data_dict[name] = self.apply(data, consistent_on_batch)
            elif isinstance(data, torch.Tensor):
                if name in self.transforms:
                    batch_size, traj_len, *shape = data.shape
                    if consistent_on_batch:
                        data = data.reshape(1, batch_size * traj_len, *shape)

                    data_dim_num = len(self.data_meta[name].shape)
                    new_data_dim_num = len(data.shape)
                    squeeze_data = False
                    if new_data_dim_num - data_dim_num == 1:
                        data = data.unsqueeze(0)
                        squeeze_data = True
                    elif new_data_dim_num - data_dim_num != 2:
                        raise ValueError(
                            f"Data {name} has more than 2 additional dimensions: {data.shape}. Currently only support (traj_len, *shape) or (batch_size, traj_len, *shape)."
                        )
                    try:
                        data = self.transforms[name](data)
                    except Exception as e:
                        print(f"Error applying transform {name} to data {data.shape}: {e}")
                        raise e
                    if squeeze_data:
                        data = data.squeeze(0)
                    if consistent_on_batch:
                        data = rearrange(data, "1 (b t) ... -> b t ...", b=batch_size, t=traj_len)  # [B,T,...]
                    data_dict[name] = data

            else:
                raise ValueError(f"Unknown data type {type(data)} for {name}")
        return data_dict
