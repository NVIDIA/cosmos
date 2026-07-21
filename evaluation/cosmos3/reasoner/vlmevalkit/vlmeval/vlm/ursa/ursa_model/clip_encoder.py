# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms
from einops import rearrange

from .sam import create_sam_vit
from .siglip_vit import create_siglip_vit


class CLIPVisionTower(nn.Module):
    def __init__(
        self,
        model_name: str = "siglip_large_patch16_384",
        image_size: Union[Tuple[int, int], int] = 336,
        select_feature: str = "patch",
        select_layer: int = -2,
        select_layers: list = None,
        ckpt_path: str = "",
        pixel_mean: Optional[List[float]] = None,
        pixel_std: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__()

        self.model_name = model_name
        self.select_feature = select_feature
        self.select_layer = select_layer
        self.select_layers = select_layers

        vision_tower_params = {
            "model_name": model_name,
            "image_size": image_size,
            "ckpt_path": ckpt_path,
            "select_layer": select_layer,
        }
        vision_tower_params.update(kwargs)
        self.vision_tower, self.forward_kwargs = self.build_vision_tower(
            vision_tower_params
        )

        if pixel_mean is not None and pixel_std is not None:
            image_norm = torchvision.transforms.Normalize(
                mean=pixel_mean, std=pixel_std
            )
        else:
            image_norm = None

        self.image_norm = image_norm

    def build_vision_tower(self, vision_tower_params):
        if self.model_name.startswith("siglip"):
            self.select_feature = "same"
            vision_tower = create_siglip_vit(**vision_tower_params)
            forward_kwargs = dict()

        elif self.model_name.startswith("sam"):
            vision_tower = create_sam_vit(**vision_tower_params)
            forward_kwargs = dict()

        else:  # huggingface
            from transformers import CLIPVisionModel

            vision_tower = CLIPVisionModel.from_pretrained(**vision_tower_params)
            forward_kwargs = dict(output_hidden_states=True)

        return vision_tower, forward_kwargs

    def feature_select(self, image_forward_outs):
        if isinstance(image_forward_outs, torch.Tensor):
            # the output has been the self.select_layer"s features
            image_features = image_forward_outs
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == "patch":
            # if the output has cls_token
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        elif self.select_feature == "same":
            image_features = image_features

        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        """

        Args:
            images (torch.Tensor): [b, 3, H, W]

        Returns:
            image_features (torch.Tensor): [b, n_patch, d]
        """

        if self.image_norm is not None:
            images = self.image_norm(images)

        image_forward_outs = self.vision_tower(images, **self.forward_kwargs)
        image_features = self.feature_select(image_forward_outs)
        return image_features


class HybridVisionTower(nn.Module):
    def __init__(
        self,
        high_res_cfg: Dict,
        low_res_cfg: Dict,
        freeze_high: bool = False,
        freeze_low: bool = False,
        concat_type: Literal["feature", "sequence", "add", "tuple"] = "tuple",
        **ignore_kwargs,
    ):
        super().__init__()

        self.vision_tower_high = CLIPVisionTower(**high_res_cfg)
        self.vision_tower_low = CLIPVisionTower(**low_res_cfg)
        self.low_res_size = low_res_cfg["image_size"]
        self.concat_type = concat_type

        self.high_layer_norm = nn.LayerNorm(high_res_cfg.get("output_dim", 1024))
        self.low_layer_norm = nn.LayerNorm(low_res_cfg.get("output_dim", 1024))

        if freeze_high:
            for p_name, p in self.vision_tower_high.named_parameters():
                p.requires_grad = False
            self.vision_tower_high = self.vision_tower_high.eval()
        else:
            # train donwsamples and neck
            for p_name, p in self.vision_tower_high.named_parameters():
                if "downsamples" in p_name or "neck" in p_name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        if freeze_low:
            for p in self.vision_tower_low.parameters():
                p.requires_grad = False
            self.vision_tower_low = self.vision_tower_low.eval()

        self.resize = torchvision.transforms.Resize(self.low_res_size, antialias=True)

    def forward(self, images: torch.Tensor):
        """

        Args:
            images (torch.Tensor): [bs, 3, H, W]

        Returns:
            res (torch.Tensor): [bs, t, c]
        """

        # [bs, c, h, w]
        high_images = images

        # [bs, c, h_low, w_low]
        low_images = self.resize(images)

        # separately run two vision towers
        # run high_res vision tower
        high_res = self.vision_tower_high(high_images)
        # [bs, c, h, w] -> [bs, h*w, c]
        high_res = rearrange(high_res, "b c h w -> b (h w) c")
        # run low_res vision tower
        low_res = self.vision_tower_low(low_images)

        if self.concat_type == "feature":
            images_features = torch.cat([high_res, low_res], dim=-1)
        elif self.concat_type == "sequence":
            images_features = torch.cat([high_res, low_res], dim=1)
        elif self.concat_type == "add":
            images_features = high_res + low_res
        elif self.concat_type == "tuple":
            images_features = (high_res, low_res)

        else:
            raise ValueError(
                "Currently only support `feature`, `sequence`, `add` and `tuple` concat type."
            )

        return images_features


if __name__ == "__main__":
    image_size = 1024
    x = torch.zeros(2, 3, image_size, image_size).bfloat16().cuda()

    high_res_cfg = dict(
        model_name="sam_b_downsample",
        select_feature="same",
        image_size=image_size,
        pixel_mean=(0.48145466, 0.4578275, 0.40821073),
        pixel_std=(0.26862954, 0.26130258, 0.27577711),
        select_layer=-1,
        ckpt_path="",
    )

    low_res_cfg = dict(
        model_name="siglip_large_patch16_384",
        select_feature="same",
        image_size=384,
        pixel_mean=(0.5, 0.5, 0.5),
        pixel_std=(0.5, 0.5, 0.5),
        select_layer=-1,
        ckpt_path="",
    )

    net = (
        HybridVisionTower(
            high_res_cfg=high_res_cfg,
            low_res_cfg=low_res_cfg,
            freeze_high=True,
            freeze_low=True,
            concat_type="tuple",
        )
        .bfloat16()
        .cuda()
    )
    high_x, low_x = net(x)
    print(x.shape, high_x.shape, low_x.shape)
