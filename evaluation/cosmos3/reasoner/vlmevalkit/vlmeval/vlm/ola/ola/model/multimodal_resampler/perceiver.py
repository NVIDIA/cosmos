import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicCompressor(nn.Module):
    def __init__(self, model_args, vision_tower):
        super().__init__()

        self.out_channels = vision_tower.hidden_size
        self.mid_channel = 256

        self.vlm_query_projector  = nn.Linear(self.out_channels, self.mid_channel)
        self.vlm_key_projector  = nn.Linear(self.out_channels, self.mid_channel)

    def downsample(self, x):
        return F.avg_pool2d(x, 2, 2)

    def downsample_4(self, x):
        return F.avg_pool2d(x, 4, 4)

    def forward(self, image_features, forward_type, image_size=None):
        if image_size is None:
            ori_W = int(math.sqrt(image_features.shape[1]))
            ori_H = int(ori_W)
        else:
            ori_H, ori_W = image_size
        T, N, C = image_features.shape
        image_features = image_features.view(T, ori_H, ori_W, C).permute(0, 3, 1, 2)  # T, C, H, W

        if forward_type == 'video':
            image_features_pool = self.downsample(image_features)
            image_feature_attn = image_features.reshape(T, C, ori_H // 2, 2, ori_W // 2, 2).permute(0, 2, 4, 3, 5, 1).reshape(T, ori_H // 2 * ori_W // 2, 4, C)
            new_image_size = (ori_H // 2, ori_W // 2)
        elif forward_type == 'image' or forward_type == 'text':
            image_features_pool = image_features
            image_feature_attn = image_features.reshape(T, C, ori_H, 1, ori_W, 1).permute(0, 2, 4, 3, 5, 1).reshape(T, ori_H * ori_W, 1, C)
            new_image_size = (ori_H, ori_W)
        elif forward_type == 'video_long':
            image_features_pool = self.downsample_4(image_features)
            image_feature_attn = image_features.reshape(T, C, ori_H // 4, 4, ori_W // 4, 4).permute(0, 2, 4, 3, 5, 1).reshape(T, ori_H // 4 * ori_W // 4, 16, C)
            new_image_size = (ori_H // 4, ori_W // 4)
        else:
            raise NotImplementedError

        image_features_pool = image_features_pool.flatten(2).permute(0, 2, 1) # T, H*W, C
        new_t, new_p, _ = image_features_pool.shape

        image_query = self.vlm_query_projector(image_features_pool).reshape(new_t*new_p, self.mid_channel)
        image_key = self.vlm_key_projector(image_feature_attn).reshape(new_t*new_p, -1, self.mid_channel)

        image_value = image_feature_attn.reshape(new_t*new_p, -1, self.out_channels)
        # import pdb;pdb.set_trace()

        image_attn = image_query[:,None] @ (image_key.transpose(-1,-2) / (image_key.shape[-1]**0.5))
        image_attn = image_attn.nan_to_num()
        attn_feat = (image_attn.softmax(-1) @ image_value).mean(1).reshape(new_t, new_p, C)

        image_features_pool = image_features_pool + attn_feat

        return image_features_pool, new_image_size

    @property
    def config(self):
        return {
            'mm_resampler_type': 'dynamic_compressor',
            'mm_out_channels': self.out_channels,
        }

    @property
    def hidden_size(self):
        return self.out_channels
