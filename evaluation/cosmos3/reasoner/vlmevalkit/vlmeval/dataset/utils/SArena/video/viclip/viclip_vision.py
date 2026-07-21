"""
Copy from https://huggingface.co/OpenGVLab/ViCLIP/blob/main/viclip_vision.py
"""

import logging
import os
from collections import OrderedDict

import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.registry import register_model
from torch import nn

# from models.utils import load_temp_embed_with_mismatch

logger = logging.getLogger(__name__)


def load_temp_embed_with_mismatch(temp_embed_old, temp_embed_new, add_zero=True):
    """
    Add/Remove extra temporal_embeddings as needed.
    https://arxiv.org/abs/2104.00650 shows adding zero paddings works.

    temp_embed_old: (1, num_frames_old, 1, d)
    temp_embed_new: (1, num_frames_new, 1, d)
    add_zero: bool, if True, add zero, else, interpolate trained embeddings.
    """
    # TODO zero pad
    num_frms_new = temp_embed_new.shape[1]
    num_frms_old = temp_embed_old.shape[1]
    logger.info(f"Load temporal_embeddings, lengths: {num_frms_old}-->{num_frms_new}")
    if num_frms_new > num_frms_old:
        temp_embed_new[:, :num_frms_old] = (
            temp_embed_old  # untrained embeddings are zeros.
        )
    elif num_frms_new < num_frms_old:
        temp_embed_new = temp_embed_old[:, :num_frms_new]
    else:  # =
        temp_embed_new = temp_embed_old
    return temp_embed_new


# On P1, model extracted from https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
MODEL_PATH = ""
_MODELS = {
    "ViT-L/14": os.path.join(MODEL_PATH, "ViClip-InternVid-10M-FLT.pth"),
}


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0.0, attn_mask=None, dropout=0.0):
        super().__init__()

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # logger.info(f'Droppath: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("drop1", nn.Dropout(dropout)),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                    ("drop2", nn.Dropout(dropout)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.drop_path1(self.attention(self.ln_1(x)))
        x = x + self.drop_path2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width, layers, heads, drop_path=0.0, checkpoint_num=0, dropout=0.0
    ):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.resblocks = nn.ModuleList()
        for idx in range(layers):
            self.resblocks.append(
                ResidualAttentionBlock(
                    width, heads, drop_path=dpr[idx], dropout=dropout
                )
            )
        self.checkpoint_num = checkpoint_num

    def forward(self, x):
        for idx, blk in enumerate(self.resblocks):
            if idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution,
        patch_size,
        width,
        layers,
        heads,
        output_dim=None,
        kernel_size=1,
        num_frames=8,
        drop_path=0,
        checkpoint_num=0,
        dropout=0.0,
        temp_embed=True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv3d(
            3,
            width,
            (kernel_size, patch_size, patch_size),
            (kernel_size, patch_size, patch_size),
            (0, 0, 0),
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        if temp_embed:
            self.temporal_positional_embedding = nn.Parameter(
                torch.zeros(1, num_frames, width)
            )

        self.transformer = Transformer(
            width,
            layers,
            heads,
            drop_path=drop_path,
            checkpoint_num=checkpoint_num,
            dropout=dropout,
        )

        self.ln_post = nn.LayerNorm(width)
        if output_dim is not None:
            self.proj = nn.Parameter(torch.empty(width, output_dim))
        else:
            self.proj = None

        self.dropout = nn.Dropout(dropout)

    def get_num_layers(self):
        return len(self.transformer.resblocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "positional_embedding",
            "class_embedding",
            "temporal_positional_embedding",
        }

    def mask_tokens(self, inputs, masking_prob=0.0):
        B, L, _ = inputs.shape

        # This is different from text as we are masking a fix number of tokens
        Lm = int(masking_prob * L)
        masked_indices = torch.zeros(B, L)
        indices = torch.argsort(torch.rand_like(masked_indices), dim=-1)[:, :Lm]
        batch_indices = (
            torch.arange(masked_indices.shape[0]).unsqueeze(-1).expand_as(indices)
        )
        masked_indices[batch_indices, indices] = 1

        masked_indices = masked_indices.bool()

        return inputs[~masked_indices].reshape(B, -1, inputs.shape[-1])

    def forward(self, x, masking_prob=0.0):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.positional_embedding.to(x.dtype)

        # temporal pos
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=T)
        if hasattr(self, "temporal_positional_embedding"):
            if x.size(1) == 1:
                # This is a workaround for unused parameter issue
                x = x + self.temporal_positional_embedding.mean(1)
            else:
                x = x + self.temporal_positional_embedding
        x = rearrange(x, "(b n) t m -> b (n t) m", b=B, t=T)

        if masking_prob > 0.0:
            x = self.mask_tokens(x, masking_prob)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # BND -> NBD
        x = self.transformer(x)

        x = self.ln_post(x)

        if self.proj is not None:
            x = self.dropout(x[0]) @ self.proj
        else:
            x = x.permute(1, 0, 2)  # NBD -> BND

        return x


def inflate_weight(weight_2d, time_dim, center=True):
    logger.info(f"Init center: {center}")
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(
    model, state_dict, input_resolution=224, patch_size=16, center=True
):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 2:
                logger.info(f"Ignore: {k}")
                continue
            logger.info(
                f"Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}"
            )
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)

    pos_embed_checkpoint = state_dict["positional_embedding"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = (input_resolution // patch_size) ** 2
    orig_size = int((pos_embed_checkpoint.shape[-2] - 1) ** 0.5)
    new_size = int(num_patches**0.5)
    if orig_size != new_size:
        logger.info(f"Pos_emb from {orig_size} to {new_size}")
        extra_tokens = pos_embed_checkpoint[:1]
        pos_tokens = pos_embed_checkpoint[1:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(0, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
        state_dict["positional_embedding"] = new_pos_embed

    message = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Load pretrained weights: {message}")


@register_model
def clip_joint_b16(
    pretrained=True,
    input_resolution=224,
    kernel_size=1,
    center=True,
    num_frames=8,
    drop_path=0.0,
):
    model = VisionTransformer(
        input_resolution=input_resolution,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        kernel_size=kernel_size,
        num_frames=num_frames,
        drop_path=drop_path,
    )
    raise NotImplementedError
    if pretrained:
        logger.info("load pretrained weights")
        state_dict = torch.load(_MODELS["ViT-B/16"], map_location="cpu", weights_only=True)
        load_state_dict(
            model,
            state_dict,
            input_resolution=input_resolution,
            patch_size=16,
            center=center,
        )
    return model.eval()


@register_model
def clip_joint_l14(
    pretrained=False,
    input_resolution=224,
    kernel_size=1,
    center=True,
    num_frames=8,
    drop_path=0.0,
    checkpoint_num=0,
    dropout=0.0,
):
    model = VisionTransformer(
        input_resolution=input_resolution,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768,
        kernel_size=kernel_size,
        num_frames=num_frames,
        drop_path=drop_path,
        checkpoint_num=checkpoint_num,
        dropout=dropout,
    )

    if pretrained:
        if isinstance(pretrained, str):
            model_name = pretrained
        else:
            model_name = "ViT-L/14"
        logger.info("load pretrained weights")
        state_dict = torch.load(_MODELS[model_name], map_location="cpu", weights_only=True)
        load_state_dict(
            model,
            state_dict,
            input_resolution=input_resolution,
            patch_size=14,
            center=center,
        )
    return model.eval()


@register_model
def clip_joint_l14_336(
    pretrained=True,
    input_resolution=336,
    kernel_size=1,
    center=True,
    num_frames=8,
    drop_path=0.0,
):
    raise NotImplementedError
    model = VisionTransformer(
        input_resolution=input_resolution,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768,
        kernel_size=kernel_size,
        num_frames=num_frames,
        drop_path=drop_path,
    )
    if pretrained:
        logger.info("load pretrained weights")
        state_dict = torch.load(_MODELS["ViT-L/14_336"], map_location="cpu", weights_only=True)
        load_state_dict(
            model,
            state_dict,
            input_resolution=input_resolution,
            patch_size=14,
            center=center,
        )
    return model.eval()


def interpolate_pos_embed_vit(state_dict, new_model):
    key = "vision_encoder.temporal_positional_embedding"
    if key in state_dict:
        vision_temp_embed_new = new_model.state_dict()[key]
        vision_temp_embed_new = vision_temp_embed_new.unsqueeze(
            2
        )  # [1, n, d] -> [1, n, 1, d]
        vision_temp_embed_old = state_dict[key]
        vision_temp_embed_old = vision_temp_embed_old.unsqueeze(2)

        state_dict[key] = load_temp_embed_with_mismatch(
            vision_temp_embed_old, vision_temp_embed_new, add_zero=False
        ).squeeze(2)

    key = "text_encoder.positional_embedding"
    if key in state_dict:
        text_temp_embed_new = new_model.state_dict()[key]
        text_temp_embed_new = text_temp_embed_new.unsqueeze(0).unsqueeze(
            2
        )  # [n, d] -> [1, n, 1, d]
        text_temp_embed_old = state_dict[key]
        text_temp_embed_old = text_temp_embed_old.unsqueeze(0).unsqueeze(2)

        state_dict[key] = (
            load_temp_embed_with_mismatch(
                text_temp_embed_old, text_temp_embed_new, add_zero=False
            )
            .squeeze(2)
            .squeeze(0)
        )
    return state_dict


if __name__ == "__main__":
    import time

    import numpy as np
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 8

    # model = clip_joint_b16(pretrained=True, kernel_size=1, num_frames=8, num_classes=400, drop_path=0.1)
    # logger.info(model)
    model = clip_joint_l14(pretrained=False)

    flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, 224, 224))
    s = time.time()
    logger.info(flop_count_table(flops, max_depth=1))
    logger.info(time.time() - s)
    # logger.info(model(torch.rand(1, 3, num_frames, 224, 224)).shape)
