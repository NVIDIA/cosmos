"""
Copy from https://huggingface.co/OpenGVLab/ViCLIP/blob/main/viclip.py
"""

import logging
import os

import torch
from torch import nn

# from .criterions import VTC_VTM_Loss
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .viclip_text import clip_text_l14
from .viclip_vision import clip_joint_l14

logger = logging.getLogger(__name__)


class ViCLIP(nn.Module):
    """docstring for ViCLIP"""

    def __init__(
        self,
        tokenizer=None,
        pretrain=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"
        ),
        freeze_text=True,
    ):
        super(ViCLIP, self).__init__()
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = _Tokenizer()
        self.max_txt_l = 32

        self.vision_encoder_name = "vit_l14"

        self.vision_encoder_pretrained = False
        self.inputs_image_res = 224
        self.vision_encoder_kernel_size = 1
        self.vision_encoder_center = True
        self.video_input_num_frames = 8
        self.vision_encoder_drop_path_rate = 0.1
        self.vision_encoder_checkpoint_num = 24
        self.is_pretrain = pretrain
        self.vision_width = 1024
        self.text_width = 768
        self.embed_dim = 768
        self.masking_prob = 0.9

        self.text_encoder_name = "vit_l14"
        self.text_encoder_pretrained = False
        self.text_encoder_d_model = 768

        self.text_encoder_vocab_size = 49408

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.text_encoder = self.build_text_encoder()

        self.temp = nn.parameter.Parameter(torch.ones([]) * 1 / 100.0)
        self.temp_min = 1 / 100.0

        if pretrain:
            logger.info(f"Load pretrained weights from {pretrain}")
            state_dict = torch.load(pretrain, map_location="cpu", weights_only=True)["model"]
            self.load_state_dict(state_dict)

        # Freeze weights
        if freeze_text:
            self.freeze_text()

    def freeze_text(self):
        """freeze text encoder"""
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        ret.update({"text_encoder." + k for k in self.text_encoder.no_weight_decay()})

        return ret

    def forward(
        self, image, text, raw_text, idx, log_generation=None, return_sims=False
    ):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict): TODO
            idx (torch.Tensor): TODO

        Returns: TODO

        """
        self.clip_contrastive_temperature()

        vision_embeds = self.encode_vision(image)
        text_embeds = self.encode_text(raw_text)
        if return_sims:
            sims = torch.nn.functional.normalize(
                vision_embeds, dim=-1
            ) @ torch.nn.functional.normalize(text_embeds, dim=-1).transpose(0, 1)
            return sims

        loss_vtc = self.clip_loss.vtc_loss(
            vision_embeds, text_embeds, idx, self.temp, all_gather=True
        )

        return dict(
            loss_vtc=loss_vtc,
        )

    def encode_vision(self, image, test=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,T,L,C].
            - pooled_vision_embeds (torch.Tensor): The pooled features. Shape: [B,T,C].

        """
        if image.ndim == 5:
            image = image.permute(0, 2, 1, 3, 4).contiguous()
        else:
            image = image.unsqueeze(2)

        if not test and self.masking_prob > 0.0:
            return self.vision_encoder(image, masking_prob=self.masking_prob)

        return self.vision_encoder(image)

    def encode_text(self, text):
        device = next(self.text_encoder.parameters()).device
        text = self.text_encoder.tokenize(text, context_length=self.max_txt_l).to(
            device
        )
        text_embeds = self.text_encoder(text)
        return text_embeds

    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training"""
        self.temp.clamp_(min=self.temp_min)

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        encoder_name = self.vision_encoder_name
        if encoder_name != "vit_l14":
            raise ValueError(f"Not implemented: {encoder_name}")
        vision_encoder = clip_joint_l14(
            pretrained=self.vision_encoder_pretrained,
            input_resolution=self.inputs_image_res,
            kernel_size=self.vision_encoder_kernel_size,
            center=self.vision_encoder_center,
            num_frames=self.video_input_num_frames,
            drop_path=self.vision_encoder_drop_path_rate,
            checkpoint_num=self.vision_encoder_checkpoint_num,
        )
        return vision_encoder

    def build_text_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        encoder_name = self.text_encoder_name
        if encoder_name != "vit_l14":
            raise ValueError(f"Not implemented: {encoder_name}")
        text_encoder = clip_text_l14(
            pretrained=self.text_encoder_pretrained,
            embed_dim=self.text_encoder_d_model,
            context_length=self.max_txt_l,
            vocab_size=self.text_encoder_vocab_size,
            checkpoint_num=0,
        )

        return text_encoder

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder

    def get_text_features(self, input_text, tokenizer, text_feature_dict={}):
        if input_text in text_feature_dict:
            return text_feature_dict[input_text]
        text_template = f"{input_text}"
        with torch.no_grad():
            # text_token = tokenizer.encode(text_template).cuda()
            text_features = self.encode_text(text_template).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_feature_dict[input_text] = text_features
        return text_features

    def get_vid_features(self, input_frames):
        with torch.no_grad():
            clip_feat = self.encode_vision(input_frames, test=True).float()
            clip_feat /= clip_feat.norm(dim=-1, keepdim=True)
        return clip_feat

    def get_predict_label(self, clip_feature, text_feats_tensor, top=5):
        label_probs = (100.0 * clip_feature @ text_feats_tensor.T).softmax(dim=-1)
        top_probs, top_labels = label_probs.cpu().topk(top, dim=-1)
        return top_probs, top_labels


if __name__ == "__main__":
    tokenizer = _Tokenizer()
