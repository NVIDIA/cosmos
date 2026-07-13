import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from transformers import WhisperFeatureExtractor

from vlmeval.vlm.ola.ola.model.speech_encoder.beats.BEATs import BEATs, BEATsConfig


class WhisperWrappedEncoder:

    @classmethod
    def load(cls, model_config):

        def replace_layer_norm(module):
            from whisper.model import LayerNorm
            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)

        encoder = whisper.load_model(name=model_config.speech_encoder, device='cpu').encoder
        replace_layer_norm(encoder)
        return encoder

class DualWrappedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.whisper_model = self.load_whisper(config)
        self.beats_model = self.load_beats(config)

    def load_whisper(cls, model_config):

        def replace_layer_norm(module):
            from whisper.model import LayerNorm
            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)

        encoder = whisper.load_model(name=model_config.speech_encoder, device='cpu').encoder
        replace_layer_norm(encoder)
        return encoder

    def load_beats(cls, model_config):
        beats_path = model_config.music_encoder
        print("Loading BEATs Model")
        beats_ckpt = torch.load(beats_path, map_location='cpu')
        beats_cfg = BEATsConfig(beats_ckpt['cfg'])
        beats = BEATs(beats_cfg)
        beats.load_state_dict(beats_ckpt['model'])
        return beats

    def forward(self, x, raw_wav=None, audio_padding_mask=None):
        with torch.no_grad():
            self.beats_model = self.beats_model.float()
            speech_embeds = self.whisper_model(x.half())
            audio_embeds, _ = self.beats_model.extract_features(raw_wav.float(), padding_mask=audio_padding_mask, feature_only=True)
        if audio_embeds.size(1) < speech_embeds.size(1):
            audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
        elif audio_embeds.size(1) > speech_embeds.size(1):
            speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
        speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
        speech_embeds = speech_embeds.to(torch.bfloat16)
        return speech_embeds
