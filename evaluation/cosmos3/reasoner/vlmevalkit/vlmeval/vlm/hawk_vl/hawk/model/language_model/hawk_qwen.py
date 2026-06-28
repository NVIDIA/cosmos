from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM,
                          Qwen2Model)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from ..hawk_arch import HawkMetaForCausalLM, HawkMetaModel
from ..vision_encoder.qwen_vit import QwenVisionConfig

logger = logging.get_logger(__name__)


class HawkQwenConfig(Qwen2Config):
    model_type = "hawk_vl"
    is_composition = True
    sub_configs = {"vision_config": QwenVisionConfig}

    def __init__(self, vision_config=None, **kwargs):
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict['vision_config'] = self.vision_config.to_dict()
        config_dict['model_type'] = self.__class__.model_type
        return config_dict


class HawkQwenModel(HawkMetaModel, Qwen2Model):
    config_class = HawkQwenConfig

    def __init__(self, config: Qwen2Config):
        super(HawkQwenModel, self).__init__(config)


class HawkQwenForCausalLM(Qwen2ForCausalLM, HawkMetaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = HawkQwenConfig
    _no_split_modules = ['Qwen2DecoderLayer']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)

        self.model = HawkQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.image_token_id = getattr(config, 'image_token_id', None)
        self.video_token_id = getattr(config, 'video_token_id', None)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (input_ids,
             position_ids,
             attention_mask,
             past_key_values,
             inputs_embeds,
             labels) = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, position_ids,
                                                                 past_key_values, labels,
                                                                 pixel_values, pixel_values_videos,
                                                                 image_grid_thw, video_grid_thw)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if pixel_values is not None or pixel_values_videos is not None:
            (input_ids, position_ids,
             attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask, position_ids,
                None, None,
                pixel_values, pixel_values_videos,
                image_grid_thw, video_grid_thw)
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        return super().generate(position_ids=position_ids,
                                attention_mask=attention_mask,
                                inputs_embeds=inputs_embeds,
                                **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):

        pixel_values = kwargs.pop("pixel_values", None)
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        inputs = super().prepare_inputs_for_generation(input_ids,
                                                       past_key_values=past_key_values,
                                                       inputs_embeds=inputs_embeds,
                                                       **kwargs)
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values
        if pixel_values_videos is not None:
            inputs["pixel_values_videos"] = pixel_values_videos
        if image_grid_thw is not None:
            inputs["image_grid_thw"] = image_grid_thw
        if video_grid_thw is not None:
            inputs["video_grid_thw"] = video_grid_thw
        return inputs


AutoConfig.register("hawk_vl", HawkQwenConfig)
AutoModelForCausalLM.register(HawkQwenConfig, HawkQwenForCausalLM)
