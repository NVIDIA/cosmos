#    Adapted from LLaVA code.
#    Copyright 2025 Shukang Yin
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from ..constants import IGNORE_INDEX
from .multimodal_projector.builder import build_vision_projector
from .vision_encoder import VISION_TRANSFORMER_CLASSES


class HawkMetaModel:

    def __init__(self, config):
        super(HawkMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            vision_config = config.vision_config
            vision_config._attn_implementation = config._attn_implementation
            self.vision_tower = VISION_TRANSFORMER_CLASSES[vision_config.model_type]._from_config(
                vision_config
            )

            self.mm_projector = build_vision_projector(vision_config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower


class HawkMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None
    ):

        vision_tower = self.get_vision_tower()

        if vision_tower is None or (pixel_values is None and pixel_values_videos is None) or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        inputs_embeds = self.get_input_embeddings()(input_ids).clone()

        ignore_flag = False

        if pixel_values is not None:
            image_embeds = vision_tower(
                pixel_values,
                grid_thw=image_grid_thw)

            image_embeds = self.get_model().mm_projector(image_embeds)

            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]

            if n_image_tokens != n_image_features:
                print(f'warning:  n_image_tokens={n_image_tokens}, while'
                      f'n_image_features={n_image_features}. '
                      f'will ignore this batch')
                image_embeds = image_embeds[:n_image_tokens]
                ignore_flag = True

            image_mask = (
                (input_ids == self.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = vision_tower(
                pixel_values_videos,
                grid_thw=video_grid_thw)

            video_embeds = self.get_model().mm_projector(video_embeds)

            n_video_tokens = (input_ids == self.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                print(f'warning:  n_video_tokens={n_video_tokens}, while'
                      f'n_video_features={n_video_features}. '
                      f'will ignore this batch')
                video_embeds = video_embeds[:n_video_tokens]
                ignore_flag = True

            video_mask = (
                (input_ids == self.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if labels is not None and ignore_flag:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels
