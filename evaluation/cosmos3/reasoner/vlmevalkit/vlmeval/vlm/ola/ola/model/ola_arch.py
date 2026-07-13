from abc import ABC, abstractmethod

import torch

from ..constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN,
                         IGNORE_INDEX, IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX)
from ..utils import lengths_to_padding_mask
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .multimodal_resampler.builder import build_vision_resampler
from .speech_encoder.builder import build_speech_encoder
from .speech_projector.builder import build_speech_projector


class OlaMetaModel:

    def __init__(self, config):
        super(OlaMetaModel, self).__init__(config)

        if hasattr(config, "speech_encoder"):
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

    def get_speech_encoder(self):
        speech_encoder = getattr(self, 'speech_encoder', None)
        if type(speech_encoder) is list:
            speech_encoder = speech_encoder[0]
        return speech_encoder

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_speech_modules(self, model_args, fsdp=None):
        self.config.speech_encoder = getattr(model_args, "speech_encoder", None)
        self.config.speech_encoder_type = getattr(model_args, "speech_encoder_type", None)
        self.config.speech_projector_type = getattr(model_args, 'speech_projector_type', 'linear')
        self.config.speech_encoder_ds_rate = getattr(model_args, 'speech_encoder_ds_rate', 5)
        self.config.speech_encoder_hidden_size = getattr(model_args, 'speech_encoder_hidden_size', 1280)

        if self.get_speech_encoder() is None:
            speech_encoder = build_speech_encoder(self.config)
            if fsdp is not None and len(fsdp) > 0:
                self.speech_encoder = [speech_encoder]
            else:
                self.speech_encoder = speech_encoder

        if getattr(self, 'speech_projector', None) is None:
            self.speech_projector = build_speech_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.speech_projector.parameters():
                p.requires_grad = True

        if model_args.pretrain_speech_projector is not None:
            pretrain_speech_projector_weights = torch.load(model_args.pretrain_speech_projector, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            print('Loading pretrain speech projector weights')

            msg = self.speech_projector.load_state_dict(get_w(pretrain_speech_projector_weights, 'speech_projector'), strict=False)
            print(msg)

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            ## Get the mm_spatial_pool_mode and  mm_spatial_pool_stride
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = getattr(vision_resampler, 'hidden_size', vision_tower.hidden_size)

        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print('Loading pretrain mm projector weights')
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, 'vision_resampler'), strict=False)
            print(incompatible_keys)

class OlaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_speech_encoder(self):
        return self.get_model().get_speech_encoder()

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_speech_projector(self):
        return self.get_model().speech_projector

    def encode_speech(self, speech, speech_lengths, speech_wav):
        # import pdb; pdb.set_trace()
        speech_encoder_type = self.config.speech_encoder_type
        speech_encoder = self.get_speech_encoder()
        if "whisper" in speech_encoder_type.lower():
            encoder_outs = speech_encoder(speech.permute(0, 2, 1))
            speech_lengths = (speech_lengths + 1) // 2
        else:
            encoder_outs = speech_encoder(speech.permute(0, 2, 1), raw_wav=speech_wav)
            speech_lengths = (speech_lengths + 1) // 2
        speech_projector_type = self.config.speech_projector_type
        speech_projector = self.get_speech_projector()
        if speech_projector_type == "linear":
            encoder_outs = speech_projector(encoder_outs)
            speech_lengths = speech_lengths // speech_projector.k
        else:
            raise ValueError(f'Unknown speech projector: {speech_projector_type}')
        # speech_features = [encoder_outs[i, :speech_lengths[i]] for i in range(len(encoder_outs))]
        return encoder_outs

    def prepare_inputs_labels_for_speech_vision_text(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        speech, speech_lengths, speech_chunks, speech_wav, images, modalities, image_sizes=None, images_highres=None
    ):
        speech_encoder = self.get_speech_encoder()
        vision_tower = self.get_vision_tower()

        if speech_encoder is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if vision_tower is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        # encode speech
        if not isinstance(speech, list):
            speech = torch.split(speech, speech_chunks.tolist(), dim=0)
            speech_lengths = torch.split(speech_lengths, speech_chunks.tolist(), dim=0)
            speech_wav = torch.split(speech_wav, speech_chunks.tolist(), dim=0)
        speech_features = []
        for idx in range(len(speech)):
            speech_features.append(self.encode_speech(speech[idx], speech_lengths[idx], speech_wav[idx]))

        # encode vision
        if isinstance(modalities, str):
            modalities = [modalities]

        video_idx_in_batch = []
        for modal in range(len(modalities)):
            if 'video' in modalities[modal]:
                video_idx_in_batch.append(modal)

        # Fix training with deepspeed zero3
        num_modality = len(modalities)
        # try:
        #     world_size = dist.get_world_size()
        #     tensor_in = torch.zeros(1,  dtype=torch.int64, device=images[0].device).fill_(num_modality)
        #     tensor_out = torch.zeros(world_size, dtype=torch.int64, device=images[0].device)
        #     dist.all_gather_into_tensor(tensor_out, tensor_in)
        #     max_num_modality = tensor_out.max().item()
        # except:
        # max_num_modality = num_modality
        aimg = images[-1]
        lowres_img = []
        for idx, img_feat in enumerate(images):
            if idx in video_idx_in_batch:
                img_feat = aimg.new(1, 3, 128, 128).fill_(0)
            lowres_img.append(img_feat)

        # Fix training with deepspeed zero3
        # if max_num_modality > num_modality:
        #     for _ in range(max_num_modality - num_modality):
        #         lowres_img.append(aimg.new(1, 3, 64, 64).fill_(0))
        #         images_highres.append(aimg.new(1, 3, 64, 64).fill_(0))
        #         modalities.append('image')
        lowres_img_features, lowres_img_sizes = self.get_model().get_vision_tower()(lowres_img)
        highres_img_features = []
        highres_img_sizes = []
        for idx, img_feat in enumerate(images_highres):
            if img_feat.ndim == 5:
                img_feat = img_feat.squeeze(1)
            highres_img_feature, highres_img_size = self.get_model().get_vision_tower()(img_feat)
            highres_img_features.append(highres_img_feature)
            highres_img_sizes.append(highres_img_size)
        image_features = []
        for idx in range(len(modalities)):
            img_feat = self.get_model().mm_projector(lowres_img_features[idx],
                                                    lowres_img_sizes[idx],
                                                    highres_img_features[idx],
                                                    highres_img_sizes[idx],
                                                    modalities[idx])
            image_features.append(img_feat.flatten(0, 1))

        # if max_num_modality > num_modality:
        #     image_features = image_features[:num_modality]
        #     modalities = modalities[:num_modality]

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_speech_idx = 0
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):

            num_speech = (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            num_speech_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() + (cur_input_ids == SPEECH_TOKEN_INDEX).sum()

            if num_speech_images == 0:
                cur_speech_features = speech_features[cur_speech_idx]
                cur_images_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_speech_features[0:0], cur_images_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_speech_idx += 1
                cur_image_idx += 1
                continue

            speech_image_token_indices = [-1] + torch.where((cur_input_ids == SPEECH_TOKEN_INDEX) | (cur_input_ids == IMAGE_TOKEN_INDEX))[0].tolist() + [cur_input_ids.shape[0]]

            cur_input_ids_nospeech_image = []
            cur_labels = labels[batch_idx]
            cur_labels_nospeech_image = []
            for i in range(len(speech_image_token_indices) - 1):
                cur_input_ids_nospeech_image.append(cur_input_ids[speech_image_token_indices[i]+1:speech_image_token_indices[i+1]])
                cur_labels_nospeech_image.append(cur_labels[speech_image_token_indices[i]+1:speech_image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_nospeech_image]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nospeech_image))
            cur_input_embeds_no_speech_image = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_speech_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_speech_image[i])
                cur_new_labels.append(cur_labels_nospeech_image[i])
                if i < num_speech_images:
                    if i < num_images:
                        cur_images_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_images_features)
                        cur_new_labels.append(torch.full((cur_images_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    else:
                        cur_speech_features = speech_features[cur_speech_idx]
                        cur_speech_idx += 1
                        cur_new_input_embeds.append(cur_speech_features)
                        cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            if num_images == 0:
                cur_new_input_embeds = torch.cat([cur_new_input_embeds, image_features[cur_image_idx][0:0]], dim=0)
                cur_image_idx += 1

            if num_speech == 0:
                cur_new_input_embeds = torch.cat([cur_new_input_embeds, speech_features[cur_speech_idx][0:0]], dim=0)
                cur_speech_idx += 1

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as speech features can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
