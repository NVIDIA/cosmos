import os.path as osp
import string
import sys
import warnings
from abc import abstractproperty

import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string, splitlen
from .base import BaseModel


class TransCoreM(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def load_pretrained_model(self, model_path, load_8bit=False, load_4bit=False, revision='main'):
        import transcorem.config_param as config_param
        from transcorem.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                          DEFAULT_IMAGE_PATCH_TOKEN)
        from transcorem.model import TransCoreMQWenForCausalLM
        kwargs = {'revision': revision}
        if load_8bit:
            kwargs['load_in_8bit'] = True
        elif load_4bit:
            kwargs['load_in_4bit'] = True
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        else:
            kwargs['torch_dtype'] = torch.float16

        config_param.model_path = model_path
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, revision=revision, trust_remote_code=True)
        model = TransCoreMQWenForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)

        image_processor = None
        mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
        mm_use_im_patch_token = getattr(model.config, 'mm_use_im_patch_token', True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device='cuda', dtype=torch.float16)
        image_processor = vision_tower.image_processor

        if hasattr(model.config, 'max_sequence_length'):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        return tokenizer, model, image_processor, context_len

    def __init__(self,
                 root=None,
                 revision='main',
                 **kwargs):

        self.root = root
        self.revision = revision
        sys.path.append(root)

        model_path = 'PCIResearch/TransCore-M'
        assert osp.exists(model_path) or splitlen(model_path) == 2
        self.tokenizer, self.model, self.image_processor, self.context_len = self.load_pretrained_model(
            model_path=model_path, revision=revision)
        self.model = self.model.cuda()
        print('==============conv_mode: transcorem_v1')
        self.conv_mode = 'transcorem_v1'

        kwargs_default = dict(do_sample=False, temperature=0.0, max_new_tokens=512, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=f) for f in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        from transcorem.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                          DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
        from transcorem.conversation import SeparatorStyle, conv_templates
        from transcorem.mm_utils import (KeywordsStoppingCriteria, highres_process_images,
                                         tokenizer_image_token)

        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_patches = highres_process_images(image, self.image_processor, args, base_reso=336)
        image_patches = [patch.unsqueeze(0).to('cuda', dtype=torch.float16) for patch in image_patches]
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt_conv = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_conv, self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_patches,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                **self.kwargs)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs
