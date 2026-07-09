import os.path as osp
import string
import warnings
from argparse import Namespace

import pandas as pd
import torch
from PIL import Image

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string, splitlen
from .base import BaseModel


class Ross(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='HaochenWang/ross-qwen2-7b',
                 **kwargs):
        from ross.mm_utils import get_model_name_from_path
        from ross.model.builder import load_pretrained_model

        assert osp.exists(model_path) or splitlen(model_path) == 2

        model_name = get_model_name_from_path(model_path)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device='cpu',
            device_map='cpu',
            torch_dtype=torch.float16,
        )
        self.model.get_vision_tower().load_model()
        self.model.eval()
        self.model.cuda()

        if 'Qwen2' in model_path:
            self.conv_mode = 'v1_qwen2'
        elif 'llama3' in model_path.lower():
            self.conv_mode = 'llama3'
        else:
            self.conv_mode = 'llava_v1'

        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
            top_p=None,
            num_beams=1,
            use_cache=True,
        ) # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset: str) -> bool:
        dataset_type = DATASET_TYPE(dataset, default=None)
        if dataset_type == 'MCQ':
            return True
        if dataset_type == 'Y/N' and dataset in {'HallusionBench'}:
            return True
        return False

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        dataset_type = DATASET_TYPE(dataset, default=None)
        if dataset_type == 'MCQ':
            return self._build_mcq_prompt(line, dataset)
        if dataset_type == 'Y/N':
            return self._build_yorn_prompt(line, dataset)
        raise ValueError(f'Unsupported dataset: {dataset}')

    def _build_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        YORN_PROMPT = "\nAnswer the question using a single word or phrase."

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += YORN_PROMPT
        return msgs

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
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

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def generate_inner(self, message, dataset=None):
        from ross.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                    DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
        from ross.conversation import SeparatorStyle, conv_templates
        from ross.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token

        # Support interleave text and image
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], 'PLACEHOLDER')
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        content, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            elif msg['type'] == 'image':
                if self.model.config.mm_use_im_start_end:
                    content += DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
                else:
                    content += DEFAULT_IMAGE_TOKEN + '\n'
                images.append(msg['value'])

        images = [Image.open(s).convert('RGB') for s in images]
        args = Namespace()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images(images, self.image_processor, args).to('cuda', dtype=torch.float16)
        prompt = prompt.replace('PLACEHOLDER', content)

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str] + ["<|im_end|>"] if stop_str == "<|im_start|>" else [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output
