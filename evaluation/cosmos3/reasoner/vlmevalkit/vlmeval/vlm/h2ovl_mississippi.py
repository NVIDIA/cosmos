import string
import warnings

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string, listinstr
from .base import BaseModel


class H2OVLChat(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='h2oai/h2ovl-mississippi-2b', **kwargs):
        assert model_path is not None

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        device = torch.cuda.current_device()
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True).eval()
        self.model = self.model.to(device)
        self.image_size = self.model.config.vision_config.image_size

        kwargs_default = dict(do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        return True

    def build_multi_choice_prompt(self, line, dataset=None):
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
            prompt += '\n请直接回答选项字母。' if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and listinstr(['MME'], dataset):
            question = line['question']
            prompt = question + ' Answer the question using a single word or phrase.'
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question']
            prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            if 'MathVista' in dataset:
                prompt = line['question']
            elif listinstr(['LLaVABench'], dataset):
                question = line['question']
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['MMVet'], dataset):
                prompt = line['question']
            else:
                question = line['question']
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        image_num = len([x for x in message if x['type'] == 'image'])
        question = ''
        image_files = [x['value'] for x in message if x['type'] == 'image']

        if image_num == 1:
            question = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])

        elif image_num > 1:
            text_part = ' '.join([x['value'] for x in message if x['type'] == 'text'])
            image_part = ' '.join([f'<image-{i + 1}>: <image>' for i in range(image_num)])
            question = image_part + '\n' + text_part

        else:
            question = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image_files = None

        response, history = self.model.chat(
            self.tokenizer,
            image_files=image_files,
            question=question,
            generation_config=self.kwargs,
            max_tiles=6,
            history=None,
            return_history=True)
        return response
