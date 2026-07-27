import logging
import os.path as osp
import re
import string
import warnings

import pandas as pd
import torch
from PIL import Image

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import listinstr, splitlen
from .base import BaseModel


class Kosmos2(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='microsoft/kosmos-2-patch14-224',
                 **kwargs):
        try:
            from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
        except Exception as e:
            logging.critical("Please install Transformers version 4.45.1 by running: pip install transformers==4.45.1")
            raise e

        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.model = (
            Kosmos2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
            .to(torch.device('cuda'))
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        default_kwargs = dict(
            max_new_tokens=512,
            use_cache=True
        )

        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        TASK_TOKEN = '<grounding> '
        QEUSTION_TOKEN = 'Question: '
        ANSWER_TOKEN = 'Answer: '
        images = []
        prompt = ''

        prompt += TASK_TOKEN
        for s in message:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                prompt += QEUSTION_TOKEN
                prompt += s['value']
                prompt += ANSWER_TOKEN

        images = [Image.open(s) for s in images]
        inputs = self.processor(text=prompt, images=images[0], return_tensors='pt').to(torch.device('cuda'))

        generated_ids = self.model.generate(
            pixel_values=inputs['pixel_values'],
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            image_embeds=None,
            image_embeds_position_mask=inputs['image_embeds_position_mask'],
            **self.kwargs
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        processed_text = self.processor.post_process_generation(generated_text, cleanup_and_extract=True)[0]
        cleaned_answer = re.sub(r'(Question:.*?Answer:|Question:.*)', '', processed_text).strip()
        return cleaned_answer

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMMU'], dataset):
            return False
        if DATASET_TYPE(dataset) == 'MCQ' or dataset == 'MMVet':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        if dataset == 'MMVet':
            prompt = question + '\nAnswer the question directly. '
        elif DATASET_TYPE(dataset) == 'MCQ':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = f'Hint: {hint}\n' if hint is not None else ''
            prompt += f'{question}\n'
            prompt += (
                f'{options_prompt}\nAnswer with the option’s letter from the given choices directly. '
                if len(options) else 'Answer the question directly. '
            )
        else:
            raise NotImplementedError

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message
