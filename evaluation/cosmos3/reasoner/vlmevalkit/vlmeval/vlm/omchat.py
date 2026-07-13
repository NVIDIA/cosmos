import re
import string

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string, listinstr
from .base import BaseModel


class OmChat(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='omlab/omchat-v2.0-13B-single-beta_hf', **kwargs):

        # Recommend to install `transformers==4.44.0`
        assert model_path is not None
        self.model_path = model_path
        print(f'load from {self.model_path}')
        model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, torch_dtype=torch.float16)
        self.model = model.cuda().eval()
        self.kwargs = kwargs
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()

        # system prompt
        self.default_system_prompt = 'You are a helpful assistant. Focus on accuracy and reliability in your response.'
        self.new1_system_prompt = 'You are a helpful assistant.'
        self.new2_system_prompt = (
            'Read the following question carefully, '
            'solve it step by step, '
            'and then output the final answer in the format of '
            "'Answer: single number or single word or phrase'.\n\n"
        )

        # suffix_prompt for MCQ
        self.mcq_suffix_prompt_en = 'Please select the correct answer from the options above. \n'
        self.mcq_suffix_prompt_cn = '请直接回答选项字母。\n'
        # suffix_prompt for Y/N
        self.yorn_suffix_prompt = ' Please answer yes or no. Answer the question using a single word or phrase.'

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ' or DATASET_TYPE(dataset) == 'Y/N':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']

        if DATASET_TYPE(dataset) == 'MCQ':
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            prompt = ''
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            prompt += f'Question: {question}\n'
            if len(options):
                prompt += options_prompt
                if not dataset.startswith('MMMU_'):
                    if not cn_string(prompt):
                        prompt += self.mcq_suffix_prompt_en
                    else:
                        prompt += self.mcq_suffix_prompt_cn

        elif DATASET_TYPE(dataset) == 'Y/N':
            prompt = question + self.yorn_suffix_prompt

        print(DATASET_TYPE(dataset))
        message = []
        if isinstance(tgt_path, list):
            message.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            message = [dict(type='image', value=tgt_path)]
        message.append(dict(type='text', value=prompt))

        return message

    def message_to_promptimg(self, message, dataset=None):
        if dataset is None or listinstr(['MMMU'], dataset):
            prompt = '\n'.join([
                re.sub(r'<image\s*\d+>', '<image>', x['value'])
                for x in message
                if x['type'] == 'text'
            ])
            image = [x['value'] for x in message if x['type'] == 'image']
        else:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = [x['value'] for x in message if x['type'] == 'image']
        return prompt, image

    def generate_inner(self, message, dataset=None):

        def replace_last_dot(input_string):
            if input_string.endswith('.'):
                return input_string[:-1]
            else:
                return input_string

        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = [Image.open(img_path).convert('RGB') for img_path in image_path]

        default_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
            top_p=1)

        if dataset is not None and listinstr(['MathVista_MINI'], dataset):
            system_prompt = self.new2_system_prompt
        elif dataset is not None and listinstr(['MMMU_DEV_VAL', 'MMStar'], dataset):
            system_prompt = self.new1_system_prompt
        else:
            system_prompt = self.default_system_prompt
        inputs = self.processor(text=prompt, system_prompt=system_prompt, images=image, return_tensors='pt').to('cuda')
        default_kwargs.update(self.kwargs)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                eos_token_id=self.model.generation_config.eos_token_id,
                **default_kwargs
            )
        res = self.processor.tokenizer.decode(output_ids[0, inputs.input_ids.shape[1]:]).strip()
        if '<|im_end|>' in res:
            res = res.split('<|im_end|>')[0].strip()

        if dataset != 'MMMU_DEV_VAL':
            if res.startswith('Answer: '):
                res = res[len('Answer: '):]

            match = re.search(r'\nThe answer is:(.+)', res)
            if match:
                res = match.group(1).strip()

        # for OCRBench
        doc_match = re.search(r'<doc>(.*?)<\/doc>', res)
        if doc_match:
            res = doc_match.group(1).strip()
        res = replace_last_dot(res)

        return res
