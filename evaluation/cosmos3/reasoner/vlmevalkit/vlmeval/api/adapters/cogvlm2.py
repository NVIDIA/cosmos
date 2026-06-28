import string

import pandas as pd

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string
from .base import ModelAdapter, register_adapter


@register_adapter('cogvlm2')
class CogVLM2Adapter(ModelAdapter):

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset, system_prompt=None):
        assert dataset is not None
        return DATASET_TYPE(dataset) in 'MCQ'

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            question = line['question']
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            if hint is not None:
                question = hint + '\n' + question

            option_candidate = string.ascii_uppercase
            options = {
                cand: line[cand]
                for cand in option_candidate
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                question += f'\n{key}. {item}'
            prompt = question

            if not cn_string(prompt):
                prompt = prompt + '\n' + "Answer with the option's letter from the given choices directly."
            else:
                prompt = prompt + '\n' + '请直接回答选项字母。'
        else:
            prompt = line['question']

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=p) for p in tgt_path])
        return message
