import string

import pandas as pd
import torch
from transformers import AutoModel, GenerationConfig

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string
from .base import BaseModel


class WeMM(BaseModel):
    def __init__(self, model_path='feipengma/WeMM', **kwargs):
        self.wemm = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.wemm.cuda()
        self.wemm.eval()
        torch.cuda.empty_cache()

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
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

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=p) for p in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        if dataset == 'HallusionBench':
            prompt = prompt + ' Please answer yes or no. Answer the question using a single word or phrase.'

        gen_config = None
        if dataset == 'MMVet':
            gen_config = GenerationConfig(
                max_new_tokens=512,
                do_sample=True,
                temperatures=0.7,
                num_beams=3,
                eos_token_id=self.wemm.tokenizer.eos_token_id,
                pad_token_id=self.wemm.tokenizer.pad_token_id
                if self.wemm.tokenizer.pad_token_id is not None else self.wemm.tokenizer.eos_token_id,
            )
        pred = self.wemm.mm_generate(image_path, prompt, gen_config)

        return pred
