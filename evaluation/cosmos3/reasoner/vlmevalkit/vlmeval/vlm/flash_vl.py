import string

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor

from ..dataset import DATASET_TYPE
from ..smp import cn_string, listinstr
from .base import BaseModel


class FlashVL(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path, **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.model = AutoModel.from_pretrained(model_path,
                                               torch_dtype=torch.bfloat16,
                                               trust_remote_code=True,
                                               device_map='cuda')
        self.model.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                             device_map='cuda')
        self.model.im_trans = CLIPImageProcessor.from_pretrained(
            model_path, trust_remote_code=True)
        self.INTERLEAVE = False

    def build_history(self, message):

        def concat_tilist(tilist):
            image_cnt = 1
            prompt = ''
            for item in tilist:
                if item['type'] == 'text':
                    prompt += item['value']
                elif item['type'] == 'image':
                    prompt += f"Picture {image_cnt}: <img>{item['value']}</img>\n"
                    image_cnt += 1
            return prompt

        assert len(message) % 2 == 0
        hist = []
        for i in range(len(message) // 2):
            m1, m2 = message[2 * i], message[2 * i + 1]
            assert m1['role'] == 'user' and m2['role'] == 'assistant'
            hist.append(
                (concat_tilist(m1['content']), concat_tilist(m2['content'])))
        return hist

    def generate_inner(self, message, dataset=None):
        text, img_path = self.message_to_promptimg(message, dataset=dataset)
        pil_image = Image.open(img_path).convert('RGB')
        messages = [{'role': 'user', 'content': text}]
        answer = self.model.chat(pil_image,
                                 messages,
                                 do_sample=False,
                                 max_new_tokens=512)
        return answer

    def chat_inner(self, message, dataset=None):
        assert len(message) % 2 == 1 and message[-1]['role'] == 'user'
        history = self.build_history(message[:-1])
        vl_list = [{
            'image': s['value']
        } if s['type'] == 'image' else {
            'text': s['value']
        } for s in message[-1]['content']]
        query = self.tokenizer.from_list_format(vl_list)
        response, _ = self.model.chat(self.tokenizer,
                                      query=query,
                                      history=history,
                                      **self.kwargs)
        return response

    def use_custom_prompt(self, dataset):

        if dataset is not None and listinstr(['MMDU'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        else:
            return True

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
            if listinstr(['MathVista', 'MathVision'], dataset):
                prompt = line['question']
            elif listinstr(['LLaVABench'], dataset):
                question = line['question']
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['MMVet', 'OCRBench'], dataset):
                prompt = line[
                    'question'] + ' Anylyze the reason for the answer.'
            elif listinstr(['MTBench_VQA'], dataset):
                prompt = line['question'] + '\n 请直接回答问题'
            else:
                question = line['question']
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def build_multi_choice_prompt(self, line, dataset=None):
        question = line['question']
        hint = line['hint'] if ('hint' in line
                                and not pd.isna(line['hint'])) else None
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
                prompt
            ) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if cn_string(
                prompt) else '\nAnswer the question directly.'

        return prompt
