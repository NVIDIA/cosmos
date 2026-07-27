import base64
import json
import os
import string

import numpy as np
import pandas as pd
import requests

from vlmeval.api.base import BaseAPI
from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string, get_logger, listinstr

logger = get_logger(__name__)


class TaiyiWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'taiyi',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 60,
                 url: str = "https://taiyi.megvii.com/v1/chat/completions",
                 max_tokens: int = 1024,
                 **kwargs):

        self.model = model
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature

        if key is None:
            key = os.environ.get('TAIYI_API_KEY', None)
        assert key is not None, ('Please set the API Key ')
        self.key = key

        self.timeout = timeout
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)
        assert url is not None, ('Please set the url ')
        self.url = url
        logger.info(f'Using url: {self.url}; API Key: {self.key}')

    def use_custom_prompt(self, dataset):
        if DATASET_TYPE(dataset) == 'Y/N' or DATASET_TYPE(dataset) == 'MCQ' or DATASET_TYPE(dataset) == 'VQA':
            return True
        return False

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    imgbytes = open(msg['value'], 'rb').read()
                    b64 = base64.b64encode(imgbytes).decode('ascii')
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}')
                    content_list.append(dict(type='image_url', image_url=img_struct))
            input_msgs.append(dict(role='user', content=content_list))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            input_msgs.append(dict(role='user', content=text))
        return input_msgs

    def image_first(self, msgs):
        nr_img = 0
        for s in msgs:
            if s['type'] == 'image':
                nr_img += 1

        if nr_img == 1:
            new_msgs = []
            img_msg = None
            for s in msgs:
                if s['type'] == 'text':
                    new_msgs.append(s)
                else:
                    img_msg = s
            new_msgs.insert(0, img_msg)
        else:
            new_msgs = msgs

        return new_msgs

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

    def build_yorn_prompt(self, line, dataset=None):
        if listinstr(['HallusionBench'], dataset):
            pre_prompt = 'Read the following question carefully, think and solve it step by step.\n\n'
        else:
            pre_prompt = ''

        prompt = pre_prompt + line['question'] + ' Please answer yes or no as the final answer.'

        return prompt

    def build_vqa_prompt(self, line, dataset=None):
        if listinstr(['OCRBench'], dataset):
            pre_prompt = 'Carefully identify the text in the image and answer the question.\n\n'
        else:
            pre_prompt = ''

        if listinstr(['MMVet'], dataset):
            post_prompt = '\nAnswer this question in detail.'
        else:
            post_prompt = ''

        prompt = pre_prompt + line['question'] + post_prompt

        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'Y/N':
            prompt = self.build_yorn_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'VQA':
            prompt = self.build_vqa_prompt(line, dataset)
        else:
            raise RuntimeError(f'Invalid dataset type: {DATASET_TYPE(dataset)}')
        message = []
        message.extend([dict(type='image', value=s) for s in tgt_path])
        message.extend([dict(type='text', value=prompt)])

        # interleave dataset
        if dataset.startswith('MMMU_'):
            from .. import MMMUDataset
            message = MMMUDataset.split_MMMU(message)
            message = self.image_first(message)

        return message

    def generate_inner(self, inputs, **kwargs) -> str:

        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)

        headers = {'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            n=1,
            temperature=temperature,
            **kwargs)
        response = requests.post(self.url, headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
        except Exception:
            pass
        return ret_code, answer, response


class TaiyiAPI(TaiyiWrapper):

    def generate(self, message, dataset=None):
        return super(TaiyiAPI, self).generate(message)
