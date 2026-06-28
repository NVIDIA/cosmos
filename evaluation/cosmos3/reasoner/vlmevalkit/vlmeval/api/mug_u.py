import json
import os

import numpy as np
import requests

from vlmeval.api.base import BaseAPI
from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE
from vlmeval.smp import encode_image_to_base64, get_logger, listinstr

logger = get_logger(__name__)


class MUGUWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str,
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 temperature: float = 0.0,
                 timeout: int = 60,
                 api_base: str = None,
                 system_prompt: str = None,
                 max_tokens: int = 4096,
                 use_mpo_prompt: bool = False,
                 **kwargs):
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.timeout = timeout

        api_base = 'https://shopee.sg/api/v1/compassllvm/v1/chat/completions'
        assert api_base is not None, 'Please set the environment variable LMDEPLOY_API_BASE.'
        self.api_base = api_base
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        model_url = ''.join([api_base.split('v1')[0], 'v1/models'])
        _ = requests.get(model_url)
        self.model = model
        if hasattr(self, 'custom_prompt'):
            logger.info(f'using custom prompt {self.custom_prompt}')
        self.temperature = temperature
        logger.info(f'Init temperature: {self.temperature}')
        self.use_mpo_prompt = use_mpo_prompt

        self.temperature = 0.0

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        assert DATASET_MODALITY(dataset) != 'VIDEO', 'not supported'
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def get_max_num(self, dataset):
        assert dataset is not None
        res_1_datasets = ['MMBench-Video', 'Video-MME', 'MVBench', 'Video', 'WorldSense']
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'VCR_EN', 'VCR_ZH', 'OCRVQA']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if listinstr(res_1_datasets, dataset):
            return 1
        elif listinstr(res_12_datasets, dataset):
            return 12
        elif listinstr(res_18_datasets, dataset):
            return 18
        elif listinstr(res_24_datasets, dataset):
            return 24
        else:
            return 6

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        from ..vlm.internvl.utils import (build_mcq_cot_prompt, build_mpo_prompt,
                                          build_multi_choice_prompt, build_qa_cot_prompt,
                                          reorganize_prompt)

        tgt_path = self.dump_image(line, dataset)
        max_num = self.get_max_num(dataset)
        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath',
                            'QSpatial', 'WeMath', 'LogicVista'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt)

        message = [dict(type='text', value=prompt)]
        image_num = len(tgt_path)
        max_num = max(1, min(max_num, 64 // image_num))
        # TODO：support upscale_flag
        message.extend([dict(type='image', value=s, max_dynamic_patch=max_num) for s in tgt_path])

        if self.use_mpo_prompt:
            message = build_mpo_prompt(message, line, dataset)

        # reorganize_prompt
        prompt = reorganize_prompt(message, image_num, dataset=dataset)
        prompt.replace('<image>', '<IMAGE_TOKEN>')
        message[0] = dict(type='text', value=prompt)
        return message

    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img)
                    extra_args = msg.copy()
                    extra_args.pop('type')
                    extra_args.pop('value')
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', **extra_args)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)

        temperature = kwargs.pop('temperature', self.temperature)
        logger.info(f'Generate temperature: {temperature}')
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        headers = {'Content-Type': 'application/json'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            top_k=1,
            temperature=temperature,
            stream=False,
            **kwargs)

        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()

            # for internvl2-8b-mpo-cot
            if getattr(self, 'use_mpo_prompt', False):
                from ..vlm.internvl.utils import mpo_post_processing

                answer = mpo_post_processing(answer, kwargs.get('dataset'))
        except Exception:
            pass
        return ret_code, answer, response


class MUGUAPI(MUGUWrapper):
    def generate(self, message, dataset=None):
        return super(MUGUAPI, self).generate(message, dataset=dataset)
