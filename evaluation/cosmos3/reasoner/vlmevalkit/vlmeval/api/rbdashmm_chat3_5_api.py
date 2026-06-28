import json
import os
import re

import numpy as np
import requests

from vlmeval.api.base import BaseAPI
from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE
from vlmeval.smp import encode_image_to_base64, get_logger, listinstr

logger = get_logger(__name__)


def process_prediction_text(text):
    if not isinstance(text, str):
        return text

    if "\\boxed{" in text:
        start = text.rfind("\\boxed{")
        i = start + len("\\boxed{")
        brace_count = 1
        content_chars = []

        while i < len(text) and brace_count > 0:
            ch = text[i]
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
            if brace_count > 0:
                content_chars.append(ch)
            i += 1

        if content_chars:
            inner = "".join(content_chars).strip()
            if inner:
                return inner

    redacted_pattern = r'<think>.*?</think>'
    if re.search(redacted_pattern, text, re.DOTALL):
        result = re.sub(redacted_pattern, '', text, flags=re.DOTALL)
        return result.strip()

    # 默认返回
    return text


R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different \
angles, potential solutions, and reason through the problem step-by-step. \
Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to \
the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the \
query. The final answer should be standalone and not reference the thinking \
section.
""".strip()


class RBdashMMChat3_78B_PromptUtil:
    def __init__(self, use_mpo_prompt=False):
        self.use_mpo_prompt = use_mpo_prompt
        self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        assert DATASET_MODALITY(dataset) != 'VIDEO', 'not supported'
        if dataset in [
            'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
            'optics_dataset', 'quantum_dataset', 'statistics_dataset'
        ]:
            return False
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT', 'MMAlignBench'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):
        use_cot = (os.getenv('USE_COT') == '1')
        use_mpo_prompt = self.use_mpo_prompt and (use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench'])

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
                prompt = question
                cot_prompt = (
                    "Answer the preceding yes-or-no question. Think step by step logically, considering all "
                    "relevant information before answering. If you are uncertain or the problem is ambiguous, make a "
                    "reasoned guess based on the information provided. The last line of your response must be exactly "
                    "in this format: 'Answer: \\boxed{Yes}' or 'Answer: \\boxed{No}' (without quotes)."
                )
                prompt = prompt + '\n' + cot_prompt
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase. \
                    The image is guaranteed to contain the correct answer. Please provide the most likely \
                    answer — do not answer "No." Let us think step by step.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath',
                            'QSpatial', 'WeMath', 'LogicVista'], dataset):
                if listinstr(['MMVet'], dataset):
                    prompt = question + "\nPlease first think and then answer the question."
                else:
                    prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
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

        if use_mpo_prompt:
            message = build_mpo_prompt(message, line, dataset)

        # reorganize_prompt
        prompt = reorganize_prompt(message, image_num, dataset=dataset)
        prompt.replace('<image>', '<IMAGE_TOKEN>')
        message[0] = dict(type='text', value=prompt)
        return message

    def get_max_num(self, dataset):
        self.total_max_num = 64
        if dataset is None:
            self.max_num = 6
            return None
        res_8_datasets = ['MMStar']
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld', 'HallusionBench',
                           'VCR_EN', 'VCR_ZH', 'OCRVQA', 'BMMR', 'MathVista_MINI']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K', 'MMVet']
        if DATASET_MODALITY(dataset) == 'VIDEO':
            self.max_num = 1
        elif listinstr(res_8_datasets, dataset):
            return 8
        elif listinstr(res_12_datasets, dataset):
            return 12
        elif listinstr(res_18_datasets, dataset):
            return 18
        elif listinstr(res_24_datasets, dataset):
            return 24
        else:
            return 6


class RBdashMMChat3_5_38B_PromptUtil:
    def __init__(self, use_mpo_prompt=False):
        self.use_mpo_prompt = use_mpo_prompt
        self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        assert DATASET_MODALITY(dataset) != 'VIDEO', 'not supported'
        if dataset in [
            'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
            'optics_dataset', 'quantum_dataset', 'statistics_dataset'
        ]:
            return False
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT', 'MMAlignBench'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):
        use_cot = (os.getenv('USE_COT') == '1')
        use_mpo_prompt = self.use_mpo_prompt and (use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench'])

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
                prompt = question
                cot_prompt = (
                    "Answer the preceding yes-or-no question. Think step by step logically, considering all "
                    "relevant information before answering. If you are uncertain or the problem is ambiguous, make "
                    "a reasoned guess based on the information provided. The last line of your response must be "
                    "exactly in this format: 'Answer: \\boxed{Yes}' or 'Answer: \\boxed{No}' (without quotes)."
                )
                prompt = prompt + '\n' + cot_prompt
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase. \
                    The image is guaranteed to contain the correct answer. Please provide the most likely \
                    answer — do not answer "No." Let us think step by step.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath',
                            'QSpatial', 'WeMath', 'LogicVista'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
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

        if use_mpo_prompt:
            message = build_mpo_prompt(message, line, dataset)

        # reorganize_prompt
        prompt = reorganize_prompt(message, image_num, dataset=dataset)
        prompt.replace('<image>', '<IMAGE_TOKEN>')
        message[0] = dict(type='text', value=prompt)
        return message

    def get_max_num(self, dataset):
        self.total_max_num = 64
        if dataset is None:
            self.max_num = 6
            return None
        res_8_datasets = ['MMStar']
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'VCR_EN', 'VCR_ZH', 'OCRVQA', 'BMMR', 'MathVista_MINI']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K', 'MMVet']
        if DATASET_MODALITY(dataset) == 'VIDEO':
            self.max_num = 1
        elif listinstr(res_8_datasets, dataset):
            return 8
        elif listinstr(res_12_datasets, dataset):
            return 12
        elif listinstr(res_18_datasets, dataset):
            return 18
        elif listinstr(res_24_datasets, dataset):
            return 24
        else:
            return 6


class RBdashMMChat3_78B_Wrapper(BaseAPI):
    is_api: bool = True
    custom_prompt = 'rbdash3'
    prompt_map = {
        'rbdash3': RBdashMMChat3_78B_PromptUtil()
    }

    def __init__(self,
                 model: str = None,
                 retry: int = 5,
                 key: str = 'sk-123456',
                 verbose: bool = True,
                 temperature: float = 0.0,
                 timeout: int = 600,
                 api_base: str = None,
                 system_prompt: str = None,
                 max_tokens: int = 20480,
                 **kwargs):
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.timeout = timeout

        key = os.environ.get('LMDEPLOY_API_KEY', key)
        api_base = os.environ.get('LMDEPLOY_API_BASE', api_base)
        assert key is not None, 'Please set the environment variable LMDEPLOY_API_KEY.'
        assert api_base is not None, 'Please set the environment variable LMDEPLOY_API_BASE.'
        self.key = key
        self.api_base = api_base
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        model_url = ''.join([api_base.split('v1')[0], 'v1/models'])
        resp = requests.get(model_url)
        model_id_list = [str(data['id']) for data in resp.json()['data']]
        self.model = model if model in model_id_list else model_id_list[0]
        logger.info(f'lmdeploy evaluate model: {self.model}')

        self.max_tokens = 20480
        self.temperature = 0.0
        self.custom_prompt = 'rbdash3'

        logger.info(f'using custom prompt {self.custom_prompt}')
        logger.info(f'Init temperature: {self.temperature}')
        self.system_prompt = R1_SYSTEM_PROMPT

    def set_dump_image(self, dump_image_func):
        if self.custom_prompt in self.prompt_map:
            self.prompt_map[self.custom_prompt].dump_image_func = dump_image_func
        self.dump_image_func = dump_image_func

    def use_custom_prompt(self, dataset):
        if self.custom_prompt in self.prompt_map:
            return self.prompt_map[self.custom_prompt].use_custom_prompt(dataset)
        return False

    def build_prompt(self, line, dataset=None):
        if self.custom_prompt in self.prompt_map:
            return self.prompt_map[self.custom_prompt].build_prompt(line, dataset)
        raise NotImplementedError

    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                    print('text msg:', msg['value'])
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
        temperature = kwargs.pop('temperature', self.temperature)
        logger.info(f'Generate temperature: {temperature}')
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        dataset = kwargs.pop('dataset', None)
        if listinstr(['MMMU_DEV_VAL', 'MathVista'], dataset):
            self.system_prompt = R1_SYSTEM_PROMPT
        else:
            self.system_prompt = None

        input_msgs = self.prepare_inputs(inputs)

        if dataset is not None and listinstr(['BMMR'], dataset):
            # BMMR dataset has a very long prompt, so we need to increase max_tokens
            max_tokens = 8196
            logger.info('BMMR dataset detected, set max_tokens to 8196')

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            do_sample=False,
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
            answer = process_prediction_text(answer)
        except Exception:
            pass
        return ret_code, answer, response


class RBdashMMChat3_5_38B_Wrapper(BaseAPI):
    is_api: bool = True
    custom_prompt = 'rbdash3_5'
    prompt_map = {
        'rbdash3_5': RBdashMMChat3_5_38B_PromptUtil()
    }

    def __init__(self,
                 model: str = None,
                 retry: int = 5,
                 key: str = 'sk-123456',
                 verbose: bool = True,
                 temperature: float = 0.0,
                 timeout: int = 600,
                 api_base: str = None,
                 system_prompt: str = None,
                 max_tokens: int = 20480,
                 **kwargs):
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.timeout = timeout

        key = os.environ.get('LMDEPLOY_API_KEY', key)
        api_base = os.environ.get('LMDEPLOY_API_BASE', api_base)
        assert key is not None, 'Please set the environment variable LMDEPLOY_API_KEY.'
        assert api_base is not None, 'Please set the environment variable LMDEPLOY_API_BASE.'
        self.key = key
        self.api_base = api_base
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        model_url = ''.join([api_base.split('v1')[0], 'v1/models'])
        resp = requests.get(model_url)
        model_id_list = [str(data['id']) for data in resp.json()['data']]
        self.model = model if model in model_id_list else model_id_list[0]
        logger.info(f'lmdeploy evaluate model: {self.model}')

        self.max_tokens = 20480
        self.temperature = 0.0
        self.custom_prompt = 'rbdash3_5'

        logger.info(f'using custom prompt {self.custom_prompt}')
        logger.info(f'Init temperature: {self.temperature}')
        self.system_prompt = R1_SYSTEM_PROMPT

    def set_dump_image(self, dump_image_func):
        if self.custom_prompt in self.prompt_map:
            self.prompt_map[self.custom_prompt].dump_image_func = dump_image_func
        self.dump_image_func = dump_image_func

    def use_custom_prompt(self, dataset):
        if self.custom_prompt in self.prompt_map:
            return self.prompt_map[self.custom_prompt].use_custom_prompt(dataset)
        return False

    def build_prompt(self, line, dataset=None):
        if self.custom_prompt in self.prompt_map:
            return self.prompt_map[self.custom_prompt].build_prompt(line, dataset)
        raise NotImplementedError

    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                    print('text msg:', msg['value'])
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
        temperature = kwargs.pop('temperature', self.temperature)
        logger.info(f'Generate temperature: {temperature}')
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        dataset = kwargs.pop('dataset', None)
        if listinstr([
                'MMMU_DEV_VAL', 'MathVista_MINI', 'MMBench_V11',
                'MMStar', 'DynaMath', 'MathVision', 'MathVerse',
                'LogicVista', 'WeMath'], dataset):
            self.system_prompt = R1_SYSTEM_PROMPT
        else:
            self.system_prompt = None

        input_msgs = self.prepare_inputs(inputs)

        if dataset is not None and listinstr(['BMMR'], dataset):
            # BMMR dataset has a very long prompt, so we need to increase max_tokens
            max_tokens = 8196
            logger.info('BMMR dataset detected, set max_tokens to 8196')

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            do_sample=False,
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
            answer = process_prediction_text(answer)
        except Exception:
            pass
        return ret_code, answer, response


class RBdashMMChat3_78B_API(RBdashMMChat3_78B_Wrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, message, dataset=None):
        return super(RBdashMMChat3_78B_API, self).generate(message, dataset=dataset)


class RBdashMMChat3_5_38B_API(RBdashMMChat3_5_38B_Wrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, message, dataset=None):
        return super(RBdashMMChat3_5_38B_API, self).generate(message, dataset=dataset)
