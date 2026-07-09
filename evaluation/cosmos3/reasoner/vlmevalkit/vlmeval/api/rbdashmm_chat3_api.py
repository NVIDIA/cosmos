import json
import os
import re

import numpy as np
import requests

from vlmeval.api.base import BaseAPI
from vlmeval.smp import encode_image_to_base64, get_logger, listinstr

logger = get_logger(__name__)


def process_prediction_text(text):
    if not isinstance(text, str):
        return text

    # ====== 处理 \boxed{}（支持嵌套）======
    if "\\boxed{" in text:
        start = text.find("\\boxed{")
        i = start + len("\\boxed{")
        brace_count = 1
        content_chars = []

        while i < len(text) and brace_count > 0:
            ch = text[i]
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
            if brace_count > 0:  # 在外层 } 之前才加入字符
                content_chars.append(ch)
            i += 1

        if content_chars:
            inner = "".join(content_chars).strip()
            if inner:
                return inner

    # ====== 删除 <think> 标签内容 ======
    redacted_pattern = r'<think>.*?</think>'
    if re.search(redacted_pattern, text, re.DOTALL):
        result = re.sub(redacted_pattern, '', text, flags=re.DOTALL)
        return result.strip()

    # ====== 长文本截断 ======
    if len(text) > 32767:
        return text[-30000:]

    # 默认返回
    return text


class RBdashMMChat3_PromptUtil:
    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        return True

    def build_prompt(self, line, dataset=None):
        from ..vlm.internvl.utils import (build_mcq_cot_prompt, build_multi_choice_prompt,
                                          build_qa_cot_prompt, build_yesorno_cot_prompt,
                                          reorganize_prompt)
        tgt_path = self.dump_image(line, dataset)
        max_num = self.get_max_num(dataset)
        if dataset is not None and listinstr(['MMBench_V11', 'MMBench_CN_V11'], dataset):
            prompt = build_multi_choice_prompt(line, dataset)
        elif dataset is not None and listinstr(['MMStar'], dataset):
            prompt = build_multi_choice_prompt(line, dataset)
        elif dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset):
            prompt = build_multi_choice_prompt(line, dataset)
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
            prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and listinstr(['MathVista_MINI'], dataset):
            prompt = line['question']
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
            prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and listinstr(['OCRBench'], dataset):
            prompt = line['question'] + '\nAnswer the question using a single word or phrase. \
            The image is guaranteed to contain the correct answer. Please provide the most likely \
            answer — do not answer "No." Let us think step by step.'
        elif dataset is not None and listinstr(['AI2D_TEST'], dataset):
            prompt = build_multi_choice_prompt(line, dataset)
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            self.cot_prompt = None
            prompt = build_yesorno_cot_prompt(line, line['question'], self.cot_prompt)
        elif dataset is not None and listinstr(['MMVet'], dataset):
            prompt = line['question'] + "\nPlease first think and then answer the question."

        message = [dict(type='text', value=prompt)]
        image_num = len(tgt_path)
        max_num = max(1, min(max_num, 64 // image_num))

        message.extend([dict(type='image', value=s, max_dynamic_patch=max_num) for s in tgt_path])

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
        elif dataset is not None and listinstr(['MMBench_V11'], dataset):
            return 6
        elif dataset is not None and listinstr(['MMStar'], dataset):
            return 8
        elif dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset):
            return 12
        elif dataset is not None and listinstr(['MathVista_MINI'], dataset):
            return 12
        elif dataset is not None and listinstr(['OCRBench'], dataset):
            return 24
        elif dataset is not None and listinstr(['AI2D_TEST'], dataset):
            return 6
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            return 12
        elif dataset is not None and listinstr(['MMVet'], dataset):
            return 24
        else:
            return 6


class RBdashMMChat3_5_PromptUtil:
    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        return True

    def build_prompt(self, line, dataset=None):
        from ..vlm.internvl.utils import (build_mcq_cot_prompt, build_multi_choice_prompt,
                                          build_qa_cot_prompt, build_yesorno_cot_prompt,
                                          reorganize_prompt)
        tgt_path = self.dump_image(line, dataset)
        max_num = self.get_max_num(dataset)
        if dataset is not None and listinstr(['MMBench_V11'], dataset):
            prompt = build_multi_choice_prompt(line, dataset)
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
            prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and listinstr(['MMStar'], dataset):
            prompt = build_multi_choice_prompt(line, dataset)
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
            prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset):
            prompt = build_multi_choice_prompt(line, dataset)
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
            prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and listinstr(['MathVista_MINI'], dataset):
            prompt = line['question']
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
            prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and listinstr(['OCRBench'], dataset):
            prompt = line['question'] + '\nAnswer the question using a single word or phrase. \
            The image is guaranteed to contain the correct answer. Please provide the most likely \
            answer — do not answer "No." Let us think step by step.'
        elif dataset is not None and listinstr(['AI2D_TEST'], dataset):
            prompt = build_multi_choice_prompt(line, dataset)
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            self.cot_prompt = None
            prompt = build_yesorno_cot_prompt(line, line['question'], self.cot_prompt)
        elif dataset is not None and listinstr(['MMVet'], dataset):
            prompt = line['question']
        elif listinstr(['MathVista', 'DynaMath', 'MathVision', 'MathVerse', 'LogicVista'], dataset):
            cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
            prompt = build_qa_cot_prompt(line, line['question'], cot_prompt)
        elif listinstr(['WeMath'], dataset):
            prompt = build_multi_choice_prompt(line, dataset)
            cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
            prompt = build_mcq_cot_prompt(line, prompt, cot_prompt)

        message = [dict(type='text', value=prompt)]
        image_num = len(tgt_path)
        max_num = max(1, min(max_num, 64 // image_num))

        message.extend([dict(type='image', value=s, max_dynamic_patch=max_num) for s in tgt_path])

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
        elif dataset is not None and listinstr(['MMBench_V11'], dataset):
            return 6
        elif dataset is not None and listinstr(['MMStar'], dataset):
            return 8
        elif dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset):
            return 12
        elif dataset is not None and listinstr(['MathVista_MINI'], dataset):
            return 12
        elif dataset is not None and listinstr(['OCRBench'], dataset):
            return 24
        elif dataset is not None and listinstr(['AI2D_TEST'], dataset):
            return 6
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            return 6
        elif dataset is not None and listinstr(['MMVet'], dataset):
            return 24
        else:
            return 6


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


class RBdashMMChat3Wrapper(BaseAPI):

    is_api: bool = True
    custom_prompt = ''
    prompt_map = {
        "rbdashmm3chat": RBdashMMChat3_PromptUtil(),
    }

    def __init__(self,
                 model: str = None,
                 retry: int = 5,
                 key: str = 'sk-123456',
                 verbose: bool = True,
                 temperature: float = 0.0,
                 timeout: int = 300,
                 api_base: str = None,
                 system_prompt: str = None,
                 max_tokens: int = 16384,
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

        self.custom_prompt = "rbdashmm3chat"
        self.max_tokens = 16384
        self.temperature = 0.0
        if hasattr(self, 'custom_prompt'):
            logger.info(f'using custom prompt {self.custom_prompt}')

        self.temperature = temperature
        logger.info(f'Init temperature: {self.temperature}')

        self.system_prompt = None

    def set_dump_image(self, dump_image_func):
        if self.custom_prompt in self.prompt_map:
            self.prompt_map[self.custom_prompt].dump_image_func = dump_image_func
        self.dump_image_func = dump_image_func

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        return True

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

    def prepare_inputs(self, inputs, dataset):
        input_msgs = []

        if listinstr(['MMMU_DEV_VAL', 'MathVista_MINI'], dataset):
            self.system_prompt = R1_SYSTEM_PROMPT
        else:
            self.system_prompt = None

        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        else:
            input_msgs.append(dict(role='system', content="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"))

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

        input_msgs = self.prepare_inputs(inputs, dataset)

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            session_len=40960,
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


class RBdashMMChat3_API(RBdashMMChat3Wrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, message, dataset=None):
        return super(RBdashMMChat3_API, self).generate(message, dataset=dataset)


class RBdashMMChat3_5_Wrapper(BaseAPI):

    is_api: bool = True
    custom_prompt = ''
    prompt_map = {
        "rbdashmm3_5chat": RBdashMMChat3_5_PromptUtil(),
    }

    def __init__(self,
                 model: str = None,
                 retry: int = 5,
                 key: str = 'sk-123456',
                 verbose: bool = True,
                 temperature: float = 0.0,
                 timeout: int = 300,
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

        self.custom_prompt = "rbdashmm3_5chat"
        self.max_tokens = 20480
        self.temperature = 0.0
        if hasattr(self, 'custom_prompt'):
            logger.info(f'using custom prompt {self.custom_prompt}')

        self.temperature = temperature
        logger.info(f'Init temperature: {self.temperature}')

        self.system_prompt = None

    def set_dump_image(self, dump_image_func):
        if self.custom_prompt in self.prompt_map:
            self.prompt_map[self.custom_prompt].dump_image_func = dump_image_func
        self.dump_image_func = dump_image_func

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        return True

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

    def prepare_inputs(self, inputs, dataset):
        input_msgs = []

        if listinstr([
                'MMMU_DEV_VAL', 'MathVista_MINI', 'MMBench_V11',
                'MMStar', 'DynaMath', 'MathVision', 'MathVerse',
                'LogicVista', 'WeMath'], dataset):
            self.system_prompt = R1_SYSTEM_PROMPT
        else:
            self.system_prompt = None

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

        input_msgs = self.prepare_inputs(inputs, dataset)

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            session_len=40960,
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


class RBdashChat3_5_API(RBdashMMChat3_5_Wrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, message, dataset=None):
        return super(RBdashChat3_5_API, self).generate(message, dataset=dataset)
