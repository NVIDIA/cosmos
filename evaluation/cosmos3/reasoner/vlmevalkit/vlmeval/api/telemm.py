import json
import os
import re

import numpy as np
import requests

from vlmeval.api.base import BaseAPI
from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE
from vlmeval.smp import encode_image_to_base64, get_logger, listinstr

logger = get_logger(__name__)

API_BASE = "https://openapi.teleagi.cn:443/aipaas/aiproxy-stream/chat/completions/8005400012"

PREDEFINED_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different \
angles, potential solutions, and reason through the problem step-by-step. \
Enclose this entire thinking process within {START} and {END} tags.

2. After the thinking section, provide a clear, concise, and direct answer to \
the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the \
query. The final answer should be standalone and not reference the thinking \
section.
""".strip()


class TeleMM2_PromptUtil:
    def __init__(self):
        self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        assert DATASET_MODALITY(dataset) != 'VIDEO', 'not supported'
        return True

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        from ..vlm.internvl.utils import (build_mcq_cot_prompt, build_multi_choice_prompt,
                                          build_qa_cot_prompt, reorganize_prompt)

        tgt_path = self.dump_image(line, dataset)
        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['HallusionBench'], dataset):
                cot_prompt = (
                    "Answer the preceding yes-or-no question. Think step by step logically, considering all "
                    "relevant information before answering. If you are uncertain or the problem is ambiguous, make a "
                    "reasoned guess based on the information provided. The last line of your response must be exactly "
                    "in this format: 'Answer: \\boxed{Yes}' or 'Answer: \\boxed{No}' (without quotes)."
                )
                prompt = question + '\n' + cot_prompt
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if listinstr(['MMMU_DEV_VAL'], dataset):
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['OCRBench'], dataset):
                OCR_prompt = (
                    "\nAnswer the question using a single word or phrase. "
                    "The image is guaranteed to contain the correct answer. Please provide the most likely "
                    "answer — do not answer \"No.\". Do not answer 'r'. "
                )
                prompt = question + OCR_prompt
            elif listinstr(['MathVista'], dataset):
                prompt = build_qa_cot_prompt(line, question, self.cot_prompt)
            elif listinstr(['MMVet'], dataset):
                prompt = question + "\nPlease first think and then answer the question. "
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']

        message = [dict(type='text', value=prompt)]
        image_num = len(tgt_path)
        message.extend([dict(type='image', value=s) for s in tgt_path])

        prompt = reorganize_prompt(message, image_num, dataset=dataset)
        prompt.replace('<image>', '<IMAGE_TOKEN>')
        message[0] = dict(type='text', value=prompt)
        return message


class TeleMM2_Wrapper(BaseAPI):
    is_api: bool = True
    custom_prompt = 'telemm2'
    prompt_map = {
        'telemm2': TeleMM2_PromptUtil()
    }

    def __init__(self,
                 model: str = None,
                 retry: int = 3,
                 verbose: bool = True,
                 timeout: int = 600,
                 authorization: str = None,
                 app_id: str = None,
                 system_prompt: str = None,
                 max_tokens: int = 20480,
                 **kwargs):
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.api_base = API_BASE
        self.authorization = os.environ.get('AUTHOR', authorization)
        self.app_id = os.environ.get('APP_ID', app_id)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        self.model = model

        logger.info(f'evaluate model: {self.model}')

        self.max_tokens = 20480
        self.temperature = 0.0
        self.custom_prompt = 'telemm2'

        logger.info(f'using custom prompt {self.custom_prompt}')
        logger.info(f'Init temperature: {self.temperature}')
        self.system_prompt = None

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

    def extract_boxed(self, text):
        if "\\boxed{" not in text:
            return None

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
        return None

    def extract_answer(self, text, dataset_name=None):
        if not isinstance(text, str):
            return text

        boxed = self.extract_boxed(text)
        if boxed is not None:
            return boxed

        if listinstr(['MMVet'], dataset_name):
            redacted_pattern = r'<think>.*?</think>'
            if re.search(redacted_pattern, text, re.DOTALL):
                result = re.sub(redacted_pattern, '', text, flags=re.DOTALL)
                return result.strip()
            return text

        elif listinstr(['MMMU_DEV_VAL'], dataset_name):
            last_end = text.rfind("</thinking>")
            if last_end != -1:
                result = text[last_end + len("</thinking>"):]
                return result.strip()
            return text
        else:
            last_end = text.rfind("</think>")
            if last_end != -1:
                result = text[last_end + len("</think>"):]
                return result.strip()
            return text

    def generate_inner(self, inputs, **kwargs) -> str:
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        dataset = kwargs.pop('dataset', None)
        if listinstr(['MathVista'], dataset):
            self.system_prompt = PREDEFINED_SYSTEM_PROMPT.format(START="<think>", END="</think>")
        elif listinstr(['MMMU_DEV_VAL'], dataset):
            self.system_prompt = PREDEFINED_SYSTEM_PROMPT.format(START="<thinking>", END="</thinking>")
        else:
            self.system_prompt = None

        input_msgs = self.prepare_inputs(inputs)

        if listinstr(['MMMU_DEV_VAL'], dataset):
            do_sample = True
            temperature = 0.6
        else:
            do_sample = False
            temperature = 0.0

        logger.info(f'Generate temperature: {temperature}')

        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.authorization,
            'X-APP-ID': self.app_id}

        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            do_sample=do_sample,
            **kwargs)
        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
            answer = self.extract_answer(answer, dataset)
        except Exception:
            pass
        return ret_code, answer, response


class TeleMM2_API(TeleMM2_Wrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, message, dataset=None):
        return super(TeleMM2_API, self).generate(message, dataset=dataset)
