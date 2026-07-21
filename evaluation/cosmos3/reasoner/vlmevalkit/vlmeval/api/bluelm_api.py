import base64
import json
import os
import re

import numpy as np
import requests

from vlmeval.api.base import BaseAPI
from vlmeval.smp import concat_images_vlmeval, get_logger

logger = get_logger(__name__)


def split_think(text: str) -> str:
    """
    提取think后的内容
    """
    if "</think>" in text:
        answer = text.split("</think>")[1]
    else:
        if "<think>" in text:
            return 'Thinking mode too long to extract answer'
        return text
    return answer


def remove_boxed(s: str):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None


def last_boxed_only_string(string: str):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def extract_boxed(pred_str: str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return pred_str  # 返回原始字符串
    answer = remove_boxed(boxed_str)
    if answer is None:
        return pred_str  # 返回原始字符串
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer


def extract_boxed_answer(pred_str: str):
    if pred_str.rfind('\\boxed') < 0 and pred_str.rfind('\\fbox') < 0:
        return pred_str
    return extract_boxed(pred_str, strip_double_curly_brace=True)


def get_streaming_response(response: requests.Response):
    for chunk in response.iter_lines(chunk_size=4096,
                                     decode_unicode=False):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data.get("result")
            yield output


def multimodal(images, text, url, key, temperature=0.6, max_tokens=32768, top_k=20, top_p=0.95, stream=True, history=[], timeout=60):  # noqa: E501
    if images:
        pics = []
        for image in images:
            with open(image, 'rb') as f:
                pic = base64.b64encode(f.read()).decode('utf-8')
            pics.append(pic)
        data = {
            'images': pics, 'text': text, 'key': key, 'temperature': temperature,
            'max_tokens': max_tokens, 'top_k': top_k, 'top_p': top_p, 'stream': stream
        }
    else:
        data = {
            'text': text, 'key': key, 'temperature': temperature,
            'max_tokens': max_tokens, 'top_k': top_k, 'top_p': top_p, 'stream': stream
        }
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"}, timeout=timeout)
    if stream:
        final_text = ''
        for h in get_streaming_response(response):
            final_text = h
    else:
        response_data = response.json()
        final_text = response_data.get("result", "")
    return final_text


class BlueLMWrapper(BaseAPI):
    is_api: bool = True

    def __init__(self,
                 model: str = 'BlueLM-2.5-3B',
                 retry: int = 5,
                 verbose: bool = True,
                 temperature: float = 0.6,
                 system_prompt: str = None,
                 max_tokens: int = 32768,
                 top_k: int = 20,
                 top_p: float = 0.95,
                 timeout: int = 60,
                 key: str = None,
                 url: str = 'http://api-ai.vivo.com.cn/multimodal',
                 **kwargs):

        self.model = model
        self.fail_msg = 'Failed to obtain answer BlueLM API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.url = url
        self.key = key
        self.timeout = timeout

        if self.key is None:
            self.key = os.environ.get('BLUELM_API_KEY', None)
        assert self.key is not None, (
            'Please set the API Key (obtain it here: '
            'contact by email : shuai.ren@vivo.com'
        )

        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def message_to_promptimg(self, message, dataset=None):

        num_images = len([x for x in message if x['type'] == 'image'])
        if num_images == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = None
        elif num_images == 1:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = [x['value'] for x in message if x['type'] == 'image']
        else:
            prompt = '\n'.join([x['value'] if x['type'] == 'text' else '<im_start><image><im_end>' for x in message])
            if dataset == 'BLINK':
                image = concat_images_vlmeval(
                    [x['value'] for x in message if x['type'] == 'image'],
                    target_size=512)
            else:
                image = [x['value'] for x in message if x['type'] == 'image']

        if dataset in ['MMBench_DEV_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_TEST_CN_V11',
                       'AI2D_TEST', 'AI2D_TEST_TO_MASK', 'MMMU_DEV_VAL', 'MMStar']:
            prompt = prompt.replace('Please select the correct answer from the options above.',
                                    'Answer with the option’s letter from the given choices directly.')
            prompt = prompt.replace('Question: Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n', '')  # noqa: E501
        elif dataset in ['ChartQA_TEST']:
            prompt = prompt.replace('Answer the question using a single word or phrase.',
                                    'Answer the question using a single number or phrase.')
        elif dataset in ['DocVQA_VAL', 'DocVQA_TEST', ]:
            prompt = prompt.replace('Answer the question using a single word or phrase.',
                                    'Give the short answer directly.')
        elif dataset in ['TextVQA_VAL']:
            prompt = prompt.replace('Answer the question using a single word or phrase.',
                                    'When the provided information is insufficient, respond with ’Unanswerable’.'
                                    'Answer the question using a single word or phrase.')
        elif dataset in ['MTVQA_TEST']:
            prompt = prompt.replace(
                '\nAnswer the question using a word or phrase in the language of the question.', '')
        elif dataset in ['MathVista_MINI']:
            if 'Choices:' in prompt:
                prompt = prompt.replace('Choices:', 'Options:').replace('Hint:', 'Context:')
                for i in range(1, 7):  # replace A ~ F
                    prompt = prompt.replace(f'({chr(64 + i)})', f'{chr(64 + i)}.')
                prompt += '\nAnswer with the option’s letter from the given choices directly.'
            else:
                prompt += '\nAnswer the question using a single word or phrase.'
        elif dataset in ['HallusionBench']:
            prompt = prompt + " Please answer yes or no."
        return prompt, image

    def generate_inner(self, inputs, **kwargs) -> str:

        assert isinstance(inputs, str) or isinstance(inputs, list)
        pure_text = np.all([x['type'] == 'text' for x in inputs])
        assert not pure_text

        prompt, image_path = self.message_to_promptimg(inputs, kwargs['dataset'])

        try:
            response = multimodal(
                images=image_path, text=prompt, url=self.url, key=self.key, temperature=self.temperature,
                max_tokens=self.max_tokens, top_k=self.top_k, top_p=self.top_p, timeout=self.timeout)
            if kwargs['dataset'] in [
                'MMBench_DEV_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_TEST_CN_V11',
                'AI2D_TEST', 'AI2D_TEST_TO_MASK', 'MMMU_DEV_VAL', 'MMStar',
                'OCRBench', 'MMVet', 'MathVista_MINI', 'HallusionBench'
            ]:

                answer = split_think(response[0])
                answer = extract_boxed_answer(answer)
            else:
                answer = split_think(response[0])
            logger.info(f'answer : {answer}')
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                logger.error(f'{type(err)}: {err}')
                logger.error(f'The input messages are {inputs}.')
            return -1, '', ''


class BlueLM_API(BlueLMWrapper):

    def generate(self, message, dataset=None):
        return super(BlueLM_API, self).generate(message, dataset=dataset)
