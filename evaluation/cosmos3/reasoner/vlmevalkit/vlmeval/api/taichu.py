import base64
import copy
import json
import os
import re
import string
from io import BytesIO

import pandas as pd
import requests
from PIL import Image

from vlmeval.api.base import BaseAPI
from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import get_logger, listinstr

logger = get_logger(__name__)


class ChatResponse(dict):
    def __getattr__(self, name):
        value = self.get(name)
        if isinstance(value, dict):
            return ChatResponse(value)  # 如果值是字典，递归包装成 DotDict
        elif isinstance(value, list):
            return [ChatResponse(v) if isinstance(v, dict) else v for v in value]  # 如果值是列表，处理其中的字典
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


class TaichuVLWrapper(BaseAPI):
    is_api: bool = True

    def __init__(self,
                 model: str = 'Taichu-VL-2B',
                 retry: int = 5,
                 verbose: bool = True,
                 temperature: float = 0.0,
                 system_prompt: str = None,
                 max_tokens: int = 4096,
                 key: str = None,
                 url: str = None,
                 **kwargs):

        self.model = model
        self.kwargs = kwargs
        self.max_tokens = max_tokens

        self.system_prompt = '[sys]You are a helpful assistant.[/sys]'
        self.hint_prompt = '|<Hint>|'
        self.mcq_prompt = '|<MCQ>|'

        self.datasets_use_system = ['MMVet']
        self.datasets_use_multichoice = [
            'MathVista', 'MathVision']

        openai_key = os.environ.get('OPENAI_API_KEY', None)
        use_openai = os.environ.get('USE_OPENAI_EVAL', True)
        self.use_openai_evaluate = (isinstance(openai_key, str) and openai_key.startswith('sk-') and use_openai)

        self.api_key = os.environ.get('TAICHU_API_KEY', key)
        self.api_url = url

        assert self.api_key is not None, 'Please set the API Key'

        super().__init__(retry=retry, system_prompt=self.system_prompt, verbose=verbose, **kwargs)

    def use_custom_prompt(self, dataset):
        if listinstr(['MCQ', 'VQA'], DATASET_TYPE(dataset)):
            return True
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            return True
        return False

    def clear_prompt(self, prompt):
        prompt = re.sub(r"Hint:.*?Question:", "", prompt, flags=re.S).strip()
        prompt = re.sub(r"\nChoices:\n.*", "", prompt, flags=re.S).strip()
        return prompt

    def encode_image(self, pil_image):
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_str

    def build_prompt(self, line, dataset=None):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        hint = None
        if listinstr(self.datasets_use_system, dataset):
            system_prompt = self.system_prompt
        else:
            system_prompt = ''
        mcq = False
        if DATASET_TYPE(dataset) == 'MCQ' or listinstr(self.datasets_use_multichoice, dataset):
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            if listinstr(self.datasets_use_multichoice, dataset):
                options = {}
                if not pd.isna(line['choices']):
                    for i, c in enumerate(eval(line['choices'])):
                        options[string.ascii_uppercase[i]] = c
                question = self.clear_prompt(question)

            # support chinese
            if listinstr(['_CN', '_cn'], dataset):
                options_prompt = '\n选项：\n'
            else:
                options_prompt = '\nOPTIONS:\n'
            options_prompt += '\n'.join(f"{key}:{value}" for key, value in options.items())
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            mcq = True if len(options) else False
            if len(options):
                prompt = question + options_prompt
            else:
                prompt = question
        else:
            prompt = question

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))

        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs.append(dict(type='image', value=tgt_path))

        if hint:
            prompt = 'Hint: ' + hint + '\n' + prompt
        msgs.append(dict(type='text', value=prompt))

        if mcq:
            msgs.append(dict(type='text', value=self.mcq_prompt))
        return msgs

    def prompt_to_request_messages(self, inputs):

        messages = [
            {'role': 'user', 'content': []}
        ]
        is_mcq = False
        for x in inputs:
            if x['type'] == 'text':
                if x['value'] == self.system_prompt:
                    messages = [{'role': 'system', 'content': [{"type": "text", "text": x['value']}]}] + messages
                elif self.mcq_prompt == x['value']:
                    is_mcq = True
                else:
                    messages[-1]['content'].append(
                        {"type": "text", "text": x['value']},
                    )
            if x['type'] == 'image':
                _url = self.encode_image(Image.open(x['value']))
                messages[-1]['content'].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_url}"}},
                )
            else:
                continue

        return messages, is_mcq

    def generate_inner(self, inputs, **kwargs) -> str:
        messages, is_mcq = self.prompt_to_request_messages(inputs)

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "top_p": 0.8,
            "stream": False,
            "extra_body": {
                "repetition_penalty": 1
            }
        }

        headers = {
            'Authorization': self.api_key,
            'Content-Type': 'application/json'
        }

        try:
            chat_response = requests.post(self.api_url, json=data, headers=headers)
            response = ChatResponse(json.loads(chat_response.content))
            result = response.choices[0].message.content
            # Extract index to exact matching when ChatGPT is unavailable.
            if self.use_openai_evaluate is False and is_mcq is True:
                try:
                    result = result[0]
                except Exception:
                    result = 'A'
            return 0, result, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                logger.error(f'{type(err)}: {err}')
                logger.error(f'The input messages are {inputs}.')
            return -1, '', ''


class TaichuVLAPI(TaichuVLWrapper):

    def generate(self, message, dataset=None):
        return super(TaichuVLAPI, self).generate(message, dataset=dataset)


class TaichuVLRWrapper(BaseAPI):
    is_api: bool = True

    def __init__(self,
                 model: str = 'taichu_vlr_3b',
                 retry: int = 5,
                 verbose: bool = True,
                 temperature: float = 0.0,
                 system_prompt: str = None,
                 max_tokens: int = 4096,
                 use_reasoning_prompt: bool = True,
                 post_process: bool = True,
                 key: str = None,
                 url: str = None,
                 **kwargs):

        self.model = model
        self.kwargs = kwargs
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.use_reasoning_prompt = use_reasoning_prompt
        self.post_process = post_process
        self.verbose = verbose

        openai_key = os.environ.get('OPENAI_API_KEY', None)
        use_openai = os.environ.get('USE_OPENAI_EVAL', True)
        self.use_openai_evaluate = (isinstance(openai_key, str) and openai_key.startswith('sk-') and use_openai)

        self.api_key = os.environ.get('TAICHU_API_KEY', key)
        self.api_url = url

        assert self.api_key is not None, 'Please set the API Key'

        super().__init__(retry=retry, system_prompt=self.system_prompt, verbose=verbose, **kwargs)

    def use_custom_prompt(self, dataset):
        return False

    def encode_image(self, pil_image):
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_str

    def post_process_func(self, response):
        resp = response.split('\\boxed{')[-1]
        lt = len(resp)
        counter, end = 1, None
        for i in range(lt):
            if resp[i] == '{':
                counter += 1
            elif resp[i] == '}':
                counter -= 1
            if counter == 0:
                end = i
                break
            elif i == lt - 1:
                end = lt
                break
        if end is not None:
            response = resp[:end]
        return response

    def prompt_to_request_messages(self, inputs):

        messages = [
            {'role': 'user', 'content': []}
        ]
        for x in inputs:
            if x['type'] == 'text':
                messages[-1]['content'].append(
                    {"type": "text", "text": x['value']},
                )
            if x['type'] == 'image':
                _url = self.encode_image(Image.open(x['value']))
                messages[-1]['content'].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_url}"}},
                )
            else:
                continue

        PROMPT = (
            "First thinks about the reasoning process in the mind and then provides the user with the answer. "
            "Put your final answer within \\boxed{}. "
            "The response of reasoning and answer are formatted in <think> reasoning </think><answer> \\boxed{answer here} </answer>.\n"  # noqa: E501
        )

        if self.use_reasoning_prompt:
            for content in messages[0]['content']:
                if content['type'] == 'text':
                    content['text'] = PROMPT + content['text']
                    break

        return messages

    def generate_inner(self, inputs, **kwargs) -> str:
        messages = self.prompt_to_request_messages(inputs)
        if self.verbose:
            verbose_messages = copy.deepcopy(messages)
            for mess in verbose_messages:
                if mess['role'] == 'user':
                    for content in mess['content']:
                        if content['type'] == 'image_url':
                            content['image_url']['url'] = ''
            print(f'\033[31m{verbose_messages}\033[0m')

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "top_p": 0.8,
            "stream": False,
            "repetition_penalty": 1.0
        }

        headers = {
            'Authorization': f"Bearer {self.api_key}",
            'Content-Type': 'application/json'
        }

        try:
            chat_response = requests.post(self.api_url, json=data, headers=headers)
            response = ChatResponse(json.loads(chat_response.content))
            result = response.choices[0].message.content
            if self.post_process:
                result = self.post_process_func(result)
            if self.verbose:
                print(f'\033[32m{result}\033[0m')

            return 0, result, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                logger.error(f'{type(err)}: {err}')
                logger.error(f'The input messages are {inputs}.')
            return -1, '', ''


class TaichuVLRAPI(TaichuVLRWrapper):

    def generate(self, message, dataset=None):
        return super(TaichuVLRAPI, self).generate(message, dataset=dataset)
