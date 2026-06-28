# flake8: noqa
import base64
import json
import os
import os.path as osp
import string

import pandas as pd
import requests

from vlmeval.api.base import BaseAPI
from vlmeval.dataset import DATASET_TYPE, img_root_map
from vlmeval.smp import (LMUDataRoot, cn_string, concat_images_vlmeval,
                         decode_base64_to_image_file, get_logger, listinstr, read_ok, toliststr)

API_ENDPOINT = "https://hl.jiutian.10086.cn/kunlun/ingress/api/hl-4a9c15/7b11a3451e1a4612a6661c3e22235df6/ai-e7d64e71b61e421b8a41c0d43424d73a/service-cd5a067331e34cfea25f5a9d7960ffe8//v1/chat/completions"
API_ENDPOINT_2B = 'https://hl.jiutian.10086.cn/kunlun/ingress/api/hl-4a9c15/7b11a3451e1a4612a6661c3e22235df6/ai-3101961c75eb47ad8cc8d8ebb62fb8b0/service-c0bf9bac00824ace8639c0e8e0a4a5da/v1/chat/completions'
APP_CODE = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiI1YjYyODY0ZjZmMWI0Yzg4YWE2ZDk1NzBhNDY1MWI3OSIsImlzcyI6ImFwaS1hdXRoLWtleSIsImV4cCI6NDg5ODU1NzAwN30.jLUbctPdJ74VC3Vwlr0nB4x9N2QxWtSGYE0vWsceZN-agDecVnH8pH8Q5SoCQ3-SBYx5jDx-UOg3kkoMqY9CdxiALauKU_UZ56CV2NKCcHUVeJIgNvfQJMb0z6yCCbSe80e1T8FxrxQXDvubyWtl4pTAhixYaEUqNG8rjUrDuA-vRgZ1e7HilBmU487OI76D9LUnU-zEdMWhzsCkh_Yy3M1Ur4PsKgMFi5QSmMuGSUGJjkpJHiGNx1QcevBLQSOCL2jvg15ifB2n2dD6zb8iPXFkfQTtmvbZofxWACSvkri-x9V3gFWg7DODwKUZsyyogPzRJVbmxDGruMsgiiCsPg"


logger = get_logger(__name__)

class JTVLChatWrapper(BaseAPI):
    is_api: bool = True
    INTERLEAVE = False

    def __init__(self,
                 model: str = 'jt-vl-chat-mini',
                 retry: int = 5,
                 wait: int = 5,
                 api_base: str = '',
                 app_code: str = '',
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0.7,
                 max_tokens: int = 2048,
                 proxy: str = None,
                 **kwargs):
        self.model = model

        self.temperature = temperature
        self.max_tokens = max_tokens
        if model == 'jt-vl-chat-mini':
            self.api_base = API_ENDPOINT
        else:
            self.api_base = API_ENDPOINT_2B
        self.app_code = APP_CODE

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def dump_image(self, line, dataset):
        """Dump the image(s) of the input line to the corresponding dataset folder.

        Args:
            line (line of pd.DataFrame): The raw input line.
            dataset (str): The name of the dataset.

        Returns:
            str | list[str]: The paths of the dumped images.
        """
        ROOT = LMUDataRoot()
        assert isinstance(dataset, str)

        img_root = os.path.join(ROOT, 'images', img_root_map(dataset) if dataset in img_root_map(dataset) else dataset)
        os.makedirs(img_root, exist_ok=True)
        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                assert 'image_path' in line
                for img, im_name in zip(line['image'], line['image_path']):
                    path = osp.join(img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = osp.join(img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])

        return tgt_path

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMMU_DEV_VAL', 'MMMU_TEST'], dataset):
            return False
        else:
            return True

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
            elif listinstr(['MMVet'], dataset):
                prompt = line['question']
            else:
                question = line['question']
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def message_to_promptimg(self, message, dataset=None):
        assert not self.INTERLEAVE
        model_name = self.__class__.__name__
        import warnings
        warnings.warn(
            f'Model {model_name} does not support interleaved input. '
            'Will use the first image and aggregated texts as prompt. ')
        num_images = len([x for x in message if x['type'] == 'image'])
        if num_images == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = None
        else:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            if dataset == 'BLINK':
                image = concat_images_vlmeval(
                    [x['value'] for x in message if x['type'] == 'image'],
                    target_size=512)
            else:
                image = [x['value'] for x in message if x['type'] == 'image'][0]
        return prompt, image

    def get_send_data(self, prompt, image_path, temperature, max_tokens, stream=False):
        image = ''
        with open(image_path, 'rb') as f:
            image = str(base64.b64encode(f.read()), 'utf-8')
        send_data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "image_base64": image,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": False,
            "stream": stream
        }
        return send_data

    def get_send_data_no_image(self, prompt, temperature, max_tokens, stream=False):
        send_data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        return send_data

    def generate_inner(self, inputs, **kwargs) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)
        inputs = [inputs] if isinstance(inputs, str) else inputs
        dataset = kwargs.get('dataset', None)
        prompt, image_path = self.message_to_promptimg(message=inputs, dataset=dataset)
        # print("prompt:",prompt)
        if image_path:
            send_data = self.get_send_data(
                prompt=prompt,
                image_path=image_path,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True)
        else:
            send_data = self.get_send_data_no_image(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True)

        json_data = json.dumps(send_data)

        header_dict = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.app_code}

        r = requests.post(self.api_base, headers=header_dict, data=json_data, timeout=3000, stream=True)
        try:
            if send_data.get('stream', False):
                # 流式处理
                chunks = []
                full_content = ""

                try:
                    for line in r.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith('data: '):
                                event_data = decoded_line[6:]
                                if event_data == '[DONE]':
                                    break
                                try:
                                    chunk = json.loads(event_data)
                                    chunks.append(chunk)

                                    # 记录最后一个有效的usage（不累加）
                                    if 'usage' in chunk:
                                        _ = chunk['usage']

                                    # 实时输出内容
                                    if 'choices' in chunk:
                                        for choice in chunk['choices']:
                                            if 'delta' in choice and 'content' in choice['delta']:
                                                content = choice['delta']['content']
                                                print(content, end='', flush=True)
                                                full_content += content
                                except json.JSONDecodeError:
                                    continue
                    print("\n")  # 换行

                    return 0, full_content, 'Succeeded! '

                except Exception as e:
                    return -1, f'Error: {str(e)}', ''
            else:
                # 非流式处理
                try:
                    r_json = r.json()
                    output = r_json['choices'][0]['message']['content']
                    return 0, output, 'Succeeded! '
                except:
                    error_msg = f'Error! code {r.status_code} content: {r.content}'
                    error_con = r.content.decode('utf-8')
                    if self.verbose:
                        logger.error(error_msg)
                        logger.error(error_con)
                        logger.error(f'The input messages are {inputs}.')
                    return -1, error_msg, ''
        except Exception as e:
            return -1, f'Error: {str(e)}', ''


class JTVLChatAPI_Mini(JTVLChatWrapper):

    def generate(self, message, dataset=None):
        return super(JTVLChatAPI_Mini, self).generate(message, dataset=dataset)


class JTVLChatAPI_2B(JTVLChatWrapper):

    def generate(self, message, dataset=None):
        return super(JTVLChatAPI_2B, self).generate(message, dataset=dataset)
