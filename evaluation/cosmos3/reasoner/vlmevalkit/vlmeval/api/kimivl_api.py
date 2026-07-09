import json
import os

import numpy as np
import requests

from vlmeval.api.base import BaseAPI
from vlmeval.smp import encode_image_to_base64, get_logger

logger = get_logger(__name__)

APIBASES = {
    'OFFICIAL': 'http://localhost:8000/v1/chat/completions',
}


def extract_summary(text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> str:
    # 输出截断, 返回空字符串
    if bot in text and eot not in text:
        return ""
    if eot in text:
        return text[text.index(eot) + len(eot):].strip()
    return text


class KimiVLAPIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'api-kimi-vl-thinking-2506',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0.8,
                 timeout: int = 360,
                 api_base: str = 'OFFICIAL',
                 max_tokens: int = 32768,
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature

        if 'kimi' in model:
            env_key = os.environ.get('KIMI_VL_API_KEY', '')
            if key is None:
                key = env_key

        self.key = key
        self.timeout = timeout

        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if 'KIMI_VL_API_BASE' in os.environ and os.environ['KIMI_VL_API_BASE'] != '':
            logger.info('Environment variable KIMI_VL_API_BASE is set. Will use it as api_base. ')
            api_base = os.environ['KIMI_VL_API_BASE']
        else:
            api_base = 'OFFICIAL'

        print(api_base)

        assert api_base is not None

        if api_base in APIBASES:
            self.api_base = APIBASES[api_base]
        elif api_base.startswith('http'):
            self.api_base = api_base
        else:
            logger.error('Unknown API Base. ')
            raise NotImplementedError

        logger.info(f'Using API Base: {self.api_base}; API Key: {self.key}')

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    if msg["value"] == "":
                        continue
                    content_list.append(dict(type='text', text=msg['value']))

                elif msg['type'] == 'image':
                    from PIL import Image

                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img)
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}')
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
        if os.environ.get("THINKING_SKIPPED", False):
            input_msgs.append({
                "role": "assistant",
                "content": "◁think▷\n\n◁/think▷",
                "partial": True
            })
            logger.info("Add skip thinking pattern")
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            n=1,
            temperature=temperature,
            **kwargs)
        print(self.model)

        payload['max_tokens'] = max_tokens
        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
            print(answer)
            length_befofe_es = len(answer.split())
            answer = extract_summary(answer)
            length_after_es = len(answer.split())
            if length_befofe_es != length_after_es:
                logger.info("Thinking length: {}".format(length_befofe_es - length_after_es))
        except Exception as err:
            if self.verbose:
                logger.error(f'{type(err)}: {err}')
                logger.error(response.text if hasattr(response, 'text') else response)

        return ret_code, answer, response


class KimiVLAPI(KimiVLAPIWrapper):

    def generate(self, message, dataset=None):
        return super(KimiVLAPI, self).generate(message)
