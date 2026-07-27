import json
import os

import numpy as np
import requests

from vlmeval.api.base import BaseAPI
from vlmeval.smp import encode_image_to_base64, get_logger

logger = get_logger(__name__)


class CWWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'cw-congrong-v2.0',
                 retry: int = 10,
                 key: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 600,
                 api_base: str = '',
                 max_tokens: int = 2048,
                 img_detail: str = 'low',
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature

        base = os.environ.get('CW_API_BASE', None)
        self.api_base = base if base is not None else api_base

        env_key = os.environ.get('CW_API_KEY', None)
        self.key = env_key if env_key is not None else key
        assert self.key is not None, 'API key not provided. Please set CW_API_KEY environment variable or \
            pass it to the constructor.'

        assert img_detail in ['high', 'low']
        self.img_detail = img_detail

        self.vision = True
        self.timeout = timeout

        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
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
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img)
                    img_struct = dict(url=f"data:image/jpeg;base64,{b64}", detail=self.img_detail)
                    content_list.append(dict(type='image_url', image_url=img_struct))
            input_msgs.append(dict(role='user', content=content_list))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            input_msgs.append(dict(role='user', content=text))
        return input_msgs

    def generate_inner(self, inputs, **kwargs):
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        if 0 < max_tokens <= 100:
            logger.warning(
                'Less than 100 tokens left, '
                'may exceed the context window with some additional meta symbols. '
            )
        if max_tokens <= 0:
            return 0, self.fail_msg + 'Input string longer than context window. ', 'Length Exceeded. '

        headers = {'Content-Type': 'application/json', 'Authorization': f'{self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            **kwargs)
        response = requests.post(self.api_base, headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
        except Exception as err:
            if self.verbose:
                logger.error(f'{type(err)}: {err}')
                logger.error(response.text if hasattr(response, 'text') else response)

        return ret_code, answer, response
