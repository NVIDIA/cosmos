import json
import os

import requests

from vlmeval.api.base import BaseAPI
from vlmeval.smp import encode_image_file_to_base64, get_logger

url = 'https://api.stepfun.com/v1/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer {}',
}


logger = get_logger(__name__)


class StepAPI_INT(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'step-1v-8k',
                 retry: int = 10,
                 key: str = None,
                 temperature: float = 0,
                 max_tokens: int = 300,
                 verbose: bool = True,
                 system_prompt: str = None,
                 **kwargs):
        self.model = model
        self.fail_msg = 'Fail to obtain answer via API.'
        self.headers = headers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        if key is not None:
            self.key = key
        else:
            self.key = os.environ.get('STEPAI_API_KEY', '')
        headers['Authorization'] = headers['Authorization'].format(self.key)

        super().__init__(retry=retry, verbose=verbose, system_prompt=system_prompt, **kwargs)

    @staticmethod
    def build_msgs(msgs_raw):
        messages = []
        message = {'role': 'user', 'content': []}

        for msg in msgs_raw:
            if msg['type'] == 'image':
                image_b64 = encode_image_file_to_base64(msg['value'])
                message['content'].append({
                    'image_url': {'url': 'data:image/webp;base64,%s' % (image_b64)},
                    'type': 'image_url'
                })
            elif msg['type'] == 'text':
                message['content'].append({
                    'text': msg['value'],
                    'type': 'text'
                })

        messages.append(message)
        return messages

    def generate_inner(self, inputs, **kwargs) -> str:
        print(inputs, '\n')
        payload = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=self.build_msgs(msgs_raw=inputs),
            **kwargs)
        response = requests.post(url, headers=headers, data=json.dumps(payload))
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


class Step1V_INT(StepAPI_INT):

    def generate(self, message, dataset=None):
        return super(StepAPI_INT, self).generate(message)
