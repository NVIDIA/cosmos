import copy as cp
import os
from http import HTTPStatus

from vlmeval.api.base import BaseAPI
from vlmeval.smp import proxy_set


# Note: This is a pure language model API.
class QwenAPI(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'qwen-max-1201',
                 retry: int = 5,
                 verbose: bool = True,
                 seed: int = 2680,
                 temperature: float = 0.0,
                 system_prompt: str = None,
                 key: str = None,
                 max_tokens: int = 2048,
                 proxy: str = None,
                 **kwargs):

        assert model in ['qwen-turbo', 'qwen-plus', 'qwen-max', 'qwen-max-1201', 'qwen-max-longcontext']
        self.model = model
        import dashscope
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        if key is None:
            key = os.environ.get('DASHSCOPE_API_KEY', None)
        assert key is not None, (
            'Please set the API Key (obtain it here: '
            'https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start)'
        )
        dashscope.api_key = key
        if proxy is not None:
            proxy_set(proxy)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    @staticmethod
    def build_msgs(msgs_raw, system_prompt=None):
        msgs = cp.deepcopy(msgs_raw)
        ret = []
        if system_prompt is not None:
            ret.append(dict(role='system', content=system_prompt))
        for i, msg in enumerate(msgs):
            role = 'user' if i % 2 == 0 else 'assistant'
            ret.append(dict(role=role, content=msg))
        return ret

    def generate_inner(self, inputs, **kwargs) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)
        inputs = [inputs] if isinstance(inputs, str) else inputs
        messages = self.build_msgs(msgs_raw=inputs, system_prompt=self.system_prompt)

        import dashscope
        response = dashscope.Generation.call(
            model=self.model,
            messages=messages,
            seed=self.seed,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            result_format='message',  # set the result to be "message" format.
        )
        if response.status_code != HTTPStatus.OK:
            return -1, 'Error: Bad Response Statuse Code. ', f'The response status code is {response.status_code}. '

        try:
            return 0, response['output']['choices'][0]['message']['content'].strip(), 'Succeeded! '
        except Exception as err:
            return -1, f'Error: Failed to parse the response. {err}', response
