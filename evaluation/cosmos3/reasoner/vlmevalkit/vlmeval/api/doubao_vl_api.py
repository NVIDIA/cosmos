import os
import os.path as osp

import numpy as np
from openai import OpenAI

from vlmeval.api.base import BaseAPI
from vlmeval.dataset import img_root_map
from vlmeval.smp import (LMUDataRoot, decode_base64_to_image_file, encode_image_to_base64,
                         get_logger, read_ok, toliststr)

logger = get_logger(__name__)


class DoubaoVLWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = '',
                 retry: int = 5,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 60,
                 max_tokens: int = 4096,
                 api_base: str = 'https://ark.cn-beijing.volces.com/api/v3',  # 使用系统推荐的服务区域地址
                 **kwargs):

        self.model = model  # This variable is unused
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.temperature = temperature
        self.max_tokens = max_tokens

        assert 'DOUBAO_VL_KEY' in os.environ, 'You may need to set the env variable DOUBAO_VL_KEY to use DOUBAO_VL.'

        key = os.environ.get('DOUBAO_VL_KEY', None)
        assert key is not None, 'Please set the environment variable DOUBAO_VL_KEY. '
        self.key = key

        assert api_base is not None, 'Please set the variable API_BASE. '
        self.api_base = api_base
        self.timeout = timeout

        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        # Models that require an EP
        # assert self.model in ['Doubao-1.5-vision-pro', 'doubao-1-5-thinking-vision-pro-250428']
        EP_KEY = 'DOUBAO_VL_ENDPOINT' + '_' + self.model.replace('.', '_').replace('-', '_').upper()
        endpoint = os.getenv(EP_KEY, None)

        if endpoint is not None:
            self.endpoint = endpoint
        else:
            logger.warning(
                f'Endpoint for model {model} is not set (can be set w. environment var {EP_KEY}. '
                f'By default, we will use the model name {model} as the EP if not set. '
            )
            self.endpoint = model

        self.client = OpenAI(
            api_key=self.key,
            base_url=self.api_base,
            timeout=self.timeout
        )

        logger.info(f'Using API Base: {self.api_base}; End Point: {self.endpoint}; API Key: {self.key}')

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

    def use_custom_prompt(self, dataset_name):
        if dataset_name == 'MathVerse_MINI_Vision_Only':
            return True
        else:
            return False

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:

        if dataset in {'MathVerse_MINI_Vision_Only'}:
            return self. _build_mathVerse_mini_vision_only_prompt(line, dataset)
        raise ValueError(f'Unsupported dataset: {dataset}')

    def _build_mathVerse_mini_vision_only_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)

        tgt_path = self.dump_image(line, dataset)

        question = line['question']

        # remove 'directly' from the prompt, so the model will answer the question in Chain-of-Thought (CoT) manner
        prompt = question.replace('directly', '', 1)

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
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
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:

        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        ret_code = -1
        answer = self.fail_msg
        response = None
        payload = dict(model=self.endpoint, messages=input_msgs, max_tokens=max_tokens, temperature=temperature)
        try:
            response = self.client.chat.completions.create(**payload)
            answer = response.choices[0].message.content.strip()
            ret_code = 0
        except Exception as err:
            logger.error(f'{type(err)}: {err}')
            logger.error(response.text if hasattr(response, 'text') else response)

        return ret_code, answer, response


class DoubaoVL(DoubaoVLWrapper):

    def generate(self, message, dataset=None):
        return super(DoubaoVL, self).generate(message)


if __name__ == '__main__':
    # export DOUBAO_VL_KEY=''
    # export DOUBAO_VL_ENDPOINT=''
    model = DoubaoVLWrapper(verbose=True)
    inputs = [
        {'type': 'image', 'value': './assets/apple.jpg'},
        {'type': 'text', 'value': '请详细描述一下这张图片。'},
    ]
    code, answer, resp = model.generate_inner(inputs)
    print(code, answer, resp)
