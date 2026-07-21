import os
import string
import time
from typing import Optional

import pandas as pd
import requests

from vlmeval.api.base import BaseAPI
from vlmeval.dataset import DATASET_TYPE, img_root_map
from vlmeval.smp import (LMUDataRoot, cn_string, decode_base64_to_image_file, get_logger,
                         listinstr, osp, read_ok, toliststr)

logger = get_logger(__name__)


class SenseChatVisionWrapper(BaseAPI):
    is_api: bool = True

    def __init__(
        self,
        base_url: str = "https://api.sensenova.cn/v1/llm/chat-completions",
        api_key: str = None,
        model: str = "SenseNova-V6-5-Pro",
        retry: int = 5,
        wait: int = 5,
        verbose: bool = True,
        system_prompt: str = None,
        max_tokens: int = 16384,
        **kwargs,
    ):
        self.base_url = base_url
        self.model = model
        self.fail_msg = "Failed to obtain answer via API. "
        self.api_key = os.getenv("SENSENOVA_API_KEY", api_key)
        assert self.api_key is not None, (
            "Please set the `SENSENOVA_API_KEY` environment variable or pass `api_key` in the config.json."
        )
        self.max_new_tokens = max_tokens
        self.thinking = False
        super().__init__(
            wait=wait,
            retry=retry,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )

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
        img_root = osp.join(ROOT, "images", img_root_map(dataset))
        os.makedirs(img_root, exist_ok=True)
        if "image" in line:
            if isinstance(line["image"], list):
                tgt_path = []
                assert "image_path" in line
                for img, im_name in zip(line["image"], line["image_path"]):
                    path = osp.join(img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = osp.join(img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line["image"], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert "image_path" in line
            tgt_path = toliststr(line["image_path"])

        return tgt_path

    def image_to_base64(self, image_path):
        import base64

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode("utf-8")

    def use_custom_prompt(self, *args, **kwargs):
        """Check if the prompt is customized."""
        return True

    def build_multi_choice_prompt(self, line, dataset=None):
        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )

        return prompt

    def build_mcq_cot_prompt(self, line, prompt):
        question = line["question"]
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = {
            'multiple-choice': "You are an expert in {}. Please solve the university-level {} examination question, which includes interleaved images and text. Answer the preceding multiple choice question. The last line of your response should follow this format: 'Answer: \\boxed LETTER', where LETTER is one of the options. If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Think step by step logically, considering all relevant information before answering.",  # noqa: E501
            'open': 'You are an expert in {}. Please solve the university-level {} examination question, which includes interleaved images and text. Your output should be divided into two parts: First, reason about the correct answer. Then write the answer in the following format where X is only the answer and nothing else: "ANSWER: X"'  # noqa: E501
        }
        subject = '_'.join(line['id'].split('_')[1:-1])
        prompt = prompt[line['question_type']].format(subject, subject) + '\n' + question
        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)

        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and listinstr(["MME"], dataset):
            question = line["question"]
            prompt = question + " Answer the question using a single word or phrase."
        elif dataset is not None and listinstr(["HallusionBench"], dataset):
            question = line["question"]
            prompt = (
                question
                + " Please answer yes or no. Answer the question using a single word or phrase."
            )
        elif dataset is not None and DATASET_TYPE(dataset) == "MCQ":
            prompt = self.build_multi_choice_prompt(line, dataset)
            if "MMMU" in dataset:
                prompt = self.build_mcq_cot_prompt(line, prompt)
                self.thinking = True
        elif dataset is not None and DATASET_TYPE(dataset) == "VQA":
            if "MathVista" in dataset:
                prompt = line["question"]
                self.thinking = True
            elif listinstr(["LLaVABench"], dataset):
                question = line["question"]
                prompt = question + "\nAnswer this question in detail."
            elif listinstr(["MMVet"], dataset):
                prompt = line["question"]
            else:
                question = line["question"]
                prompt = (
                    question
                    + "\nPlease reason step by step, and put your final answer within \\boxed{}."
                )
        else:
            prompt = line["question"]

        message = [dict(type="text", value=prompt)]
        message.extend([dict(type="image", value=s) for s in tgt_path])

        return message

    def message_to_promptimg(self, message, dataset=None):
        if dataset is None or listinstr(["MMMU", "BLINK"], dataset):
            prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])
            image = [[x["value"] for x in message if x["type"] == "image"][0]]
        else:
            prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])
            image = [x["value"] for x in message if x["type"] == "image"]
        return prompt, image

    def set_max_num(self, dataset: Optional[str] = None) -> None:
        """Set the max_num based on the dataset."""
        if dataset is not None and listinstr(
            [
                "ChartQA_TEST",
                "MMMU_DEV_VAL",
                "MMMU_TEST",
                "MME-RealWorld",
                "VCR_EN",
                "VCR_ZH",
                "OCRVQA",
            ],
            dataset,
        ):
            self.max_num = 12
        elif dataset is not None and listinstr(
            ["DocVQA_VAL", "DocVQA_TEST", "DUDE", "MMLongBench_DOC", "SLIDEVQA"],
            dataset,
        ):
            self.max_num = 18
        elif dataset is not None and listinstr(
            ["InfoVQA_VAL", "InfoVQA_TEST", "OCRBench", "HRBench4K", "HRBench8K"],
            dataset,
        ):
            self.max_num = 24
        else:
            self.max_num = 6

    def generate_inner(self, inputs, **kwargs) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)
        inputs = [inputs] if isinstance(inputs, str) else inputs
        dataset = kwargs.get("dataset", None)

        self.set_max_num(dataset=dataset)

        prompt, image = self.message_to_promptimg(message=inputs, dataset=dataset)
        content = [
            {
                "image_base64": self.image_to_base64(item),
                "type": "image_base64",
            }
            for item in image
        ]

        content.append(
            {
                "text": prompt,
                "type": "text",
            }
        )

        message = [{"content": content, "role": "user"}]
        data = {
            "messages": message,
            "max_new_tokens": self.max_new_tokens,
            "model": self.model,
            "stream": False,
            "image_split_count": self.max_num,
            "thinking": {
                "enabled": self.thinking,
            }
        }

        headers = {
            "Content-type": "application/json",
            "Authorization": self.api_key,
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            json=data,
        )
        request_id = response.headers.get("x-request-id", "")
        logger.info(f"Request-id: {request_id}")

        time.sleep(1)
        try:
            assert response.status_code == 200
            response = response.json()["data"]["choices"][0]["message"].strip()
            if self.verbose:
                logger.info(f"inputs: {inputs}\nanswer: {response}")
            return 0, response, "Succeeded! "
        except Exception as err:
            if self.verbose:
                logger.error(
                    "---------------------------ERROR---------------------------"
                )
                logger.error(response.json())
                logger.error(err)
                logger.error(
                    "---------------------------request_id---------------------------"
                    + request_id
                )
                logger.error(
                    "api error"
                    + response.json()["error"]["message"]
                    + str(
                        [
                            input["value"] if input["type"] == "image" else None
                            for input in inputs
                        ]
                    )
                )
                logger.error(f"The input messages are {inputs}.")
            return -1, response.json()["error"]["message"], ""


class SenseChatVisionAPI(SenseChatVisionWrapper):
    def generate(self, message, dataset=None):
        return super(SenseChatVisionAPI, self).generate(message, dataset=dataset)


class SenseChatVisionV2API(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'SenseNova-V6-5-Pro-20251215',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 300,
                 api_base: str = "https://api.sensenova.cn/compatible-mode/v2/chat/completions",
                 max_completion_tokens: int = 4096,
                 img_size: int = -1,
                 **kwargs):

        self.model = model
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.api_base = api_base
        self.key = key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        self.timeout = timeout
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        logger.info(f'Using API Base: {self.api_base}; API Key: {self.key}')

    def generate(self, message, dataset=None):
        return super(SenseChatVisionV2API, self).generate(message)

    def prepare_itlist(self, inputs):
        import numpy as np

        from vlmeval.smp import encode_image_to_base64

        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        image_num = len([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img, target_size=int(self.img_size / (image_num ** 0.5)))
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
        assert all(['type' in x for x in inputs]) or all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        import json
        input_msgs = self.prepare_inputs(inputs)

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}

        payload = dict(model=self.model, messages=input_msgs, stream=False, **kwargs)

        proxies = {}
        if os.getenv('http_proxy'):
            proxies['http'] = os.getenv('http_proxy')
        if os.getenv('https_proxy'):
            proxies['https'] = os.getenv('https_proxy')
        proxies = proxies or None

        response = requests.post(
            self.api_base,
            headers=headers,
            data=json.dumps(payload),
            proxies=proxies,
            timeout=self.timeout * 1.1,
        )
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
