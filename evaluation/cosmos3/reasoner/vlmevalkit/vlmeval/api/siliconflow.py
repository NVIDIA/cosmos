import base64
import io
import json
import math
import os
import os.path as osp

import requests
from PIL import Image

from vlmeval.api.base import BaseAPI
from vlmeval.dataset import img_root_map
from vlmeval.smp import LMUDataRoot, decode_base64_to_image_file, get_logger, read_ok, toliststr

logger = get_logger(__name__)

API_BASE = "https://api.siliconflow.cn/v1/chat/completions"


def resize_image(image: Image.Image, max_height: int, max_width: int) -> Image.Image:
    width, height = image.size
    if min(width, height) < 50:
        scale = 50 / min(width, height)
        image = image.resize((int(width * scale), int(height * scale)))
    current_pixels = width * height

    if current_pixels <= max_height * max_width:
        return image

    scale = math.sqrt(max_height * max_width / current_pixels)
    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def encode_image(path: str, max_height: int = 1024, max_width: int = 1024) -> str:
    image = Image.open(path).convert("RGB")
    image = resize_image(image, max_height, max_width)
    width, height = image.size
    if min(height, width) < 50:
        scale = 50 / min(width, height)
        image = image.resize((int(width * scale), int(height * scale)))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


class SiliconFlowAPI(BaseAPI):

    is_api: bool = True

    def __init__(
        self,
        model: str = "deepseek-ai/DeepSeek-V2.5",
        retry: int = 5,
        key: str = None,
        api_base: str = API_BASE,
        verbose: bool = True,
        system_prompt: str = None,
        timeout: int = 60,
        reasoning: bool = False,  # If set, will return results in the format of {'content': '...', 'reasoning': '...'}
        **kwargs,
    ):

        self.model = model
        self.api_base = api_base
        self.reasoning = reasoning
        self.timeout = timeout

        default_kwargs = {
            "stream": False,
            "temperature": 0,
            "n": 1,
            "max_tokens": 1280,
        }
        for k, v in default_kwargs.items():
            if k not in kwargs:
                kwargs[k] = default_kwargs[k]
        if key is not None:
            self.key = key
        else:
            self.key = os.environ.get("SiliconFlow_API_KEY", "")
        headers = {"Authorization": "Bearer {}", "Content-Type": "application/json"}
        headers["Authorization"] = headers["Authorization"].format(self.key)
        self.headers = headers
        super().__init__(
            retry=retry,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )

    @staticmethod
    def build_msgs(msgs_raw):
        messages = []
        message = {"role": "user", "content": []}
        image_b64 = None
        for msg in msgs_raw:
            if msg["type"] == "image" and not image_b64:
                image_b64 = encode_image(msg["value"])
                message["content"].append({
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    },
                    "type": "image_url"
                })
            elif msg["type"] == "text":
                message["content"].append({"text": msg["value"], "type": "text"})

        messages.append(message)
        return messages

    def generate_inner(self, inputs, **kwargs) -> str:
        default_kwargs = self.default_kwargs
        default_kwargs.update(kwargs)

        payload = dict(
            model=self.model,
            messages=self.build_msgs(msgs_raw=inputs),
            **default_kwargs,
        )

        response = requests.post(
            self.api_base, headers=self.headers, data=json.dumps(payload), timeout=self.timeout * 1.1
        )
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            msg = resp_struct["choices"][0]["message"]
            if self.reasoning and 'reasoning_content' in msg:
                answer = {'content': msg['content'], 'reasoning': msg['reasoning_content']}
            else:
                answer = resp_struct["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        return ret_code, answer, response


class TeleMMAPI(SiliconFlowAPI):

    is_api: bool = True

    def __init__(
        self,
        model: str = "TeleAI/TeleMM",
        key: str = None,
        max_height: int = 1280,
        max_width: int = 784,
        **kwargs,
    ):
        super().__init__(model=model, key=key, **kwargs)
        self.max_height = max_height
        self.max_width = max_width

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
        # img_root = osp.join(ROOT, 'images', img_root_map[dataset] if dataset in img_root_map else dataset)
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

    def _prepare_content(
        self, inputs: list[dict[str, str]], dataset: str = None
    ) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        has_image = False
        for s in inputs:
            if s["type"] == "image":
                if not has_image:
                    item = {
                        "type": "image_url",
                        "image_url": {
                            "url": encode_image(
                                s["value"],
                                max_height=self.max_height,
                                max_width=self.max_width,
                            )
                        },
                    }
                    has_image = True
                else:
                    continue
            elif s["type"] == "text":
                prompt = s["value"]
                if len(prompt) == 0:
                    continue
                if dataset == "HallusionBench":
                    prompt += " Please answer yes or no directly, without any unnecessary explanation."
                elif dataset == "OCRBench":
                    prompt = (
                        prompt + "\nExtract the text from the image intactly and "
                        + "answer the question concisely and clearly if possible."
                    )

                elif (
                    dataset == "AI2D_TEST"
                    or dataset == "MMStar"
                    or dataset == "MMBench_TEST_EN_V11"
                    or dataset == "MMVet"
                ):
                    prompt = prompt.replace(
                        "Please select the correct answer from the options above. \n",
                        "Please select the correct option from the above choices based on the "
                        + "input image and question. The final output should only be one option, such as 'A'",
                    )
                elif dataset == "MMBench_TEST_CN_V11":
                    prompt = prompt.replace(
                        "Please select the correct answer from the options above. \n",
                        "请根据输入图像和问题从上述选项中选择正确选项，最终的输出只有一个选项，例如'A'",
                    )
                item = {"type": "text", "text": prompt}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)

        return content

    def generate_inner(self, inputs, **kwargs) -> str:
        default_kwargs = self.default_kwargs
        default_kwargs.update(kwargs)

        messages = []
        messages.append(
            {
                "role": "user",
                "content": self._prepare_content(
                    inputs, dataset=kwargs.get("dataset", None)
                ),
            }
        )

        payload = dict(model=self.model, messages=messages, **default_kwargs)

        response = requests.post(
            self.api_base, headers=self.headers, data=json.dumps(payload)
        )
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct["choices"][0]["message"]["content"].strip()
            return ret_code, answer, response
        except Exception as err:
            import traceback

            traceback.print_exc()
            if self.verbose:
                logger.error(f"{type(err)}: {err}")
                logger.error(f"The input messages are {inputs}.")
            return -1, "", ""
