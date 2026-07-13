"""
Together AI API support for vision models.
OpenAI-compatible chat completions with image_url (base64 or URL).
Set TOGETHER_API_KEY or pass key=...

Requires: pip install together (optional; or use requests with api_base)
"""
import json
import os

import numpy as np
import requests

from ..smp import encode_image_to_base64, get_logger
from .base import BaseAPI

logger = get_logger(__name__)

TOGETHER_API_BASE = "https://api.together.xyz/v1/chat/completions"


class TogetherAPI(BaseAPI):
    """VLM API using Together AI (OpenAI-compatible; supports vision)."""

    is_api: bool = True

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        key: str = None,
        api_base: str = None,
        retry: int = 10,
        wait: int = 1,
        system_prompt: str = None,
        verbose: bool = True,
        temperature: float = 0,
        max_tokens: int = 2048,
        timeout: int = 300,
        img_size: int = -1,
        **kwargs,
    ):
        self.model = model
        self.key = key or os.environ.get("TOGETHER_API_KEY")
        self.api_base = api_base or os.environ.get("TOGETHER_API_BASE", TOGETHER_API_BASE)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.img_size = img_size

        if not self.key:
            raise ValueError(
                "Together API key is required. Set TOGETHER_API_KEY or pass key=..."
            )

        super().__init__(
            retry=retry,
            wait=wait,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )

        logger.info(f"TogetherAPI: model={self.model}, api_base={self.api_base}")

    def _prepare_content(self, inputs):
        """Build OpenAI-style content list (text + image_url with base64)."""
        assert all(isinstance(x, dict) for x in inputs)
        has_images = np.sum([x["type"] == "image" for x in inputs])
        if has_images:
            content_list = []
            for item in inputs:
                if item["type"] == "text":
                    content_list.append({"type": "text", "text": item["value"]})
                elif item["type"] == "image":
                    from PIL import Image

                    img = Image.open(item["value"])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
            return content_list
        text = "\n".join([x["value"] for x in inputs if x["type"] == "text"])
        return [{"type": "text", "text": text or ""}]

    def _prepare_messages(self, inputs):
        if self.system_prompt:
            out = [{"role": "system", "content": self.system_prompt}]
        else:
            out = []
        if inputs and "role" in inputs[0]:
            for item in inputs:
                out.append({
                    "role": item["role"],
                    "content": self._prepare_content(item["content"]),
                })
        else:
            out.append({"role": "user", "content": self._prepare_content(inputs)})
        return out

    def generate_inner(self, inputs, **kwargs):
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        messages = self._prepare_messages(inputs)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                self.api_base,
                headers={
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=self.timeout * 1.1,
            )
        except Exception as err:
            if self.verbose:
                logger.error(f"{type(err).__name__}: {err}")
            return -1, self.fail_msg, str(err)

        ret_code = response.status_code
        ret_code = 0 if (200 <= ret_code < 300) else ret_code
        answer = self.fail_msg

        try:
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
        except Exception as err:
            if self.verbose:
                logger.error(f"{type(err).__name__}: {err}")
                logger.error(response.text)

        return ret_code, answer, response
