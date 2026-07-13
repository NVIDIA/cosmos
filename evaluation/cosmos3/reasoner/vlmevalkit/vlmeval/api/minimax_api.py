"""
MiniMax API support for text-based LLM evaluation.
OpenAI-compatible chat completions endpoint.
Set MINIMAX_API_KEY or pass key=...

Models: MiniMax-M2.7, MiniMax-M2.5, MiniMax-M2.5-highspeed
API docs: https://platform.minimaxi.com/document/guides/chat-model/text-generation
"""
import json
import os

import requests

from vlmeval.smp.log import get_logger
from .base import BaseAPI

MINIMAX_API_BASE = "https://api.minimax.io/v1/chat/completions"
logger = get_logger(__name__)


class MiniMaxAPI(BaseAPI):
    """Text LLM API using MiniMax (OpenAI-compatible)."""

    is_api: bool = True

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        key: str = None,
        api_base: str = None,
        retry: int = 10,
        wait: int = 1,
        system_prompt: str = None,
        verbose: bool = True,
        temperature: float = 0,
        max_tokens: int = 2048,
        timeout: int = 300,
        **kwargs,
    ):
        self.model = model
        self.key = key or os.environ.get("MINIMAX_API_KEY")
        self.api_base = api_base or os.environ.get("MINIMAX_API_BASE", MINIMAX_API_BASE)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        if not self.key:
            raise ValueError(
                "MiniMax API key is required. Set MINIMAX_API_KEY or pass key=..."
            )

        super().__init__(
            retry=retry,
            wait=wait,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )

        logger.info(f"MiniMaxAPI: model={self.model}, api_base={self.api_base}")

    def _prepare_messages(self, inputs):
        """Build OpenAI-style messages from VLMEvalKit input format."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Handle multi-turn chat format
        if inputs and isinstance(inputs[0], dict) and "role" in inputs[0]:
            for item in inputs:
                content_parts = item.get("content", [])
                text = "\n".join(
                    x["value"] for x in content_parts if x["type"] == "text"
                )
                messages.append({"role": item["role"], "content": text or ""})
        else:
            # Single-turn: extract text from inputs
            text = "\n".join(
                x["value"] for x in inputs if x["type"] == "text"
            )
            messages.append({"role": "user", "content": text or ""})

        return messages

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
