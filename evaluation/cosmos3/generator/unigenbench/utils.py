# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
# -----------------------------------------------------------------------------

import asyncio
import atexit
import base64
import json
import os
import random
import tempfile
import threading
from typing import Any, Optional
from urllib.parse import urlparse

from loguru import logger
from tqdm import tqdm
from openai import AsyncOpenAI


def image_bytes_to_data_url(image_bytes: bytes, image_fmt: str = "webp") -> str:
    mime_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    mime_type = mime_types.get(image_fmt.lower(), f"image/{image_fmt}")
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"


class GatewayVLM:
    def __init__(
        self,
        gateway_url: str,
        gateway_api: str,
        model_str: str,
        num_concurrency: int = 4,
        num_max_retry: int = 1,
        timeout: int = 100,
    ):
        self.client = AsyncOpenAI(api_key=gateway_api, base_url=gateway_url)
        self.model_str = model_str
        self.num_concurrency = num_concurrency
        self.num_max_retry = num_max_retry
        self.timeout = timeout
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    async def query_core(self, request: dict) -> str:
        for attempt in range(self.num_max_retry):
            try:

                async def stream_response():
                    stream = await self.client.chat.completions.create(**request)
                    result = ""
                    async for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            result += chunk.choices[0].delta.content
                    return result

                return await asyncio.wait_for(stream_response(), timeout=self.timeout)
            except KeyboardInterrupt:
                raise
            except asyncio.TimeoutError:
                if attempt == 0 or attempt == self.num_max_retry - 1:
                    logger.warning(
                        f"VLM API for {request['model']} timeout (attempt {attempt + 1}/{self.num_max_retry}): exceeded {self.timeout}s"
                    )
            except Exception as e:
                if attempt == 0 or attempt == self.num_max_retry - 1:
                    logger.warning(
                        f"VLM API for {request['model']} error (attempt {attempt + 1}/{self.num_max_retry}): {type(e).__name__}: {e}"
                    )
        logger.warning("VLM query failed after max retries")
        return ""

    def query(self, request_list: list[dict[str, Any]], pbar_desc: Optional[str] = None) -> list[str]:
        pbar = tqdm(total=len(request_list), desc=pbar_desc, disable=pbar_desc is None)

        async def coroutine_gather() -> list[str]:
            sem = asyncio.Semaphore(self.num_concurrency)

            async def process_one(req: dict[str, Any]) -> str:
                async with sem:
                    result = await self.query_core(req)
                    pbar.update(1)
                    return result

            return await asyncio.gather(*(process_one(r) for r in request_list))

        future = asyncio.run_coroutine_threadsafe(coroutine_gather(), self.loop)
        result = future.result()
        pbar.close()
        return result

    def build_request(
        self,
        system_prompt: str,
        content_prompt: str,
        image_bytes: Optional[bytes] = None,
        image_fmt: str = "webp",
        temperature: float = 0.0,
        max_tokens: int = 32768,
    ) -> dict[str, Any]:
        request = {
            "model": self.model_str,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content_prompt},
                    ],
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        extra_session_list = []
        if image_bytes is not None:
            extra_session_list.append(
                {"type": "image_url", "image_url": {"url": image_bytes_to_data_url(image_bytes, image_fmt)}}
            )
        if extra_session_list:
            request["messages"][-1]["content"].extend(extra_session_list)
        return request

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
