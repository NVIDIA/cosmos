"""
AWS Bedrock API support via the Converse API.
Supports vision models (e.g. Claude on Bedrock) with image inputs.
Set AWS credentials via environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
or default profile.

Requires: pip install boto3
"""
import base64
import os
import os.path as osp

from ..smp import get_logger
from .base import BaseAPI

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    boto3 = None
    BotoCoreError = ClientError = Exception

logger = get_logger(__name__)

# Common Bedrock vision model IDs (region-specific; use full ID in config)
BEDROCK_VISION_MODELS = {
    "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
    "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
}


def _image_path_to_format(path):
    ext = osp.splitext(path)[-1].lower()
    if ext in (".jpg", ".jpeg"):
        return "jpeg"
    if ext in (".png", ".gif", ".webp"):
        return ext[1:]
    return "jpeg"


class BedrockAPI(BaseAPI):
    """VLM API using AWS Bedrock Converse API (supports text + image)."""

    is_api: bool = True

    def __init__(
        self,
        model_id: str,
        region_name: str = None,
        retry: int = 10,
        wait: int = 1,
        system_prompt: str = None,
        verbose: bool = True,
        temperature: float = 0,
        max_tokens: int = 2048,
        img_size: int = -1,
        **kwargs,
    ):
        if boto3 is None:
            raise ImportError("boto3 is required for BedrockAPI. Install with: pip install boto3")

        self.model_id = BEDROCK_VISION_MODELS.get(model_id, model_id)
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.img_size = img_size

        super().__init__(
            retry=retry,
            wait=wait,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )

        self._client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region_name,
        )
        logger.info(
            f"BedrockAPI: model_id={self.model_id}, region={self.region_name}"
        )

    def _encode_image_to_bytes(self, image_path, target_size=-1):
        from ..smp import encode_image_file_to_base64

        suffix = osp.splitext(image_path)[-1].lower()
        fmt = "JPEG" if suffix in (".jpg", ".jpeg") else "PNG"
        b64 = encode_image_file_to_base64(
            image_path, target_size=target_size, fmt=fmt
        )
        return base64.b64decode(b64), _image_path_to_format(image_path)

    def _build_content_blocks(self, inputs):
        """Build Converse API content list from VLMEvalKit message list."""
        blocks = []
        for item in inputs:
            if item["type"] == "text" and item["value"]:
                blocks.append({"text": item["value"]})
            elif item["type"] == "image":
                path = item["value"]
                raw_bytes, fmt = self._encode_image_to_bytes(
                    path, target_size=self.img_size
                )
                blocks.append({
                    "image": {
                        "format": fmt,
                        "source": {"bytes": raw_bytes},
                    }
                })
        return blocks

    def _prepare_messages(self, inputs):
        """Convert VLMEvalKit message list to Converse API messages.
        inputs: either [ {type, value}, ... ] (single turn) or
                 [ {role, content: [ {type, value}, ... ] }, ... ] (multi-turn).
        """
        if inputs and "role" in inputs[0]:
            # Multi-turn chat
            messages = []
            for msg in inputs:
                role = msg["role"]
                content = self._build_content_blocks(msg["content"])
                if not content:
                    content = [{"text": ""}]
                messages.append({"role": role, "content": content})
            return messages
        # Single turn
        content = self._build_content_blocks(inputs)
        if not content:
            content = [{"text": ""}]
        return [{"role": "user", "content": content}]

    def generate_inner(self, inputs, **kwargs):
        messages = self._prepare_messages(inputs)
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        request = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if self.system_prompt:
            request["system"] = [{"text": self.system_prompt}]

        try:
            response = self._client.converse(**request)
        except (BotoCoreError, ClientError) as e:
            err_msg = str(e)
            if self.verbose:
                logger.error(f"Bedrock API error: {err_msg}")
            return -1, self.fail_msg, err_msg

        answer = self.fail_msg
        try:
            content_list = response.get("output", {}).get("message", {}).get("content", [])
            texts = [
                block["text"]
                for block in content_list
                if isinstance(block, dict) and "text" in block
            ]
            answer = "".join(texts).strip() if texts else self.fail_msg
        except Exception as e:
            if self.verbose:
                logger.error(f"Parse response: {e}")

        return 0, answer, response
