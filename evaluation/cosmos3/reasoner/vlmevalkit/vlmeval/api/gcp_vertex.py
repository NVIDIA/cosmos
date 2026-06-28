"""
GCP (Google Cloud Platform) Vertex AI API support.
Single class for both Gemini and Claude on Vertex. Backend is chosen by model name:
- model starts with "claude-" -> AnthropicVertex (anthropic[vertex])
- else -> Vertex AI GenerativeModel (google-cloud-aiplatform)
Uses Application Default Credentials (ADC). Set GOOGLE_CLOUD_PROJECT and optionally GOOGLE_CLOUD_LOCATION.

Gemini: pip install google-cloud-aiplatform
Claude: pip install -U google-cloud-aiplatform "anthropic[vertex]"
"""
import mimetypes
import os
import os.path as osp

import numpy as np

from ..smp import encode_image_to_base64, get_logger
from .base import BaseAPI

# Vertex AI (Gemini)
try:
    import vertexai
    from vertexai.generative_models import GenerationConfig, GenerativeModel
    from vertexai.generative_models import Image as VertexImage
    from vertexai.generative_models import Part
except ImportError:
    vertexai = None
    GenerativeModel = Part = GenerationConfig = VertexImage = None

# Claude on Vertex
try:
    from anthropic import AnthropicVertex
except ImportError:
    AnthropicVertex = None

logger = get_logger(__name__)


def _encode_image_file_to_base64(image_path, target_size=-1, fmt=".jpg"):
    from PIL import Image

    image = Image.open(image_path)
    if fmt in (".jpg", ".jpeg"):
        pil_fmt = "JPEG"
    elif fmt == ".png":
        pil_fmt = "PNG"
    else:
        pil_fmt = "JPEG"
    return encode_image_to_base64(image, target_size=target_size, fmt=pil_fmt)


def _is_claude_model(model: str) -> bool:
    return model.strip().lower().startswith("claude-")


class GCPVertexAPI(BaseAPI):
    """Single API for GCP Vertex AI: Gemini or Claude, chosen by model name (claude-* -> Claude)."""

    is_api: bool = True

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        project_id: str = None,
        location: str = None,
        retry: int = 10,
        wait: int = 1,
        system_prompt: str = None,
        verbose: bool = True,
        temperature: float = 0,
        max_tokens: int = 2048,
        **kwargs,
    ):
        self.model = model
        self._is_claude = _is_claude_model(model)
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.environ.get(
            "GOOGLE_CLOUD_LOCATION",
            "global" if self._is_claude else "us-central1",
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.project_id:
            raise ValueError(
                "GCP project_id is required. Set GOOGLE_CLOUD_PROJECT or pass project_id=..."
            )

        super().__init__(
            retry=retry,
            wait=wait,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )

        if self._is_claude:
            if AnthropicVertex is None:
                raise ImportError(
                    "Claude on Vertex requires: pip install -U google-cloud-aiplatform \"anthropic[vertex]\""
                )
            self._client = AnthropicVertex(
                project_id=self.project_id,
                region=self.location,
            )
            self._gemini_model = None
        else:
            if vertexai is None or GenerativeModel is None:
                raise ImportError(
                    "GCP Vertex AI (Gemini) requires google-cloud-aiplatform. "
                    "Install with: pip install google-cloud-aiplatform"
                )
            vertexai.init(project=self.project_id, location=self.location)
            model_id = "gemini-1.0-pro-vision" if model == "gemini-1.0-pro" else model
            self._gemini_model = GenerativeModel(model_id)
            self._client = None

        logger.info(
            f"GCPVertexAPI: model={self.model}, project={self.project_id}, "
            f"location={self.location} ({'Claude' if self._is_claude else 'Gemini'})"
        )

    def _build_gemini_contents(self, inputs):
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        for inp in inputs:
            if inp["type"] == "text" and inp["value"]:
                parts.append(inp["value"])
            elif inp["type"] == "image":
                parts.append(Part.from_image(VertexImage.load_from_file(inp["value"])))
        return parts

    def _generate_gemini(self, inputs, temperature, max_tokens):
        contents = self._build_gemini_contents(inputs)
        if not contents:
            contents = [""]
        resp = self._gemini_model.generate_content(
            contents,
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        answer = resp.text.strip() if resp.text else self.fail_msg
        return 0, answer, resp

    def _prepare_claude_content(self, inputs):
        assert all(isinstance(x, dict) for x in inputs)
        has_images = np.sum([x["type"] == "image" for x in inputs])
        if has_images:
            content_list = []
            for item in inputs:
                if item["type"] == "text" and item["value"]:
                    content_list.append({"type": "text", "text": item["value"]})
                elif item["type"] == "image":
                    pth = item["value"]
                    suffix = osp.splitext(pth)[-1].lower()
                    media_type = mimetypes.types_map.get(suffix) or "image/jpeg"
                    content_list.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": _encode_image_file_to_base64(
                                pth, target_size=4096, fmt=suffix
                            ),
                        },
                    })
            return content_list
        text = "\n".join([x["value"] for x in inputs if x["type"] == "text"])
        return [{"type": "text", "text": text or ""}]

    def _prepare_claude_messages(self, inputs):
        if inputs and "role" in inputs[0]:
            return [
                {"role": item["role"], "content": self._prepare_claude_content(item["content"])}
                for item in inputs
            ]
        return [{"role": "user", "content": self._prepare_claude_content(inputs)}]

    def _generate_claude(self, inputs, temperature, max_tokens):
        messages = self._prepare_claude_messages(inputs)
        create_kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if self.system_prompt:
            create_kwargs["system"] = self.system_prompt
        resp = self._client.messages.create(**create_kwargs)
        answer = (
            resp.content[0].text.strip()
            if resp.content and getattr(resp.content[0], "text", None)
            else self.fail_msg
        )
        return 0, answer, resp

    def generate_inner(self, inputs, **kwargs):
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        try:
            if self._is_claude:
                return self._generate_claude(inputs, temperature, max_tokens)
            return self._generate_gemini(inputs, temperature, max_tokens)
        except Exception as err:
            if self.verbose:
                logger.error(f"{type(err).__name__}: {err}")
            return -1, self.fail_msg, str(err)
