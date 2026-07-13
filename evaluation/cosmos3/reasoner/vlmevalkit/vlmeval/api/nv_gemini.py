"""NV-deployed Gemini wrapper for gcp/google/gemini-3.1-{pro,flash-lite}-preview.

Routes through NV inference gateway's Vertex passthrough
(inference-api.nvidia.com/vertex_ai/v1/.../models/{MODEL}:generateContent).
Native Gemini schema (contents/parts/inlineData) — the OpenAI-compat path on
the same gateway resamples video to ~1 fps server-side, so Vertex passthrough
is the only path that preserves the client's encoded fps end-to-end.

Reuses cosmos_reason.process_video_info_to_video_url for client-side
resample + lossless ffmpeg re-encode; sets videoMetadata.fps to the
re-encoded fps so Vertex samples at exactly that rate (same semantic as the
vLLM --media-io-kwargs num_frames=-1, fps=-1 production setup).

The same class targets vanilla GCP Vertex (aiplatform.googleapis.com); only
$OPENAI_API_KEY needs to hold a different bearer ($(gcloud auth
print-access-token)).
"""

import base64
import copy
import json
import os

import requests

from ..smp import get_logger
from .base import BaseAPI
from .cosmos_reason import _process_video_with_timeout

logger = get_logger(__name__)


_VIDEO_KW_KEYS = (
    'nframes', 'fps', 'total_pixels', 'max_pixels', 'min_pixels', 'max_frames',
)


class NVGemini(BaseAPI):
    is_api: bool = True
    VIDEO_LLM: bool = True
    allowed_types = ['text', 'image', 'video']

    def __init__(
        self,
        model: str,
        api_base: str,
        key: str | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        retry: int = 10,
        wait: int = 5,
        timeout: int = 600,
        video_proc_timeout: int = 300,
        use_video_cache: bool = True,
        verbose: bool = False,
        image_patch_size: int = 16,
        system_prompt: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.api_base = api_base
        self.key = key or os.environ.get('OPENAI_API_KEY', '')
        self.thinking_budget = self._resolve_thinking_budget(
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            chat_template_kwargs=kwargs.get('chat_template_kwargs'),
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.video_proc_timeout = video_proc_timeout
        self.use_video_cache = use_video_cache
        self.image_patch_size = image_patch_size
        self._last_error_type = None

        super().__init__(
            retry=retry,
            wait=wait,
            verbose=verbose,
            system_prompt=system_prompt,
        )
        logger.info(
            f'NVGemini: model={self.model}, api_base={self.api_base}, '
            f'thinking_budget={self.thinking_budget}'
        )
        self._preflight()

    @staticmethod
    def _resolve_thinking_budget(enable_thinking, thinking_budget, chat_template_kwargs):
        if thinking_budget is not None:
            return int(thinking_budget)
        flag = enable_thinking
        if flag is None and chat_template_kwargs:
            flag = chat_template_kwargs.get('enable_thinking')
        return 8192 if flag else 0

    def _preflight(self):
        """One-shot :generateContent probe replacing CosmosReason's
        _check_endpoint_health + _resolve_model_name. Logs loudly on failure
        but does not raise — transient startup blips shouldn't kill the job,
        and a genuinely broken endpoint will surface as http_error counts on
        every subsequent request via the normal retry loop.
        """
        payload = {
            'contents': [{'role': 'user', 'parts': [{'text': 'ping'}]}],
            'generationConfig': {
                'maxOutputTokens': 8,
                'thinkingConfig': {'thinkingBudget': 0},
            },
        }
        headers = {
            'Authorization': f'Bearer {self.key}',
            'Content-Type': 'application/json',
        }
        try:
            r = requests.post(
                self.api_base, headers=headers,
                data=json.dumps(payload), timeout=30,
            )
        except Exception as e:
            logger.warning(
                f'⚠️  NVGemini preflight network error: {type(e).__name__}: {e}. '
                f'Proceeding; live requests will surface the issue.'
            )
            return
        if r.status_code == 200 and r.json().get('candidates'):
            logger.info(f'✓ NVGemini preflight passed: {self.api_base}')
            return
        logger.error(
            f'⚠️  NVGemini preflight failed (HTTP {r.status_code}): {r.text[:400]}. '
            f'Proceeding; live requests will surface the issue.'
        )

    # ---- part builders ------------------------------------------------------

    def _video_kwargs(self, msg: dict) -> dict:
        kw = copy.deepcopy(msg)
        kw.pop('type', None)
        kw.pop('value', None)
        return {k: v for k, v in kw.items() if k in _VIDEO_KW_KEYS and v is not None}

    def _video_part(self, msg: dict) -> dict:
        video_url, video_kwargs = _process_video_with_timeout(
            msg['value'],
            image_patch_size=self.image_patch_size,
            use_cache=self.use_video_cache,
            kwargs=self._video_kwargs(msg),
            timeout=self.video_proc_timeout,
        )
        b64 = video_url.split('base64,', 1)[1]
        part = {'inlineData': {'mimeType': 'video/mp4', 'data': b64}}
        fps = (video_kwargs or {}).get('fps')
        if fps is not None:
            part['videoMetadata'] = {'fps': fps if not isinstance(fps, list) else fps[0]}
        return part

    def _image_part(self, msg: dict) -> dict:
        with open(msg['value'], 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        suffix = msg['value'].lower().rsplit('.', 1)[-1]
        if suffix in ('jpg', 'jpeg'):
            mime = 'image/jpeg'
        elif suffix in ('png', 'webp', 'gif'):
            mime = f'image/{suffix}'
        else:
            mime = 'image/png'
        return {'inlineData': {'mimeType': mime, 'data': b64}}

    def _part_for(self, msg: dict) -> dict:
        t = msg['type']
        if t == 'text':
            return {'text': msg['value']}
        if t == 'image':
            return self._image_part(msg)
        if t == 'video':
            return self._video_part(msg)
        raise ValueError(f'NVGemini: unsupported message type {t!r}')

    # ---- generate -----------------------------------------------------------

    def generate_inner(self, inputs, **kwargs):
        self._last_error_type = None
        try:
            parts = [self._part_for(x) for x in inputs]
        except Exception as e:
            logger.warning(f'NVGemini: input preparation failed: {type(e).__name__}: {e}')
            return 0, 'VIDEO_LOAD_FAILED', None

        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        payload = {
            'contents': [{'role': 'user', 'parts': parts}],
            'generationConfig': {
                'maxOutputTokens': max_tokens,
                'temperature': temperature,
                'thinkingConfig': {'thinkingBudget': self.thinking_budget},
            },
        }
        if self.system_prompt:
            payload['systemInstruction'] = {'parts': [{'text': self.system_prompt}]}

        headers = {
            'Authorization': f'Bearer {self.key}',
            'Content-Type': 'application/json',
        }
        try:
            r = requests.post(
                self.api_base, headers=headers,
                data=json.dumps(payload), timeout=self.timeout * 1.1,
            )
        except requests.exceptions.Timeout:
            logger.error(f'⚠️  ENDPOINT TIMEOUT: {self.api_base}')
            self._last_error_type = 'timeout'
            return 408, self.fail_msg, None
        except requests.exceptions.ConnectionError as e:
            logger.error(f'⚠️  ENDPOINT DOWN: {self.api_base}: {e}')
            self._last_error_type = 'http_error'
            return 503, self.fail_msg, None
        except Exception as e:
            logger.error(f'⚠️  REQUEST FAILED: {type(e).__name__}: {e}')
            self._last_error_type = 'http_error'
            return 500, self.fail_msg, None

        if r.status_code != 200:
            self._last_error_type = 'timeout' if r.status_code in (408, 504) else 'http_error'
            logger.error(f'⚠️  HTTP {r.status_code} from {self.api_base}: {r.text[:400]}')
            return r.status_code, self.fail_msg + r.text[:400], r.text

        try:
            data = r.json()
        except Exception as e:
            self._last_error_type = 'http_error'
            return -1, self.fail_msg + f'json decode: {e}', r.text

        cands = data.get('candidates') or []
        if not cands:
            self._last_error_type = 'http_error'
            return -1, self.fail_msg + json.dumps(data)[:400], json.dumps(data)

        finish = cands[0].get('finishReason')
        text = ''.join(
            p.get('text', '')
            for p in cands[0].get('content', {}).get('parts', [])
        )
        if finish == 'MAX_TOKENS':
            self._last_error_type = 'truncated'
            logger.info(f"⚠️  Answer truncated at maxOutputTokens={max_tokens}.")
        elif finish == 'SAFETY':
            self._last_error_type = 'content_filter'
            logger.info('⚠️  Response blocked by Gemini safety filter (finishReason=SAFETY).')

        if self.verbose:
            logger.info(f'NVGemini response: {text[:200]}')

        return 0, text.strip(), r.text

    def generate(self, message, **kwargs):
        """Record per-sample inference outcome to $INFERENCE_STATS_FILE, mirroring
        CosmosReason.generate so run_metric.py aggregation works unchanged."""
        result = super().generate(message, **kwargs)
        stats_file = os.environ.get('INFERENCE_STATS_FILE')
        if stats_file:
            is_success = result is not None and self.fail_msg not in str(result)
            if self._last_error_type == 'truncated':
                category = 'truncated'
            elif is_success:
                category = 'success'
            else:
                category = self._last_error_type or 'other'
            try:
                with open(stats_file, 'a') as f:
                    f.write(json.dumps({'outcome': category}) + '\n')
            except Exception:
                pass
        return result
