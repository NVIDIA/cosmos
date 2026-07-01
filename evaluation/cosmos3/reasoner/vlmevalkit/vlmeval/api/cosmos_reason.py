import copy
import os
import random
import re
import string
from typing import Literal

import requests

import numpy as np
from qwen_vl_utils import process_vision_info

from vlmeval.dataset import img_root_map
from ..dataset import DATASET_MODALITY, DATASET_TYPE
from ..smp import *
from ..utils.video_cache import get_video_cache
from .base import BaseAPI

logger = get_logger(__name__)


def build_multi_choice_prompt(line, dataset=None):
    question = line['question']
    hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
    if hint is not None:
        question = hint + '\n' + question

    options = {
        cand: line[cand]
        for cand in string.ascii_uppercase
        if cand in line and not pd.isna(line[cand])
    }
    for key, item in options.items():
        question += f'\n{key}. {item}'
    prompt = question

    if len(options):
        if listinstr(['MMMU'], dataset):
            prompt += '\n请直接回答选项字母。' if cn_string(prompt) else "\nPlease answer directly with only the letter of the correct option and nothing else."
        else:
            prompt += '\n请直接回答选项字母。' if cn_string(prompt) else "\nPlease answer directly with only the letter of the correct option and nothing else. "
    else:
        if listinstr(['MMMU'], dataset):
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

    return prompt


APIBASES = {
    # 'OFFICIAL': 'https://<your-endpoint>/v1/chat/completions',
}


import base64
import subprocess
import threading

import torch


def tensor_video_to_base64(
    video_tensor: torch.Tensor,
    fps: int,
    crf: int = 0, # NOTE: yilzhao: the quality of writing the video is important since artifacts may be introduced and lead to wrong answers, for example, with crf=14, tailgating f1=0.7, with crf=0, tailgating f1=0.74
    base64_decode_code: str = "utf-8"
) -> str:
    """
    Convert a (T, C, H, W) RGB tensor to an MP4 data URL using ffmpeg via pipes.
    - video_tensor: RGB in [0,255] or [0,1], shape (T,C,H,W), C=3
    - fps: frames per second, fps could not be infered from video_tensor, so it is required
    - crf: quality (lower is better; 18–28 reasonable)
    """
    assert video_tensor.ndim == 4, "Expected (T,C,H,W)"
    T, C, H, W = video_tensor.shape
    assert C == 3, "Expected 3 channels (RGB)"

    # Ensure uint8 RGB [0,255]
    if video_tensor.dtype != torch.uint8:
        # If it's float, assume [0,1] or [0,255]; clamp & scale safely
        v = video_tensor
        if torch.is_floating_point(v) and v.max() <= 1.0:
            v = (v * 255.0)
        video_u8 = v.clamp(0, 255).to(torch.uint8)
    else:
        video_u8 = video_tensor

    # (T,C,H,W) -> (T,H,W,C), contiguous
    video_thwc = video_u8.permute(0, 2, 3, 1).contiguous()

    # ffmpeg command: read raw rgb24, encode H.264, write MP4 to stdout
    # Use fragmented MP4 so MP4 can be streamed to stdout (no moov at end)
    cmd = [
        "ffmpeg",
        "-threads", "1",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(fps),
        "-i", "-",                # stdin
        "-an",
        "-vcodec", "libx264",
        "-preset", "fast",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",    # broad compatibility
        "-movflags", "frag_keyframe+empty_moov",
        "-f", "mp4",
        "-"                       # stdout
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**6)
    try:
        out, err = proc.communicate(input=video_thwc.numpy().tobytes(), timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise RuntimeError(f"ffmpeg timed out after 120s (shape={video_tensor.shape}, fps={fps})")
    except Exception as e:
        raise RuntimeError(f"ffmpeg communication failed: {e}")

    if proc.returncode != 0:
        err_msg = err.decode('utf-8', errors='ignore') if err else "No error output"
        raise RuntimeError(f"ffmpeg failed (returncode={proc.returncode}): {err_msg}\nVideo shape: {video_tensor.shape}, fps={fps}")

    video_base64 = base64.b64encode(out).decode(base64_decode_code)
    return video_base64


def process_video_info_to_video_url(
    video_path,
    image_patch_size=16,
    use_cache=True,
    **kwargs,
):
    """
    Process video to base64 data URL with caching support.

    Args:
        video_path: Path to video file
        image_patch_size: Patch size for video processing
        use_cache: Whether to use cache (default: True)
        **kwargs: Additional processing parameters (fps, nframes, etc.)

    Returns:
        tuple: (video_url, video_kwargs)
    """
    # Check cache first if enabled
    if use_cache:
        cache = get_video_cache()
        cached_result = cache.get(video_path, image_patch_size=image_patch_size, **kwargs)
        if cached_result[0] is not None:
            # Normalize for current vLLM (same rules as the miss path below).
            # Older cache entries may have list-form fps and/or do_resize set.
            cached_kwargs = cached_result[1] or {}
            cached_kwargs.setdefault("do_sample_frames", False)
            if isinstance(cached_kwargs.get("fps"), list):
                cached_kwargs["fps"] = cached_kwargs["fps"][0]
            cached_kwargs.pop("do_resize", None)
            return cached_result[0], cached_kwargs
    # Cache miss or caching disabled - process video
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    **kwargs,
                }
            ]
        }
    ]
    # NOTE: tensor is large (1-2GB for 60s video at 1fps)
    # This is the memory bottleneck - caching avoids this entirely
    _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True, image_patch_size=image_patch_size)
    video_base64 = tensor_video_to_base64(
        video_inputs[0],
        video_kwargs['fps'][0]
    )
    video_url = f"data:video/mp4;base64,{video_base64}"

    # vLLM Qwen3-VL needs scalar fps (qwen-vl-utils returns list) and rejects
    # do_resize (removed in vLLM PR #26193). Normalize before cache.set so
    # cache-hit retries see the same shape we send on the live request.
    video_kwargs.setdefault("do_sample_frames", False)
    if isinstance(video_kwargs.get("fps"), list):
        video_kwargs["fps"] = video_kwargs["fps"][0]
    video_kwargs.pop("do_resize", None)

    if use_cache:
        cache.set(video_path, video_url, video_kwargs, image_patch_size=image_patch_size, **kwargs)

    return video_url, video_kwargs


def _process_video_with_timeout(
    video_path,
    image_patch_size,
    use_cache,
    kwargs,
    timeout=300,
):
    """Run process_video_info_to_video_url on a daemon thread with a hard timeout.

    Needed because decord.VideoReader can stall indefinitely on non-faststart
    MP4s over NFS; SIGALRM cannot interrupt C-extension blocking I/O.
    """
    result: list = [None, None]  # [payload, exception]

    def _worker():
        try:
            result[0] = process_video_info_to_video_url(
                video_path,
                image_patch_size=image_patch_size,
                use_cache=use_cache,
                **kwargs,
            )
        except Exception as e:
            result[1] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise RuntimeError(
            f"Video processing timed out after {timeout}s for {video_path}"
        )
    if result[1] is not None:
        raise result[1]
    return result[0]


class CosmosReason(BaseAPI):
    is_api: bool = True
    VIDEO_LLM: bool = True # sometimes used by video datasets
    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text

    # video llm, default settings
    # NOTE: the priority of this set of video is lower than the one in video datasets, suggested to use total_pixels only for corpus videos (varying length and resolution) while using dataset specific setting for the others when dealing with certain use case (e.g. same length and resolution)
    nframes: int = None
    fps: float = None
    total_pixels: int = None
    max_pixels: int = None
    min_pixels: int = None
    max_frames: int = None

    def __init__(self,
                 model: str = 'qwen3_30b_a3b',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 top_p: float = None,
                 top_k: int = None,
                 repetition_penalty: float = None,
                 presence_penalty: float = None,
                 timeout: int = 60,
                 api_base: str = None,
                 max_tokens: int = 2048,
                 seed: int = 1,
                 img_size: int = -1,
                 img_detail: str = 'high',
                 use_video_cache: bool = True,
                 **kwargs):
        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed

        key = os.environ.get('COSMOS_API_KEY', '')

        self.key = key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail
        self.timeout = timeout
        # args for video processing
        self.nframes = kwargs.pop('nframes', None)
        self.fps = kwargs.pop('fps', None)
        self.total_pixels = kwargs.pop('total_pixels', None)
        self.max_pixels = kwargs.pop('max_pixels', None)
        self.min_pixels = kwargs.pop('min_pixels', None)
        self.max_frames = kwargs.pop('max_frames', None)
        self.use_nim = kwargs.pop('use_nim', False)
        self.chat_template_kwargs = kwargs.pop('chat_template_kwargs', None)
        self.response_format = kwargs.pop('response_format', None)
        # Generic payload passthrough for vLLM-specific fields (e.g. structured_outputs).
        self.extra_body = kwargs.pop('extra_body', None)
        # Default Qwen3 / Reason2 setup.
        self.image_patch_size = kwargs.pop('image_patch_size', None)
        self.key = key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail
        self.timeout = timeout
        self.use_video_cache = use_video_cache

        self.api_base = api_base

        # Track the last error type for inference stats reporting
        self._last_error_type = None

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)
        logger.info(f'Using API Base: {self.api_base}; API Key: {"set" if self.key else "EMPTY"}')

        # Check if endpoint is accessible
        self._check_endpoint_health()

        # Resolve model name from endpoint (auto-detect if declared name doesn't match)
        self._resolve_model_name()

    def _resolve_model_name(self, max_retries: int = 5, retry_wait: int = 10):
        """Query /v1/models and remap self.model if declared name doesn't match.

        Retries on transient errors (timeout, connection) up to max_retries times.
        Raises RuntimeError if all retries are exhausted or the endpoint has no models.
        """
        import time
        models_url = self.api_base.rsplit('/chat/completions', 1)[0] + '/models'
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(models_url, headers={'Authorization': f'Bearer {self.key}'}, timeout=10)
                resp.raise_for_status()
                models = resp.json().get('data', [])
                if not models:
                    raise RuntimeError(
                        f'No models found at endpoint {models_url}. '
                        f'Please verify the endpoint is serving a model at {self.api_base}'
                    )
                server_model = models[0]['id']
                if server_model != self.model:
                    logger.warning(
                        f'Model name remapped: {self.model!r} -> {server_model!r} (auto-detected from endpoint)'
                    )
                    self.model = server_model
                if len(models) > 1:
                    ids = [m['id'] for m in models]
                    logger.warning(f'Multiple models at endpoint: {ids}. Using first: {self.model!r}')
                return
            except RuntimeError:
                raise
            except Exception as e:
                last_error = e
                logger.warning(
                    f'Failed to query /v1/models (attempt {attempt}/{max_retries}): {e}'
                )
                if attempt < max_retries:
                    time.sleep(retry_wait)
        raise RuntimeError(
            f'Cannot resolve model from endpoint after {max_retries} attempts: {last_error}. '
            f'Please verify the endpoint is serving a model at {self.api_base}'
        ) from last_error

    def _check_endpoint_health(self):
        """Check if the API endpoint is accessible"""
        try:
            # Try a simple GET request to check if endpoint is up
            response = requests.get(self.api_base.rsplit('/', 1)[0], timeout=5)
            logger.info(f'✓ Endpoint health check passed: {self.api_base}')
        except requests.exceptions.Timeout:
            logger.warning(f'⚠️  WARNING: Endpoint health check timed out: {self.api_base}')
            logger.warning(f'⚠️  The endpoint may be slow or down. Evaluation will proceed but may fail.')
        except requests.exceptions.ConnectionError:
            logger.error(f'⚠️  ERROR: Cannot connect to endpoint: {self.api_base}')
            logger.error(f'⚠️  Please verify the endpoint is running before starting evaluation.')
            logger.error(f'⚠️  Evaluation will proceed but all requests will likely fail.')
        except Exception as e:
            logger.warning(f'⚠️  WARNING: Endpoint health check failed: {type(e).__name__}: {str(e)}')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['BlinkSpatial', 'BlinkDepth', 'RealWorldQA', 'MMBench'], dataset):
            return True
        if DATASET_TYPE(dataset) == 'MCQ' or DATASET_TYPE(dataset).startswith('MCQ'):
            # Use VLMEvalKit's standard MCQ prompt for fair benchmark comparison.
            # CosmosReason's "only the letter...nothing else" directive suppresses
            # model reasoning and causes accuracy drop on MCQ benchmarks such as MMMU.
            return False
        if listinstr([
            'SparBench', 'EgoPlanBench2', 'CVBench', 'EmbSpatialBench',
            'RoboSpatialHome', 'RefSpatial', 'SATBench', 'Where2Place',
            'CosmosERQA', 'TemporalLocalization', 'AETCBench',
            'VANTAGE_SOT', 'MetropolisEventVerification',
            'MetropolisVQA', 'VANTAGE_VQA',
            'CountBenchQA',
            'AVSpecialOODReasoningBench',
        ], dataset):
            return False
        if listinstr(['MathVision', 'MathVerse'], dataset):
            # Math VQA benchmarks (TYPE='VQA'): use dataset's default prompt (raw question only).
            # CosmosReason's "Answer using a single word or phrase" suppresses reasoning,
            # which is critical for multi-step math computation.
            return False
        if listinstr(['IFBench'], dataset):
            # IFBench (TYPE='OPENENDED'): use dataset's own build_prompt() (raw prompt).
            # CosmosReason's OPENENDED fallthrough sets max_new_tokens=128, far too low
            # for instruction-following tasks that require long responses.
            return False
        return True

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        kwargs_default = dict(do_sample=False, max_new_tokens=128, top_p=None, num_beams=1)
        self.kwargs = kwargs_default

        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no.'
            else:
                prompt = question
        elif dataset is not None and (DATASET_TYPE(dataset) == 'MCQ' or DATASET_TYPE(dataset).startswith('MCQ')):
            prompt = build_multi_choice_prompt(line, dataset)

        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['MathVista'], dataset):
                prompt = question + ' Answer the question with a step-by-step process if the problem is complex, otherwise answer directly. Finally give the final answer with "The answer is ..."'
            elif listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['Omni3D'], dataset):
                prompt = question
            elif listinstr(['Astro2D', 'Metropolis2D', 'Metropolis2DGrounding'], dataset):
                prompt = question
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']
        # The base model expects image first instead of text first.
        message = [dict(type='image', value=s) for s in tgt_path]
        message.extend([dict(type='text', value=prompt)])
        return message

    def parse_answer(self, answer: str) -> str:
        return answer

    def _get_video_processing_kwargs(self, msg) -> dict:
        process_video_kwargs = copy.deepcopy(msg)
        process_video_kwargs.pop('type')
        process_video_kwargs.pop('value')
        video_processing_kwargs_keys = ['nframes', 'fps', 'total_pixels', 'max_pixels', 'min_pixels', 'max_frames'] # if the msg provided the kwargs, do not override anything
        if any(video_processing_kwargs_key in process_video_kwargs for video_processing_kwargs_key in video_processing_kwargs_keys):
            return process_video_kwargs
        for video_processing_kwargs_key in video_processing_kwargs_keys:
            if video_processing_kwargs_value := getattr(self, video_processing_kwargs_key, None):
                process_video_kwargs[video_processing_kwargs_key] = video_processing_kwargs_value

        return process_video_kwargs

    def prepare_itlist(self, inputs):
        video_kwargs = None
        assert np.all([isinstance(x, dict) for x in inputs])
        has_multimodal = np.sum([
            x['type'] == 'image' or x['type'] == 'video' or x['type'] == 'video_base64' for x in inputs
        ])
        if has_multimodal:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    b64 = encode_image_file_to_base64(msg['value'], target_size=self.img_size)
                    import mimetypes
                    mime_type = 'image/jpeg' if self.img_size > 0 else (
                        mimetypes.guess_type(msg['value'])[0] or 'image/png')
                    img_struct = dict(url=f'data:{mime_type};base64,{b64}', detail=self.img_detail)
                    content_list.append(dict(type='image_url', image_url=img_struct))
                elif msg['type'] == 'video_base64':
                    # Pre-encoded video URL (avoid subprocess in multiprocessing)
                    content_list.append(dict(
                        type='video_url',
                        video_url=dict(url=msg['value'])
                    ))
                elif msg['type'] == 'video':
                    assert self.image_patch_size is not None, "please set image_patch_size, 14 for qwen2.5vl series, 16 for qwen3.vl series"
                    video_url, video_kwargs = _process_video_with_timeout(
                        msg['value'],
                        image_patch_size=self.image_patch_size,
                        use_cache=self.use_video_cache,
                        kwargs=self._get_video_processing_kwargs(msg),
                        timeout=getattr(self, 'video_proc_timeout', 300),
                    )
                    content_list.append(dict( # NOTE: confirm this: video being processed to ideal size + fps, recorded within the base64 encoded video bytes, assuming the server will use the video file's parameters as default
                        type='video_url',
                        video_url=dict(
                            url=video_url,
                        )
                    ))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list, video_kwargs

    def prepare_inputs(self, inputs):
        input_msgs, video_kwargs = [], None
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                output = self.prepare_itlist(item['content'])
                if video_kwargs is None:
                    video_kwargs = output[1]
                input_msgs.append(dict(role=item['role'], content=output[0]))
        else:
            output = self.prepare_itlist(inputs)
            if video_kwargs is None:
                video_kwargs = output[1]
            input_msgs.append(dict(role='user', content=output[0]))
        return input_msgs, video_kwargs

    def generate_inner(self, inputs, **kwargs) -> str:
        self._last_error_type = None  # reset per-sample; prevents leaking across samples
        try:
            input_msgs, video_kwargs = self.prepare_inputs(inputs)
        except Exception as e:
            # Non-fail_msg sentinel skips BaseAPI retry (it retries only when fail_msg in answer).
            logger.warning(f'Video preparation failed (skipping): {type(e).__name__}: {e}')
            return 0, 'VIDEO_LOAD_FAILED', None
        kwargs.pop('dataset', None)
        temperature = kwargs.pop('temperature', self.temperature)
        top_p = kwargs.pop('top_p', self.top_p)
        top_k = kwargs.pop('top_k', self.top_k)
        repetition_penalty = kwargs.pop('repetition_penalty', self.repetition_penalty)
        presence_penalty = kwargs.pop('presence_penalty', self.presence_penalty)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        seed = kwargs.pop('seed', self.seed)
        # Will send request if use Azure, dk how to use openai client for it
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            n=1,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_tokens=max_tokens,
            **kwargs,
        )
        if self.chat_template_kwargs is not None:
            payload['chat_template_kwargs'] = self.chat_template_kwargs
        if self.response_format is not None:
            payload['response_format'] = self.response_format
        if self.extra_body:
            payload.update(self.extra_body)

        if video_kwargs is not None:
            if self.use_nim:
                # TODO: NIM evaluation haven't been fully aligned yet.
                payload['mm_processor_kwargs'] = {
                    "fps": video_kwargs['fps'],
                }
                payload['media_io_kwargs'] = { # for NIM deployments
                    "fps": video_kwargs['fps'][0],
                }
            else:
                # Verified that this branch will bring expected number of tokens to the vllm server. (from server output)
                # with previously used container (vllm 0.10.0), this is correct
                payload.update({
                    'mm_processor_kwargs': video_kwargs
                })

        THINKING_RETRY = 1
        for i in range(THINKING_RETRY):
            payload['seed'] += i
            if self.use_nim:
                # TODO: NIM evaluation haven't been fully aligned yet.
                payload['nvext'] = { # these go into nvext for NIMs
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                }
            else:
                payload['top_k'] = top_k
                payload['repetition_penalty'] = repetition_penalty
                payload['presence_penalty'] = presence_penalty
            try:
                response = requests.post(self.api_base, headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
            except requests.exceptions.Timeout:
                logger.error(f'⚠️  ENDPOINT TIMEOUT: {self.api_base}')
                logger.error(f'⚠️  The endpoint took longer than {self.timeout}s to respond. It may be down or overloaded.')
                self._last_error_type = "timeout"
                return 408, self.fail_msg, None
            except requests.exceptions.ConnectionError as e:
                logger.error(f'⚠️  ENDPOINT DOWN: {self.api_base}')
                logger.error(f'⚠️  Cannot connect to the endpoint. Please check if the service is running.')
                logger.error(f'⚠️  Error: {str(e)}')
                self._last_error_type = "http_error"
                return 503, self.fail_msg, None
            except Exception as e:
                logger.error(f'⚠️  REQUEST FAILED: {type(e).__name__}: {str(e)}')
                self._last_error_type = "http_error"
                return 500, self.fail_msg, None


            ret_code = response.status_code
            ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

            # Check for common error status codes
            if ret_code == 401:
                logger.error(f'⚠️  AUTHENTICATION FAILED (401): Invalid API key for {self.api_base}')
            elif ret_code == 403:
                logger.error(f'⚠️  ACCESS FORBIDDEN (403): API key lacks permissions for {self.api_base}')
            elif ret_code == 404:
                logger.error(f'⚠️  ENDPOINT NOT FOUND (404): {self.api_base}')
            elif ret_code == 429:
                logger.error(f'⚠️  RATE LIMIT EXCEEDED (429): Too many requests to {self.api_base}')
            elif ret_code == 500:
                logger.error(f'⚠️  SERVER ERROR (500): The endpoint {self.api_base} encountered an internal error')
            elif ret_code == 503:
                logger.error(f'⚠️  SERVICE UNAVAILABLE (503): {self.api_base} is temporarily unavailable')
            elif ret_code != 0:
                logger.error(f'⚠️  HTTP ERROR ({ret_code}): Request to {self.api_base} failed')
                logger.error(f'⚠️  Response: {response.text[:1000] if hasattr(response, "text") else "N/A"}')

            if ret_code in (408, 504):
                self._last_error_type = "timeout"
            elif ret_code != 0:
                self._last_error_type = "http_error"

            answer = self.fail_msg
            try:
                resp_struct = json.loads(response.text)
                answer = resp_struct['choices'][0]['message']['content'].strip()
                finish_reason = resp_struct['choices'][0]['finish_reason']
                if finish_reason == 'stop':
                    if self.verbose:
                        logger.info(f"prompt: {payload['messages'][0]['content'][-1]}")
                        logger.info(f'API Response: {answer}')
                    answer = self.parse_answer(answer)
                    if self._last_error_type is not None and self._last_error_type != "truncated":
                        logger.info(f'✅ Recovered after {self._last_error_type}')
                    self._last_error_type = None
                    break
                else:
                    self._last_error_type = "truncated"
                    logger.info(f"⚠️  Answer truncated at max_tokens={payload['max_tokens']} (finish_reason={finish_reason}).")
            except Exception as err:
                if self.verbose:
                    logger.error(f'{type(err)}: {err}')
                    logger.error(response.text if hasattr(response, 'text') else response)
        return ret_code, answer, response

    def generate(self, message, **kwargs):
        """Wrap parent generate() to record per-sample inference outcome."""
        result = super().generate(message, **kwargs)
        stats_file = os.environ.get("INFERENCE_STATS_FILE")
        if stats_file:
            is_success = result is not None and self.fail_msg not in str(result)
            # Truncated responses return valid content (for scoring) but are still
            # tracked as "truncated" so we can monitor max_tokens adequacy.
            if self._last_error_type == "truncated":
                category = "truncated"
            elif is_success:
                category = "success"
            else:
                category = self._last_error_type or "other"
            try:
                with open(stats_file, "a") as f:
                    f.write(json.dumps({"outcome": category}) + "\n")
            except Exception:
                pass
        return result


class CosmosReason1(CosmosReason):

    def __init__(self, model: str = 'cosmos_reason1_7b', **kwargs):
        image_patch_size = 14
        super().__init__(model=model, image_patch_size=image_patch_size, **kwargs)

    def parse_answer(self, answer: str) -> str:
        pattern = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>(?:\n|\n\n| |)<answer>([\s\S]*?)<\/answer>$"
        match = re.search(pattern, answer, re.DOTALL)
        if match:
            return match.group(2).strip()
        else:
            return answer

class CosmosReason1Think(CosmosReason1):
    def prepare_inputs(self, inputs):
        # NOTE: yilzhao: following cosmos reason1 thinking convention, adding the thinking prompt to the system prompt
        input_msgs, video_kwargs = super().prepare_inputs(inputs)
        reasoning_prompt = "\n".join([
            "",
            "Make your response in the following format:",
            "<think>",
            "your reasoning",
            "</think>",
            "<answer>",
            "your answer",
            "</answer>",
        ])
        for msg in input_msgs:
            if msg['role'] == 'system':
                if isinstance(msg['content'], list):
                    for content in msg['content']:
                        if content['type'] == 'text':
                            content['text'] += reasoning_prompt
                elif isinstance(msg['content'], str):
                    msg['content'] += reasoning_prompt
        return input_msgs, video_kwargs

class CosmosReason2(CosmosReason):

    def __init__(self, model: str = 'qwen3_30b_a3b', **kwargs):
        image_patch_size = 16
        super().__init__(model=model, image_patch_size=image_patch_size, **kwargs)

    def parse_answer(self, answer: str) -> str:
        pattern = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>(?:\n|\n\n| |)([\s\S]*?)$"
        match = re.search(pattern, answer, re.DOTALL)
        if match:
            return match.group(2).strip() # answer
        else:
            return answer


class CosmosReason2Think(CosmosReason):
    def __init__(self, model: str = 'qwen3_30b_a3b', **kwargs):
        image_patch_size = 16
        self.pattern = r"^<think>(.*)<\/think>(.+)$"
        super().__init__(model=model, image_patch_size=image_patch_size, **kwargs)

    def parse_answer(self, answer: str) -> str:
        # pattern = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>(?:\n|\n\n| |)([\s\S]*?)$"
        # match = re.search(pattern, answer, re.DOTALL)
        match = re.match(self.pattern, answer, re.DOTALL)
        if match:
            return match.group(2).strip()  # answer
        else:
            return answer

    def prepare_inputs(self, inputs):
        input_msgs, video_kwargs = super().prepare_inputs(inputs)
        for msg in input_msgs:
            if msg['role'] == 'user':
                for content in msg['content']:
                    if content['type'] == 'text':
                        content['text'] += '\nPlease provide your detailed reasoning within <think> and </think> tags before giving the final answer.'
        return input_msgs, video_kwargs

class Qwen3VLThink(CosmosReason2Think):
    def parse_answer(self, answer: str) -> str:
        # NOTE: yilzhao: Qwen3VLThink series do not return the first <think> token since it is embedded, overriding the parsing logic to handle the different format of return
        pattern = r"^([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>(?:\n|\n\n| |)([\s\S]*?)$"
        match = re.search(pattern, answer, re.DOTALL)
        if match:
            return match.group(2).strip() # answer
        else:
            return answer
