import logging
import os
import warnings

import torch

from ..base import BaseModel
from .prompt import HawkVLPromptMixin


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


class HawkVL(HawkVLPromptMixin, BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(
        self,
        model_path: str,
        min_pixels: int = None,
        max_pixels: int = None,
        max_new_tokens=1024,
        top_p=0.8,
        top_k=20,
        temperature=0.7,
        repetition_penalty=1.0,
        use_custom_prompt: bool = False,
        system_prompt: str = None,
        verbose: bool = False,
    ):
        try:
            from .hawk.model import HawkQwenForCausalLM
        except Exception as e:
            logging.critical(
                "Please move the code ('hawk' directory) to $VLMEVALKIT/vlm/hawk_vl."
            )
            raise e
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.fps = 1.0
        self.nframe = 64
        self.FRAME_FACTOR = 1

        self.model_path = model_path
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.model = HawkQwenForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map=None)

        self.model.cuda().eval()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt')

        inputs = inputs.to('cuda')

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )

        out = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0].strip()

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response
