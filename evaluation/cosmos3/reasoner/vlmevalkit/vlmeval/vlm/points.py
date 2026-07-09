import re
import string
from typing import List

import pandas as pd
import torch
import transformers
from PIL import Image

from ..dataset import DATASET_TYPE
from ..smp import cn_string, listinstr
from .base import BaseModel


class POINTS(BaseModel):
    """Official implementation of POINTS: Improving Your Vision-language Model with Affordable Strategies # noqa

    Paper link: https://arxiv.org/abs/2409.04828
    POINTS is a vision-language model developed by researchers at WeChat AI. This model represents the inaugural version in our
    series of multimodal models, known as WePOINTS.

    Args:
        model_path (str): The path or the name (the unique huggingface id) of the model.
    """

    def __init__(self, model_path: str, **kwargs) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor

        version = transformers.__version__
        use_fast = True
        if 'yi' in model_path.lower():
            assert version == '4.38.2', f'The version of transformers for Yi-1.5 should be 4.38.2, but got {version}.'  # noqa
            use_fast = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=use_fast)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          trust_remote_code=True,  # noqa
                                                          device_map='cuda'
                                                          ).to(torch.bfloat16)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_path)

    def use_custom_prompt(self, dataset: str) -> bool:
        """Whether to use custom prompt for the dataset.

        Args:
            dataset (str): The name of the dataset.

        Returns:
            bool: Whether to use custom prompt for the dataset.
        """
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line: str, dataset: str) -> List[dict]:
        """Build prompt for multi-choice dataset.

        Args:
            line (str): one line of the dataset.
            dataset (str): The name of the dataset.

        Returns:
            List[dict]: A list of elements constructed for current line.
        """
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if (
            'hint' in line and not pd.isna(line['hint'])) else None
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
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else  # noqa
                "\nAnswer with the option\'s letter from the given choices directly."  # noqa
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(  # noqa
                prompt) else '\nAnswer the question directly.'
        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def generate_inner(self, message: List[dict], dataset: str = None) -> str:
        """Generate response for the given message.

        Args:
            message (List[dict]): A list of elements constructed for
                current line.
            dataset (str): The name of the dataset.

        Returns:
            str: The generated response.
        """
        prompt, image_path = self.message_to_promptimg(message)
        catty = True  # whether to use catty
        if dataset == 'HallusionBench':
            prompt = prompt + \
                ' Please answer yes or no. Answer the question using a single word or phrase.'  # noqa
        elif dataset == 'MMVet':
            prompt = prompt + ' Answer this question in detail.'
            catty = False
        else:
            # use default setting
            pass

        if dataset is None:
            max_splits = 8
        elif listinstr(['MMBench', 'OCRBench'], dataset):
            max_splits = 12
        else:
            max_splits = 8

        image = Image.open(image_path).convert('RGB')
        generation_config = {
            'max_new_tokens': 2048,
            'temperature': 0.0,
            'top_p': 0.0,
            'num_beams': 1,
        }
        response = self.model.chat(image,
                                   prompt,
                                   self.tokenizer,
                                   self.image_processor,
                                   catty,
                                   generation_config,
                                   max_splits)
        return response


class POINTSV15(BaseModel):
    """Official implementation of POINTSv1.5

    This implementation is based on the official implementation of POINTSv1.5
    (https://github.com/WePOINTS/WePOINTS)

    Args:
        model_path (str): The path or the name (the unique huggingface id)
            of the model.
    """

    def __init__(self, model_path: str, **kwargs) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        quant_config = QuantoConfig(modules_to_not_convert=['vision_encoder'])
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          trust_remote_code=True,  # noqa
                                                          device_map='cuda',
                                                          torch_dtype=torch.bfloat16,
                                                          quantization_config=quant_config
                                                          )
        try:
            from wepoints.utils.images import Qwen2ImageProcessorForPOINTSV15
        except ImportError:
            print('Please install WePOINTS, and refer to https://github.com/WePOINTS/WePOINTS')
        self.image_processor = Qwen2ImageProcessorForPOINTSV15.from_pretrained(model_path) # noqa

    def use_custom_prompt(self, dataset: str) -> bool:
        """Whether to use custom prompt for the dataset.

        Args:
            dataset (str): The name of the dataset.

        Returns:
            bool: Whether to use custom prompt for the dataset.
        """
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line: str, dataset: str) -> List[dict]:
        """Build prompt for multi-choice dataset.

        Args:
            line (str): one line of the dataset.
            dataset (str): The name of the dataset.

        Returns:
            List[dict]: A list of elements constructed for current line.
        """
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if (
            'hint' in line and not pd.isna(line['hint'])) else None
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
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else  # noqa
                "\nAnswer with the option\'s letter from the given choices directly."  # noqa
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(  # noqa
                prompt) else '\nAnswer the question directly.'
        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def set_image_processor(self, dataset: str) -> None:
        """Set the image processor for the dataset.

        Args:
            dataset (str): The name of the dataset.
        """
        if dataset in ['OCRBench']:
            self.image_processor.min_pixels = 280 * 280
        elif dataset in ['MMMU_DEV_VAL']:
            self.image_processor.min_pixels = 1280 * 28 * 28
            self.image_processor.max_pixels = 16384 * 28 * 28
        elif dataset in ['MathVista_MINI']:
            self.image_processor.min_pixels = 56 * 56
        elif dataset in ['MMVet', 'HallusionBench',
                         'MMBench_TEST_EN_V11', 'MMBench_TEST_CN_V11']:
            self.image_processor.min_pixels = 1280 * 28 * 28
        else:
            self.image_processor.min_pixels = 840 * 840

    def construct_messages(self, prompt: str,
                           image_paths: List[str]) -> List[dict]:
        """Construct messages for the given prompt and image paths.

        Args:
            prompt (str): The prompt for the generation.
            image_paths (List[str]): A list of image paths.

        Returns:
            List[dict]: A list of elements constructed for current line.
        """
        content = []
        for image_path in image_paths:
            content.append(
                dict(type='image', image=image_path)
            )
        content.append(
            dict(type='text', text=prompt)
        )
        messages = [
            {
                'role': 'user',
                'content': content
            }
        ]
        return messages

    def generate_inner(self, message: List[dict], dataset: str = None) -> str:
        """Generate response for the given message.

        Args:
            message (List[dict]): A list of elements constructed for
                current line.
            dataset (str): The name of the dataset.

        Returns:
            str: The generated response.
        """
        self.set_image_processor(dataset)
        prompt, image_paths = self.message_to_promptimg(message)
        image_paths = [image_paths]
        if dataset == 'HallusionBench':
            prompt = prompt + \
                ' Please answer yes or no. Answer the question using a single word or phrase.'  # noqa
        elif dataset == 'MMVet':
            prompt = prompt + ' Answer this question in detail.'
        else:
            # use default setting
            pass
        pattern = r'<image \d+>'
        prompt = re.sub(pattern, '\n', prompt)
        messages = self.construct_messages(prompt, image_paths)

        generation_config = {
            'max_new_tokens': 2048,
            'temperature': 0.0,
            'top_p': 0.0,
            'num_beams': 1,
        }
        response = self.model.chat(messages,
                                   self.tokenizer,
                                   self.image_processor,
                                   generation_config)
        return response
