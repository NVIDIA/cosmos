import logging
import re
import string

import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor, set_seed

from vlmeval.dataset import DATASET_TYPE
from ..base import BaseModel


class Valley2Chat(BaseModel):
    """
    A multimodal chat model interface for the Valley2 architecture. (https://arxiv.org/abs/2501.05901)

    Attributes:
        - seed (int): Random seed used for reproducibility. Default is 42.
        - torch_dtype (torch.dtype): Data type for model weights (e.g., torch.bfloat16). Default is bfloat16.
        - model_path (str): Hugging Face model path or local checkpoint directory for loading
            the Valley2-DPO model. Default is 'bytedance-research/Valley2-DPO'.
        - cot_output_trunc (bool): Whether to truncate model output by extracting the part between
            <CONCLUSION> and </CONCLUSION> tags, if present. This is useful for structured
            chain-of-thought outputs. Default is False.
        - max_new_tokens (int): Maximum number of tokens to be generated during inference. Default is 2048.
        - min_pixels (int): Minimum number of image pixels allowed after preprocessing. Default is 1280*28*28.
        - max_pixels (int): Maximum number of image pixels allowed after preprocessing. Default is 16384*28*28.
        - use_llava_cot_prompt (bool): Whether to enable LLaVA-CoT (Chain-of-Thought) prompting. If enabled,
            "Please think step by step." is appended to the prompt to trigger detailed reasoning-style
            outputs as in LLaVA-CoT (https://arxiv.org/abs/2411.10440). Default is False.
        - use_custom_prompt (bool): Whether to enable custom prompts tailored to specific benchmarks. These prompts
            have been manually optimized for individual datasets. Can be combined with `use_llava_cot_prompt`.
            Default is True.
        - use_custom_pixels (bool): Whether to enable benchmark-specific custom image pixel limits during
            preprocessing. Some datasets benefit from lower or higher resolution constraints. Default is True.
    """

    def __init__(self,
                 model_path='bytedance-research/Valley2-DPO',
                 cot_output_trunc=False,
                 max_new_tokens: int = 2048,
                 seed=42,
                 torch_dtype=torch.float16,
                 min_pixels=1280 * 28 * 28,
                 max_pixels=16384 * 28 * 28,
                 use_llava_cot_prompt=False,
                 use_custom_prompt=True,
                 use_custom_pixels_limit=True,
                 **kwargs):
        set_seed(seed)
        self.torch_dtype = torch_dtype
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logging.info(f"Start loading valley model from {model_path}")
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=self.torch_dtype, trust_remote_code=True)
        self.model = self.model.to(self.device).half()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            anyres=self.model.config.anyres,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            trust_remote_code=True
        )

        self._use_llava_cot_prompt = use_llava_cot_prompt
        self._use_custom_prompt = use_custom_prompt
        self._use_custom_pixels_limit = use_custom_pixels_limit
        self.cot_output_trunc = cot_output_trunc
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        kwargs_default = dict(do_sample=False, max_new_tokens=max_new_tokens, repetition_penalty=1.0) # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

    def use_custom_prompt(self, dataset):
        if not self._use_custom_prompt:
            return False
        if any(dataset.startswith(prefix) for prefix in ['TextVQA', 'ChartQA', 'InfoVQA', 'DocVQA', 'CMMMU']):
            return True
        if any(dataset.startswith(prefix) for prefix in ['MMVet', 'MathVista', 'MathVerse', 'MathVision']):
            return True
        # if DATASET_TYPE(dataset) == 'Y/N' or DATASET_TYPE(dataset) == 'MCQ':
        #     return True
        return False

    def build_yorn_prompt(self, line, dataset=None):
        prompt = line['question']
        prompt += '\nAnswer the question using a single word or phrase.'
        return prompt

    def build_multi_choice_prompt(self, line, dataset=None):
        prompt = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            prompt = hint + '\n' + prompt

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            prompt += f'\n{key}. {item}'

        if len(options):
            prompt += "\nAnswer with the option's letter from the given choices directly."

        return prompt

    def build_mmvet_prompt(self, line, dataset=None):
        prompt = line['question']
        prompt += "\nProvide a step-by-step solution to the problem carefully."
        return prompt

    def build_math_prompt(self, line, dataset=None):
        prompt = line['question']
        prompt += (
            "\nProvide a step-by-step solution to the problem, "
            "and conclude with 'the answer is' followed by the final solution."
        )
        return prompt

    def build_ocr_cmmmu_prompt(self, line, dataset=None):
        prompt = line['question']
        prompt += "\nPlease try to answer the question with short words or phrases if possible."
        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset == 'MMVet':
            prompt = self.build_mmvet_prompt(line, dataset)
        elif any(dataset.startswith(prefix) for prefix in ('TextVQA', 'ChartQA', 'InfoVQA', 'DocVQA', 'CMMMU')):
            prompt = self.build_ocr_cmmmu_prompt(line, dataset)
        elif any(dataset.startswith(prefix) for prefix in ('MathVista', 'MathVerse', 'MathVision')):
            prompt = self.build_math_prompt(line, dataset)
        # elif DATASET_TYPE(dataset) == 'Y/N':
        #     prompt = self.build_yorn_prompt(line, dataset)
        # elif DATASET_TYPE(dataset) == 'MCQ':
        #     prompt = self.build_multi_choice_prompt(line, dataset)
        else:
            raise RuntimeError(f'Invalid dataset type: {DATASET_TYPE(dataset)}')

        # add image to message
        message = []
        if isinstance(tgt_path, list):
            message = [dict(type='image', value=p) for p in tgt_path]
        else:
            message = [dict(type='image', value=tgt_path)]

        # add text to message
        message.append(dict(type='text', value=prompt))
        return message

    def post_process(self, generation_text, dataset=None):
        if any(dataset.startswith(prefix) for prefix in ('TextVQA', 'ChartQA', 'InfoVQA', 'DocVQA', 'CMMMU')):
            self.cot_output_trunc = True  # froce truncate cot output

        if self.cot_output_trunc:
            pattern = r"(?:<CONCLUSION>)([\s\S]*?)(?:</CONCLUSION>)"
            match = re.search(pattern, generation_text)
            if match:
                generation_text = match.group(1)

        if any(dataset.startswith(prefix) for prefix in ('TextVQA', 'ChartQA', 'InfoVQA', 'DocVQA', 'CMMMU')):
            generation_text = generation_text.strip()
            if generation_text.endswith('.'):
                generation_text = generation_text[:-1]
        return generation_text

    def get_pixels_limit(self, dataset=None):
        if self._use_custom_pixels_limit and \
                any(dataset.startswith(prefix) for prefix in ['OCRBench', 'HallusionBench', 'MMMU', 'MMBench']):
            min_pixels = 100 * 28 * 28
            max_pixels = 1280 * 28 * 28
        else:
            min_pixels = self.min_pixels
            max_pixels = self.max_pixels
        return min_pixels, max_pixels

    def generate_inner(self, message, dataset=None):
        # Construct messages and images
        messages = []
        text, images = '', []
        for item in message:
            if item['type'] == 'text':
                text += item['value']
            elif item['type'] == 'image':
                text += ' <image> '
                images.append(item['value'])

        # Add CoT prompt, trigger the llava-cot output format
        LLAVA_COT_PROMPT = "\nPlease think step by step."
        if self._use_llava_cot_prompt:
            messages.append({"role": 'user', "content": text + LLAVA_COT_PROMPT})
        elif self._use_custom_prompt and dataset in ["MMMU_DEV_VAL", "MMStar", "OCRBench", "MMVet"]:
            messages.append({"role": 'user', "content": text + LLAVA_COT_PROMPT})
        else:
            messages.append({"role": 'user', "content": text})

        # Preprocess the messages and images
        min_pixels, max_pixels = self.get_pixels_limit(dataset)
        data_dict = self.processor(
            {
                "conversations": messages,
                "images": images
            },
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )

        # Inference
        with torch.inference_mode():
            self.model.to(dtype=self.torch_dtype, device=self.device)

            if type(data_dict["images"][0]) is list:
                images = [
                    [item.to(dtype=self.torch_dtype, device=self.device) for item in img]
                    for img in data_dict["images"]
                ]
            else:
                images = [img.to(dtype=self.torch_dtype, device=self.device) for img in data_dict["images"]]

            output_ids = self.model.generate(
                input_ids=data_dict["input_ids"].to(self.device),
                images=images,
                image_sizes=data_dict["image_sizes"],
                pixel_values=data_dict["pixel_values"].to(dtype=self.torch_dtype, device=self.device),
                image_grid_thw=data_dict["image_grid_thw"].to(self.device),
                do_sample=self.kwargs["do_sample"],
                max_new_tokens=self.kwargs["max_new_tokens"],
                repetition_penalty=self.kwargs["repetition_penalty"],
                return_dict_in_generate=True,
                output_scores=True,
            )
        input_token_len = data_dict["input_ids"].shape[1]
        generation_text = self.processor.batch_decode(output_ids.sequences[:, input_token_len:])[0]
        generation_text = generation_text.replace("<|im_end|>", "")

        # Postprocess
        generation_text = self.post_process(generation_text, dataset)
        return generation_text
