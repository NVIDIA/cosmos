import logging
import re
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, set_seed

from vlmeval.dataset import DATASET_TYPE
from ..base import BaseModel


class UrsaChat(BaseModel):
    def __init__(self, model_path: str, **kwargs) -> None:
        from .ursa_model import UrsaForConditionalGeneration, UrsaProcessor
        super().__init__()
        self.model: UrsaForConditionalGeneration = UrsaForConditionalGeneration.from_pretrained(model_path,
                                                                                                torch_dtype=torch.bfloat16).to('cuda')
        self.image_processor: UrsaProcessor = UrsaProcessor.from_pretrained(model_path)
        self.prompts = {
            'SYSTEM_PROMPT' : 'You are a helpful assistant.',
            'ORIGINAL_PROMPT' : 'you are given a math problem image, please solve the problem step by step. \nQuestion:',
            'EXTRACT_PROMPT' : 'you are given a math problem image, please solve the problem step by step. When you get an answer, please return the correspond option instead of the text content.\nQuestion:',
        }


    def use_custom_prompt(self, dataset: str) -> bool:
        """Whether to use custom prompt for the dataset.

        Args:
            dataset (str): The name of the dataset.

        Returns:
            bool: Whether to use custom prompt for the dataset.
        """
        if 'VQA' in DATASET_TYPE(dataset):
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
        # use only question
        if dataset in ['MathVista_MINI', 'MathVista', 'MathVision']:
            question = question[question.find('Question: ') + len('Question: ') : ]
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question
        if dataset in ['DynaMath']:
            prompt = '{}\n{}'.format(self.prompts['EXTRACT_PROMPT'], prompt)
        else:
            prompt = '{}\n{}'.format(self.prompts['ORIGINAL_PROMPT'], prompt)
        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message


    def extract_answer(self, text:str) -> str:
        """Extract final answer which places after '†Answer:'

        Args:
            text (str): original response

        Returns:
            str: the extracted answer
        """
        patterns = [
            r'Answer: \\boxed\{(.*?)\}',
            r'\\boxed\{(.*?)\}',
            r'†Answer:(.*?)(?:\n|$)',  # Match †Answer: to the next newline or end of string
            r'†Answer: (.*?)(?:\n|$)',  # Match †Answer: to the next newline or end of string
            r'†Answer: \n(.*?)(?:\n|$)',  # Match †Answer: to the next newline or end of string
            r'correct answer：(.*?)(?:\n|$)',  # Match correct answer：to the next newline or end of string
            r'Correct answer：(.*?)(?:\n|$)',  # Match Correct answer：to the next newline or end of string
            r'Answer：(.*?)(?:\n|$)',
            r"correct answer is:(.*?)(?:\n|$)",
            r"correct answer is:\n(.*?)(?:\n|$)",
            r"correct answer is:\n\n(.*?)(?:\n|$)",
            r"correct answer is:\n\n\n(.*?)(?:\n|$)",
            r'correct answer:(.*?)(?:\n|$)',  # Match correct answer：to the next newline or end of string
            r'Correct answer:(.*?)(?:\n|$)',  # Match Correct answer：to the next newline or end of string
            r'correct answer is(.*?)(?:\n|$)',  # Match correct answer is to the next newline or end of string
            r'Answer:(.*?)(?:\n|$)',
            r'†(.*?)(?:\n|$)',  # Match † to the next newline or end of string
            r'Answer: (.*?)(?:\n|$)',
            r'The answer is (.*?)(?:\n|$)'
        ]

        res_list = []

        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if i == 0:  # Only concatenate results for the first pattern
                    for match in matches:
                        res = match.strip()
                        if len(res) == 1 and (not res.isalpha()) and (not res.isdigit()):
                            continue
                        if len(res) == 0:
                            continue
                        res_list.append(res)
                    if res_list:
                        return '\n'.join(res_list)
                else:
                    for match in matches:
                        res = match.strip()
                        if len(res) == 1 and (not res.isalpha()) and (not res.isdigit()):
                            continue
                        if len(res) == 0:
                            continue
                        return res

        return text


    def generate_inner(self, message: List[dict], dataset: str=None) -> str:
        """Generate response for the given message.

        Args:
            message (List[dict]): A list of elements constructed for
                current line.
            dataset (str): The name of the dataset.

        Returns:
            str: The generated response.
        """
        prompt, image_path = self.message_to_promptimg(message)
        conv = [{"role": "system", "content": self.prompts['SYSTEM_PROMPT']}]
        conv.append({
            "role": "user",
            "content": "<|image|>" + prompt
        })
        prompt = self.image_processor.apply_chat_template(conv, add_generation_prompt=True)
        if isinstance(image_path, str):
            raw_image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            raw_image = image_path.convert('RGB')
        else:
            raw_image = None
        raw_image = [raw_image]
        inputs = self.image_processor(prompt, raw_image, return_tensors='pt').to('cuda', torch.bfloat16)
        generation_config = dict(
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.,
            pad_token_id=self.image_processor.tokenizer.eos_token_id,
        )
        outputs = self.model.generate(**inputs, **generation_config)
        response = self.image_processor.decode(outputs[0], skip_special_tokens=True).split('assistant\n')[-1]
        if dataset in ['MathVista_MINI', 'MathVista', 'DynaMath', 'MathVision', 'LogicVista', 'WeMath', 'MathVerse_MINI_Vision_Only', 'MathVerse']:
            response = 'Answer: {}'.format(self.extract_answer(response))
        return response
