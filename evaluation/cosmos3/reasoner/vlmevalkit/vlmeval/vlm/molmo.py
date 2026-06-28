import logging
import string

import pandas as pd
import torch
from PIL import Image

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import listinstr
from .base import BaseModel

TYPE_PROMPTS = {
    'Y/N': 'vqa2:',
    'VQA': 'vqa2:',
    'MCQ': 'a_okvqa_mc:',
}

DATASET_PROMPTS = {
    'AI2D_TEST': 'ai2_diagram:',
    'AI2D_TEST_NO_MASK': 'ai2_diagram:',
    'COCO_VAL': 'coco_captioning:',
    'ChartQA_TEST': 'chart_qa:',
    'ChartQA_VAL': 'chart_qa:',
    'DocVQA_VAL': 'doc_qa:',
    'DocVQA_TEST': 'doc_qa:',
    'InfoVQA_TEST': 'info_qa:',
    'InfoVQA_VAL': 'info_qa:',
    'OCRVQA_TEST': 'ocr_vqa:',
    'OCRVQA_TESTCORE': 'ocr_vqa:',
    'ScienceQA_VAL': 'science_qa:',
    'ScienceQA_TEST': 'science_qa:',
    'TableVQABench': 'tabwmp_da:',
    'TextVQA_VAL': 'text_vqa:'
}


class molmo(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='allenai/Molmo-7B-D-0924', **kwargs):
        try:
            import einops  # noqa: F401
            from transformers import (AutoModelForCausalLM, AutoProcessor,  # noqa: F401
                                      GenerationConfig)
        except Exception as e:
            logging.critical('Please install transformer and einops before using molmo.')
            raise e

        if '72b' not in model_path.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='cuda')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto")

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.kwargs = kwargs
        self.model_name = model_path
        # set default maximum number of crops to 36
        self.max_crops = kwargs.get('max_crops', 36)

    def use_custom_prompt(self, dataset):
        if DATASET_TYPE(dataset) in ['Y/N', 'MCQ', 'VQA']:
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        prefix = None
        if dataset in ['MMMU_DEV_VAL', 'MMMU_TEST']:
            prompt = self.build_prompt_mcq_vqa(line)
        elif dataset in ['MathVista_MINI']:
            prompt = self.build_prompt_mathvista(line)
        elif dataset in ['AI2D_TEST', 'AI2D_TEST_NO_MASK']:
            prompt = self.build_prompt_ai2d(line)
        elif dataset is not None and listinstr(list(DATASET_PROMPTS.keys()), dataset):
            prefix = DATASET_PROMPTS[dataset]  # rest of supervised datasets are in VQA format
            prompt = self.build_prompt_vqa(line, prefix)
        elif dataset is not None and listinstr(['MCQ'], DATASET_TYPE(dataset)):
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        # interleave dataset
        if dataset.startswith('MMMU_'):
            from .. import MMMUDataset
            message = MMMUDataset.split_MMMU(message)
        return message

    def build_prompt_mathvista(self, line):
        if line['question_type'] == 'multi_choice':
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)
        return prompt

    def build_prompt_ai2d(self, line):
        def option_is_abc(line):
            for cand in string.ascii_uppercase:
                if cand in line and not pd.isna(line[cand]):
                    # check if option is single letter
                    if not line[cand].strip().isalpha() or len(line[cand].strip()) > 1:
                        return False
            return True

        if line['abcLabel'] and option_is_abc(line):
            prompt = line['question']
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                prompt += f'\n{item}'
            prompt = f"ai2_diagram_no_letter: {prompt}"
            # prompt = self.build_prompt_multiple_choice(line, prefix='ai2_diagram_no_letter:')
        else:
            prompt = self.build_prompt_multiple_choice(line, prefix='ai2_diagram:')
        return prompt

    def build_prompt_mcq_vqa(self, line):
        if line['question_type'] == 'multiple-choice':
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)
        return prompt

    def build_prompt_multiple_choice(self, line, prefix=None):
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
            question += f'\n{key}: {item}'
        if prefix is None:
            prompt = f"{TYPE_PROMPTS['MCQ']} {question}"
        else:
            prompt = f"{prefix} {question}"

        return prompt

    def build_prompt_vqa(self, line, prefix=None):
        question = line['question']
        if prefix is None:
            prompt = f"{TYPE_PROMPTS['VQA']} {question}"
        else:
            prompt = f"{prefix} {question}"
        return prompt

    def generate_inner(self, message, dataset=None):
        from transformers import GenerationConfig
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # process the image and text
        max_crops = self.max_crops
        inputs = self.processor.process(
            images=[image],
            text=prompt,
            images_kwargs={
                "max_crops": max_crops
            }
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer,
                use_cache=False
            )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # AI2D: map direct answer to letter option
        if dataset in ['AI2D_TEST', 'AI2D_TEST_NO_MASK']:
            # 'ai2_diagram_no_letter: Which of the following is the magma chamber?\nK\nB\nC\nH'
            if 'ai2_diagram_no_letter' in prompt:
                options = prompt.split('\n')[1:]
                answer = options.index(generated_text)
                generated_text = chr(answer + ord('A'))

        # print(dataset, prompt, generated_text, inputs['images'].size()) # uncomment to debug

        return generated_text
