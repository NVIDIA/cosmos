import copy as cp
import re
import string
import warnings

import pandas as pd
import torch
from PIL import Image

from ..dataset import DATASET_MODALITY, DATASET_TYPE
from ..smp import listinstr
from .base import BaseModel


class Aria(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='rhymes-ai/Aria', **kwargs):
        from transformers import AutoModelForCausalLM, AutoProcessor
        assert model_path is not None
        self.model_path = model_path
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.unk_token_id
        self.processor = processor
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()
        default_kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=1,
            num_return_sequences=1,
            use_cache=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.unk_token_id,
            stop_strings=["<|im_end|>"],
            tokenizer=processor.tokenizer,
        )
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'ChartQAPro', 'ChartMuseum'], dataset):  # noqa: E501
            # For Multi-Turn and some special datasets we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

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
            prompt += (
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            if listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse'], dataset):
                prompt = prompt
            elif listinstr(['LLaVABench', 'MMBench-Video'], dataset):
                prompt += '\nAnswer this question in detail.'
            elif listinstr(['DocVQA'], dataset):
                prompt += '\nAnswer briefly and directly.'
            else:
                prompt += '\nAnswer the question using a single word or phrase.'

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def build_video_prompt(self, prompt, dataset=None):
        if listinstr(['MMBench-Video'], dataset):
            prompt = prompt.replace('\nAnswer:', '')
            prompt = prompt.replace(
                'Question: ',
                'Please carefully check the video and then answer the following question with details:'
            )
        elif listinstr(['Video-MME'], dataset):
            prompt = prompt.replace('\nAnswer:', '')
            prompt += "\nAnswer with the option's letter from the given choices directly."
        elif listinstr(['MVBench'], dataset):
            prompt = prompt.replace('Best option:(', '')
            system_prompt = 'Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n'  # noqa: E501
            prompt = prompt.replace(system_prompt, '')

        return prompt

    def adjust_kwargs(self, dataset):
        kwargs = cp.deepcopy(self.kwargs)
        kwargs["temperature"] = 0.0
        kwargs["do_sample"] = False

        if DATASET_MODALITY(dataset) == "VIDEO":
            kwargs["max_image_size"] = 490
        else:
            kwargs["max_image_size"] = 980

        kwargs["split_image"] = False

        if listinstr(['MMMU', 'MMStar', 'Math'], dataset):
            # These datasets may lead the model to work as a CoT-alike behaviour.
            # Allow to output longer.
            kwargs['max_new_tokens'] = 512
            return kwargs
        if DATASET_TYPE(dataset) in ['MCQ', 'Y/N']:
            kwargs['max_new_tokens'] = 64
        elif DATASET_TYPE(dataset) == 'Caption' and 'COCO' in dataset:
            kwargs['max_new_tokens'] = 64
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['OCRVQA', 'ChartQA', 'DocVQA'], dataset):
                kwargs['max_new_tokens'] = 128
            elif listinstr(['TextVQA'], dataset):
                kwargs['max_new_tokens'] = 32

        if listinstr(['OCR', 'ChartQA', 'DocVQA', 'InfoVQA', 'TextVQA'], dataset):
            # OCR-related datasets that need to split image
            kwargs["split_image"] = True

        return kwargs

    def generate_inner(self, message, dataset=None):
        if dataset is not None:
            kwargs = self.adjust_kwargs(dataset)
        else:
            kwargs = self.kwargs

        max_image_size = kwargs.pop("max_image_size")
        split_image = kwargs.pop("split_image")

        prompt = '<|im_start|>user\n'
        images = []
        last_message_modality = "text"

        if listinstr(['MLVU', 'TempCompass', 'MVBench'], dataset):  # re-arrange the data
            new_message = []
            for s in message:
                if s['type'] == 'image':
                    new_message.append(s)
            for s in message:
                if s['type'] == 'text':
                    new_message.append(s)
            message = new_message

        for s in message:
            if s['type'] == 'image':
                prompt += '<fim_prefix><|img|><fim_suffix>'
                images.append(s['value'])
                last_message_modality = "image"
            elif s['type'] == 'text':
                text = re.sub(r"<image \d+>", "", s["value"])
                if last_message_modality == "image":
                    prompt += "\n"
                    last_message_modality = "text"
                prompt += text

        if DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = self.build_video_prompt(prompt, dataset)

        prompt += '<|im_end|>\n<|im_start|>assistant\n'
        if images:
            images = [Image.open(s).convert('RGB') for s in images]
            encoded = self.processor(
                text=prompt,
                images=images,
                return_tensors='pt',
                padding='longest',
                max_image_size=max_image_size,
                split_image=split_image,
            )
        else:
            encoded = self.processor(text=prompt, return_tensors='pt', padding='longest')
        encoded["pixel_values"] = encoded["pixel_values"].to(self.model.dtype)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        pred = self.model.generate(**encoded, **kwargs)
        answer = self.tokenizer.decode(pred[0][encoded['input_ids'].size(1):].cpu(), skip_special_tokens=True).strip()
        answer = answer.replace('<|im_end|>', '')
        return answer
