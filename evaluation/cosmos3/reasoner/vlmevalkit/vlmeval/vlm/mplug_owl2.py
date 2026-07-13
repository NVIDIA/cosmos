import copy as cp
import logging
import string
import warnings

import pandas as pd
import torch
from PIL import Image

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import listinstr
from .base import BaseModel


class mPLUG_Owl2(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_path='MAGAer13/mplug-owl2-llama2-7b', **kwargs):
        try:
            from mplug_owl2.mm_utils import get_model_name_from_path
            from mplug_owl2.model.builder import load_pretrained_model
        except Exception as e:
            logging.critical('Please install mPLUG_Owl2 before using mPLUG_Owl2. ')
            raise e

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name, load_8bit=False, load_4bit=False, device='cpu')

        self.model = model.cuda()
        self.device = self.model.device
        self.image_processor = image_processor
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.context_len = context_len

        kwargs_default = dict(
            max_new_tokens=512, do_sample=False, num_beams=1,
            min_new_tokens=1, length_penalty=1, num_return_sequences=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMMU'], dataset):
            return False
        if DATASET_TYPE(dataset) == 'MCQ' or dataset == 'MMVet':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        if dataset == 'MMVet':
            prompt = question + '\nAnswer the question directly. '
        elif DATASET_TYPE(dataset) == 'MCQ':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = f'Hint: {hint}\n' if hint is not None else ''
            prompt += f'{question}\n'
            prompt += (
                f'{options_prompt}\nAnswer with the option’s letter from the given choices directly. '
                if len(options) else 'Answer the question directly. '
            )
        else:
            raise NotImplementedError

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token
        kwargs = cp.deepcopy(self.kwargs)
        if dataset in ['MMVet', 'LLaVABench']:
            kwargs['length_penalty'] = 0
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            kwargs['length_penalty'] = 0
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            kwargs['max_new_tokens'] = 10
        num_images = len([x for x in message if x['type'] == 'image'])
        assert num_images >= 0
        prompt_full = 'USER: '
        images = []
        if num_images == 1:
            prompt, image = self.message_to_promptimg(message, dataset=dataset)
            prompt_full += f'<|image|>{prompt} \nASSISTANT: '
            images.append(image)
        else:
            for msg in message:
                if msg['type'] == 'image':
                    images.append(msg['value'])
                    prompt_full += '<|image|>'
                elif msg['type'] == 'text':
                    prompt_full += msg['value']
            prompt_full += '\nASSISTANT: '

        def preproc_image(fname):
            image = Image.open(fname).convert('RGB')
            max_edge = max(image.size)
            image = image.resize((max_edge, max_edge))
            return image
        images = [preproc_image(fname) for fname in images]
        image_tensor = process_images(images, self.image_processor)
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(
            prompt_full, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=image_tensor,
                output_hidden_states=True,
                use_cache=True,
                **kwargs)
        answer = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        return answer.split('</s>')[0]
