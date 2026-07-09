import logging
import math
import os
import random
import re
import string

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE
from vlmeval.smp import encode_image_to_base64, listinstr
from .base import BaseModel


class MiniCPM_V(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='openbmb/MiniCPM-V', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        print(f'load from {self.model_path}')
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = self.model.to(dtype=torch.bfloat16)
        self.model.eval().cuda()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        self.num_beams = 3

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'MMAlignBench', 'ChartQAPro', 'ChartMuseum'], dataset):  # noqa: E501
            # For Multi-Turn we don't have custom prompt
            return False
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'{question}\n'
        if len(options):
            prompt += options_prompt
            prompt = 'Study the image carefully and pick the option associated with the correct answer. \
                Focus solely on selecting the option and avoid including any other content.\n' + prompt
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=p) for p in tgt_path])

        return message

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')
        msgs = [{'role': 'user', 'content': prompt}]
        if DATASET_TYPE(dataset) == 'MCQ':
            max_new_tokens = 20
        elif DATASET_TYPE(dataset) == 'Y/N':
            max_new_tokens = 100
        else:
            max_new_tokens = 1024

        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams
        )
        default_kwargs.update(self.kwargs)
        res, _, _ = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            **default_kwargs
        )
        return res


class MiniCPM_Llama3_V(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='openbmb/MiniCPM-Llama3-V-2_5', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        print(f'load from {self.model_path}')
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = self.model.to(dtype=torch.float16)
        self.model.eval().cuda()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        self.num_beams = 3
        self.options_system_prompt = ('Carefully read the following question and select the letter corresponding '
                                      'to the correct answer. Highlight the applicable choices without giving '
                                      'explanations.')
        self.wo_options_system_prompt = 'Carefully read the following question Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'

    def use_custom_prompt(self, dataset):
        if listinstr(['MCQ', 'VQA'], DATASET_TYPE(dataset)):
            return True
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            return True
        return False

    def build_prompt(self, line, dataset=None):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt = ''

        question = line['question']
        if DATASET_TYPE(dataset) == 'MCQ':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = ''
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            prompt += f'Question: {question}\n'
            if len(options):
                prompt += options_prompt
                system_prompt = self.options_system_prompt + '\nPlease just indicate your choice.'
            else:
                system_prompt = self.wo_options_system_prompt
            if 'MMMU' in dataset:  # Corner Case
                prompt = system_prompt + '\n' + prompt
                system_prompt = ''
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question'] + ' Yes or No?'
            prompt = question
        elif dataset is not None and listinstr(['MME'], dataset):
            question = line['question'] + ' Yes or No?'
            prompt = question
        elif dataset is not None and listinstr(['OCRBench'], dataset):
            system_prompt = self.vqa_prompt
            question = line['question']
            prompt = question
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['LLaVABench', 'MMLongBench_DOC'], dataset):
                system_prompt = ''
                prompt = question
            elif listinstr(['MMVet'], dataset):
                system_prompt = self.detail_system_prompt
                prompt = question
            else:
                system_prompt = self.vqa_prompt
                prompt = question

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def generate_inner(self, message, dataset=None):
        if DATASET_TYPE(dataset) == 'MCQ':
            max_new_tokens = 200
        elif DATASET_TYPE(dataset) == 'Y/N':
            max_new_tokens = 3
        else:
            max_new_tokens = 1024

        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        content = []
        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                content.append(image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            msgs=msgs,
            context=None,
            image=None,
            tokenizer=self.tokenizer,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]
        return res

    def chat_inner(self, message, dataset=None):
        max_new_tokens = 1024

        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        msgs = []
        for msg in message:
            content = []
            if len(msg['content']) == 1 and msg['content'][0]['type'] == 'text':
                msg_new = {'role': msg['role'], 'content': msg['content'][0]['value']}
                msgs.append(msg_new)
                continue

            for x in msg['content']:
                if x['type'] == 'text':
                    content.append(x['value'])
                elif x['type'] == 'image':
                    image = Image.open(x['value']).convert('RGB')
                    content.append(image)
            msg_new = {'role': msg['role'], 'content': content}
            msgs.append(msg_new)

        res = self.model.chat(
            msgs=msgs,
            context=None,
            image=None,
            tokenizer=self.tokenizer,
            **default_kwargs)

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]
        return res


class MiniCPM_V_2_6(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='openbmb/MiniCPM-V-2_6', **kwargs):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        assert model_path is not None
        self.model_path = model_path
        print(f'load from path {self.model_path}')
        if self.use_lmdeploy:
            logging.warning(
                'Currently LMDeploy does not support interleaved text-image prompt. '
                'All images will be placed at the beginning of the prompt, '
                'which may lead to performance degradation.'
            )
            from lmdeploy import TurbomindEngineConfig, pipeline
            num_gpus = torch.cuda.device_count()
            self.model = pipeline(
                model_path,
                backend_config=TurbomindEngineConfig(session_len=32768, cache_max_entry_count=0.1, tp=num_gpus)
            )
            torch.cuda.set_device(0)
            self.device = 'cuda'
        else:
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = self.model.to(dtype=torch.bfloat16)
            self.model.eval().cuda()

            self.kwargs = kwargs
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            torch.cuda.empty_cache()

        self.num_beams = 3

        self.options_suffix_prompt = '''\nAnswer with the option's letter from the given choices directly.'''
        self.wo_options_system_prompt = 'Carefully read the following question Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'

        self.multi_choice_cot_prompt = ('''Carefully read the following multichoice question, solve it step '''
                                        '''by step and finally pick the option associated with the correct '''
                                        '''answer in the format of "Answer: selected option\n\n''')
        self.short_ans_cot_prompt = ('''Read the following question carefully, solve it step by step, and '''
                                     '''then output the final answer in the format of "Answer: single number '''
                                     '''or single word or phrase".\n\n''')

    def use_custom_prompt(self, dataset=None):
        if dataset is None:
            return False
        if DATASET_TYPE(dataset) in ['MCQ', 'VQA', 'Y/N']:
            return True
        return False

    def use_cot(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MMMU', 'HallusionBench', 'OCRBench', 'ChartQA'], dataset):
            return True
        elif listinstr(['MathVista', 'MMVet', 'MMBench', 'MMStar', 'AI2D', 'RealWorldQA',
                        'POPE', 'ScienceQA', 'TextVQA', 'DocVQA'], dataset):
            return False
        else:
            return False

    def use_upsize(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MMVet', 'MMBench', 'MMStar', 'AI2D', 'OCRBench'], dataset):
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt, prompt = '', ''

        question = line['question']

        if not self.use_cot(dataset):
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = 'Options:\n'
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'Question: {question}\n'
                if len(options):
                    prompt += options_prompt
                    prompt += self.options_suffix_prompt
                else:
                    system_prompt = self.wo_options_system_prompt

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            elif dataset is not None and listinstr(['HallusionBench'], dataset):
                question += ' Yes or No?'
                prompt = question
            elif dataset is not None and listinstr(['OCRBench'], dataset):
                system_prompt = self.vqa_prompt
                prompt = question
            elif DATASET_TYPE(dataset) == 'VQA':
                if listinstr(['LLaVABench'], dataset):
                    system_prompt = ''
                elif listinstr(['MMVet'], dataset):
                    system_prompt = self.detail_system_prompt
                else:
                    system_prompt = self.vqa_prompt
                prompt = question
            else:
                prompt = question
        else:
            has_options = True
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = ''
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'{question}\n'

                if len(options):
                    prompt += options_prompt
                else:
                    has_options = False

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            else:
                prompt = question

            if DATASET_TYPE(dataset) in ['MCQ', 'Y/N', 'VQA']:
                if DATASET_TYPE(dataset) == 'MCQ':
                    if has_options:
                        prompt = self.multi_choice_cot_prompt + prompt
                    else:
                        prompt = self.short_ans_cot_prompt + prompt
                elif DATASET_TYPE(dataset) == 'Y/N':
                    prompt = self.short_ans_cot_prompt + prompt
                else:
                    prompt = self.short_ans_cot_prompt + prompt

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def message_to_lmdeploy(self, messages, system_prompt=None):
        """
        TODO:
        Support interleaved text-image prompt
        after LMDeploy supports it.
        """
        from PIL import Image
        prompt, image_path = '', []
        for msg in messages:
            if msg['type'] == 'text':
                prompt += msg['value']
            elif msg['type'] == 'image':
                image_path.append(msg['value'])
        content = [{'type': 'text', 'text': prompt}]
        for image in image_path:
            img = Image.open(image).convert('RGB')
            b64 = encode_image_to_base64(img)
            img_struct = dict(url=f'data:image/jpeg;base64,{b64}')
            content.append(dict(type='image_url', image_url=img_struct))
        ret = []
        if system_prompt is not None:
            ret.append(dict(role='system', content=system_prompt))
        ret.append(dict(role='user', content=content))
        return [ret]

    def generate_inner_transformers(self, message, dataset=None):
        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            max_slice_nums = 1
            use_image_id = False
            max_inp_length = 2048 * 10
        else:
            max_slice_nums = None
            use_image_id = True
            max_inp_length = 8192

        max_new_tokens = 2048
        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        content = []

        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                if not self.use_upsize(dataset):
                    content.append(image)
                else:
                    img_width, img_height = image.width, image.height
                    if (img_width * img_height) >= (1344 * 1344):
                        content.append(image)
                    else:
                        ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                        max_img_width = int(img_width * ratio)
                        new_img_width = random.randint(img_width, max_img_width)
                        new_img_height = int(new_img_width / img_width * img_height)
                        resized_image = image.resize((new_img_width, new_img_height))
                        content.append(resized_image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            image=None,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            max_inp_length=max_inp_length,
            use_image_id=use_image_id,
            max_slice_nums=max_slice_nums,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]

        return res

    def generate_inner_lmdeploy(self, message, dataset=None):
        from lmdeploy import GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=2048,
            top_p=0.001,
            top_k=1,
            temperature=0.01,
            repetition_penalty=1.0,
        )
        gen_config.random_seed = None
        messages_list = self.message_to_lmdeploy(message, system_prompt=None)
        assert len(messages_list) == 1
        response = self.model(messages_list, gen_config=gen_config)[0]
        response = response.text
        return response

    def generate_inner(self, message, dataset=None):
        if self.use_lmdeploy:
            return self.generate_inner_lmdeploy(message, dataset)
        else:
            return self.generate_inner_transformers(message, dataset)


class MiniCPM_o_2_6(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='openbmb/MiniCPM-o-2_6', **kwargs):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        assert model_path is not None
        self.model_path = model_path
        print(f'load from path {self.model_path}')
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=False,
            init_tts=False
        )

        self.model.eval().cuda()

        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()

        self.num_beams = int(os.getenv("NUM_BEAMS", "3"))

        repetition_penalty = float(os.getenv("PENALTY", "1.2"))
        self.repetition_penalty = repetition_penalty

        self.options_suffix_prompt = '''\nAnswer with the option's letter from the given choices directly.'''
        self.wo_options_system_prompt = 'Carefully read the following question Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'

        self.multi_choice_cot_prompt = ('''Carefully read the following multichoice question, solve it step '''
                                        '''by step and finally pick the option associated with the correct '''
                                        '''answer in the format of "Answer: selected option\n\n''')
        self.short_ans_cot_prompt = ('''Read the following question carefully, solve it step by step, and '''
                                     '''then output the final answer in the format of "Answer: single number '''
                                     '''or single word or phrase".\n\n''')

    def use_custom_prompt(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MCQ', 'VQA', 'Y/N'], DATASET_TYPE(dataset)) and not listinstr(['Video'], DATASET_TYPE(dataset)):
            return True
        return False

    def use_cot(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MMMU', 'MathVista', 'OCRBench', 'ChartQA', 'MathVision', 'MathVerse_MINI_Vision_Only'], dataset):
            return True
        elif listinstr(['MMVet', 'MMBench', 'MMStar', 'HallusionBench', 'AI2D', 'RealWorldQA',
                        'POPE', 'ScienceQA', 'TextVQA', 'DocVQA'], dataset):
            return False
        else:
            return False

    def use_upsize(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MathVista', 'MMBench_TEST_CN', 'MMStar', 'AI2D', 'OCRBench', 'DynaMath'], dataset):
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt, prompt = '', ''

        question = line['question']

        if not self.use_cot(dataset):
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = 'Options:\n'
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'Question: {question}\n'
                if len(options):
                    prompt += options_prompt
                    prompt += self.options_suffix_prompt
                else:
                    system_prompt = self.wo_options_system_prompt

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            elif dataset is not None and listinstr(['HallusionBench'], dataset):
                question += ' Yes or No?'
                prompt = question
            elif dataset is not None and listinstr(['OCRBench'], dataset):
                system_prompt = self.vqa_prompt
                prompt = question
            elif DATASET_TYPE(dataset) == 'VQA':
                if listinstr(['LLaVABench'], dataset):
                    system_prompt = ''
                elif listinstr(['MMVet'], dataset):
                    system_prompt = self.detail_system_prompt
                else:
                    system_prompt = self.vqa_prompt
                prompt = question
            else:
                prompt = question
        else:
            has_options = True
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = ''
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'{question}\n'

                if len(options):
                    prompt += options_prompt
                else:
                    has_options = False

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            else:
                prompt = question

            if DATASET_TYPE(dataset) in ['MCQ', 'Y/N', 'VQA']:
                if DATASET_TYPE(dataset) == 'MCQ':
                    if has_options:
                        prompt = self.multi_choice_cot_prompt + prompt
                    else:
                        prompt = self.short_ans_cot_prompt + prompt
                elif DATASET_TYPE(dataset) == 'Y/N':
                    prompt = self.short_ans_cot_prompt + prompt
                else:
                    prompt = self.short_ans_cot_prompt + prompt

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def extract_answer(self, res, dataset=None):
        if dataset is None:
            return res
        if self.use_cot(dataset):
            if DATASET_TYPE(dataset) == 'MCQ':
                pattern = r'Answer:\s*([A-Ia-i])(?![A-Za-z])'
                matches = re.findall(pattern, res, re.DOTALL)
                if matches:
                    extracted_res = matches[-1].strip()
                else:
                    extracted_res = res
                return extracted_res
            elif DATASET_TYPE(dataset) == 'VQA' and not listinstr(['OCRBench'], dataset):
                pattern = r'Answer:\s*(.*)\s*$'
                match = re.search(pattern, res, re.DOTALL)
                if match:
                    extracted_res = match.group(1)
                else:
                    extracted_res = res
                return extracted_res
        return res

    def generate_inner(self, message, dataset=None):
        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            max_slice_nums = 1
            use_image_id = False
            max_inp_length = 2048 * 10
        else:
            max_slice_nums = None
            use_image_id = True
            max_inp_length = 8192

        max_new_tokens = 2048
        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            repetition_penalty=self.repetition_penalty,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        content = []

        if dataset is not None and DATASET_TYPE(dataset) == 'Video-MCQ':
            message.append(dict(type='text', value=self.options_suffix_prompt))

        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                if not self.use_upsize(dataset):
                    content.append(image)
                else:
                    img_width, img_height = image.width, image.height
                    if (img_width * img_height) >= (1344 * 1344):
                        content.append(image)
                    else:
                        ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                        max_img_width = int(img_width * ratio)
                        new_img_width = random.randint(img_width, max_img_width)
                        new_img_height = int(new_img_width / img_width * img_height)
                        resized_image = image.resize((new_img_width, new_img_height))
                        content.append(resized_image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            image=None,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            max_inp_length=max_inp_length,
            use_image_id=use_image_id,
            max_slice_nums=max_slice_nums,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]

        res = self.extract_answer(res, dataset)

        return res


class MiniCPM_V_4(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='openbmb/MiniCPM-V-4', **kwargs):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        assert model_path is not None
        self.model_path = model_path
        print(f'load from path {self.model_path}')
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = self.model.to(dtype=torch.bfloat16)
        self.model.eval().cuda()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()

        self.num_beams = 3
        self.max_new_tokens = 2048
        self.options_suffix_prompt = '''\nAnswer with the option's letter from the given choices directly.'''
        self.wo_options_system_prompt = 'Carefully read the following question. Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'
        self.multi_choice_cot_prompt = ('''Carefully read the following multichoice question, solve it step '''
                                        '''by step and finally pick the option associated with the correct '''
                                        '''answer in the format of "Answer: selected option\n\n''')
        self.short_ans_cot_prompt = ('''Read the following question carefully, solve it step by step, and '''
                                     '''then output the final answer in the format of "Answer: single number '''
                                     '''or single word or phrase".\n\n''')
        self.ocrbench_cot_prompt = 'Carefully observe the image and answer the OCR-related questions below. \n\n'

    def use_custom_prompt(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MCQ', 'VQA', 'Y/N'], DATASET_TYPE(dataset)):
            return True
        return False

    def use_cot(self, dataset=None):
        if dataset is None:
            return False
        if listinstr([
                'MMMU', 'MathVista', 'MMStar', 'HallusionBench', 'OCRBench',
                'ChartQA', 'MathVision', 'MathVerse_MINI_Vision_Only'
        ], dataset):
            return True
        elif listinstr([
                'MMVet', 'MMBench', 'AI2D', 'RealWorldQA', 'POPE', 'ScienceQA',
                'TextVQA', 'DocVQA'
        ], dataset):
            return False
        else:
            return False

    def use_upsize(self, dataset=None):
        if dataset is None:
            return False
        if listinstr([
                'MathVista', 'MMVet', 'MMStar', 'AI2D', 'OCRBench', 'ChartQA',
                'TextVQA'
        ], dataset):
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt, prompt = '', ''

        question = line['question']

        if not self.use_cot(dataset):
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = 'Options:\n'
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'Question: {question}\n'
                if len(options):
                    prompt += options_prompt
                    prompt += self.options_suffix_prompt
                else:
                    system_prompt = self.wo_options_system_prompt

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            elif dataset is not None and listinstr(['HallusionBench'], dataset):
                question += ' Yes or No?'
                prompt = question
            elif dataset is not None and listinstr(['OCRBench'], dataset):
                system_prompt = self.vqa_prompt
                prompt = question
            elif DATASET_TYPE(dataset) == 'VQA':
                if listinstr(['LLaVABench'], dataset):
                    system_prompt = ''
                elif listinstr(['MMVet'], dataset):
                    system_prompt = self.detail_system_prompt
                else:
                    system_prompt = self.vqa_prompt
                prompt = question
            else:
                prompt = question
        else:
            has_options = True
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = ''
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'{question}\n'

                if len(options):
                    prompt += options_prompt
                else:
                    has_options = False

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            else:
                prompt = question

            if DATASET_TYPE(dataset) in ['MCQ', 'Y/N', 'VQA']:
                if DATASET_TYPE(dataset) == 'MCQ':
                    if has_options:
                        prompt = self.multi_choice_cot_prompt + prompt
                    else:
                        prompt = self.short_ans_cot_prompt + prompt
                elif DATASET_TYPE(dataset) == 'Y/N':
                    prompt = self.short_ans_cot_prompt + prompt
                elif listinstr(['OCRBench'], dataset):
                    prompt = self.ocrbench_cot_prompt + prompt
                else:
                    prompt = self.short_ans_cot_prompt + prompt

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        if dataset.startswith('MMMU_'):
            from .. import MMMUDataset
            msgs = MMMUDataset.split_MMMU(msgs)

        return msgs

    def extract_answer(self, res, dataset=None):
        if dataset is None:
            return res
        if self.use_cot(dataset):
            if DATASET_TYPE(dataset) == 'MCQ':
                pattern = r'Answer:\s*([A-Ia-i])(?![A-Za-z])'
                matches = re.findall(pattern, res, re.DOTALL)
                if matches:
                    extracted_res = matches[-1].strip()
                else:
                    extracted_res = res
                return extracted_res
            elif DATASET_TYPE(dataset) == 'VQA' and not listinstr(['OCRBench', 'MMVet'], dataset):
                pattern = r'Answer:\s*(.*)\s*$'
                match = re.search(pattern, res, re.DOTALL)
                if match:
                    extracted_res = match.group(1)
                else:
                    extracted_res = res
                return extracted_res
            elif DATASET_TYPE(dataset) == 'Y/N':
                pattern = r'Answer:\s*(.*)\s*$'
                match = re.search(pattern, res, re.DOTALL)
                if match:
                    extracted_res = match.group(1)
                else:
                    extracted_res = res
                return extracted_res
        return res

    def generate_inner(self, message, dataset=None):
        if self.use_cot(dataset):
            max_new_tokens = self.max_new_tokens
        else:
            max_new_tokens = 1024
        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        content = []

        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                if not self.use_upsize(dataset):
                    content.append(image)
                else:
                    img_width, img_height = image.width, image.height
                    if (img_width * img_height) >= (1344 * 1344):
                        content.append(image)
                    else:
                        ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                        max_img_width = int(img_width * ratio)
                        new_img_width = random.randint(img_width, max_img_width)
                        new_img_height = int(new_img_width / img_width * img_height)
                        resized_image = image.resize((new_img_width, new_img_height))
                        content.append(resized_image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            image=None,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            max_inp_length=8192,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]
        res = self.extract_answer(res, dataset)

        return res


class MiniCPM_V_4_5(MiniCPM_V_4):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='openbmb/MiniCPM-V-4_5', **kwargs):
        super().__init__(model_path, **kwargs)
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._original_chat_template = self.tokenizer.chat_template
        self._long_cot_chat_template = self._original_chat_template.replace(
            "{{- '<think>\\n' }}", "{{- '<think>\\nI' }}")

    def use_long_cot(self, dataset=None):
        if dataset is None:
            return False
        if listinstr([
                'MMMU', 'MathVista', 'MMVet', 'MMBench', 'HallusionBench',
                'MMStar', 'MathVision', 'MathVerse_MINI',
                'MathVerse_MINI_Vision_Only', 'DynaMath', 'LogicVista'
        ], dataset):
            return True
        else:
            return False

    def use_cot(self, dataset=None):
        if dataset is None:
            return False
        if listinstr([
                'MMMU', 'MathVista', 'MMBench', 'HallusionBench', 'MMStar',
                'OCRBench', 'ChartQA', 'MathVision', 'MathVerse_MINI',
                'MathVerse_MINI_Vision_Only', 'DynaMath', 'LogicVista'
        ], dataset):
            return True
        else:
            return False

    def use_upsize(self, dataset=None):
        if dataset is None:
            return False
        if self.use_long_cot(dataset):
            return True
        if listinstr(['AI2D', 'OCRBench', 'ChartQA', 'TextVQA'], dataset):
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        if self.use_long_cot(dataset):
            self.tokenizer.chat_template = self._long_cot_chat_template
        else:
            self.tokenizer.chat_template = self._original_chat_template

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt, prompt = '', ''
        question = line['question']

        if not self.use_cot(dataset):
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = 'Options:\n'
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'Question: {question}\n'
                if len(options):
                    prompt += options_prompt
                    prompt += self.options_suffix_prompt
                else:
                    system_prompt = self.wo_options_system_prompt

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            elif dataset is not None and listinstr(['HallusionBench'], dataset):
                question += ' Yes or No?'
                prompt = question
            elif dataset is not None and listinstr(['OCRBench'], dataset):
                system_prompt = self.vqa_prompt
                prompt = question
            elif DATASET_TYPE(dataset) == 'VQA':
                if listinstr(['LLaVABench'], dataset):
                    system_prompt = ''
                elif listinstr(['MMVet'], dataset):
                    system_prompt = self.detail_system_prompt
                else:
                    system_prompt = self.vqa_prompt
                prompt = question
            else:
                prompt = question
        else:
            has_options = True
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = ''
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'{question}\n'

                if len(options):
                    prompt += options_prompt
                else:
                    has_options = False

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            else:
                prompt = question

            if DATASET_TYPE(dataset) in ['MCQ', 'Y/N', 'VQA']:
                if DATASET_TYPE(dataset) == 'MCQ':
                    if has_options:
                        prompt = self.multi_choice_cot_prompt + prompt
                    else:
                        prompt = self.short_ans_cot_prompt + prompt
                elif DATASET_TYPE(dataset) == 'Y/N':
                    prompt = self.short_ans_cot_prompt + prompt
                elif listinstr(['OCRBench'], dataset):
                    prompt = self.ocrbench_cot_prompt + prompt
                else:
                    prompt = self.short_ans_cot_prompt + prompt

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        if dataset.startswith('MMMU_'):
            from .. import MMMUDataset
            msgs = MMMUDataset.split_MMMU(msgs)

        return msgs

    def generate_inner(self, message, dataset=None):
        if self.use_long_cot(dataset):
            default_kwargs = dict(
                enable_thinking=True,
                max_new_tokens=8192,
                sampling=True,
                temperature=0.7,
                num_beams=1,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0
            )
        elif self.use_cot(dataset):
            default_kwargs = dict(
                max_new_tokens=2048,
                sampling=False,
                num_beams=self.num_beams,
                repetition_penalty=1.2
            )
        else:
            default_kwargs = dict(
                max_new_tokens=1024,
                sampling=False,
                num_beams=self.num_beams,
                repetition_penalty=1.2
            )

        default_kwargs.update(self.kwargs)

        content = []
        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                if not self.use_upsize(dataset):
                    content.append(image)
                else:
                    img_width, img_height = image.width, image.height
                    if (img_width * img_height) >= (1344 * 1344):
                        content.append(image)
                    else:
                        ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                        max_img_width = int(img_width * ratio)
                        new_img_width = random.randint(img_width, max_img_width)
                        new_img_height = int(new_img_width / img_width * img_height)
                        resized_image = image.resize((new_img_width, new_img_height))
                        content.append(resized_image)
        msgs = [{'role': 'user', 'content': content}]

        self.processor.tokenizer = self.tokenizer

        res = self.model.chat(
            image=None,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_inp_length=8192,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]

        res = res.replace('<think>\n', '<think>\nI ')
        res = self.extract_answer(res, dataset)

        return res


class MiniCPM_o_4_5(MiniCPM_V_4):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='openbmb/MiniCPM-o-4_5', **kwargs):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        assert model_path is not None
        self.model_path = model_path
        print(f'load from path {self.model_path}')
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=False,
            init_tts=False,
        )
        self.model.eval().cuda()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()

        self.num_beams = 3
        self.max_new_tokens = 2048
        self.options_suffix_prompt = '''\nAnswer with the option's letter from the given choices directly.'''
        self.wo_options_system_prompt = 'Carefully read the following question. Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'
        self.multi_choice_cot_prompt = ('''Carefully read the following multichoice question, solve it step '''
                                        '''by step and finally pick the option associated with the correct '''
                                        '''answer in the format of "Answer: selected option\n\n''')
        self.short_ans_cot_prompt = ('''Read the following question carefully, solve it step by step, and '''
                                     '''then output the final answer in the format of "Answer: single number '''
                                     '''or single word or phrase".\n\n''')
        self.ocrbench_cot_prompt = 'Carefully observe the image and answer the OCR-related questions below. \n\n'

    def use_long_cot(self, dataset=None):
        if dataset is None:
            return False
        if listinstr([
                'MMMU', 'MathVista', 'MMStar', 'MathVision',
                'MathVerse_MINI', 'MathVerse_MINI_Vision_Only',
                'DynaMath', 'LogicVista',
                'MMBench', 'OCRBench', 'AI2D', 'HallusionBench'
        ], dataset):
            return True
        else:
            return False

    def use_cot(self, dataset=None):
        if dataset is None:
            return False
        if listinstr([
                'MMMU', 'MathVista', 'MMStar', 'OCRBench',
                'MathVision', 'MathVerse_MINI',
                'DynaMath', 'LogicVista',
                'MMBench', 'AI2D'
        ], dataset):
            return True
        elif listinstr([
                'MMVet',
                'RealWorldQA', 'POPE', 'ScienceQA', 'TextVQA', 'DocVQA'
        ], dataset):
            return False
        else:
            return False

    def use_upsize(self, dataset=None):
        if dataset is None:
            return False
        if listinstr([
                'MMMU', 'MathVista', 'MMStar', 'OCRBench',
                'ChartQA', 'TextVQA', 'DocVQA',
                'AI2D', 'MMBench', 'MMVet',
        ], dataset):
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        return MiniCPM_V_4.build_prompt(self, line, dataset)

    def generate_inner(self, message, dataset=None):
        if self.use_long_cot(dataset):
            default_kwargs = dict(
                enable_thinking=True,
                max_new_tokens=8192,
                do_sample=True,
                temperature=0.7,
                num_beams=1,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0
            )
        else:
            default_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=self.num_beams,
            )

        default_kwargs.update(self.kwargs)

        content = []
        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                if not self.use_upsize(dataset):
                    content.append(image)
                else:
                    img_width, img_height = image.width, image.height
                    if (img_width * img_height) >= (1344 * 1344):
                        content.append(image)
                    else:
                        ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                        max_img_width = int(img_width * ratio)
                        new_img_width = random.randint(img_width, max_img_width)
                        new_img_height = int(new_img_width / img_width * img_height)
                        resized_image = image.resize((new_img_width, new_img_height))
                        content.append(resized_image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            image=None,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_inp_length=8192,
            use_tts_template=False,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]

        res = self.extract_answer(res, dataset)

        return res
