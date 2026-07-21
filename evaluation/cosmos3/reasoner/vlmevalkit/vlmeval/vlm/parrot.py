import logging
import os
import string
from abc import abstractproperty

import pandas as pd
import torch
from PIL import Image

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string, listinstr
from .base import BaseModel


class Parrot(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='AIDC-AI/Parrot-7B', **kwargs):
        try:
            from parrot.model.conversation_formatter import ConversationFormatter
            from parrot.model.parrot_arch import ParrotMetaForCausalLM
            from parrot.utils.constants import BEGIN_LINE, DEFAULT_IMAGE_TOKEN, END_LINE
            from parrot.utils.mm_utils import process_images
        except Exception as e:
            logging.critical('Please install Parrot before using Parrot')
            logging.critical('Please install Parrot from https://github.com/AIDC-AI/Parrot')
            logging.critical('Using `pip install -e . --no-deps` in the Parrot directory')
            logging.critical('Recommend to install transformers==4.39.0')
            raise e

        self.process_images = process_images
        self.ConversationFormatter = ConversationFormatter
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.BEGIN_LINE = BEGIN_LINE
        self.END_LINE = END_LINE

        try:
            model_name = 'parrot_qwen2'
            model, tokenizer, conversation_formatter = ParrotMetaForCausalLM.build(
                model_name, model_path, mm_vision_tower='openai/clip-vit-large-patch14-336'
            )
            self.model = model.cuda()
            self.vision_tower = self.model.get_vision_tower()
            self.tokenizer = tokenizer
            self.conversation_formatter = conversation_formatter
            self.image_processor = self.model.get_vision_tower().image_processor
        except Exception as e:
            logging.critical('Error when loading Parrot model:')
            raise e

        self.kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            repetition_penalty=None,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        if int(os.environ.get('LOCAL_RANK', '0')) == 0:
            print(f'Following kwargs {self.kwargs} will be used as generation config.')

        self.count = 0

    def use_custom_prompt(self, dataset):
        if DATASET_TYPE(dataset) == 'Y/N' or DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if DATASET_TYPE(dataset) == 'Y/N':
            prompt = self.built_yorn_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        else:
            raise ValueError(f'Invalid dataset type: {DATASET_TYPE(dataset)}')

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=p) for p in tgt_path])
        return message

    def built_yorn_prompt(self, line, dataset=None):
        prompt = line['question']
        previous_suffixs = [' Please answer yes or no.', ' Yes or No', ' Answer in one sentence.']
        for previous_suffix in previous_suffixs:
            if prompt.endswith(previous_suffix):
                prompt = prompt[:-len(previous_suffix)]
                break
        prompt += '\n请直接回答Yes或No。请用单个词或短语回答问题。' if cn_string(
            prompt) else '\nPlease strictly answer Yes or No. Answer the question using a single word or phrase.'
        return prompt

    def build_multi_choice_prompt(self, line, dataset=None):
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
            default_prompt = "\nAnswer with the option's letter from the given choices directly."
            if dataset[-3:] == '_cn' or cn_string(prompt):
                default_prompt = '\n请直接用给定选项中的选项字母回答。'
            elif dataset[-3:] == '_pt':
                default_prompt = '\nResponda diretamente com a letra da opção das escolhas dadas.'
            elif dataset[-3:] == '_ar':
                default_prompt = '\nأجب مباشرةً بحرف الخيار من الاختيارات المعطاة.'
            elif dataset[-3:] == '_ru':
                default_prompt = '\nОтветьте буквой варианта из предложенных вариантов напрямую.'
            elif dataset[-3:] == '_tr':
                default_prompt = '\nVerilen seçeneklerden doğrudan seçeneğin harfi ile cevap verin.'
            prompt += default_prompt
            # prompt += (
            #     '\n请直接回答选项字母。' if cn_string(prompt) else
            #     "\nAnswer with the option's letter from the given choices directly."
            # )
        else:
            prompt += '\n请用单个词或短语回答问题。' if cn_string(
                prompt) else '\nAnswer the question using a single word or phrase.'

        return prompt

    def process_answer_prefix(self, answer, prefixes):
        for prefix in prefixes:
            if prefix in answer.lower():
                return answer[answer.lower().find(prefix) + len(prefix):]
        return answer

    def generate_inner(self, message, dataset=None):
        query, image_paths = self.prepare_inputs(message)
        images_list = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensors = self.process_images(images_list, self.image_processor, args).cuda()
        prompt, input_ids = self.conversation_formatter.format_query(query)
        input_ids = input_ids.unsqueeze(0).cuda()

        with torch.inference_mode():
            kwargs = dict(
                images=image_tensors,
            )
            kwargs.update(self.kwargs)
            output_ids = self.model.generate(input_ids, **kwargs)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        response = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                               skip_special_tokens=True)[0].strip(string.whitespace)
        answer = response

        if query.endswith("Answer with the option's letter from the given choices directly.") or query.endswith(
                '请直接回答选项字母。'):
            qtype = 'multiple-choice'
            while True:
                answer = answer.strip(string.punctuation + string.whitespace)
                if len(answer) > 1:
                    if answer[0] in string.ascii_uppercase and answer[1] in string.whitespace + string.punctuation:
                        answer = answer[0]
                        break
                    elif answer[-1] in string.ascii_uppercase and answer[-2] in string.whitespace + string.punctuation:
                        answer = answer[-1]
                        break
                    elif listinstr(['answer is', 'answer:'], answer.lower()):
                        answer = self.process_answer_prefix(answer, ['answer is', 'answer:'])
                        answer = self.process_answer_prefix(answer, ['option'])
                    else:
                        break
                else:
                    break
        else:
            qtype = 'open'

        if self.count % 50 == 0 and int(os.environ.get('LOCAL_RANK', '0')) == 0:
            print(f'\n{self.BEGIN_LINE}')
            print(f'image_paths: {image_paths}\n')
            print(f'prompt: {prompt}\n')
            print(f'qtype: {qtype}\n')
            print(f'output: {response}\n')
            print(f'answer: {answer}\n')
            print(f'{self.END_LINE}\n', flush=True)

        self.count += 1

        return answer

    def prepare_inputs(self, message):
        prompt = ''
        image_paths = []
        image_count = 0
        text_count = 0
        pure_text = ''
        for msg in message:
            if msg['type'] == 'text':
                text_count += 1
                prompt += msg['value']
                pure_text += msg['value']
            elif msg['type'] == 'image':
                image_count += 1
                prompt += self.DEFAULT_IMAGE_TOKEN
                image_paths.append(msg['value'])

        if image_count == 1 and text_count == 1:
            prompt = self.DEFAULT_IMAGE_TOKEN + '\n' + pure_text

        return prompt, image_paths
