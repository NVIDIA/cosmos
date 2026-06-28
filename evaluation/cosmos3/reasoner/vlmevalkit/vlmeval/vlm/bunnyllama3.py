import re
import string
import warnings

import pandas as pd
import torch
import transformers
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import listinstr
from .base import BaseModel


class BunnyLLama3(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='BAAI/Bunny-v1_1-Llama-3-8B-V', **kwargs):
        assert model_path is not None
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        warnings.filterwarnings('ignore')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True)
        self.kwargs = kwargs

    def use_custom_prompt(self, dataset):
        if listinstr(['MCQ', 'Y/N'], DATASET_TYPE(dataset)) or listinstr(['mathvista'], dataset.lower()):
            return True
        else:
            return False

    def build_prompt(self, line, dataset):
        if dataset is None:
            dataset = self.dataset

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)

        prompt = line['question']

        if DATASET_TYPE(dataset) == 'MCQ':
            if listinstr(['mmmu'], dataset.lower()):
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                assert hint is None

                question = line['question']
                question = re.sub(r'<image (\d+)>', lambda x: x.group(0)[1:-1], question)

                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = '\n'
                for key, item in options.items():
                    options_prompt += f'({key}) {item}\n'

                prompt = question
                if len(options):
                    prompt += options_prompt
                    prompt += "\nAnswer with the option's letter from the given choices directly."
                else:
                    prompt += '\nAnswer the question using a single word or phrase.'
            else:
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                prompt = ''
                if hint is not None:
                    prompt += f'{hint}\n'

                question = line['question']

                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = '\n'
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'

                prompt += question + options_prompt
                if listinstr(['cn', 'ccbench'], dataset.lower()):
                    prompt += '请直接回答选项字母。'
                else:
                    prompt += "Answer with the option's letter from the given choices directly."
        elif DATASET_TYPE(dataset) == 'Y/N':
            if listinstr(['mme'], dataset.lower()):
                if not listinstr(
                        ['code_reasoning', 'commonsense_reasoning', 'numerical_calculation', 'text_translation'],
                        line['category']):
                    prompt = prompt.replace(' Please answer yes or no.',
                                            '\nAnswer the question using a single word or phrase.')
            elif listinstr(['pope'], dataset.lower()):
                prompt = prompt.replace(' Please answer yes or no.',
                                        '\nAnswer the question using a single word or phrase.')
        elif listinstr(['mathvista'], dataset.lower()):
            match = re.search(r'Hint: (.*?)\nQuestion: (.*?)\n(Choices:\n(.*))?', prompt + '\n', re.DOTALL)

            prompt = match.group(2)
            if match.group(4) is not None:
                prompt += '\n' + match.group(4).rstrip('\n')
            prompt += '\n' + match.group(1)
        else:
            raise ValueError(
                f"Bunny doesn't implement a custom prompt for {dataset}. It should use the default prompt, but didn't.")

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def generate_inner(self, message, dataset=None):

        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        text = (f'A chat between a curious user and an artificial intelligence assistant. '
                f"The assistant gives helpful, detailed, and polite answers to the user's questions. "
                f'USER: <image>\n{prompt} ASSISTANT:')

        text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0)
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype)

        output_ids = self.model.generate(input_ids, images=image_tensor, max_new_tokens=128, use_cache=True)[0]
        response = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True)
        return response
