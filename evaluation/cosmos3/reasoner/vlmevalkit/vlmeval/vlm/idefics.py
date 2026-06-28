import os.path as osp
import warnings

import torch
from PIL import Image
from transformers.image_utils import load_image

from ..smp import listinstr, splitlen
from .base import BaseModel


class IDEFICS(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='HuggingFaceM4/idefics-9b-instruct', **kwargs):
        assert osp.exists(model_path) or splitlen(model_path) == 2
        from transformers import AutoProcessor, IdeficsForVisionText2Text

        self.model = IdeficsForVisionText2Text.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        kwargs_default = {'max_new_tokens': 512}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        self.file_root = osp.dirname(__file__)
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )

    def generate_inner(self, message, dataset=None):
        prompts = (
            ['Users:']
            + [msg['value'] if msg['type'] == 'text' else Image.open(msg['value']) for msg in message]
            + ['<end_of_utterance>', '\nAssistant: ']
        )
        inputs = self.processor(
            prompts, add_end_of_utterance_token=False, return_tensors='pt'
        ).to('cuda')
        exit_condition = self.processor.tokenizer(
            '<end_of_utterance>', add_special_tokens=False
        ).input_ids
        bad_words_ids = self.processor.tokenizer(
            ['<image>', '<fake_token_around_image>'], add_special_tokens=False
        ).input_ids

        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            **self.kwargs,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        text = generated_text[0].split('\nAssistant: ')[-1]
        return text


class IDEFICS2(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='HuggingFaceM4/idefics2-8b', **kwargs):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        assert model_path is not None
        self.model_path = model_path
        if 'Idefics3' in self.model_path.lower():
            warnings.warn('Install transfomers from source: PR https://github.com/open-compass/VLMEvalKit/pull/379')
            warnings.warn('Reference: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3')
        self.processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            device_map="auto")
        self.model = model

        kwargs_default = {'max_new_tokens': 1024}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )
        torch.cuda.empty_cache()

    def _process(self, formatted_messages, formatted_images):
        inputs = self.processor(
            text=formatted_messages, images=formatted_images, return_tensors='pt'
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    def build_prompt_default(self, message, add_brief=False, add_yes_or_no=False, change_the_img_place=False):
        if change_the_img_place:
            new_message = []
            for s in message:
                if s['type'] == 'image':
                    new_message.append(s)
            for s in message:
                if s['type'] == 'text':
                    new_message.append(s)
            message = new_message
        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                prompt += msg['value'].strip()
        if add_brief:
            prompt += '\nGive a very brief answer.'
        if add_yes_or_no:
            prompt += '\nAnswer yes or no.'
        prompt += '<end_of_utterance>\nAssistant:'
        return prompt, images

    def build_prompt_puremcq(self, message):
        replace_mapping = {
            '\nOptions:': '\nChoices:',
            'Please select the correct answer from the options above.': 'Answer with the letter.',
        }

        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction
        prompt += '<end_of_utterance>\nAssistant: Answer:'
        return prompt, images

    def build_prompt_mt(self, message):
        prompt, images = '', []
        for msg in message:
            if msg['role'] == 'user':
                prompt += 'User: '
            elif msg['role'] == 'assistant':
                prompt += 'Assistant: '
            for item in msg['content']:
                if item['type'] == 'image':
                    img = load_image(item['value'])
                    images.append(img)
                    prompt += '<image>'
                elif item['type'] == 'text':
                    prompt += item['value'].strip()
                prompt += '<end_of_utterance>\n'
        return prompt + 'Assistant: '

    def build_prompt_mmbench(self, message):
        replace_mapping = {
            '\nOptions:': '\nChoices:',
            'Please select the correct answer from the options above.': 'Answer with a letter.',
        }

        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                # Swap hint and question
                if instruction.startswith('Hint:'):
                    hint, question = instruction.split('\nQuestion:')
                    question, choices = question.split('\nChoices:')
                    instruction = (
                        'Question:' + question + '\n' + hint + '\nChoices:' + choices
                    )
                prompt += instruction
        prompt += '<end_of_utterance>\nAssistant: Answer:'
        return prompt, images

    def build_prompt_mmmu(self, message):
        replace_mapping = {
            'Question:': '',
            'Please select the correct answer from the options above.': 'Answer with the letter.',
            '\nOptions:': '\nChoices:',
        }

        prompt, images, img_counter = 'User: Question: ', [], 1
        for msg in message:
            if msg['type'] == 'image':
                prompt += f'<image {img_counter}>:<image>\n'
                img_counter += 1
        img_counter = 1

        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += f' <image {img_counter}> '
                img_counter += 1
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()
        prompt += '<end_of_utterance>\nAssistant:'
        if 'A.' in prompt and 'B.' in prompt:
            prompt += ' Answer:'
        return prompt, images

    def build_prompt_mathvista(self, message):
        replace_mapping = {
            '(A) ': 'A. ',
            '(B) ': 'B. ',
            '(C) ': 'C. ',
            '(D) ': 'D. ',
            '(E) ': 'E. ',
            '(F) ': 'F. ',
            '(G) ': 'G. ',
            '(H) ': 'H. ',
            '\nOptions:': '\nChoices:',
            'Hint: ': '',
        }

        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()
        if 'A.' in prompt and 'B.' in prompt:
            prompt += '\nAnswer with the letter.'
        prompt += '<end_of_utterance>\nAssistant:'
        if 'A.' in prompt and 'B.' in prompt:
            prompt += ' Answer:'
        return prompt, images

    def chat_inner(self, message, dataset=None):
        formatted_messages, formatted_images = self.build_prompt_mt(message)
        inputs = self._process(formatted_messages, formatted_images)

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True
        )[0]
        response = generated_text.strip()
        # print(dataset, " | ", formatted_messages.replace("\n", "\\n"), " | ", response.replace("\n", "\\n"))
        return response

    def generate_inner(self, message, dataset=None):
        if dataset in [
            'MMBench_DEV_EN', 'MMBench_DEV_EN_V11',
            'MMBench_TEST_EN', 'MMBench_TEST_EN_V11',
            'MMBench_DEV_CN', 'MMBench_DEV_CN_V11',
            'MMBench_TEST_CN', 'MMBench_TEST_CN_V11',
            'MMBench', 'MMBench_V11', 'MMBench_CN', 'MMBench_CN_V11'
        ]:
            formatted_messages, formatted_images = self.build_prompt_mmbench(message)
        elif dataset in ['MMMU_DEV_VAL', 'MMMU_TEST']:
            formatted_messages, formatted_images = self.build_prompt_mmmu(message)
        elif dataset in ['MathVista_MINI']:
            formatted_messages, formatted_images = self.build_prompt_mathvista(message)
        elif dataset in [
            'MME',
            'MMVet',
            'OCRVQA_TEST',
            'OCRVQA_TESTCORE',
            'TextVQA_VAL',
            'ChartQA_TEST',
            'DocVQA_VAL',
            'DocVQA_TEST',
            'InfoVQA_VAL',
            'InfoVQA_TEST',
        ]:
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_brief=True
            )
        elif dataset == 'HallusionBench':
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_yes_or_no=True
            )
        elif dataset in [
            'MMStar',
            'SEEDBench_IMG',
            'AI2D_TEST',
            'ScienceQA_VAL',
            'ScienceQA_TEST',
        ]:
            formatted_messages, formatted_images = self.build_prompt_puremcq(message)
        elif dataset is not None and listinstr(['MLVU', 'TempCompass', 'MVBench'], dataset):
            formatted_messages, formatted_images = self.build_prompt_default(message, change_the_img_place=True)
        else:
            formatted_messages, formatted_images = self.build_prompt_default(message)

        inputs = self._process(formatted_messages, formatted_images)

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True
        )[0]
        response = generated_text.strip()
        # print(dataset, " | ", formatted_messages.replace("\n", "\\n"), " | ", response.replace("\n", "\\n"))
        return response
