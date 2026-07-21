import re
import string

import pandas as pd
import torch
import torchvision
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import listinstr
from ..base import BaseModel

pattern = re.compile(r'[A-Z]')


def __padding__(image):
    width, height = image.size
    tar = max(width, height)
    top_padding = int((tar - height) / 2)
    bottom_padding = tar - height - top_padding
    left_padding = int((tar - width) / 2)
    right_padding = tar - width - left_padding
    image = torchvision.transforms.functional.pad(image, [left_padding, top_padding, right_padding, bottom_padding])
    return image


meta_instruction = """
You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).
- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by
Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language
chosen by the user such as English and 中文.
- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively
based on the provided image.
"""


def model_gen(model, text, images, need_bos=True, padding=False, beams=3, max_token=500):
    pt1 = 0
    embeds = []
    im_mask = []
    images = [images]
    images_loc = [0]
    for i, pts in enumerate(images_loc + [len(text)]):
        subtext = text[pt1:pts]
        if need_bos or len(subtext) > 0:
            text_embeds = model.encode_text(subtext, add_special_tokens=need_bos)
            embeds.append(text_embeds)
            im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
            need_bos = False
        if i < len(images):
            try:
                image = Image.open(images[i]).convert('RGB')
            except Exception:
                image = images[i].convert('RGB')
            if padding:
                image = __padding__(image)
            image = model.vis_processor(image).unsqueeze(0).cuda()
            image_embeds = model.encode_img(image)
            embeds.append(image_embeds)
            im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
        pt1 = pts
    embeds = torch.cat(embeds, dim=1)
    im_mask = torch.cat(im_mask, dim=1)
    im_mask = im_mask.bool()

    outputs = model.generate(
        inputs_embeds=embeds,
        im_mask=im_mask,
        temperature=1.0,
        max_new_tokens=max_token,
        num_beams=beams,
        do_sample=False,
        repetition_penalty=1.0)

    output_token = outputs[0]
    if output_token[0] == 0 or output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('[UNUSED_TOKEN_145]')[0].strip()
    return output_text


class XComposer2(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='internlm/internlm-xcomposer2-vl-7b', **kwargs):
        assert model_path is not None
        self.model_path = model_path

        model = AutoModel.from_pretrained(self.model_path, device_map='cpu', trust_remote_code=True).cuda().eval()
        model.half()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model.tokenizer = tokenizer
        self.model = model
        self.device = self.model.model.tok_embeddings.weight.device

    def generate_mme(self, image_path, text):
        text = text.split('Please answer')[0].strip()
        text = f'{text} Answer this question briefly'
        text = f'[UNUSED_TOKEN_146]user\n{text}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'

        return model_gen(self.model, text, image_path, need_bos=True, padding=True, beams=5)

    def generate_multichoice(self, image_path, text, dataset):
        out = model_gen(self.model, text, image_path, need_bos=True, padding=False, beams=5, max_token=5)
        if 'mmmu' in dataset.lower():
            return out
        res = pattern.findall(out)
        if len(res) == 0:
            print('Error:', out)
            res = 'Z'
        return res[0]

    def generate_vqa(self, image_path, text):
        out = model_gen(self.model, text, image_path, need_bos=True)
        return out

    def generate_vanilla(self, image_path, text):
        text = (
            '[UNUSED_TOKEN_146]system\n{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]user\n{}'
            'Answer this question in detail.[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
        ).format(meta_instruction, text)
        out = model_gen(self.model, text, image_path, need_bos=True, max_token=500)
        return out

    def generate_brief(self, image_path, text):
        text = (
            '[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase.{}'
            '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
        ).format(text)
        out = model_gen(self.model, text, image_path, need_bos=True, max_token=10)
        return out

    def generate_driectly(self, image_path, text):
        text = '[UNUSED_TOKEN_146]user\n{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'.format(text)
        out = model_gen(self.model, text, image_path, need_bos=True, max_token=500)
        return out

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        with torch.cuda.amp.autocast():
            if dataset is None:
                return self.generate_vanilla(image_path, prompt)
            assert isinstance(dataset, str)
            if dataset == 'MME':
                return self.generate_mme(image_path, prompt)

            elif listinstr(['hallu'], dataset.lower()):
                return self.generate_brief(image_path, prompt)

            elif listinstr(['llava'], dataset.lower()):
                return self.generate_vanilla(image_path, prompt)

            elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
                return self.generate_multichoice(image_path, prompt, dataset)

            elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
                return self.generate_vqa(image_path, prompt)

            else:
                return self.generate_vanilla(image_path, prompt)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ' or DATASET_TYPE(dataset) == 'VQA':
            return True
        return False

    def build_mcqa(self, line):
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        img_prompt = '[UNUSED_TOKEN_146]user\n'
        if len(options):
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item} '
            options_prompt = options_prompt.strip()
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None

            context = 'N/A' if hint is None else hint
            mid_prompt = 'Question: ' + question + '\nContext: ' + context + '\nOptions: ' + options_prompt
            ans_prompt = '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nThe answer is'
            prompt = img_prompt + mid_prompt + ans_prompt
        else:
            mid_prompt = f'Answer the question using a single word or phrase.{question}'
            ans_prompt = '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
            prompt = img_prompt + mid_prompt + ans_prompt

        return prompt

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        if DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_mcqa(line)
        elif DATASET_TYPE(dataset) == 'VQA':
            if 'mathvista' in dataset.lower():
                q = line['question']
                prompt = f'[UNUSED_TOKEN_146]user\n{q}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
            else:
                q = line['question']
                prompt = (
                    f'[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase.{q}'
                    '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
                )
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message
