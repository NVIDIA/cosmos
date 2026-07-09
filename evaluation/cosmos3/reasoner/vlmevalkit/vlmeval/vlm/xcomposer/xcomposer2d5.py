import logging
import re
import string
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import listinstr
from ..base import BaseModel

pattern = re.compile(r'[A-Z]')
conv_pattern = '\\[UNUSED_TOKEN_146\\]user\\\n|\\[UNUSED_TOKEN_146\\]assistant\\\n|\\[UNUSED_TOKEN_145\\]'


def get_font():
    try:
        truetype_url = "https://opencompass.openxlab.space/utils/Fonts/SimHei.ttf"
        ff = urlopen(truetype_url)
        font = ImageFont.truetype(ff, size=40)
    except Exception as e:
        logging.warning(f'{type(e)}: {e}')
        logging.warning("Fail to download the font. Use the default one.")
        font = ImageFont.load_default(size=40)
    return font


def padding_560(b):
    width, height = b.size
    tar = int(np.ceil(height / 560) * 560)
    top_padding = int((tar - height) / 2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = transforms.functional.pad(
        b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255, 255, 255])

    return b


def Identity_transform(img, hd_num=25):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width / height)
    scale = 1
    new_h = int(scale * 560)
    new_w = int(new_h * ratio)
    # print (new_h, new_w)

    img = transforms.functional.resize(img, [new_h, new_w],)
    img = img.transpose(Image.TRANSPOSE)
    img = padding_560(img)
    width, height = img.size
    if not trans:
        img = img.transpose(Image.TRANSPOSE)

    return img


def HD_transform(img, im_num=36, id_scale=1.5):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width / height)
    scale = 1
    while scale * np.ceil(scale / ratio) <= im_num:
        scale += 1
    scale -= 1

    scale = min(np.ceil(width * id_scale / 560), scale)
    new_w = int(scale * 560)
    new_h = int(new_w / ratio)

    try:
        img = transforms.functional.resize(img, [new_h, new_w])
    except Exception:
        print((height, width))
        print((new_h, new_w))
        print(im_num)
    img = padding_560(img)
    width, height = img.size
    assert width * height <= im_num * 560 * 560
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img


def img_process(imgs):
    new_imgs = []
    for img in imgs:
        w, h = img.size
        scale = w / h
        if w > h:
            new_w = 560 * 2
            new_h = int(560 * 2 / scale)
        else:
            new_w = int(560 * 2 * scale)
            new_h = 560 * 2
        img = transforms.functional.resize(img, [new_h, new_w],)
        new_imgs.append(img)
    imgs = new_imgs
    new_w = 0
    new_h = 0
    pad = 40
    if w > h:
        for im in imgs:
            w, h = im.size
            new_w = max(new_w, w)
            new_h += h + 10 + pad
        font = get_font()
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_h = 0
        for idx, im in enumerate(imgs):
            w, h = im.size
            new_img.paste(im, (0, pad + curr_h))
            draw.text((0, curr_h), f'<IMAGE {idx}>', font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(0, pad + curr_h + h + 5), (new_w, pad + curr_h + h + 5)], fill='black', width=2)
            curr_h += h + 10 + pad
        # print (new_w, new_h)
    else:
        for im in imgs:
            w, h = im.size
            new_w += w + 10
            new_h = max(new_h, h)
        new_h += pad
        font = get_font()
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_w = 0
        for idx, im in enumerate(imgs):
            w, h = im.size
            new_img.paste(im, (curr_w, pad))
            draw.text((curr_w, 0), f'<IMAGE {idx}>', font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(curr_w + w + 5, 0), (curr_w + w + 5, new_h)], fill='black', width=2)
            curr_w += w + 10
    return new_img


meta_instruction = """You are an AI assistant whose name is InternLM (书生·浦语).\n" + "- InternLM (书生·浦语) \
is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室).
It is designed to be helpful, honest, and harmless.\n"+"- InternLM (书生·浦语) \
can understand and communicate fluently in the language chosen by the user such as English and 中文."""


def model_gen(model, text, images, need_bos=True, padding=False, beams=3, max_token=2048, video_input=False):
    embeds = []
    im_mask = []
    # print(text)

    im_idx = 0
    sub_q = text.split('<IM_POS>')
    add_im = len(sub_q) - 1
    for subtext in sub_q:
        if need_bos or len(subtext) > 0:
            text_embeds = model.encode_text(
                subtext, add_special_tokens=need_bos)
            embeds.append(text_embeds)
            im_mask.append(torch.zeros(text_embeds.shape[:2]).to(model.device))
            need_bos = False

        if im_idx < len(images) and add_im:
            image = images[im_idx]
            if video_input:
                image = Identity_transform(image)
            else:
                if len(images) > 1:
                    image = HD_transform(image, im_num=max(4, model.hd_num // len(images)), id_scale=model.id_scale)
                else:
                    image = HD_transform(
                        image, im_num=model.hd_num, id_scale=model.id_scale)
            # print(image.size)
            image = model.vis_processor(image).unsqueeze(0).to(model.device)
            image_embeds = model.encode_img(image)
            im_idx += 1
            add_im -= 1
            embeds.append(image_embeds)
            im_mask.append(torch.ones(
                image_embeds.shape[:2], dtype=torch.long).to(model.device))

    embeds = torch.cat(embeds, dim=1)
    im_mask = torch.cat(im_mask, dim=1)
    im_mask = im_mask.bool()

    outputs = model.generate(inputs_embeds=embeds, im_mask=im_mask,
                             temperature=1.0, max_new_tokens=max_token, num_beams=beams,
                             do_sample=False, repetition_penalty=1.0)

    output_token = outputs[0]
    if output_token[0] == 0 or output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('[UNUSED_TOKEN_145]')[0].strip().split('<|im_end|>')[0].strip().split('The answer is')[-1].strip()  # noqa
    # print(output_text)
    return output_text


class XComposer2d5(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='internlm/internlm-xcomposer2d5-7b', id_scale=1.5, beam=3, **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.id_scale = id_scale
        self.beam = beam

        model = AutoModel.from_pretrained(
            self.model_path, device_map='cpu', trust_remote_code=True, local_files_only=True).cuda().eval()
        model.half()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        model.tokenizer = tokenizer
        self.model = model
        self.device = self.model.model.tok_embeddings.weight.device
        self.org_hd_num = 36
        self.model.hd_num = 36
        self.model.id_scale = self.id_scale

    def message_to_promptimg(self, message, dataset=None, video_input=False):
        num_images = len([x for x in message if x['type'] == 'image'])
        if num_images == 0:
            prompt = '\n'.join([x['value']
                               for x in message if x['type'] == 'text'])
            image = None

        else:
            image = [Image.open(x['value']).convert('RGB') for x in message if x['type'] == 'image']
            if len(image) > 1:
                self.model.hd_num = max(1, self.org_hd_num // len(image))
            else:
                self.set_max_num(dataset)

            if video_input:
                im_prompt = '<IM_POS>Here are some frames of a video.'
                if len(image) > 64:
                    step = len(image) / 64
                    image = [image[int(i * step)] for i in range(64)]
                image = [img_process(image)]

            else:
                if len(image) > 1:
                    im_prompt = ' '.join([
                        f'Image{im_idx + 1}: <IM_POS>;' for im_idx in range(len(image))])
                else:
                    im_prompt = '<IM_POS>'

            prompt = ''
            for x in message:
                if x['type'] == 'text' and x.get('role', '') != 'system':
                    prompt += x['value']
            sp = [i for i in re.split(conv_pattern, prompt) if i != '' and i != '\n']
            assert len(sp) <= 2
            q = sp[0]
            prompt = f'[UNUSED_TOKEN_146]user\n{im_prompt}{q}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'

            for idx in range(10):
                idx = chr(65 + idx)
                prompt = prompt.replace(f'({idx})', f'{idx}.')

        return prompt, image

    def generate_mme(self, image_path, text):
        text = text.split('Please answer')[0].strip()
        text = f'{text} Answer this question briefly'
        text = f'[UNUSED_TOKEN_146]user\n{text}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'

        return model_gen(self.model, text, image_path, need_bos=True, padding=True, beams=self.beam)

    def generate_multichoice(self, image_path, text, dataset):
        out = model_gen(self.model, text, image_path,
                        need_bos=True, padding=False, beams=self.beam, max_token=5)
        if 'mmmu' in dataset.lower():
            return out
        res = pattern.findall(out)
        if len(res) == 0:
            print('Error:', out)
            res = 'Z'
        return res[0]

    def generate_vqa(self, image_path, text):
        out = model_gen(self.model, text, image_path, beams=self.beam,
                        need_bos=True, max_token=100)
        return out

    def generate_vanilla(self, image_path, text):
        out = model_gen(self.model, text, image_path, beams=self.beam,
                        need_bos=True, max_token=500)
        return out

    def generate_brief(self, image_path, text):
        text = '[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase.{}\
               [UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'.format(text)
        out = model_gen(self.model, text, image_path, beams=self.beam,
                        need_bos=True, max_token=10)
        return out

    def generate_video(self, image_path, text):
        out = model_gen(
            self.model, text, image_path, beams=1,  # self.beam,
            need_bos=True, max_token=100, video_input=True)
        return out

    def set_max_num(self, dataset):
        if dataset is not None and listinstr(['MME-RealWorld', 'MME-RealWorld-CN'], dataset):
            self.model.hd_num = 25
        else:
            self.model.hd_num = self.org_hd_num

    def generate_inner(self, message, dataset=None):
        with torch.cuda.amp.autocast():
            if dataset is None:
                prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
                return self.generate_vanilla(image_path, prompt)
            assert isinstance(dataset, str)

            if listinstr(['video', 'mvbench'], dataset.lower()):
                prompt, image_path = self.message_to_promptimg(message, dataset=dataset, video_input=True)
                return self.generate_video(image_path, prompt)
            else:
                prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
                if dataset == 'MME':
                    return self.generate_mme(image_path, prompt)
                elif listinstr(['hallu', 'pope'], dataset.lower()):
                    return self.generate_brief(image_path, prompt)
                elif listinstr(['llava', 'mmvet'], dataset.lower()):
                    return self.generate_vanilla(image_path, prompt)
                elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
                    return self.generate_multichoice(image_path, prompt, dataset)
                elif listinstr(['MME-RealWorld', 'MME-RealWorld-CN'], dataset):
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
            hint = line['hint'] if (
                'hint' in line and not pd.isna(line['hint'])) else None

            context = 'N/A' if hint is None else hint
            mid_prompt = 'Question: ' + question + '\nContext: ' + \
                context + '\nOptions: ' + options_prompt
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
            elif listinstr(['llava', 'mmvet'], dataset.lower()):
                q = line['question']
                prompt = '[UNUSED_TOKEN_146]system\n{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]user\n{}\
                         Answer this question in detail.[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]\
                         assistant\n'.format(meta_instruction, q)
            elif listinstr(['mmlongbench_doc', 'dude', 'slidevqa'], dataset.lower()):
                q = line['question']
                prompt = f'[UNUSED_TOKEN_146]user\n{q}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
            else:
                q = line['question']
                prefix = 'Answer the question using a single word or phrase.'
                prompt = f'[UNUSED_TOKEN_146]user\n{prefix}{q}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
        ret = [dict(type='text', value=prompt)]
        ret.extend([dict(type='image', value=s) for s in tgt_path])
        return ret
