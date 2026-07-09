import os
import re
import string
import warnings

import pandas as pd
import torch
import torchvision.transforms as T
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.utils import logging

from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE
from vlmeval.smp import cn_string, listinstr, version_cmp
from .base import BaseModel

logger = logging.get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def build_transform(input_size, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6, upscale=False, normalize_type="imagenet"):
    image = Image.open(image_file).convert('RGB')
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
    transform = build_transform(input_size=input_size, normalize_type=normalize_type)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def extract_answer(text):
    match = re.search(r'(Final answer:|Answer:)\s*(.*)', text, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return text


class Ristretto(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self,
                 model_path='',
                 load_in_8bit=False,
                 cot_prompt=False,
                 **kwargs):

        assert model_path is not None
        assert version_cmp(transformers.__version__, '4.36.2', 'ge')

        self.cot_prompt = cot_prompt
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if self.config.vision_config.model_type == "siglip_vision_model":
            self.normalize_type = "siglip"
        else:
            self.normalize_type = "imagenet"
        self.image_size = self.config.vision_config.image_size

        # Regular expression to match the pattern 'Image' followed by a number, e.g. Image1
        self.pattern = r'Image(\d+)'
        # Replacement pattern to insert a hyphen between 'Image' and the number, e.g. Image-1
        self.replacement = r'Image-\1'

        # Regular expression to match the pattern 'Image-' followed by a number
        self.reverse_pattern = r'Image-(\d+)'
        # Replacement pattern to remove the hyphen (Image-1 -> Image1)
        self.reverse_replacement = r'Image\1'
        self.device = 'cuda'

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
            trust_remote_code=True).eval()
        if not load_in_8bit:
            self.model = self.model.to('cuda')
        self.model = self.model.to(torch.bfloat16)
        self.image_size = self.model.config.vision_config.image_size
        kwargs_default = dict(do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

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
            prompt += '\n请直接回答选项字母。' if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        return prompt

    def build_video_prompt(self, prompt, dataset=None, max_frames=64):
        for start in range(0, max_frames, 8):
            images_to_remove = ''.join([f'<Image-{i}>' for i in range(start + 1, start + 9)])
            prompt = prompt.replace(images_to_remove, '')
        for i in range(max_frames):
            prompt = prompt.replace(f'Image-{i + 1}', f'Frame-{i + 1}')
        if listinstr(['MMBench-Video'], dataset):
            prompt = prompt.replace('\nAnswer:', '')
        elif listinstr(['Video-MME'], dataset):
            prompt = prompt.replace('\nAnswer:', '')
            prompt += "\nAnswer with the option's letter from the given choices directly."
        elif listinstr(['MVBench'], dataset):
            prompt = prompt.replace('Best option:(', '')

        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        kwargs_default = dict(do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1)
        if int(os.environ.get("MAX_NEW_TOKENS", 0)) != 0:
            kwargs_default["max_new_tokens"] = int(os.environ.get("MAX_NEW_TOKENS", 0))
        self.kwargs = kwargs_default

        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse'], dataset):
                prompt = question
            elif listinstr(['LLaVABench'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']

        if self.cot_prompt and not listinstr(['LLaVABench'], dataset):
            cot_prompt_with_final_answer = (
                "Your task is to answer the question below. "
                "Give step by step reasoning before you answer, and when you're ready to answer, "
                "please use the format \"Final answer: ..\""
                "\n\n"
                "Question:"
                "\n\n"
                "{question}"
            )
            cot_prompt_wo_final_answer = (
                "Your task is to answer the question below. "
                "Give step by step reasoning. "
                "\n\n"
                "Question:"
                "\n\n"
                "{question}"
            )

            if listinstr(['MMVet'], dataset):
                cot_prompt = cot_prompt_wo_final_answer
            else:
                cot_prompt = cot_prompt_with_final_answer

            question_orig = line['question']
            if listinstr(['MathVerse', 'MathVision'], dataset):
                question_orig = question_orig.split('Question:', 1)[-1].strip()
                question_orig = question_orig.replace('Choices:\n', '').strip()

            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            if options_prompt.strip():
                question_orig = f'{question_orig}\n{options_prompt}'

            prompt = cot_prompt.format(question=question_orig)

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def set_max_num(self, dataset):
        if int(os.environ.get("MAX_PATCH_NUM", 0)) != 0:
            max_patch_num = int(os.environ.get("MAX_PATCH_NUM", None))
            self.max_num = max_patch_num
            return None

        if dataset is None:
            self.max_num = 6
            return None
        # res_1_datasets = ['MMBench-Video', 'Video-MME', 'MVBench', 'Video']
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'MME-RealWorld', 'VCR_EN', 'VCR_ZH']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if DATASET_MODALITY(dataset) == 'VIDEO':
            self.max_num = 1
        elif listinstr(res_12_datasets, dataset):
            self.max_num = 12
        elif listinstr(res_18_datasets, dataset):
            self.max_num = 18
        elif listinstr(res_24_datasets, dataset):
            self.max_num = 24
        else:
            self.max_num = 6

    def _generate(self, message, dataset=None):
        image_num = len([x for x in message if x['type'] == 'image'])
        if image_num == 1:
            prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        else:
            prompt, image_idx = '', 1
            for x in message:
                if x['type'] == 'text':
                    prompt += x['value']
                elif x['type'] == 'image':
                    prompt += f'<Image-{image_idx}>'
                    image_idx += 1
            prompt = '\n'.join([f'Image-{i + 1}: <image>' for i in range(image_num)]) + '\n' + prompt

        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = self.build_video_prompt(prompt, dataset)

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list = []
            pixel_values_list = []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset)
                curr_pixel_values = load_image(
                    file_name, input_size=self.image_size, max_num=self.max_num,
                    upscale=upscale_flag, normalize_type=self.normalize_type
                ).to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            upscale_flag = dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset)
            pixel_values = load_image(
                image_path, input_size=self.image_size, max_num=self.max_num,
                upscale=upscale_flag, normalize_type=self.normalize_type
            ).to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        num_image_token = 256
        if dataset is not None:
            if listinstr(['MMBench_DEV_EN_V11', 'MathVista_MINI', 'MMVet'], dataset):
                num_image_token = 144
            elif listinstr(['HallusionBench'], dataset):
                num_image_token = 576

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                question=prompt,
                num_image_token=num_image_token,
                generation_config=self.kwargs,
                verbose=False
            )

        if (
            self.cot_prompt
            and dataset is not None
            and (
                DATASET_TYPE(dataset) in ['Y/N', 'MCQ']
                or listinstr(['CRPE'], dataset)
            )
        ):
            response = extract_answer(response).strip()

        return response

    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        return self._generate(message, dataset)
