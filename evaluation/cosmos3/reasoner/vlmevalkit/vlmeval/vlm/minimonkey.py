import re
import string
import warnings

import pandas as pd
import torch
import torchvision.transforms as T
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string, listinstr, version_cmp
from .base import BaseModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
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


def dynamic_preprocess(image, min_num=5, max_num=6, image_size=448, use_thumbnail=False):
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
    return processed_images, target_aspect_ratio


def dynamic_preprocess2(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False, prior_aspect_ratio=None):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    new_target_ratios = []
    if prior_aspect_ratio is not None:
        for i in target_ratios:
            if prior_aspect_ratio[0] % i[0] != 0 or prior_aspect_ratio[1] % i[1] != 0:
                new_target_ratios.append(i)
            else:
                continue
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, new_target_ratios, orig_width, orig_height, image_size)

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


def load_image(image_file, input_size=448, min_num=1, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images, target_aspect_ratio = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, min_num=min_num, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, target_aspect_ratio


def load_image2(image_file, input_size=448, target_aspect_ratio=(1, 1), min_num=1, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess2(
        image,
        image_size=input_size,
        prior_aspect_ratio=target_aspect_ratio,
        use_thumbnail=True,
        min_num=min_num,
        max_num=max_num)

    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# To revert changes
class MiniMonkey(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='mx262/MiniMokney', **kwargs):
        assert model_path is not None
        assert version_cmp(transformers.__version__, '4.36.2', 'ge')

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        # Regular expression to match the pattern 'Image' followed by a number, e.g. Image1
        self.pattern = r'Image(\d+)'
        # Replacement pattern to insert a hyphen between 'Image' and the number, e.g. Image-1
        self.replacement = r'Image-\1'

        # Convert InternVL2 response to dataset format
        # e.g. Image1 -> Image-1

        # Regular expression to match the pattern 'Image-' followed by a number
        self.reverse_pattern = r'Image-(\d+)'
        # Replacement pattern to remove the hyphen (Image-1 -> Image1)
        self.reverse_replacement = r'Image\1'

        self.device = 'cuda'
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            load_in_8bit=False,
            device_map='cuda').eval()

        self.image_size = self.model.config.vision_config.image_size
        self.kwargs = kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        if dataset is None:
            return False
        if listinstr(['MMDU'], dataset):
            # For Multi-Turn we don't have custom prompt
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

    def build_video_prompt(self, prompt, dataset=None, max_nframe=64):
        for start in range(0, max_nframe, 8):
            images_to_remove = ''.join([f'<image-{i}>' for i in range(start + 1, start + 9)])
            prompt = prompt.replace(images_to_remove, '')
        for i in range(max_nframe):
            prompt = prompt.replace(f'<image-{i + 1}>', f'Frame{i + 1}')
        if listinstr(['MMBench-Video'], dataset):
            prompt = prompt.replace('\nAnswer:', '')
            prompt += '\nAnswer the question using a single word or phrase.'
        elif listinstr(['Video-MME'], dataset):
            prompt = prompt.replace('\nAnswer:', '')
            prompt += "\nAnswer with the option's letter from the given choices directly."
        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        kwargs_default = dict(do_sample=False, max_new_tokens=512, top_p=None, num_beams=1)
        self.kwargs = kwargs_default

        if dataset is not None and listinstr(['MME'], dataset):
            question = line['question']
            prompt = question + ' Answer the question using a single word or phrase.'
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question']
            prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['MathVista', 'MathVision'], dataset):
                prompt = line['question']
            elif listinstr(['LLaVABench'], dataset):
                question = line['question']
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['MMVet'], dataset):
                prompt = line['question']
            else:
                question = line['question']
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def set_max_num(self, dataset):
        if dataset is None:
            self.max_num = 12
            self.max_num2 = 7
            self.min_num = 4
            self.min_num2 = 3
            return

        if dataset is not None and listinstr(['ChartQA_TEST'], dataset):
            self.max_num = 12
            self.max_num2 = 3
        elif dataset is not None and listinstr(['DocVQA_VAL', 'DocVQA_TEST', 'TextVQA_VAL'], dataset):
            self.max_num = 23
            self.max_num2 = 15
            self.min_num = 14
            self.min_num2 = 5
        elif dataset is not None and listinstr(['InfoVQA_VAL', 'InfoVQA_TEST', 'SEEDBench_IMG'], dataset):
            self.max_num = 23
            self.max_num2 = 5
            self.min_num = 15
            self.min_num2 = 3
        elif dataset is not None and listinstr(['OCRBench', 'POPE'], dataset):
            self.max_num = 24
            self.max_num2 = 8
            self.min_num = 9
            self.min_num2 = 5
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            self.max_num = 11
            self.max_num2 = 6
            self.min_num = 4
            self.min_num2 = 2
        elif dataset is not None and listinstr(['MME'], dataset):
            self.max_num = 11
            self.max_num2 = 6
            self.min_num = 5
            self.min_num2 = 2
        elif dataset is not None and listinstr(['AI2D_TEST'], dataset):
            self.max_num = 12
            self.max_num2 = 6
            self.min_num = 5
            self.min_num2 = 2
        elif dataset is not None and listinstr(['CCBench'], dataset):
            self.max_num = 24
            self.max_num2 = 8
            self.min_num = 9
            self.min_num2 = 4
        elif dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset):
            self.max_num = 12
            self.max_num2 = 7
            self.min_num = 5
            self.min_num2 = 3
        else:
            self.max_num = 12
            self.max_num2 = 7
            self.min_num = 4
            self.min_num2 = 3

    def generate_v2(self, message, dataset=None):
        image_num = len([x for x in message if x['type'] == 'image'])
        if image_num == 1:
            prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        else:
            prompt, image_idx = '', 1
            for x in message:
                if x['type'] == 'text':
                    prompt += x['value']
                elif x['type'] == 'image':
                    prompt += f'<image-{image_idx}>'
                    image_idx += 1
            prompt = ' '.join([f'<image-{i + 1}>: <image>' for i in range(image_num)]) + '\n' + prompt

        if dataset is not None and listinstr(['Video'], dataset):
            prompt = self.build_video_prompt(prompt, dataset)

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list = []
            pixel_values_list = []
            for image_idx, file_name in enumerate(image_path):
                curr_pixel_values, target_aspect_ratio = load_image(
                    file_name, min_num=self.min_num, max_num=self.max_num)
                curr_pixel_values = curr_pixel_values.cuda().to(torch.bfloat16)
                curr_pixel_values2 = load_image2(
                    file_name, target_aspect_ratio=target_aspect_ratio, min_num=self.min_num2, max_num=self.max_num2)
                curr_pixel_values2 = curr_pixel_values2.cuda().to(torch.bfloat16)
                curr_pixel_values = torch.cat(
                    (curr_pixel_values[:-1], curr_pixel_values2[:-1], curr_pixel_values[-1:]), 0)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            pixel_values, target_aspect_ratio = load_image(image_path, min_num=self.min_num, max_num=self.max_num)
            pixel_values = pixel_values.cuda().to(torch.bfloat16)
            pixel_values2 = load_image2(
                image_path, target_aspect_ratio=target_aspect_ratio, min_num=self.min_num2, max_num=self.max_num2)
            pixel_values2 = pixel_values2.cuda().to(torch.bfloat16)
            pixel_values = torch.cat((pixel_values[:-1], pixel_values2[:-1], pixel_values[-1:]), 0)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                target_aspect_ratio=(1, 1),
                num_patches_list=num_patches_list,
                question=prompt,
                generation_config=self.kwargs,
                verbose=False
            )
        return response

    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        return self.generate_v2(message, dataset)

    def build_history(self, message):
        # Global Variables
        image_path = []
        image_cnt = 0

        def concat_tilist(tilist):
            nonlocal image_cnt  # Declare image_cnt as nonlocal to modify it
            prompt = ''
            for item in tilist:
                # Substitute the pattern in the text
                if item['type'] == 'text':
                    prompt += re.sub(self.pattern, self.replacement, item['value'])
                elif item['type'] == 'image':
                    image_cnt += 1
                    prompt += '<image>\n'
                    image_path.append(item['value'])
            return prompt

        # Only previous messages
        assert len(message) % 2 == 0
        history = []
        for i in range(len(message) // 2):
            m1, m2 = message[2 * i], message[2 * i + 1]
            assert m1['role'] == 'user' and m2['role'] == 'assistant'
            history.append((concat_tilist(m1['content']), concat_tilist(m2['content'])))

        return history, image_path, image_cnt

    def chat_inner_v2(self, message, dataset=None):

        image_cnt = 0
        if len(message) > 1:
            history, image_path, image_cnt = self.build_history(message[:-1])
        else:
            history, image_path, image_cnt = None, [], 1
        current_msg = message[-1]
        question = ''

        # If message is just text in the conversation
        if len(current_msg['content']) == 1 and current_msg['content'][0]['type'] == 'text':
            question = current_msg['content'][0]['value']
            question = re.sub(self.pattern, self.replacement, question)  # Fix pattern as per InternVL
        else:
            for msg in current_msg['content']:
                if msg['type'] == 'text':
                    question += re.sub(self.pattern, self.replacement, msg['value'])
                elif msg['type'] == 'image':
                    image_cnt += 1
                    question += '<image>\n'
                    image_path.append(msg['value'])

        if image_cnt > 1:
            num_patches_list = []
            pixel_values_list = []
            for image_idx, file_name in enumerate(image_path):
                curr_pixel_values, target_aspect_ratio = load_image(
                    file_name, min_num=self.min_num, max_num=self.max_num)
                curr_pixel_values = curr_pixel_values.cuda().to(torch.bfloat16)
                curr_pixel_values2 = load_image2(
                    file_name, target_aspect_ratio=target_aspect_ratio, min_num=self.min_num2, max_num=self.max_num2)
                curr_pixel_values2 = curr_pixel_values2.cuda().to(torch.bfloat16)
                curr_pixel_values = torch.cat(
                    (curr_pixel_values[:-1], curr_pixel_values2[:-1], curr_pixel_values[-1:]), 0)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_cnt == 1:
            pixel_values, target_aspect_ratio = load_image(image_path, min_num=self.min_num, max_num=self.max_num)
            pixel_values = pixel_values.cuda().to(torch.bfloat16)
            pixel_values2 = load_image2(
                image_path, target_aspect_ratio=target_aspect_ratio, min_num=self.min_num2, max_num=self.max_num2)
            pixel_values2 = pixel_values2.cuda().to(torch.bfloat16)
            pixel_values = torch.cat((pixel_values[:-1], pixel_values2[:-1], pixel_values[-1:]), 0)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        response, history = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            target_aspect_ratio=target_aspect_ratio,
            num_patches_list=num_patches_list,
            question=question,
            generation_config=self.kwargs,
            history=history,
            return_history=True
        )

        response = re.sub(self.reverse_pattern, self.reverse_replacement, response)

        return response

    def chat_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        kwargs_default = dict(do_sample=False, max_new_tokens=512, top_p=None, num_beams=1)
        self.kwargs = kwargs_default
        return self.chat_inner_v2(message, dataset)
