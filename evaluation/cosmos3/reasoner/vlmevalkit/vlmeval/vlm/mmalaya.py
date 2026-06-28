import string
import warnings

import pandas as pd
import torch
import torchvision.transforms as T
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from vlmeval.smp import cn_string, listinstr, version_cmp
from ..dataset import DATASET_TYPE
from .base import BaseModel


class MMAlaya(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='DataCanvas/MMAlaya', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda", trust_remote_code=True
        ).eval()
        # need initialize tokenizer
        model.initialize_tokenizer(self.tokenizer)
        self.model = model

        self.kwargs = kwargs
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        # read image
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')
        # tokenize prompt, and proprecess image
        input_ids, image_tensor, stopping_criteria = self.model.prepare_for_inference(
            prompt, self.tokenizer, image, return_tensors='pt'
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids.cuda(),
                images=image_tensor.cuda(),
                do_sample=False,
                max_new_tokens=512,
                num_beams=1,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
            # truncate input_ids in generate_ids and then decode to text
            input_token_len = input_ids.shape[1]
            response = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:].cpu(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        return response


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
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


def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

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
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6, upscale=False):
    image = Image.open(image_file).convert('RGB')
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class MMAlaya2(BaseModel):
    """
    This implementation fine-tunes 20 LoRA modules based on the InternVL-Chat-V1-5 model.
    The fine-tuned LoRA modules are then merged with the InternVL-Chat-V1-5 model
    using the PEFT model merging method, TIES.
    The code is based on the implementation in `vlmeval/vlm/internvl_chat.py`.
    """

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path='DataCanvas/MMAlaya2',
        load_in_8bit=False,
        **kwargs,
    ):
        assert model_path is not None
        assert version_cmp(transformers.__version__, '4.36.2', 'ge')

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

        # Regular expression to match the pattern "Image" followed by a number, e.g. Image1
        self.pattern = r'Image(\d+)'
        # Replacement pattern to insert a hyphen between "Image" and the number, e.g. Image-1
        self.replacement = r'Image-\1'

        # Convert InternVL2 response to dataset format
        # e.g. Image1 -> Image-1

        # Regular expression to match the pattern "Image-" followed by a number
        self.reverse_pattern = r'Image-(\d+)'
        # Replacement pattern to remove the hyphen (Image-1 -> Image1)
        self.reverse_replacement = r'Image\1'

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            load_in_8bit=load_in_8bit,
            device_map="cuda"
        ).eval()

        self.image_size = self.model.config.vision_config.image_size

        kwargs_default = dict(
            do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN'], dataset):
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
            prompt += (
                '\n请直接回答选项字母。'
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                '\n请直接回答问题。'
                if cn_string(prompt)
                else '\nAnswer the question directly.'
            )

        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and listinstr(['MME'], dataset):
            question = line['question']
            prompt = question + ' Answer the question using a single word or phrase.'
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question']
            prompt = (
                question
                + ' Please answer yes or no. Answer the question using a single word or phrase.'
            )
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['MathVista', 'MathVision', 'MathVerse'], dataset):
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
        if dataset is not None and listinstr(['ChartQA_TEST', 'MMMU_DEV_VAL'], dataset):
            self.max_num = 12
        elif dataset is not None and listinstr(['DocVQA_VAL', 'DocVQA_TEST'], dataset):
            self.max_num = 18
        elif dataset is not None and listinstr(
            ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench'], dataset
        ):
            self.max_num = 24
        elif dataset is not None and listinstr(
            ['MMBench-Video', 'Video-MME', 'Video'], dataset
        ):
            self.max_num = 1
        else:
            self.max_num = 6

    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        image_num = len([x for x in message if x['type'] == 'image'])
        prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            pixel_values_list = []
            max_num = max(1, self.max_num // image_num)
            for file_name in image_path:
                pixel_values_list.append(load_image(file_name, max_num=max_num).cuda().to(torch.bfloat16))
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            pixel_values = (
                load_image(image_path, max_num=self.max_num).cuda().to(torch.bfloat16)
            )
        else:
            pixel_values = None
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=self.kwargs,
                # verbose=False,
            )
        return response


if __name__ == '__main__':
    model = MMAlaya2(max_new_tokens=1024, do_sample=False)
    response = model.generate_inner(
        [
            {'type': 'image', 'value': './assets/apple.jpg'},
            {'type': 'text', 'value': '请详细描述一下这张图片。'},
        ]
    )
    print(response)
