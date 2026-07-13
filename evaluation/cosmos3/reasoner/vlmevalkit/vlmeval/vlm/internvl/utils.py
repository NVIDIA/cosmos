import os
import re
import string

import pandas as pd
import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string, listinstr

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


def load_image(image_file, input_size=448, max_num=6, upscale=False):
    image = Image.open(image_file).convert('RGB')
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_local_rank_and_local_world_size():
    if not dist.is_available():
        return 0, 1
    if not dist.is_initialized():
        return 0, 1

    if 'SLURM_LOCALID' in os.environ:
        local_rank = int(os.environ['SLURM_LOCALID'])
        local_world_size = int(os.environ['SLURM_NTASKS_PER_NODE'])
        return local_rank, local_world_size

    if 'LOCAL_RANK' in os.environ and 'LOCAL_WORLD_SIZE' in os.environ:
        return int(os.environ['LOCAL_RANK']), int(os.environ['LOCAL_WORLD_SIZE'])

    raise NotImplementedError(
        "Fail to get local_rank and local_world_size! "
        "Please ensure that you set the environment variable "
        "`LOCAL_RANK` and `LOCAL_WORLD_SIZE`"
    )


def build_yesorno_cot_prompt(line, prompt, cot_prompt=None):
    if cot_prompt is None:
        cot_prompt = (
            "Answer the preceding yes-or-no question. Think step by step logically, considering all "
            "relevant information before answering. If you are uncertain or the problem is ambiguous, make a "
            "reasoned guess based on the information provided. The last line of your response must be exactly "
            "in this format: 'Answer: \\boxed{Yes}' or 'Answer: \\boxed{No}' (without quotes)."
        )
    prompt = prompt.replace("Please answer yes or no. Answer the question using a single word or phrase.", '').strip()
    prompt = prompt + '\n' + cot_prompt

    return prompt


def build_mcq_cot_prompt(line, prompt, cot_prompt=None):
    if cot_prompt is None:
        cot_prompt = (
            "Answer the preceding multiple choice question. The last line of your response should follow "
            "this format: 'Answer: \\boxed{$LETTER}' (without quotes), where LETTER is one of the options. "
            "If you are uncertain or the problem is too complex, make a reasoned guess based on the "
            "information provided. Avoid repeating steps indefinitely—provide your best guess even if "
            "unsure. Think step by step logically, considering all relevant information before answering."
        )
    prompt = prompt.replace("Answer with the option's letter from the given choices directly.", '').strip()
    prompt = prompt + '\n' + cot_prompt

    return prompt


def build_qa_cot_prompt(line, prompt, cot_prompt=None):
    if cot_prompt is None:
        cot_prompt = (
            "Answer the preceding question. The last line of your response should follow this format: "
            "'Answer: \\boxed{$FINAL_ANSWER}' (without quotes), where 'FINAL_ANSWER' is your conclusion "
            "based on the reasoning provided. If you are uncertain or the problem is too complex, make "
            "a reasoned guess based on the information provided. Avoid repeating steps indefinitely—"
            "provide your best guess even if unsure. Think step by step logically, considering all "
            "relevant information before answering."
        )
    prompt = prompt + '\n' + cot_prompt

    return prompt


def build_multi_choice_prompt(line, dataset=None):
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


def build_video_prompt(prompt, dataset=None, max_frames=64):
    for start in range(0, max_frames, 8):
        images_to_remove = ''.join([f'<Image-{i}>' for i in range(start + 1, start + 9)])
        prompt = prompt.replace(images_to_remove, '')
    for i in range(max_frames):
        prompt = prompt.replace(f'Image-{i + 1}', f'Frame-{i + 1}')
    if listinstr(['MMBench-Video'], dataset):
        prompt = prompt.replace('\nAnswer:', '')
    elif listinstr(['Video-MME', 'WorldSense'], dataset):
        prompt = prompt.replace('\nAnswer:', '')
        prompt += "\nAnswer with the option's letter from the given choices directly."
    elif listinstr(['MVBench'], dataset):
        prompt = prompt.replace('Best option:(', '')

    return prompt


def reorganize_prompt(message, image_num, dataset=None):
    if dataset is not None and listinstr(['MUIRBench'], dataset):
        prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        images_to_remove = ' '.join(['<image>'] * image_num)
        prompt = prompt.replace(images_to_remove, '')
        for i in range(image_num):
            prompt = prompt.replace('<image>', f'<Image-{i + 1}>', 1)
        prompt = ''.join([f'Image-{i + 1}: <image>\n' for i in range(image_num)]) + prompt
    elif dataset is not None and listinstr(["bmmr"], dataset.lower()):
        if image_num == 1:
            prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])
        else:
            prompt, image_idx = "", 1
            for x in message:
                if x["type"] == "text":
                    prompt += x["value"]
                elif x["type"] == "image":
                    image_idx += 1
    elif dataset is not None and listinstr(["mmsivideo"], dataset.lower()):
        prompt, image_idx = '', 1
        for x in message:
            if x['type'] == 'text':
                prompt += x['value']
            elif x['type'] == 'image':
                prompt += '<image>'
    elif image_num == 1:
        prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
    else:
        prompt, image_idx = '', 1
        for x in message:
            if x['type'] == 'text':
                prompt += x['value']
            elif x['type'] == 'image':
                prompt += f'<Image-{image_idx}>'
                image_idx += 1
        prompt = ''.join([f'Image-{i + 1}: <image>\n' for i in range(image_num)]) + prompt
        images_to_remove = ''.join([f'<Image-{i + 1}>' for i in range(image_num)])
        prompt = prompt.replace(images_to_remove, '')
    return prompt


mpo_prompt_with_final_answer = (
    "Your task is to answer the question below. "
    "Give step by step reasoning before you answer, and when you're ready to answer, "
    "please use the format \"Final answer: ..\""
    "\n\n"
    "Question:"
    "\n\n"
    "{question}"
)

mpo_prompt_without_final_answer = (
    "Your task is to answer the question below. "
    "Give step by step reasoning. "
    "\n\n"
    "Question:"
    "\n\n"
    "{question}"
)


def mpo_post_processing(response, dataset):

    def extract_answer(text):
        match = re.search(r'(Final answer:|Answer:)\s*(.*)', text, re.IGNORECASE)
        if match:
            return match.group(2).strip()
        return text

    if dataset is not None and (DATASET_TYPE(dataset) in ['Y/N', 'MCQ'] or listinstr(['CRPE'], dataset)):
        response = extract_answer(response).strip()
    return response


def parse_bbox_internvl(response):
    # 使用正则表达式匹配bounding box
    # pattern = r"<box>\[\[(\d+), (\d+), (\d+), (\d+)\]\]</box>"
    pattern = r"\[\[(\d+), (\d+), (\d+), (\d+)\]\]"
    match = re.search(pattern, response)
    if match:
        # 提取匹配到的坐标值并转换为整数
        x1, y1, x2, y2 = map(int, match.groups())
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    else:
        return response


def build_mpo_prompt(message, line, dataset):
    if listinstr(['LLaVABench', 'MMVet'], dataset):
        return message

    question_orig = line['question']
    if listinstr(['MathVerse', 'MathVision'], dataset):
        question_orig = question_orig.split('Question:', 1)[-1].strip()
        question_orig = question_orig.replace('Choices:\n', '').strip()
    if listinstr(['WeMath'], dataset):
        question_orig = question_orig.replace('Regarding the format, please answer following the template below, and be sure to include two <> symbols:\n<Thought process>: <<your thought process>> <Answer>: <<your option>>', '').strip()  # noqa: E501
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

    cot_prompt = mpo_prompt_with_final_answer
    prompt = cot_prompt.format(question=question_orig).strip()
    message[0]['value'] = prompt
    return message


def format_nav_prompt(template, placeholders, **kwargs):
    prompt = template
    for placeholder in placeholders:
        value = kwargs.get(placeholder, '')
        prompt = prompt.replace(f"{{{placeholder}}}", str(value))
    return prompt


def pile_action_history(history, max_num=4):
    if len(history) > 0:
        return '\n'.join(history[-max_num:])
    else:
        return 'None'
