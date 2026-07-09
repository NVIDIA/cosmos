import math
import os
import re
import string
import warnings

import pandas as pd
import torch
import torch.distributed as dist
import torchvision.transforms as T
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer

from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE
from vlmeval.smp import cn_string, get_rank_and_world_size, listinstr, version_cmp
from ..base import BaseModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def auto_split_flag():
    flag = os.environ.get('AUTO_SPLIT', '0')
    if flag == '1':
        return True
    _, world_size = get_rank_and_world_size()
    try:
        import torch
        device_count = torch.cuda.device_count()
        if device_count > world_size and device_count % world_size == 0:
            return True
        else:
            return False
    except Exception:
        return False


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


######################################################################################
def preprocess_padding_global(image, net_input_size=448):
    # 不形变的resize方式
    orig_width, orig_height = image.size
    if orig_height > orig_width:
        scale_ratio = net_input_size / orig_height
        scale_h = net_input_size
        scale_w = int(orig_width * scale_ratio)
    else:
        scale_ratio = net_input_size / orig_width
        scale_h = int(orig_height * scale_ratio)
        scale_w = net_input_size

    resized = image.resize((scale_w, scale_h), Image.BILINEAR)
    new_image = Image.new('RGB', (net_input_size, net_input_size), (0, 0, 0))
    paste_x = (net_input_size - scale_w) // 2
    paste_y = (net_input_size - scale_h) // 2
    new_image.paste(resized, (paste_x, paste_y))
    return new_image


def dynamic_preprocess(image, min_num=5, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    padding_global = preprocess_padding_global(image, image_size)

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
    processed_images.append(padding_global)
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


def load_image(image_file, input_size=448, max_num=6, upscale=False):
    image = Image.open(image_file).convert('RGB')
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
    transform = build_transform(input_size=input_size)
    images, target_aspect_ratio = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
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


def dynamic_preprocess3(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
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


def load_image3(image_file, input_size=448, max_num=6, upscale=False):
    image = Image.open(image_file).convert('RGB')
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess3(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
######################################################################################


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


def split_model(model_path):
    num_gpus_per_node = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    try:
        local_rank, local_world_size = get_local_rank_and_local_world_size()
    except Exception:
        local_rank = rank

    if 'GPUS_PER_PROCESS' in os.environ:
        gpus_per_process = int(os.environ['GPUS_PER_PROCESS'])
    else:
        gpus_per_process = 8  # default to use 8 GPUs for one model
    gpus_per_process = min(gpus_per_process, num_gpus_per_node // local_world_size)
    start_gpu = local_rank * gpus_per_process
    end_gpu = start_gpu + gpus_per_process

    assert end_gpu <= num_gpus_per_node, f"Process {local_rank} tries to access GPU {end_gpu}, " \
                                         f"but only {num_gpus_per_node} GPUs are available per node."

    visible_devices = list(range(start_gpu, end_gpu))

    device_map = {}
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    num_gpus_for_vit = 0.5
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (len(visible_devices) - num_gpus_for_vit))
    num_layers_per_gpu = [num_layers_per_gpu] * len(visible_devices)
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = visible_devices[i]
            layer_cnt += 1
    device_map['vision_model'] = visible_devices[0]
    device_map['mlp1'] = visible_devices[0]
    device_map['language_model.model.tok_embeddings'] = visible_devices[0]
    device_map['language_model.model.embed_tokens'] = visible_devices[0]
    device_map['language_model.output'] = visible_devices[0]
    device_map['language_model.model.norm'] = visible_devices[0]
    device_map['language_model.model.rotary_emb'] = visible_devices[0]
    device_map['language_model.lm_head'] = visible_devices[0]
    device_map[f'language_model.model.layers.{num_layers - 1}'] = visible_devices[0]

    return device_map, visible_devices


def build_mcq_cot_prompt(line, prompt):
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


def build_qa_cot_prompt(line, prompt):
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


class QTuneVLChat(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self,
                 model_path='hanchaow/QTuneVL1-2B',
                 load_in_8bit=False,
                 use_mpo_prompt=False,
                 version='V1.0',
                 # Best-of-N parameters
                 best_of_n=1,
                 reward_model_path=None,
                 **kwargs):

        assert best_of_n >= 1
        assert model_path is not None
        assert version_cmp(transformers.__version__, '4.37.2', 'ge')

        self.use_mpo_prompt = use_mpo_prompt
        self.use_cot = (os.getenv('USE_COT') == '1')

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        # Regular expression to match the pattern 'Image' followed by a number, e.g. Image1
        self.pattern = r'Image(\d+)'
        # Replacement pattern to insert a hyphen between 'Image' and the number, e.g. Image-1
        self.replacement = r'Image-\1'

        # Convert QtuneVL2 response to dataset format
        # e.g. Image1 -> Image-1

        # Regular expression to match the pattern 'Image-' followed by a number
        self.reverse_pattern = r'Image-(\d+)'
        # Replacement pattern to remove the hyphen (Image-1 -> Image1)
        self.reverse_replacement = r'Image\1'

        if auto_split_flag():
            device_map, visible_devices = split_model(model_path=model_path)
            self.device = visible_devices[0]
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=device_map).eval()
        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
                low_cpu_mem_usage=True).eval().to('cuda')
            self.device = 'cuda'

        if best_of_n > 1:
            assert reward_model_path is not None

            if auto_split_flag():
                rm_device_map, visible_devices = split_model(model_path=reward_model_path)
                rm_kwargs = {'device_map': rm_device_map}
            else:
                rm_kwargs = {}

            self.reward_tokenizer = AutoTokenizer.from_pretrained(
                reward_model_path, trust_remote_code=True, use_fast=False
            )
            self.reward_model = AutoModel.from_pretrained(
                reward_model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
                low_cpu_mem_usage=True, **rm_kwargs).eval()

            if not auto_split_flag():
                self.reward_model = self.reward_model.to(self.device)

            if not self.use_cot:
                os.environ['USE_COT'] = '1'
                self.use_cot = True
                print('[Warning] Since Best-of-N is enabled, USE_COT is forced to be set to 1.')

            print(f'Enable Best-of-N evaluation with PRM: {reward_model_path}')

        self.image_size = self.model.config.vision_config.image_size
        self.version = version
        self.best_of_n = best_of_n
        kwargs_default = dict(do_sample=False, max_new_tokens=4096, top_p=None)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT', 'MMAlignBench', 'ChartQAPro', 'ChartMuseum'], dataset):  # noqa: E501
            # For Multi-Turn and some special datasets we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):
        use_mpo_prompt = self.use_mpo_prompt and (self.use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench'])

        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath', 'QSpatial',
                            'WeMath', 'LogicVista'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt)

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        if use_mpo_prompt:
            message = build_mpo_prompt(message, line, dataset)
        return message

    def set_max_num(self, dataset):
        # The total limit on the number of images processed, set to avoid Out-of-Memory issues.
        self.total_max_num = 64
        if dataset is None:
            self.max_num = 6
            return None
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'VCR_EN', 'VCR_ZH', 'OCRVQA']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
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

    @torch.no_grad()
    def generate_v2(self, message, dataset=None):

        use_mpo_prompt = self.use_mpo_prompt and (self.use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench'])

        image_num = len([x for x in message if x['type'] == 'image'])
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        prompt = reorganize_prompt(message, image_num, dataset=dataset)

        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = build_video_prompt(prompt, dataset)

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list, pixel_values_list = [], []
            for image_idx, file_name in enumerate(image_path):

                if dataset is not None and listinstr(['MMBench', 'MMStar', 'MMVet'], dataset):
                    upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                    curr_pixel_values = load_image3(
                        file_name, max_num=max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                    num_patches_list.append(curr_pixel_values.size(0))
                    pixel_values_list.append(curr_pixel_values)
                else:
                    upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                    curr_pixel_values, target_aspect_ratio = load_image(
                        file_name, max_num=max_num, upscale=upscale_flag)
                    curr_pixel_values = curr_pixel_values.cuda().to(torch.bfloat16)
                    curr_pixel_values2 = load_image2(
                        file_name, target_aspect_ratio=target_aspect_ratio, max_num=max_num)
                    curr_pixel_values2 = curr_pixel_values2.cuda().to(torch.bfloat16)
                    if listinstr(['MathVista', 'HallusionBench', 'OCRBench'], dataset):
                        curr_pixel_values = torch.cat(
                            (curr_pixel_values[:-1], curr_pixel_values2[:-1], curr_pixel_values[-1:]), 0)
                    else:
                        curr_pixel_values = torch.cat(
                            (curr_pixel_values2[:-1], curr_pixel_values[-1:]), 0)
                    num_patches_list.append(curr_pixel_values.size(0))
                    pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            if dataset is not None and listinstr(['MMBench', 'MMStar', 'MMVet'], dataset):
                image_path = [x['value'] for x in message if x['type'] == 'image'][0]
                upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
                pixel_values = load_image3(
                    image_path, max_num=max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                num_patches_list = [pixel_values.size(0)]
            else:
                image_path = [x['value'] for x in message if x['type'] == 'image'][0]
                upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
                pixel_values, target_aspect_ratio = load_image(
                    image_path, max_num=max_num, upscale=upscale_flag)
                pixel_values = pixel_values.to(self.device).to(torch.bfloat16)
                pixel_values2 = load_image2(
                    image_path, target_aspect_ratio=target_aspect_ratio, max_num=max_num)
                pixel_values2 = pixel_values2.cuda().to(torch.bfloat16)
                if listinstr(['MathVista', 'HallusionBench', 'OCRBench'], dataset):
                    pixel_values = torch.cat((pixel_values[:-1], pixel_values2[:-1], pixel_values[-1:]), 0)
                else:
                    pixel_values = torch.cat((pixel_values2[:-1], pixel_values[-1:]), 0)
                num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []
        self.model.vision_model.to(self.device)
        self.model.to(self.device)
        response_list = []
        for idx in range(self.best_of_n):
            kwargs_default = self.kwargs.copy()
            kwargs_default['do_sample'] = idx > 0
            kwargs_default['temperature'] = 0.7
            kwargs_default['top_p'] = 0.95

            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                question=prompt,
                generation_config=kwargs_default,
                verbose=idx == 0,
            )
            response_list.append(response)

        if self.best_of_n > 1:
            response_list = self.reward_model.select_best_response(
                tokenizer=self.reward_tokenizer,
                question=prompt,
                response_list=response_list,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
            )
        response = response_list[0]

        if use_mpo_prompt:
            response = mpo_post_processing(response, dataset)
        return response

    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        print(f'QtuneVL model version: {self.version}')
        if self.version in ['V1.0', 'V1.5']:
            return self.generate_v2(message, dataset)
        else:
            raise ValueError(f'Unsupported version: {self.version}')

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

        if len(message) > 1:
            history, image_path, image_cnt = self.build_history(message[:-1])
        else:
            history, image_path, image_cnt = None, [], 1
        current_msg = message[-1]
        question = ''

        # If message is just text in the conversation
        if len(current_msg['content']) == 1 and current_msg['content'][0]['type'] == 'text':
            question = current_msg['content'][0]['value']
            question = re.sub(self.pattern, self.replacement, question)  # Fix pattern as per QtuneVL
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
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset)
                curr_pixel_values = load_image(
                    file_name, max_num=self.max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_cnt == 1:
            upscale_flag = listinstr(['MMMU_DEV_VAL'], dataset)
            pixel_values = load_image(
                image_path, max_num=self.max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        response, history = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
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

        if self.version in ['V1.0', 'V1.5']:
            kwargs_default = dict(do_sample=False, max_new_tokens=512, top_p=None, num_beams=1)
            self.kwargs = kwargs_default
            return self.chat_inner_v2(message, dataset)
        else:
            raise ValueError(f'Unsupported version for Multi-Turn: {self.version}')
