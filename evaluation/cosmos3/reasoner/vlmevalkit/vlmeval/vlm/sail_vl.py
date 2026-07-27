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
from transformers import AutoModel, AutoTokenizer

from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE
from vlmeval.smp import cn_string, listinstr, version_cmp
from .base import BaseModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def get_cot_answer(response):
    pattern1 = r"<think>(?P<think>.*?)</think>.*?\\boxed\{(?P<answer>[^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    match = re.search(pattern1, response, re.DOTALL)
    if match:
        return match.group("answer")
    pattern2 = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern2, response, re.DOTALL)
    if matches:
        return matches[-1]
    return response


def process_response(response, dataset_name, use_cot=False):
    if use_cot:
        response = get_cot_answer(response)
    if dataset_name is None:
        return response
    if listinstr(["ChartQA", "OCRVQA"], dataset_name):
        if len(response) >= 1 and response[-1] == ".":
            response = response[:-1]
    return response


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
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
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
    image = Image.open(image_file).convert("RGB")
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

    if "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
        local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
        return local_rank, local_world_size

    if "LOCAL_RANK" in os.environ and "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_RANK"]), int(os.environ["LOCAL_WORLD_SIZE"])

    raise NotImplementedError(
        "Fail to get local_rank and local_world_size! "
        "Please ensure that you set the environment variable "
        "`LOCAL_RANK` and `LOCAL_WORLD_SIZE`"
    )


def build_multi_choice_prompt(line, dataset=None):
    question = line["question"]
    hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
    if hint is not None:
        question = hint + "\n" + question

    options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
    for key, item in options.items():
        question += f"\n{key}. {item}"
    prompt = question

    if len(options):
        prompt += (
            "\n请直接回答选项字母。"
            if cn_string(prompt)
            else "\nAnswer with the option's letter from the given choices directly."
        )
    else:
        prompt += "\n请直接回答问题。" if cn_string(prompt) else "\nAnswer the question directly."

    return prompt


def build_video_prompt(prompt, dataset=None, max_frames=64):
    for start in range(0, max_frames, 8):
        images_to_remove = "".join([f"<Image-{i}>" for i in range(start + 1, start + 9)])
        prompt = prompt.replace(images_to_remove, "")
    for i in range(max_frames):
        prompt = prompt.replace(f"Image-{i + 1}", f"Frame-{i + 1}")
    if listinstr(["MMBench-Video"], dataset):
        prompt = prompt.replace("\nAnswer:", "")
    elif listinstr(["Video-MME"], dataset):
        prompt = prompt.replace("\nAnswer:", "")
        prompt += "\nAnswer with the option's letter from the given choices directly."
    elif listinstr(["MVBench"], dataset):
        prompt = prompt.replace("Best option:(", "")

    return prompt


def reorganize_prompt(message, image_num, dataset=None):
    if dataset is not None and listinstr(["MUIRBench"], dataset):
        prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])
        images_to_remove = " ".join(["<image>"] * image_num)
        prompt = prompt.replace(images_to_remove, "")
        for i in range(image_num):
            prompt = prompt.replace("<image>", f"<Image-{i + 1}>", 1)
        prompt = "".join([f"Image-{i + 1}: <image>\n" for i in range(image_num)]) + prompt
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
    elif image_num == 1:
        prompt = "<image>\n" + "\n".join([x["value"] for x in message if x["type"] == "text"])
    else:
        prompt, image_idx = "", 1
        for x in message:
            if x["type"] == "text":
                prompt += x["value"]
            elif x["type"] == "image":
                prompt += f"<Image-{image_idx}>"
                image_idx += 1
        prompt = "".join([f"Image-{i + 1}: <image>\n" for i in range(image_num)]) + prompt
        images_to_remove = "".join([f"<Image-{i + 1}>" for i in range(image_num)])
        prompt = prompt.replace(images_to_remove, "")
    return prompt


def dynamic_preprocess_msac1(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
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
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
    return processed_images, target_aspect_ratio


def dynamic_preprocess_msac2(
    image,
    min_num=1,
    max_num=6,
    image_size=448,
    use_thumbnail=False,
    prior_aspect_ratio=None,
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

    new_target_ratios = []
    if prior_aspect_ratio is not None:
        for i in target_ratios:
            if prior_aspect_ratio[0] % i[0] != 0 or prior_aspect_ratio[1] % i[1] != 0:
                new_target_ratios.append(i)
            else:
                continue

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, new_target_ratios, orig_width, orig_height, image_size
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


def load_image_msac(image_file, input_size=448, max_num=10, upscale=False):
    image = Image.open(image_file).convert("RGB")
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
    transform = build_transform(input_size=input_size)
    images, target_aspect_ratio = dynamic_preprocess_msac1(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    images = (
        images[:-1]
        + dynamic_preprocess_msac2(
            image,
            max_num=max_num,
            image_size=input_size,
            use_thumbnail=False,
            prior_aspect_ratio=target_aspect_ratio,
        )
        + images[-1:]
    )

    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class SailVL(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path="BytedanceDouyinContent/SAIL-VL-2B",
        load_in_8bit=False,
        use_msac=True,
        use_cot=False,
        max_new_tokens=1024,
        **kwargs,
    ):
        assert model_path is not None
        assert version_cmp(transformers.__version__, "4.36.2", "ge")
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.use_msac = use_msac
        # Regular expression to match the pattern 'Image' followed by a number, e.g. Image1
        self.pattern = r"Image(\d+)"
        # Replacement pattern to insert a hyphen between 'Image' and the number, e.g. Image-1
        self.replacement = r"Image-\1"
        self.use_cot = use_cot
        self.cot_prompt = (
            "You FIRST think about the reasoning process as an internal monologue "
            "and then provide the final answer. The reasoning process MUST BE "
            "enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
        )
        # Convert InternVL2 response to dataset format
        # e.g. Image1 -> Image-1

        # Regular expression to match the pattern 'Image-' followed by a number
        self.reverse_pattern = r"Image-(\d+)"
        # Replacement pattern to remove the hyphen (Image-1 -> Image1)
        self.reverse_replacement = r"Image\1"

        self.model = (
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            .eval()
            .cuda()
        )
        self.device = "cuda"

        self.image_size = self.model.config.vision_config.image_size
        kwargs_default = dict(do_sample=False, max_new_tokens=max_new_tokens, top_p=None)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(["MMDU", "MME-RealWorld", "MME-RealWorld-CN", "ChartQAPro", "ChartMuseum"], dataset):  # noqa: E501
            # For Multi-Turn and some special datasets we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == "VIDEO":
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and DATASET_TYPE(dataset) == "Y/N":
            question = line["question"]
            if listinstr(["MME"], dataset):
                prompt = question + " Answer the question using a single word or phrase."
            elif listinstr(["HallusionBench", "AMBER"], dataset):
                prompt = question + " Please answer yes or no. Answer the question using a single word or phrase."
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == "MCQ":
            prompt = build_multi_choice_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == "VQA":
            question = line["question"]
            if listinstr(["LLaVABench", "WildVision"], dataset):
                prompt = question + "\nAnswer this question in detail."
            elif listinstr(
                [
                    "OCRVQA",
                    "TextVQA",
                    "ChartQA",
                    "DocVQA",
                    "InfoVQA",
                    "OCRBench",
                    "DUDE",
                    "SLIDEVQA",
                    "GQA",
                    "MMLongBench_DOC",
                ],
                dataset,
            ):
                prompt = question + "\nAnswer the question using a single word or phrase."
            elif listinstr(
                [
                    "MathVista",
                    "MathVision",
                    "VCR",
                    "MMVet",
                    "MTVQA",
                    "MathVerse",
                    "MMDU",
                    "CRPE",
                    "MIA-Bench",
                    "MM-Math",
                    "DynaMath",
                    "QSpatial",
                ],
                dataset,
            ):
                prompt = question
            else:
                prompt = question + "\nAnswer the question using a single word or phrase."
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line["question"]
        message = [dict(type="text", value=prompt)]
        message.extend([dict(type="image", value=s) for s in tgt_path])
        return message

    def set_max_num(self, dataset):
        # The total limit on the number of images processed, set to avoid Out-of-Memory issues.
        self.total_max_num = 64
        if dataset is None:
            self.max_num = 10
            return None
        res_12_datasets = [
            "ChartQA_TEST",
            "MMMU_DEV_VAL",
            "MMMU_TEST",
            "MME-RealWorld",
            "VCR_EN",
            "VCR_ZH",
            "OCRVQA",
        ]
        res_18_datasets = [
            "DocVQA_VAL",
            "DocVQA_TEST",
            "DUDE",
            "MMLongBench_DOC",
            "SLIDEVQA",
        ]
        res_24_datasets = [
            "InfoVQA_VAL",
            "InfoVQA_TEST",
            "OCRBench",
            "HRBench4K",
            "HRBench8K",
        ]
        if DATASET_MODALITY(dataset) == "VIDEO":
            self.max_num = 1
        elif listinstr(res_12_datasets, dataset):
            self.max_num = 12
        elif listinstr(res_18_datasets, dataset):
            self.max_num = 18
        elif listinstr(res_24_datasets, dataset):
            self.max_num = 24
        else:
            self.max_num = 10

    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        image_num = len([x for x in message if x["type"] == "image"])
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        prompt = reorganize_prompt(message, image_num, dataset=dataset)

        if dataset is not None and DATASET_MODALITY(dataset) == "VIDEO":
            prompt = build_video_prompt(prompt, dataset)

        if image_num > 1:
            image_path = [x["value"] for x in message if x["type"] == "image"]
            num_patches_list, pixel_values_list = [], []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(["MMMU"], dataset)
                curr_pixel_values = (
                    load_image(file_name, max_num=max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                )
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x["value"] for x in message if x["type"] == "image"][0]
            upscale_flag = dataset is not None and listinstr(["MMMU"], dataset)
            if self.use_msac:
                pixel_values = (
                    load_image_msac(
                        image_path,
                        max_num=self.max_num,
                        upscale=upscale_flag,
                        input_size=self.image_size,
                    )
                    .cuda()
                    .to(torch.bfloat16)
                )
            else:
                pixel_values = (
                    load_image(image_path, max_num=max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                )
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []
        if self.use_cot:
            prompt += f"\n{self.cot_prompt}"
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                question=prompt,
                generation_config=self.kwargs,
                verbose=True,
            )
        response = process_response(response, dataset_name=dataset, use_cot=self.use_cot)
        return response

    def build_history(self, message):
        # Global Variables
        image_path = []
        image_cnt = 0

        def concat_tilist(tilist):
            nonlocal image_cnt  # Declare image_cnt as nonlocal to modify it
            prompt = ""
            for item in tilist:
                # Substitute the pattern in the text
                if item["type"] == "text":
                    prompt += re.sub(self.pattern, self.replacement, item["value"])
                elif item["type"] == "image":
                    image_cnt += 1
                    prompt += "<image>\n"
                    image_path.append(item["value"])
            return prompt

        # Only previous messages
        assert len(message) % 2 == 0
        history = []
        for i in range(len(message) // 2):
            m1, m2 = message[2 * i], message[2 * i + 1]
            assert m1["role"] == "user" and m2["role"] == "assistant"
            history.append((concat_tilist(m1["content"]), concat_tilist(m2["content"])))

        return history, image_path, image_cnt

    def chat_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        kwargs_default = dict(do_sample=False, max_new_tokens=512, top_p=None, num_beams=1)
        self.kwargs = kwargs_default
        if len(message) > 1:
            history, image_path, image_cnt = self.build_history(message[:-1])
        else:
            history, image_path, image_cnt = None, [], 1
        current_msg = message[-1]
        question = ""

        # If message is just text in the conversation
        if len(current_msg["content"]) == 1 and current_msg["content"][0]["type"] == "text":
            question = current_msg["content"][0]["value"]
            question = re.sub(self.pattern, self.replacement, question)  # Fix pattern as per InternVL
        else:
            for msg in current_msg["content"]:
                if msg["type"] == "text":
                    question += re.sub(self.pattern, self.replacement, msg["value"])
                elif msg["type"] == "image":
                    image_cnt += 1
                    question += "<image>\n"
                    image_path.append(msg["value"])

        if image_cnt > 1:
            num_patches_list = []
            pixel_values_list = []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(["MMMU_DEV_VAL"], dataset)
                curr_pixel_values = (
                    load_image(file_name, max_num=self.max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                )
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_cnt == 1:
            upscale_flag = listinstr(["MMMU_DEV_VAL"], dataset)
            pixel_values = (
                load_image(image_path, max_num=self.max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
            )
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
            return_history=True,
        )
        response = re.sub(self.reverse_pattern, self.reverse_replacement, response)
        return response
