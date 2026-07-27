# flake8: noqa
import copy as cp
import math
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from ..dataset import DATASET_TYPE
from .base import BaseModel

IMG_TAG_TOKEN = "<image>"
VID_TAG_TOKEN = "<video>"
AUD_TAG_TOKEN = "<audio>"

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'

VID_CONTEXT_TOKEN = '<VID_CONTEXT>'
VID_START_TOKEN = '<vid>'
VID_END_TOKEN = '</vid>'

PATCH_CONTEXT_TOKEN = '<PATCH_CONTEXT>'
PATCH_START_TOKEN = '<patch>'
PATCH_END_TOKEN = '</patch>'

AUD_START_TOKEN = '<|begin_of_audio|>'
AUD_END_TOKEN = '<|end_of_audio|>'

QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'


IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = IMG_CONTEXT_TOKEN
DEFAULT_IMAGE_PATCH_TOKEN = PATCH_CONTEXT_TOKEN
DEFAULT_IM_START_TOKEN = IMG_START_TOKEN
DEFAULT_IM_END_TOKEN = IMG_END_TOKEN


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )

        # Calculate effective and wasted resolutions
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


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
        processed_images = [
            thumbnail_img,
        ] + processed_images
    return processed_images, (target_width, target_height)


def get_external_inputs(tokens, tokenizer, image_processor, image_list=None, image_path_list=None, video_path_list=None, max_num_frame=4096, max_fps=1, image_token_length=256):
    tokens = tokens.tolist()

    IMG_CONTEXT_ID = tokenizer(IMG_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    IMG_START_ID = tokenizer(IMG_START_TOKEN, add_special_tokens=False).input_ids
    IMG_END_ID = tokenizer(IMG_END_TOKEN, add_special_tokens=False).input_ids

    VID_CONTEXT_ID = tokenizer(VID_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    VID_START_ID = tokenizer(VID_START_TOKEN, add_special_tokens=False).input_ids
    VID_END_ID = tokenizer(VID_END_TOKEN, add_special_tokens=False).input_ids

    PATCH_CONTEXT_ID = tokenizer(PATCH_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    PATCH_START_ID = tokenizer(PATCH_START_TOKEN, add_special_tokens=False).input_ids
    PATCH_END_ID = tokenizer(PATCH_END_TOKEN, add_special_tokens=False).input_ids

    IMG_TAG_ID = tokenizer(IMG_TAG_TOKEN, add_special_tokens=False).input_ids
    VID_TAG_ID = tokenizer(VID_TAG_TOKEN, add_special_tokens=False).input_ids

    assert len(IMG_CONTEXT_ID) == 1
    assert len(IMG_START_ID) == 1
    assert len(IMG_END_ID) == 1

    assert len(VID_CONTEXT_ID) == 1
    assert len(VID_START_ID) == 1
    assert len(VID_END_ID) == 1

    assert len(PATCH_CONTEXT_ID) == 1
    assert len(PATCH_START_ID) == 1
    assert len(PATCH_END_ID) == 1

    IMG_CONTEXT_ID = IMG_CONTEXT_ID[0]
    IMG_START_ID = IMG_START_ID[0]
    IMG_END_ID = IMG_END_ID[0]

    VID_CONTEXT_ID = VID_CONTEXT_ID[0]
    VID_START_ID = VID_START_ID[0]
    VID_END_ID = VID_END_ID[0]

    PATCH_CONTEXT_ID = PATCH_CONTEXT_ID[0]
    PATCH_START_ID = PATCH_START_ID[0]
    PATCH_END_ID = PATCH_END_ID[0]

    IMG_TAG_ID = IMG_TAG_ID[0]
    VID_TAG_ID = VID_TAG_ID[0]

    nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids

    image_indices = []
    images = []

    # ----------------------------------------------------------------
    # image
    for batch_idx, input_ids in enumerate(tokens):
        img_positions = [i for i, x in enumerate(input_ids) if x == IMG_TAG_ID]
        if len(img_positions) == 0:
            continue
        if image_path_list is not None:
            assert len(img_positions) == len(image_path_list), f"{img_positions} {image_path_list} {IMG_CONTEXT_TOKEN} {IMG_CONTEXT_ID} {tokens}"
        if image_list is not None:
            assert len(img_positions) == len(image_list), f"{img_positions} {image_list} {IMG_CONTEXT_TOKEN} {IMG_CONTEXT_ID} {tokens}"

        new_input_ids = []
        st = 0
        for img_idx, img_pos in enumerate(img_positions):
            if image_path_list is not None:
                image_patches, (best_width, best_height) = image_processor.process_images_with_subpatch(image_path_list[img_idx])
            if image_list is not None:
                image_patches, (best_width, best_height) = image_processor.process_images_with_subpatch(image_list[img_idx])
            images.append(image_patches)

            new_input_ids += input_ids[st:img_pos]

            new_input_ids += [IMG_START_ID]

            image_indice_b = torch.zeros(
                1, image_token_length, dtype=torch.int64
            )  # This will change in collate_fn
            image_indice_s = (
                torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                .unsqueeze(0)
                .repeat(1, 1)
            )
            image_indice_b_s = torch.stack(
                [image_indice_b, image_indice_s], dim=0
            )  # 2, num_image, image_length
            image_indices.append(image_indice_b_s)

            new_input_ids += [IMG_CONTEXT_ID] * image_token_length

            new_input_ids += [IMG_END_ID]

            if len(image_patches) > 1:
                for i in range(0, best_height, image_processor.patch_size):
                    new_input_ids += nl_tokens

                    for j in range(0, best_width, image_processor.patch_size):
                        new_input_ids += [PATCH_START_ID]

                        image_indice_b = torch.zeros(
                            1, image_token_length, dtype=torch.int64
                        )  # This will change in collate_fn
                        image_indice_s = (
                            torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                            .unsqueeze(0)
                            .repeat(1, 1)
                        )
                        image_indice_b_s = torch.stack(
                            [image_indice_b, image_indice_s], dim=0
                        )  # 2, num_image, image_length
                        image_indices.append(image_indice_b_s)

                        new_input_ids += [PATCH_CONTEXT_ID] * image_token_length

                        new_input_ids += [PATCH_END_ID]

            st = img_pos + 1

        new_input_ids += input_ids[st:]

        input_ids = new_input_ids
        tokens[batch_idx] = input_ids

    # ----------------------------------------------------------------
    # video
    for batch_idx, input_ids in enumerate(tokens):
        vid_positions = [i for i, x in enumerate(input_ids) if x == VID_TAG_ID]
        if len(vid_positions) == 0:
            continue
        if video_path_list is not None:
            assert len(vid_positions) == len(video_path_list), f"{vid_positions} {video_path_list} {VID_CONTEXT_TOKEN} {VID_CONTEXT_ID} {tokens}"
        if image_path_list is not None:
            assert len(vid_positions) == len(image_path_list), f"{vid_positions} {image_path_list} {VID_CONTEXT_TOKEN} {VID_CONTEXT_ID} {tokens}"
        if image_list is not None:
            assert len(vid_positions) == len(image_list), f"{vid_positions} {image_list} {VID_CONTEXT_TOKEN} {VID_CONTEXT_ID} {tokens}"

        new_input_ids = []
        st = 0
        for vid_idx, vid_pos in enumerate(vid_positions):
            if video_path_list is not None:
                video_frames, _ = image_processor.process_video(video_path_list[vid_idx], max_num_frame, max_fps)
            if image_path_list is not None:
                video_frames = image_processor.process_images([image_path_list[vid_idx]])
            if image_list is not None:
                video_frames = image_processor.process_images([image_list[vid_idx]])

            images.append(video_frames)

            new_input_ids += input_ids[st:vid_pos]

            for _ in video_frames:
                new_input_ids += [VID_START_ID]

                image_indice_b = torch.zeros(
                    1, image_token_length, dtype=torch.int64
                )  # This will change in collate_fn
                image_indice_s = (
                    torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                    .unsqueeze(0)
                    .repeat(1, 1)
                )
                image_indice_b_s = torch.stack(
                    [image_indice_b, image_indice_s], dim=0
                )  # 2, num_image, image_length
                image_indices.append(image_indice_b_s)

                new_input_ids += [VID_CONTEXT_ID] * image_token_length

                new_input_ids += [VID_END_ID]

            st = vid_pos + 1

        new_input_ids += input_ids[st:]

        input_ids = new_input_ids
        tokens[batch_idx] = input_ids

    images = torch.cat(images, dim=0)
    images = torch.tensor(images, dtype=torch.bfloat16).contiguous().to(torch.cuda.current_device())

    image_indices = torch.cat(image_indices, dim=1)
    image_indices = image_indices.contiguous().to(torch.cuda.current_device())

    tokens = torch.tensor(tokens, dtype=torch.long, device='cuda')

    return tokens, images, image_indices


class ImageProcessor:
    def __init__(
        self,
        process_type,
        image_size=448,
        normalize_type="imagenet",
        min_patch_grid=1,
        max_patch_grid=6,
    ):
        self.process_type = process_type
        self.image_size = image_size

        if normalize_type == "imagenet":
            MEAN, STD = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        elif normalize_type == "clip":
            MEAN, STD = OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
        elif normalize_type == "siglip":
            MEAN, STD = IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
        else:
            raise NotImplementedError
        self.mean = MEAN
        self.std = STD

        self.patch_size = image_size
        self.min_patch_grid = min_patch_grid
        self.max_patch_grid = max_patch_grid

        if self.process_type == "anyres":
            self.grid_pinpoints = [
                (i, j)
                for i in range(min_patch_grid, max_patch_grid + 1)
                for j in range(min_patch_grid, max_patch_grid + 1)
            ]
            self.possible_resolutions = [
                [dim * self.patch_size for dim in pair] for pair in self.grid_pinpoints
            ]

        if self.process_type == "dynamic":
            max_num = self.max_patch_grid
            min_num = self.min_patch_grid
            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            )
            self.target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
            self.possible_resolutions = [
                [dim * self.patch_size for dim in pair] for pair in self.target_ratios
            ]

    def get_frame_paths(self, frame_root, num_frames=8):
        os.makedirs(frame_root, exist_ok=True)

        self.frame_tmpl = "frame-{}-of-{}.jpg"
        return [
            os.path.join(frame_root, self.frame_tmpl.format(i, num_frames))
            for i in range(1, num_frames + 1)
        ]

    def save_video_frames(self, vid_path, max_fps=1, num_frames=8):
        import decord
        vid = decord.VideoReader(vid_path, num_threads=1)

        step_size = len(vid) / (num_frames + 1)
        fps = vid.get_avg_fps()
        step_size = max(fps / max_fps, step_size)

        indices = [int(i * step_size) for i in range(0, num_frames)]
        indices = [i for i in indices if i < len(vid)]

        num_frames = len(indices)

        frame_paths = self.get_frame_paths(vid_path + ".saved_frames", num_frames)
        flag = np.all([os.path.exists(p) for p in frame_paths])
        if flag:
            return frame_paths

        images = [vid[i].asnumpy() for i in indices]
        images = [Image.fromarray(arr) for arr in images]

        for im, pth in zip(images, frame_paths):
            im.save(pth)

        return frame_paths

    def get_video_frames(self, vid_path, max_fps=1, num_frames=8):
        import decord
        vid = decord.VideoReader(vid_path, num_threads=1)

        step_size = len(vid) / (num_frames + 1)
        fps = vid.get_avg_fps()
        step_size = max(fps / max_fps, step_size)

        indices = [int(i * step_size) for i in range(0, num_frames)]
        indices = [i for i in indices if i < len(vid)]

        images = [vid[i].asnumpy() for i in indices]
        images = [Image.fromarray(arr) for arr in images]
        print(f"for this video, video duration is {len(vid)/fps}s, get {len(images)} frames for inference")

        return images

    def process_video(self, video_file_or_dir, max_num_frame=8, max_fps=1):
        import natsort
        if os.path.isdir(video_file_or_dir):
            all_filepath = []
            for root, dirs, files in os.walk(video_file_or_dir):
                for filename in files:
                    if (
                        filename.endswith("png")
                        or filename.endswith("jpeg")
                        or filename.endswith("jpg")
                    ):
                        filepath = os.path.join(root, filename)
                        all_filepath.append(filepath)

            if len(all_filepath) == 0:
                return None

            all_filepath = natsort.natsorted(all_filepath)
            total_frame = len(all_filepath)
            if "ShareGPTVideo" in video_file_or_dir:
                fps = 2
            else:
                fps = 1
            target_frame = int(min(total_frame / fps * max_fps, max_num_frame))
            index = [int(1.0 * total_frame / target_frame) * x for x in range(target_frame)]

            selected_filepath = [all_filepath[x] for x in index]

            img_or_path_list = selected_filepath
        elif os.path.isfile(video_file_or_dir):
            img_or_path_list = self.get_video_frames(
                video_file_or_dir, num_frames=max_num_frame, max_fps=max_fps
            )
        else:
            raise NotImplementedError

        return self.process_images(img_or_path_list), img_or_path_list

    def process_images(self, img_or_path_list):
        if isinstance(img_or_path_list[0], str):
            images = [Image.open(x).convert("RGB") for x in img_or_path_list]
        elif isinstance(img_or_path_list[0], Image.Image):
            images = [x.convert("RGB") for x in img_or_path_list]
        else:
            images = img_or_path_list

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image_tensor = torch.ones([len(images), 3, self.image_size, self.image_size])

        for i, image in enumerate(images):
            image = expand2square(image, tuple(int(x * 255) for x in self.mean))

            image = image.resize(
                (self.image_size, self.image_size), resample=Image.Resampling.BICUBIC
            )

            image = np.array(image, dtype=np.float32)
            image = image * 1.0 / 255.0

            mean = np.array(self.mean, dtype=image.dtype)
            std = np.array(self.std, dtype=image.dtype)
            image = (image - mean) / std

            image = torch.tensor(image, dtype=torch.float32)
            image = image.permute(2, 0, 1)

            image_tensor[i] = image

        return image_tensor

    def process_images_with_subpatch(self, img_or_path):
        if self.process_type == "anyres":
            return self.process_anyres(img_or_path)
        if self.process_type == "dynamic":
            return self.process_dynamic(img_or_path)

        if isinstance(img_or_path, str):
            image = Image.open(img_or_path).convert("RGB")
        elif isinstance(img_or_path, Image.Image):
            image = img_or_path.convert("RGB")
        else:
            image = img_or_path

        return self.process_images([image])

    def process_anyres(self, img_or_path):
        if isinstance(img_or_path, str):
            image = Image.open(img_or_path).convert("RGB")
        elif isinstance(img_or_path, Image.Image):
            image = img_or_path.convert("RGB")
        else:
            image = img_or_path

        best_resolution = select_best_resolution(image.size, self.possible_resolutions)
        image_padded = resize_and_pad_image(image, best_resolution)
        patches = divide_to_patches(image_padded, self.patch_size)

        if best_resolution == (self.patch_size, self.patch_size):
            image_patches = [image]
        else:
            image_patches = [image] + patches

        image_patches = self.process_images(image_patches)

        return image_patches, best_resolution

    def process_dynamic(self, img_or_path):
        if isinstance(img_or_path, str):
            image = Image.open(img_or_path).convert("RGB")
        elif isinstance(img_or_path, Image.Image):
            image = img_or_path.convert("RGB")
        else:
            image = img_or_path

        image_patches, best_resolution = dynamic_preprocess(
            image,
            min_num=self.min_patch_grid,
            max_num=self.max_patch_grid,
            image_size=self.patch_size,
            use_thumbnail=True,
        )

        image_patches = self.process_images(image_patches)

        return image_patches, best_resolution


class LongVITAWrapper(BaseModel):
    allowed_types = ['text', 'image', 'video']
    is_api: bool = False

    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(self, model_path='VITA-MLLM/Long-VITA-16K_HF', max_num_frame=4096, **kwargs):
        assert model_path is not None

        self.default_params = {
            'top_p': 0,
            'top_k': 0,
            'temperature': 1.0,
            'repetition_penalty': 1.0,
            'do_sample': False,
            'max_new_tokens': 1024,
        }

        chat_template = """
        {%- for message in messages %} {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %} {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }} {%- elif message.role == "assistant" %} {{- '<|im_start|>' + message.role }} {%- if message.content %} {{- '\n' + message.content }} {%- endif %} {%- for tool_call in message.tool_calls %} {%- if tool_call.function is defined %} {%- set tool_call = tool_call.function %} {%- endif %} {{- '\n<tool_call>\n{"name": "' }} {{- tool_call.name }} {{- '", "arguments": ' }} {{- tool_call.arguments | tojson }} {{- '}\n</tool_call>' }} {%- endfor %} {{- '<|im_end|>\n' }} {%- elif message.role == "tool" %} {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %} {{- '<|im_start|>user' }} {%- endif %} {{- '\n<tool_response>\n' }} {{- message.content }} {{- '\n</tool_response>' }} {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %} {{- '<|im_end|>\n' }} {%- endif %} {%- endif %} {%- endfor %} {%- if add_generation_prompt %} {{- '<|im_start|>assistant\n' }} {%- endif %}
        """

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            chat_template=chat_template,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()

        model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        model.generation_config.max_new_tokens = 1024
        model.generation_config.chat_format = "chatml"
        model.generation_config.max_window_size = 1310720
        model.generation_config.do_sample = False
        model.generation_config.use_cache = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        image_processor = ImageProcessor(
            process_type="dynamic",
            image_size=448,
            normalize_type="imagenet",
            min_patch_grid=1,
            max_patch_grid=12,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        max_num_frame = os.environ.get('MAX_NUM_FRAME', default=max_num_frame)
        self.max_num_frame = int(max_num_frame)
        print(f"max_num_frame {self.max_num_frame}")

    def build_msgs(self, msgs_raw, system_prompt=None, dataset=None):
        msgs = cp.deepcopy(msgs_raw)
        image_list = []
        image_path_list = []
        video_path_list = []
        text = ""
        image_count = 1
        for i, msg in enumerate(msgs):
            if msg['type'] == 'text':
                text += msg['value']
            elif msg['type'] == 'image':
                image_path_list.append(msg['value'])
                if dataset == "Video-MME":
                    if image_count == 1:
                        text += f"<video>"
                    else:
                        text += f"<video>"
                else:
                    text += f"<image>\n"
                image_count += 1

            elif msg['type'] == 'video':
                video_path_list.append(msg['value'])
                text += f"<video>"
            else:
                raise ValueError(f"Invalid message type: {msg['type']}, {msg}")

        # VideoMME
        text = text.replace("\nAnswer: ", "\n")

        if dataset == "OCRBench":
            text += "\nAnswer this question using the text in the image directly without any other context."

        elif dataset in ['MMMU_DEV_VAL', 'MMMU_TEST', "MMStar"]:
            text = text.replace("Please select the correct answer from the options above.", "").strip() + "\n"
            text += "Answer with the option's letter from the given choices directly."

        elif dataset in ['MVBench',]:
            text = text.replace("Only give the best option.Best option:(", "")
            text += "Answer with the letter."

        elif dataset in ['MMVet',]:
            pass

        elif dataset in ['MathVista_MINI',]:
            text += "\nAnswer the question using a single word or phrase."

        elif dataset is not None and DATASET_TYPE(dataset) in ['Y/N',]:
            text = text.replace("Answer the question with Yes or No.", "").strip() + "\n"
            text += "Answer yes or no."

        elif dataset is not None and DATASET_TYPE(dataset) in ['MCQ',]:
            text = text.replace("Please select the correct answer from the options above.", "").strip() + "\n"
            text += "Answer with the letter."

        elif dataset is not None and DATASET_TYPE(dataset) in ['VQA',]:
            pass

        elif dataset is not None and DATASET_TYPE(dataset) in ['Video-MCQ',]:
            text += "Offer a very short reply."

        else:
            text = text.replace("Answer the question using a single word or phrase.", "").strip() + "\n"
            text += "Answer the question using a single word or phrase."

        return text, image_list, image_path_list, video_path_list

    def generate_inner(self, inputs, dataset=None) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)
        inputs = [inputs] if isinstance(inputs, str) else inputs

        print("")
        print("-" * 100, flush=True)

        message, image_list, image_path_list, video_path_list = self.build_msgs(msgs_raw=inputs, dataset=dataset)

        messages = [
            {
                "role": "user",
                "content": message,
            }
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        print("input", self.tokenizer.decode(inputs[0], skip_special_tokens=False), flush=True)

        inputs, images, image_indices = get_external_inputs(inputs, self.tokenizer, self.image_processor, image_path_list=image_path_list if len(image_path_list) > 0 else None, video_path_list=video_path_list if len(video_path_list) > 0 else None, max_num_frame=self.max_num_frame)
        outputs = self.model.generate(inputs=inputs, images=images, image_indices=image_indices, **self.default_params)
        output = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print(f"output", output, flush=True)

        answer = output

        return answer


class LongVITA(LongVITAWrapper):

    def generate(self, message, dataset=None):
        return super(LongVITA, self).generate(message, dataset=dataset)
