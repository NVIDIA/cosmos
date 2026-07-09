import ast
import base64
import io
import math
import os

import torch
from PIL import Image
from transformers import StoppingCriteria

if 'HIGHRES_BASE' in os.environ:
    # highresxpatch
    HIGHRES_BASE = os.environ['HIGHRES_BASE']
    highres_base, highres_ps = HIGHRES_BASE.split('x')
    highres_base = int(highres_base)
    highres_ps = int(highres_ps)
    print(f"HIGHRES_BASE is set as {HIGHRES_BASE}, {highres_base}, {highres_ps}")
else:
    HIGHRES_BASE = None

if 'MAXRES' in os.environ:
    # highresxpatch
    MAXRES = int(os.environ['MAXRES'])
    print(f"MAXRES is set as {MAXRES}")
else:
    MAXRES = 1536

if 'MINRES' in os.environ:
    # highresxpatch
    MINRES = int(os.environ['MINRES'])
    print(f"MINRES is set as {MINRES}")
else:
    MINRES = 0

if 'PAD2STRIDE' in os.environ:
    # highresxpatch
    PAD2STRIDE = True
    print(f"PAD2STRIDE is set")
else:
    PAD2STRIDE = False

if 'LOWRES_RESIZE' in os.environ:
    LOWRES_RESIZE = os.environ['LOWRES_RESIZE']
    print(f"LOWRES_RESIZE is set as {LOWRES_RESIZE}")
    if 'x' in LOWRES_RESIZE:
        size, ps = LOWRES_RESIZE.split('x')
        size = int(size)
        ps = int(ps)
        LOWRES_RESIZE = (size, ps)
    else:
        LOWRES_RESIZE = int(LOWRES_RESIZE)
else:
    LOWRES_RESIZE = None


def pad_image(image, target_resolution, value=0):
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
    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new('RGB', (target_width, target_height), (value, value, value))
    paste_x = (target_width - original_width) // 2
    paste_y = (target_height - original_height) // 2
    new_image.paste(image, (paste_x, paste_y))
    return new_image

def resize_images(image, patch_size=14, base_size=896):
    h, w = image.size
    if base_size == 0:
        if h * w > MAXRES * MAXRES:
            scale = MAXRES * MAXRES / (h * w)
            scale = math.sqrt(scale)
        elif h * w < MINRES * MINRES:
            scale = MINRES * MINRES / (h * w)
            scale = math.sqrt(scale)
        else:
            scale = None
    else:
        scale = base_size * base_size / (h * w)
        scale = math.sqrt(scale)


    if scale is not None:
        new_h = int(h * scale / patch_size) * patch_size
        new_w = int(w * scale / patch_size) * patch_size
        new_h = max(new_h, patch_size)
        new_w = max(new_w, patch_size)
        image = image.resize((new_h, new_w))
    elif PAD2STRIDE:
        if h % patch_size == 0:
            new_h = h
        else:
            new_h = (h // patch_size + 1) * patch_size

        if w % patch_size == 0:
            new_w = w
        else:
            new_w = (w // patch_size + 1) * patch_size
        image = pad_image(image, (new_h, new_w), value=127)
    else:
        scale = 1.0
        new_h = int(h * scale / patch_size) * patch_size
        new_w = int(w * scale / patch_size) * patch_size
        new_h = max(new_h, patch_size)
        new_w = max(new_w, patch_size)
        image = image.resize((new_h, new_w))

    return image

def process_anyres_highres_image_genli(image, processor):
    h, w = image.size
    if h < 32 and w < 32:
        min_size = min(h, w)
        ratio = 64 / min_size
        image = image.resize((int(h * ratio), int(w * ratio)))
    elif h < 32:
        ratio = 64 / h
        image = image.resize((int(h * ratio), int(w * ratio)))
    elif w < 32:
        ratio = 64 / w
        image = image.resize((int(h * ratio), int(w * ratio)))
    if HIGHRES_BASE is not None:
        image = resize_images(image, patch_size=highres_ps, base_size=highres_base)

    if LOWRES_RESIZE is not None:
        image_original_resize = resize_images(image, patch_size=LOWRES_RESIZE[1], base_size=LOWRES_RESIZE[0])
    else:
        image_original_resize = image.resize((384, 384))

    image_patches = processor.preprocess(image_original_resize, return_tensors='pt')['pixel_values'][0]
    image_padded = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    return image_patches.unsqueeze(0), image_padded.unsqueeze(0)



def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
