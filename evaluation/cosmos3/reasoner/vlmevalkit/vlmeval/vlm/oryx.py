import logging
import os
import os.path as osp
import string
import warnings

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPImageProcessor

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import splitlen
from .base import BaseModel

os.environ['LOWRES_RESIZE'] = "384x32"
os.environ['HIGHRES_BASE'] = "0x32"
os.environ['MAXRES'] = "1536"
os.environ['MINRES'] = "0"
os.environ['SIMPLE_ARCH'] = "1"
os.environ['PAD2STRIDE'] = "1"
os.environ['REGIONAL_POOL'] = '2x'
os.environ['FORCE_NO_DOWNSAMPLE'] = "1"


def preprocess_qwen(
    sources, tokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant."
):
    from oryx.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX

    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start = tokenizer("<|im_start|>").input_ids[0]
    im_end = tokenizer("<|im_end|>").input_ids[0]
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(DEFAULT_IMAGE_TOKEN.findall(sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i == IMAGE_TOKEN_INDEX for i in _input_id]) == num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens  # noqa: E501
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1: -2] + [im_end] + nl_tokens  # noqa: E501
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


class Oryx(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='liuhaotian/llava_v1.5_7b',
                 **kwargs):
        try:
            from oryx.mm_utils import get_model_name_from_path
            from oryx.model.builder import load_pretrained_model
        except Exception as err:
            logging.critical('Please install requirements on https://github.com/Oryx-mllm/Oryx before using Oryx')
            raise err

        assert osp.exists(model_path) or splitlen(model_path) == 2

        model_name = get_model_name_from_path(model_path)

        overwrite_config = {}
        overwrite_config["mm_resampler_type"] = "dynamic_compressor"
        overwrite_config["patchify_video_feature"] = False
        overwrite_config["attn_implementation"] = "sdpa" if torch.__version__ >= "2.1.2" else "eager"

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name, device_map="cuda:0", overwrite_config=overwrite_config
        )

        if self.image_processor is None:
            self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
            print('Using default image processor. ')

        self._config = self.model.config

        self.model = self.model.cuda()
        self.conv_mode = 'qwen_1_5'

        self.device = torch.device('cuda')

        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1, use_cache=True)  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

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
            prompt += '\nAnswer with the option letter from the given choices directly.'
        else:
            prompt += '\nAnswer the question using a single word or phrase.'

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def generate_inner(self, message, dataset=None):
        from oryx.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                    DEFAULT_IMAGE_TOKEN)
        from oryx.conversation import conv_templates
        from oryx.mm_utils import KeywordsStoppingCriteria, process_anyres_highres_image_genli

        # Support interleave text and image
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], 'PLACEHOLDER')
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        content, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            elif msg['type'] == 'image':
                if self.model.config.mm_use_im_start_end:
                    content += DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
                else:
                    content += DEFAULT_IMAGE_TOKEN + '\n'
                images.append(msg['value'])

        images = [Image.open(s).convert('RGB') for s in images]
        image_sizes = [img.size for img in images]
        self.image_processor.do_resize = False
        self.image_processor.do_center_crop = False
        image_tensor, image_highres_tensor = [], []
        for visual in images:
            image_tensor_, image_highres_tensor_ = process_anyres_highres_image_genli(visual, self.image_processor)
            image_tensor.append(image_tensor_)
            image_highres_tensor.append(image_highres_tensor_)
        if type(image_tensor) is list:
            image_tensor = [_image.bfloat16().to("cuda") for _image in image_tensor]
        else:
            image_tensor = image_tensor.bfloat16().to("cuda")
        if type(image_highres_tensor) is list:
            image_highres_tensor = [_image.bfloat16().to("cuda") for _image in image_highres_tensor]
        else:
            image_highres_tensor = image_highres_tensor.bfloat16().to("cuda")
        prompt = prompt.replace('PLACEHOLDER', content)

        input_ids = preprocess_qwen(
            [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': None}], self.tokenizer, has_image=True
        ).cuda()
        stop_str = '<|im_end|>'
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id  # noqa: E501
        attention_masks = input_ids.ne(pad_token_ids).to(self.device)
        print(self.kwargs)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, images_highres=image_highres_tensor, image_sizes=image_sizes,
                modalities=['image'] * len(image_tensor),
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                stopping_criteria=[stopping_criteria], **self.kwargs)

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(content, output)
        return output
