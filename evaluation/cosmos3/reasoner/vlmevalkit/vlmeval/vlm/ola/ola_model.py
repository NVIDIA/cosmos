import os
import os.path as osp
import string
import sys
import warnings
from abc import abstractproperty

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPImageProcessor

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import splitlen
from ..base import BaseModel

os.environ['LOWRES_RESIZE']="384x32"
os.environ['HIGHRES_BASE']="0x32"
os.environ['MAXRES']="1536"
os.environ['MINRES']="0"
os.environ['SIMPLE_ARCH']="1"
os.environ['PAD2STRIDE']="1"
os.environ['REGIONAL_POOL']='2x'
os.environ['FORCE_NO_DOWNSAMPLE']="1"
os.environ['LOAD_VISION_EARLY']="1"
os.environ['SKIP_LOAD_VIT']="1"

class Ola(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='liuhaotian/llava_v1.5_7b',
                 **kwargs):

        from .ola.mm_utils import get_model_name_from_path
        from .ola.model.builder import load_pretrained_model

        assert osp.exists(model_path) or splitlen(model_path) == 2

        model_name = get_model_name_from_path(model_path)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            use_flash_attn=True,
        )

        if self.image_processor is None:
            self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
            print('Using default image processor. ')

        self._config = self.model.config


        self.model = self.model.cuda()
        self.conv_mode = 'v1_qwen2'

        self.device = torch.device('cuda')

        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1, use_cache=True) # noqa E501
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
        from .ola.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                    DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
        from .ola.conversation import SeparatorStyle, conv_templates
        from .ola.datasets.preprocess import tokenizer_image_token
        from .ola.mm_utils import KeywordsStoppingCriteria, process_anyres_highres_image_genli

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
                if 'MMVet' in msg['value'] or 'MMMU' in msg['value']:
                    os.environ['USE_HIGHRES_ONLY'] = '1'
                if self.model.config.mm_use_im_start_end:
                    content += DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
                else:
                    content += DEFAULT_IMAGE_TOKEN + '\n'
                images.append(msg['value'])

        images = [Image.open(s).convert('RGB') for s in images]
        image_sizes = [img.size for img in images]
        # args = abstractproperty()
        # args.image_aspect_ratio = 'pad'
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

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = '<|im_end|>'
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        attention_masks = input_ids.ne(pad_token_ids).to(self.device)

        speechs = [torch.zeros(1, 3000, 128).bfloat16().to('cuda')]
        speech_lengths = [torch.LongTensor([3000]).to('cuda')]
        speech_wavs = [torch.zeros([1, 480000]).to('cuda')]
        speech_chunks = [torch.LongTensor([1]).to('cuda')]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, images_highres=image_highres_tensor, image_sizes=image_sizes,
                modalities=['image'] * len(image_tensor),
                speech=speechs,
                speech_lengths=speech_lengths,
                speech_chunks=speech_chunks,
                speech_wav=speech_wavs,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                stopping_criteria=[stopping_criteria], **self.kwargs)

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(content, output)
        return output
