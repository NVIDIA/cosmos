import logging
import os
import os.path as osp
import sys
import warnings

import torch
from PIL import Image

from .base import BaseModel

'''
    Please follow the instructions to download ckpt.
    https://github.com/dvlab-research/MGM?tab=readme-ov-file#pretrained-weights
'''


class Mini_Gemini(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_path, root=None, conv_mode='llava_v1', **kwargs):
        if root is None:
            warnings.warn('Please set `root` to Mini_Gemini code directory, \
                which is cloned from here: "https://github.com/dvlab-research/MGM?tab=readme-ov-file" ')
            raise ValueError
        warnings.warn('Please follow the instructions of Mini_Gemini to put the ckpt file in the right place, \
            which can be found at https://github.com/dvlab-research/MGM?tab=readme-ov-file#structure')
        assert model_path == 'YanweiLi/MGM-7B-HD', 'We only support MGM-7B-HD for now'
        self.model_path = model_path
        sys.path.append(root)
        try:
            from mgm.mm_utils import get_model_name_from_path
            from mgm.model.builder import load_pretrained_model
        except Exception as e:
            logging.critical(
                'Please first install Mini_Gemini and set the root path to use Mini_Gemini, '
                'which is cloned from here: "https://github.com/dvlab-research/MGM?tab=readme-ov-file" '
            )
            raise e

        VLMEvalKit_path = os.getcwd()
        os.chdir(root)
        warnings.warn('Please set `root` to Mini_Gemini code directory, \
            which is cloned from here: "https://github.com/dvlab-research/MGM?tab=readme-ov-file" ')
        model_path = osp.join(root, 'work_dirs', 'MGM', 'MGM-7B-HD')
        try:
            model_name = get_model_name_from_path(model_path)
        except Exception as e:
            logging.critical(
                'Please follow the instructions of Mini_Gemini to put the ckpt file in the right place, '
                'which can be found at https://github.com/dvlab-research/MGM?tab=readme-ov-file#structure'
            )
            raise e

        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
        os.chdir(VLMEvalKit_path)
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = conv_mode

        kwargs_default = dict(temperature=float(0), num_beams=1, top_p=None, max_new_tokens=1024, use_cache=True)
        kwargs_default.update(kwargs)
        do_sample = kwargs_default['temperature'] > 0
        kwargs_default.update({'do_sample': do_sample})
        self.kwargs = kwargs_default

    def generate_inner(self, message, dataset=None):
        try:
            from mgm.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                       DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
            from mgm.conversation import conv_templates
            from mgm.mm_utils import process_images, tokenizer_image_token
        except Exception as e:
            logging.critical(
                'Please first install Mini_Gemini and set the root path to use Mini_Gemini, '
                'which is cloned from here: "https://github.com/dvlab-research/MGM?tab=readme-ov-file" '
            )
            raise e

        prompt, image = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image)
        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda()
        if hasattr(self.model.config, 'image_size_aux'):
            if not hasattr(self.image_processor, 'image_size_raw'):
                self.image_processor.image_size_raw = self.image_processor.crop_size.copy()
            self.image_processor.crop_size['height'] = self.model.config.image_size_aux
            self.image_processor.crop_size['width'] = self.model.config.image_size_aux
            self.image_processor.size['shortest_edge'] = self.model.config.image_size_aux
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        image_grid = getattr(self.model.config, 'image_grid', 1)
        if hasattr(self.model.config, 'image_size_aux'):
            raw_shape = [
                self.image_processor.image_size_raw['height'] * image_grid,
                self.image_processor.image_size_raw['width'] * image_grid
            ]
            image_tensor_aux = image_tensor
            image_tensor = torch.nn.functional.interpolate(
                image_tensor[None],
                size=raw_shape,
                mode='bilinear',
                align_corners=False
            )[0]
        else:
            image_tensor_aux = []
        if image_grid >= 2:
            raw_image = image_tensor.reshape(
                3, image_grid, self.image_processor.image_size_raw['height'],
                image_grid, self.image_processor.image_size_raw['width']
            )
            raw_image = raw_image.permute(1, 3, 0, 2, 4)
            raw_image = raw_image.reshape(
                -1, 3, self.image_processor.image_size_raw['height'], self.image_processor.image_size_raw['width']
            )

            if getattr(self.model.config, 'image_global', False):
                global_image = image_tensor
                if len(global_image.shape) == 3:
                    global_image = global_image[None]
                global_image = torch.nn.functional.interpolate(
                    global_image,
                    size=[
                        self.image_processor.image_size_raw['height'],
                        self.image_processor.image_size_raw['width']
                    ],
                    mode='bilinear',
                    align_corners=False
                )
                # [image_crops, image_global]
                raw_image = torch.cat([raw_image, global_image], dim=0)
            image_tensor = raw_image.contiguous()

        images = image_tensor[None].to(dtype=self.model.dtype, device='cuda', non_blocking=True)
        if len(image_tensor_aux) > 0:
            images_aux = image_tensor_aux[None].to(dtype=self.model.dtype, device='cuda', non_blocking=True)
        else:
            images_aux = None

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                images_aux=images_aux,
                # no_repeat_ngram_size=3,
                bos_token_id=self.tokenizer.bos_token_id,  # Begin of sequence token
                eos_token_id=self.tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=self.tokenizer.pad_token_id,  # Pad token
                **self.kwargs
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
