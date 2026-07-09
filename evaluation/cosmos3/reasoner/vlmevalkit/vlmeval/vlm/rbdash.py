import logging
import os
import os.path as osp
import string
import sys
import warnings

import pandas as pd
import torch
from PIL import Image

from vlmeval.smp import download_file, listinstr
from .base import BaseModel

'''
    Please follow the instructions to download ckpt.
    https://github.com/RBDash-Team/RBDash?tab=readme-ov-file#pretrained-weights
'''


class RBDash(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_path, root=None, conv_mode='qwen', **kwargs):
        from huggingface_hub import snapshot_download
        if root is None:
            raise ValueError('Please set `root` to RBDash code directory, \
                which is cloned from here: "https://github.com/RBDash-Team/RBDash?tab=readme-ov-file" ')
        warnings.warn('Please follow the instructions of RBDash to put the ckpt file in the right place, \
            which can be found at https://github.com/RBDash-Team/RBDash?tab=readme-ov-file#structure')
        assert model_path == 'RBDash-Team/RBDash-v1.5', 'We only support RBDash-v1.5 for now'
        sys.path.append(root)
        try:
            from rbdash.mm_utils import get_model_name_from_path
            from rbdash.model.builder import load_pretrained_model
        except Exception as err:
            logging.critical(
                'Please first install RBdash and set the root path to use RBdash, '
                'which is cloned from here: "https://github.com/RBDash-Team/RBDash?tab=readme-ov-file" '
            )
            raise err

        VLMEvalKit_path = os.getcwd()
        os.chdir(root)
        warnings.warn('Please set `root` to RBdash code directory, \
            which is cloned from here: "https://github.com/RBDash-Team/RBDash?tab=readme-ov-file" ')
        try:
            model_name = get_model_name_from_path(model_path)
        except Exception as err:
            logging.critical(
                'Please follow the instructions of RBdash to put the ckpt file in the right place, '
                'which can be found at https://github.com/RBDash-Team/RBDash?tab=readme-ov-file#structure'
            )
            raise err

        download_model_path = snapshot_download(model_path)

        internvit_local_dir = './model_zoo/OpenGVLab/InternViT-6B-448px-V1-5'
        os.makedirs(internvit_local_dir, exist_ok=True)
        snapshot_download('OpenGVLab/InternViT-6B-448px-V1-5', local_dir=internvit_local_dir)

        convnext_local_dir = './model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup'
        os.makedirs(convnext_local_dir, exist_ok=True)
        snapshot_download('laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup', local_dir=convnext_local_dir)
        preprocessor_url = 'https://huggingface.co/openai/clip-vit-large-patch14-336/blob/main/preprocessor_config.json'
        download_file_path = osp.join(convnext_local_dir, 'preprocessor_config.json')
        if not osp.exists(download_file_path):
            print(f'download preprocessor to {download_file_path}')
            download_file(preprocessor_url, download_file_path)

        tokenizer, model, image_processor, image_processor_aux, context_len = load_pretrained_model(
            download_model_path, None, model_name, device_map="auto"
        )
        os.chdir(VLMEvalKit_path)
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_processor_aux = image_processor_aux
        self.conv_mode = conv_mode

        if tokenizer.unk_token is None:
            tokenizer.unk_token = '<|endoftext|>'
        tokenizer.pad_token = tokenizer.unk_token

        kwargs_default = dict(temperature=float(0.2), num_beams=1, top_p=None, max_new_tokens=128, use_cache=True)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

    def generate_inner(self, message, dataset=None):
        try:
            from rbdash.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                          DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
            from rbdash.conversation import conv_templates
            from rbdash.mm_utils import process_images, tokenizer_image_token
        except Exception as err:
            logging.critical(
                'Please first install RBdash and set the root path to use RBdash, '
                'which is cloned from here: "https://github.com/RBDash-Team/RBDash?tab=readme-ov-file" '
            )
            raise err

        prompt, image = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image).convert('RGB')

        if self.model.config.mm_use_im_start_end:
            prompt = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + '\n'
                + prompt
            )
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt

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
            self.image_processor_aux.crop_size['height'] = self.model.config.image_size_aux
            self.image_processor_aux.crop_size['width'] = self.model.config.image_size_aux
            self.image_processor_aux.size[
                'shortest_edge'
            ] = self.model.config.image_size_aux
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        image_grid = getattr(self.model.config, 'image_grid', 1)
        if hasattr(self.model.config, 'image_size_aux'):
            raw_shape = [
                self.image_processor.image_size_raw['height'] * image_grid,
                self.image_processor.image_size_raw['width'] * image_grid
            ]
            if self.image_processor is not self.image_processor_aux:
                image_tensor_aux = process_images([image], self.image_processor_aux, self.model.config)[
                    0
                ]
            else:
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
                max_new_tokens=512,
                images=images,
                images_aux=images_aux,
                do_sample=True if self.kwargs['temperature'] > 0 else False,
                temperature=self.kwargs['temperature'],
                top_p=self.kwargs['top_p'],
                num_beams=self.kwargs['num_beams']
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if 'mme' in dataset.lower():
            return True
        elif 'hallusionbench' in dataset.lower():
            return True
        elif 'mmmu' in dataset.lower():
            return True
        elif 'mmbench' in dataset.lower():
            return True
        return False

    def build_mme(self, line):
        question = line['question']
        prompt = question + 'Answer the question using a single word or phrase.'
        return prompt

    def build_hallusionbench(self, line):
        question = line['question']
        prompt = question + '\nAnswer the question using a single word or phrase.'
        return prompt

    def build_mmbench(self, line):
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += "Answer with the option's letter from the given choices directly."
        else:
            prompt += 'Answer the question using a single word or phrase.'
        return prompt

    def build_mmmu(self, line):
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'({key}) {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += "Answer with the option's letter from the given choices directly."
        else:
            prompt += 'Answer the question using a single word or phrase.'
        return prompt

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        if 'mme' in dataset.lower():
            prompt = self.build_mme(line)
        elif 'hallusionbench' in dataset.lower():
            prompt = self.build_hallusionbench(line)
        elif 'mmmu' in dataset.lower():
            prompt = self.build_mmmu(line)
        elif 'mmbench' in dataset.lower():
            prompt = self.build_mmbench(line)

        ret = [dict(type='text', value=prompt)]
        ret.extend([dict(type='image', value=s) for s in tgt_path])
        return ret
