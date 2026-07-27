import logging

import torch
from PIL import Image

from .base import BaseModel

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'


class Cambrian(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_path='nyu-visionx/cambrian-8b', **kwargs):
        assert model_path is not None
        try:
            from cambrian.conversation import SeparatorStyle, conv_templates  # noqa: F401
            from cambrian.mm_utils import (get_model_name_from_path, process_images,
                                           tokenizer_image_token)
            from cambrian.model.builder import load_pretrained_model
        except Exception as e:
            logging.critical('Please install cambrian from https://github.com/cambrian-mllm/cambrian.')
            raise e

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map="cuda"
        )

        if '8b' in model_path:
            self.conv_mode = 'llama_3'
        elif '13b' in model_path:
            self.conv_mode = 'vicuna_v1'
        else:
            self.conv_mode = 'chatml_direct'

        self.model_config = model.config
        self.conv_templates = conv_templates
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = process_images

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model = model

    def process(self, image, question):
        if self.model_config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        image_size = [image.size]
        image_tensor = self.process_images([image], self.image_processor, self.model_config)
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda()
        return input_ids, image_tensor, image_size, prompt

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')
        input_ids, image_tensor, image_sizes, prompt = self.process(image, prompt)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                num_beams=1,
                max_new_tokens=2048,
                use_cache=True
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
