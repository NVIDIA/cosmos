import copy
import logging

import torch
from PIL import Image

from .base import BaseModel


class SliME(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = '<image>'
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path='yifanzhang114/SliME-Llama3-8B', **kwargs):
        assert model_path is not None
        try:
            from llava.conversation import conv_templates
            from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
            from llava.model.builder import load_pretrained_model
        except Exception as err:
            logging.critical('Please install requirements on https://github.com/yfzhang114/SliME before using SliME')
            raise err

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name, device_map=None)
        model.cuda().eval()
        model.tie_weights()

        if 'llama3' in model_path.lower():
            conv_mode = 'llama3'
        elif 'vicuna' in model_path.lower():
            conv_mode = 'v1'
        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token

    def generate_inner(self, message, dataset=None):
        content, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            else:
                images.append(Image.open(msg['value']).convert('RGB'))
                content += (self.DEFAULT_IMAGE_TOKEN + '\n')

        preprocess = self.image_processor.preprocess
        image_tokenizer = self.tokenizer_image_token
        image_tensor = [
            preprocess(f, return_tensors='pt')['pixel_values'][0].half().cuda() for f in images
        ]
        image_tensor = torch.stack(image_tensor)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.messages = list(conv.messages)
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = image_tokenizer(prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda()

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs
