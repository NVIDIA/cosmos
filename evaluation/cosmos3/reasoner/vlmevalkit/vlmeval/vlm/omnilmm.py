import string
import sys

import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer

from ..dataset import DATASET_TYPE
from ..smp.log import get_logger
from .base import BaseModel

logger = get_logger(__name__)


DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'


def init_omni_lmm(model_path):
    from omnilmm.model.omnilmm import OmniLMMForCausalLM
    from omnilmm.model.utils import build_transform
    from omnilmm.utils import disable_torch_init

    torch.backends.cuda.matmul.allow_tf32 = True
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=2048)

    model = OmniLMMForCausalLM.from_pretrained(
        model_path, tune_clip=True, torch_dtype=torch.bfloat16, device_map='cpu'
    )
    model = model.to(device='cuda', dtype=torch.bfloat16)

    image_processor = build_transform(
        is_train=False, input_size=model.model.config.image_size, std_mode='OPENAI_CLIP'
    )

    mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    assert mm_use_im_start_end

    tokenizer.add_tokens(
        [DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
        special_tokens=True,
    )

    vision_config = model.model.vision_config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    vision_config.im_start_token, vision_config.im_end_token = (
        tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    )
    image_token_len = model.model.config.num_query

    return model, image_processor, image_token_len, tokenizer


def expand_question_into_multimodal(
    question_text, image_token_len, im_st_token, im_ed_token, im_patch_token
):
    if '<image>' in question_text[0]['content']:
        question_text[0]['content'] = question_text[0]['content'].replace(
            '<image>', im_st_token + im_patch_token * image_token_len + im_ed_token
        )
    else:
        question_text[0]['content'] = (
            im_st_token
            + im_patch_token * image_token_len
            + im_ed_token
            + '\n'
            + question_text[0]['content']
        )
    return question_text


def wrap_question_for_omni_lmm(question, image_token_len, tokenizer):
    from omnilmm.train.train_utils import omni_preprocess

    question = expand_question_into_multimodal(
        question,
        image_token_len,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IMAGE_PATCH_TOKEN,
    )

    conversation = question
    data_dict = omni_preprocess(
        sources=[conversation], tokenizer=tokenizer, generation=True
    )

    data_dict = dict(input_ids=data_dict['input_ids'][0], labels=data_dict['labels'][0])
    return data_dict


class OmniLMM12B(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_path, root, **kwargs) -> None:
        sys.path.append(root)
        model, img_processor, image_token_len, tokenizer = init_omni_lmm(model_path)
        self.model = model
        self.image_token_len = image_token_len
        self.image_transform = img_processor
        self.tokenizer = tokenizer
        self.model.eval()
        default_kwargs = dict(
            max_new_tokens=512,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            repetition_penalty=1.1,
        )
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            logger.error('Image Decode Error')
            return 'Image Decode Error'

        msgs = [dict(role='user', content=prompt)]
        input_ids = wrap_question_for_omni_lmm(
            msgs, self.image_token_len, self.tokenizer
        )['input_ids']
        input_ids = torch.as_tensor(input_ids)
        image = self.image_transform(image)

        with torch.inference_mode():
            output = self.model.generate_vllm(
                input_ids=input_ids.unsqueeze(0).cuda(),
                images=image.unsqueeze(0).half().cuda(),
                **self.kwargs,
            )

            response = self.tokenizer.decode(
                output.sequences[0], skip_special_tokens=True
            )
            response = response.strip()
            return response

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

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
        prompt += f'{question}\n'
        if len(options):
            prompt += options_prompt
            prompt = (
                """
Study the image carefully and pick the option associated with the correct answer.
Focus solely on selecting the option and avoid including any other content.\n
"""
                + prompt
            )

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message
