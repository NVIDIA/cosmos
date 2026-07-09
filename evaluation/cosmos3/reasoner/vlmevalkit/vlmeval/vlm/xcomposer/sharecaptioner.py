import string

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string
from ..base import BaseModel


class ShareCaptioner(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='Lin-Chen/ShareCaptioner', **kwargs):
        assert model_path is not None
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map='cuda', trust_remote_code=True).eval()
        self.model.tokenizer = tokenizer
        self.model.cuda()
        self.model.half()

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            question = line['question']
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            if hint is not None:
                question = hint + '\n' + question

            option_candidate = string.ascii_uppercase
            options = {
                cand: line[cand]
                for cand in option_candidate
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                question += f'\n{key}. {item}'
            prompt = question

            if not cn_string(prompt):
                prompt = prompt + '\n' + "Answer with the option's letter from the given choices directly."
            else:
                prompt = prompt + '\n' + '请直接回答选项字母。'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        seg1 = '<|User|>:'
        seg2 = f'{prompt}{self.model.eoh}\n<|Bot|>:'
        self.seg_emb1 = self.model.encode_text(seg1, add_special_tokens=True)
        self.seg_emb2 = self.model.encode_text(seg2, add_special_tokens=False)

        image = Image.open(image_path).convert('RGB')
        image = self.model.vis_processor(image).unsqueeze(0)
        image = image.to(self.model.device)
        tmp_bs = image.shape[0]
        tmp_seg_emb1 = self.seg_emb1.repeat(tmp_bs, 1, 1)
        tmp_seg_emb2 = self.seg_emb2.repeat(tmp_bs, 1, 1)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                image = self.model.encode_img(image)
                input_emb = torch.cat(
                    [tmp_seg_emb1, image, tmp_seg_emb2], dim=1)
                out_embeds = self.model.internlm_model.generate(
                    inputs_embeds=input_emb,
                    max_length=500,
                    num_beams=3,
                    min_length=1,
                    do_sample=True,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1.,
                    eos_token_id=self.model.tokenizer.eos_token_id,
                    num_return_sequences=1)

        for j, out in enumerate(out_embeds):
            out[out == -1] = 2
            response = self.model.decode_text([out])
        return response
