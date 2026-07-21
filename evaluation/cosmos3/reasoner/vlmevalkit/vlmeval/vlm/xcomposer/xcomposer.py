import copy as cp
import string

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from vlmeval.dataset import DATASET_TYPE
from ..base import BaseModel


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


class XComposer(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='internlm/internlm-xcomposer-vl-7b', **kwargs):
        assert model_path is not None
        self.model_path = model_path

        model = AutoModel.from_pretrained(self.model_path, device_map='cpu', trust_remote_code=True).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model.tokenizer = tokenizer
        self.model = model
        self.device = self.model.internlm_model.model.embed_tokens.weight.device
        self.eoh = '<TOKENS_UNUSED_0>'
        self.eoa = '<TOKENS_UNUSED_1>'
        stop_words_ids = [
            torch.tensor([103027]).to(self.device),  # end of human
            torch.tensor([103028]).to(self.device),  # end of bot
        ]
        default_kwargs = {
            'max_new_tokens': 512, 'num_beams': 5, 'do_sample': False,
            'min_length': 1, 'repetition_penalty': 1.5, 'length_penalty': 1.0
        }
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def generate_inner(self, message, dataset=None):
        if len(message) == 2:
            if message[0]['type'] == 'text' and message[1]['type'] == 'image':
                message = [message[1], message[0]]
        kwargs = cp.deepcopy(self.kwargs)
        if dataset is not None:
            if DATASET_TYPE(dataset) == 'MCQ':
                kwargs['max_new_tokens'] = 5
                kwargs['num_beams'] = 5

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt_embs = self.message_to_prompt_embs(message, dataset)
                outputs = self.model.internlm_model.generate(
                    inputs_embeds=prompt_embs,
                    stopping_criteria=self.stopping_criteria,
                    **kwargs
                )

        output_token = outputs[0]
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.model.tokenizer.decode(output_token, add_special_tokens=False)

        output_text = output_text.split(self.model.eoa)[0]
        output_text = output_text.split('<|Bot|>')[-1].strip()
        return output_text

    def message_to_prompt_embs(self, message, dataset=None):
        assert isinstance(message, list)
        img_embeds = []
        prompt_full = '<|User|>: '
        for msg in message:
            if msg['type'] == 'text':
                prompt_full += msg['value']
            elif msg['type'] == 'image':
                image = Image.open(msg['value']).convert('RGB')
                image = self.model.vis_processor(image).unsqueeze(0).to(self.device)
                img_embeds.append(self.model.encode_img(image))
                prompt_full += '<ImageHere>'

        prompt_full += self.model.eoh + ' <|Bot|>: '
        if dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt_full += 'Answer: The answer is '
        elif dataset is not None and DATASET_TYPE(dataset) in ['VQA', 'QA', 'Y/N']:
            prompt_full += 'Answer: '

        prompt_segs = prompt_full.split('<ImageHere>')
        assert len(prompt_segs) == len(img_embeds) + 1

        prompt_seg_tokens = [
            self.model.tokenizer(seg, return_tensors='pt', add_special_tokens=(i == 0)).to(self.device).input_ids.long()
            for i, seg in enumerate(prompt_segs)
        ]
        prompt_seg_embs = [self.model.internlm_model.model.embed_tokens(seg) for seg in prompt_seg_tokens]
        all_embeddings = []
        for i in range(len(img_embeds)):
            all_embeddings.extend([prompt_seg_embs[i], img_embeds[i]])
        all_embeddings.append(prompt_seg_embs[-1])
        prompt_embs = torch.cat(all_embeddings, dim=1)
        return prompt_embs

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
        options_prompt = ''
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        context = 'N/A' if hint is None else hint
        mid_prompt = 'Context: ' + context + '\nQuestion: ' + question
        if len(options_prompt):
            mid_prompt += '\nOptions: ' + options_prompt

        if len(options):
            txt_prompt = 'Please answer this question by choosing the correct choice.'
        else:
            txt_prompt = 'Please answer this question directly. '
        prompt = txt_prompt + mid_prompt
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message
