import logging
import os
import os.path as osp
import string

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string, get_cache_path
from ..base import BaseModel


class LLaVA_XTuner(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self,
                 llava_path,
                 llm_path=None,
                 visual_encoder_path='openai/clip-vit-large-patch14-336',
                 visual_select_layer=-2,
                 prompt_template=None,
                 stop_words=[],
                 torch_dtype=torch.float16):
        try:
            from peft import PeftModel
            from xtuner.utils import PROMPT_TEMPLATE, StopWordStoppingCriteria
        except Exception as err:
            logging.critical(
                'Please install xtuner with `pip install -U xtuner` before '
                'using LLaVA_XTuner')
            raise err

        from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,  # noqa
                                  StoppingCriteriaList)

        if not osp.isdir(llava_path):
            cache_path = get_cache_path(llava_path)
            if cache_path is not None:
                llava_path = cache_path
            else:
                llava_path = snapshot_download(repo_id=llava_path)
        assert osp.exists(llava_path) and osp.isdir(llava_path)

        # build visual_encoder
        if 'llm' in os.listdir(llava_path):
            assert llm_path is None, (
                "Please don't specify the `llm_path` since passed "
                '`llava_path` contains a LLM!')
            llm_path = osp.join(llava_path, 'llm')
        else:
            assert llm_path is not None, 'Please specify the `llm_path`!'

        llm = AutoModelForCausalLM.from_pretrained(llm_path,
                                                   trust_remote_code=True,
                                                   torch_dtype=torch_dtype,
                                                   device_map='cpu')
        tokenizer = AutoTokenizer.from_pretrained(llm_path,
                                                  trust_remote_code=True,
                                                  encode_special_tokens=True)
        print(f'Load LLM from {llm_path}')

        # build visual_encoder
        if 'visual_encoder' in os.listdir(llava_path):
            assert visual_encoder_path is None, (
                "Please don't specify the `visual_encoder_path` since passed "
                '`llava_path` contains a visual encoder!')
            visual_encoder_path = osp.join(llava_path, 'visual_encoder')
        else:
            assert visual_encoder_path is not None, (
                'Please specify the `visual_encoder_path`!')

        from transformers import CLIPImageProcessor, CLIPVisionModel
        visual_encoder = CLIPVisionModel.from_pretrained(
            visual_encoder_path, torch_dtype=torch_dtype, device_map='cpu')
        image_processor = CLIPImageProcessor.from_pretrained(
            visual_encoder_path)
        print(f'Load visual_encoder from {visual_encoder_path}')

        # load adapter
        if 'llm_adapter' in os.listdir(llava_path):
            adapter_path = osp.join(llava_path, 'llm_adapter')
            llm = PeftModel.from_pretrained(llm,
                                            adapter_path,
                                            trust_remote_code=True,
                                            device_map='cpu')
            print(f'Load LLM adapter from {llava_path}')
        if 'visual_encoder_adapter' in os.listdir(llava_path):
            adapter_path = osp.join(llava_path, 'visual_encoder_adapter')
            visual_encoder = PeftModel.from_pretrained(visual_encoder,
                                                       adapter_path,
                                                       trust_remote_code=True,
                                                       device_map='cpu')
            print(f'Load visual_encoder adapter from {llava_path}')

        # build projector
        projector_path = osp.join(llava_path, 'projector')
        projector = AutoModel.from_pretrained(projector_path,
                                              trust_remote_code=True,
                                              torch_dtype=torch_dtype,
                                              device_map='cpu')
        print(f'Load projector from {llava_path}')

        llm.eval()
        visual_encoder.eval()
        projector.eval()

        self.llm = llm.cuda()
        self.tokenizer = tokenizer
        self.visual_encoder = visual_encoder.cuda()
        self.image_processor = image_processor
        self.projector = projector.cuda()
        self.visual_select_layer = visual_select_layer
        if prompt_template is not None:
            # modified prompt template
            if prompt_template == 'llama3_chat':
                self.prompt_template = dict(
                    SYSTEM=('<|start_header_id|>system<|end_header_id|>\n\n'
                            '{system}<|eot_id|>'),
                    INSTRUCTION=(
                        '<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>'
                        '<|start_header_id|>assistant<|end_header_id|>\n\n'),
                    SUFFIX='<|eot_id|>',
                    SUFFIX_AS_EOS=True,
                    STOP_WORDS=['<|eot_id|>'])
            else:
                self.prompt_template = PROMPT_TEMPLATE[prompt_template]
            stop_words += self.prompt_template.get('STOP_WORDS', [])
        else:
            self.prompt_template = None

        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, word))

    def build_gen_config(self, dataset):
        from transformers import GenerationConfig
        gen_kwargs = dict(max_new_tokens=512,
                          do_sample=True,
                          temperature=1,
                          num_beams=5,
                          eos_token_id=self.tokenizer.eos_token_id,
                          pad_token_id=self.tokenizer.pad_token_id
                          if self.tokenizer.pad_token_id is not None else
                          self.tokenizer.eos_token_id)
        # For single word generation
        if (dataset is not None
                and DATASET_TYPE(dataset) in ['MCQ', 'Y/N']):
            gen_kwargs.update(
                dict(max_new_tokens=5, do_sample=False, num_beams=1))
        return GenerationConfig(**gen_kwargs)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line
                                and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'

        if not cn_string(question):
            prompt = question + '\n' + ("Answer with the option's letter "
                                        'from the given choices directly.')
        else:
            prompt = question + '\n' + '请直接回答选项字母。'

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        from xtuner.dataset.utils import expand2square
        from xtuner.model.utils import prepare_inputs_labels_for_multimodal
        from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        prompt = prompt.replace('<image>', '')
        image = Image.open(image_path).convert('RGB')
        image = expand2square(
            image,
            tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        image = image.cuda().unsqueeze(0)
        visual_outputs = self.visual_encoder(image, output_hidden_states=True)
        pixel_values = self.projector(
            visual_outputs.hidden_states[self.visual_select_layer][:, 1:])

        inputs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        if self.prompt_template:
            inputs = self.prompt_template['INSTRUCTION'].format(input=inputs)

        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer(chunk)
            else:
                cur_encode = self.tokenizer(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode['input_ids'])
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=self.llm, input_ids=ids, pixel_values=pixel_values)

        gen_config = self.build_gen_config(dataset)
        generate_output = self.llm.generate(
            **mm_inputs,
            generation_config=gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria)
        predict = self.tokenizer.decode(generate_output[0],
                                        skip_special_tokens=True).strip()
        return predict
