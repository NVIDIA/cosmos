import logging
import string
import warnings

import pandas as pd
import torch
from PIL import Image

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string
from .base import BaseModel


class Mantis(BaseModel):
    """
    Mantis Model
    This implementation is adpated from the Llava model from llava.py and the Idefics model from idefics.py
    """
    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = '<image>'
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path='TIGER-Lab/Mantis-8B-siglip-llama3', **kwargs):
        assert model_path is not None
        try:
            from mantis.models.conversation import conv_mllava_v1 as default_conv
            from mantis.models.conversation import conv_templates
            from mantis.models.mfuyu import MFuyuForCausalLM, MFuyuProcessor
            from mantis.models.mllava import LlavaForConditionalGeneration, MLlavaProcessor
        except Exception as e:
            logging.critical(
                "Mantis is not installed. Please install Mantis to use this model.Please use 'pip install "
                "git+https://github.com/TIGER-AI-Lab/Mantis.git' to install"
            )
            raise e

        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except Exception as e:
            logging.critical(f'{type(e)}: {e}')
            logging.critical("Upgrade transformers to use Mantis's idefics model.\nError: %s" % e)

        # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2".
        # Seems FA2 is not effective during inference:
        # https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        # if is_flash_attn_2_available:
        #     best_fit_attn_implementation = "flash_attention_2"
        # flash_attn has a bug that says: ERROR Error query and key must have the same dtype in generating

        try:
            import flash_attn  # noqa: F401
            best_fit_attn_implementation = 'flash_attention_2'
        except ImportError:
            best_fit_attn_implementation = 'eager'
        self.model_path = model_path
        attn_implementation = best_fit_attn_implementation
        self._is_idefics = 'idefics' in model_path.lower()
        # Here load the "non-idefics" Mantis model.
        if not self._is_idefics:
            if 'fuyu' in model_path.lower():
                self.processor = MFuyuProcessor.from_pretrained(self.model_path)
                model = MFuyuForCausalLM.from_pretrained(
                    self.model_path,
                    device_map='cuda',
                    attn_implementation=attn_implementation,
                    torch_dtype=torch.float16
                )
            else:
                self.processor = MLlavaProcessor.from_pretrained(self.model_path)
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path,
                    device_map='cuda',
                    attn_implementation=attn_implementation,
                    torch_dtype=torch.float16
                )
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                device_map='cuda',
                torch_dtype=torch.float16
            )

        model = model.eval()
        self.model = model.cuda()
        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=1024, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

        self.tokenizer = self.processor.tokenizer
        self.default_conv = default_conv
        self.conv_templates = conv_templates

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
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'
        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def output_process(self, answer):
        if '<s>' in answer:
            answer = answer.replace('<s>', '').strip()
        if '[/INST]' in answer:
            answer = answer.split('[/INST]')[1].strip()
        elif 'ASSISTANT:' in answer:
            answer = answer.split('ASSISTANT:')[1].strip()
        elif 'assistant\n' in answer:
            answer = answer.split('assistant\n')[1].strip()
        elif '<|end_header_id|>\n\n' in answer:
            answer = answer.split('<|end_header_id|>\n\n')[2].strip()

        if '</s>' in answer:
            answer = answer.split('</s>')[0].strip()
        elif '<|im_end|>' in answer:
            answer = answer.split('<|im_end|>')[0].strip()
        elif '<|eot_id|>' in answer:
            answer = answer.split('<|eot_id|>')[0].strip()
        elif '<end_of_utterance>' in answer:
            answer = answer.split('<end_of_utterance>')[0].strip()
        elif '|ENDOFTEXT|' in answer:
            answer = answer.split('|ENDOFTEXT|')[0].strip()
        return answer

    def generate_inner(self, message, dataset=None):
        content, images = '', []
        ide_content, question = [], ''
        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
                question += msg['value']
            else:
                images.append(Image.open(msg['value']).convert('RGB'))
                content += (self.DEFAULT_IMAGE_TOKEN + '\n')
                ide_content.append({'type': 'image'})
        if self._is_idefics:
            # Follow the idefics implementation:
            ide_content.append({'type': 'text', 'text': question})
            prompt = [{'role': 'user', 'content': ide_content}]
            prompt = self.processor.apply_chat_template(prompt, add_generation_prompt=True)
        else:
            # Follow the Mantis code base to make sure they are consistent:
            # https://github.com/TIGER-AI-Lab/Mantis/blob/main/mantis/models/mllava/utils.py#L33
            # Users don't need to define chat template as it is done here
            if 'llama-3' in self.model.language_model.name_or_path.lower():
                conv = self.conv_templates['llama_3']
                terminators = [
                    self.processor.tokenizer.eos_token_id,
                    self.processor.tokenizer.convert_tokens_to_ids('<|eot_id|>')
                ]
            else:
                conv = self.default_conv
                terminators = [self.processor.tokenizer.eos_token_id]

            # Using EOT because end of *text* is more accurate for what we're doing than end of *sentence*
            if 'eos_token_id' not in self.kwargs:
                self.kwargs['eos_token_id'] = terminators

            conv = conv.copy()
            conv.append_message(conv.roles[0], content)
            conv.append_message(conv.roles[1], '')
            assert conv.messages[-1][0] == conv.roles[1] and conv.messages[-1][1] == '', 'Format check'
            prompt = conv.get_prompt()

        inputs = self.processor(prompt, images, return_tensors='pt', truncation=True)
        # FIXME: Fuyu model would return a list instead of a pytorch tensor. This weird behavior needs fixing.
        if 'image_patches' in inputs.keys():
            inputs['image_patches'] = inputs['image_patches'][0]
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        output = self.model.generate(**inputs, **self.kwargs)
        output = output[0]
        generated_ids = output[inputs['input_ids'].shape[-1]:]
        answer = self.processor.decode(generated_ids, skip_special_token=True)
        answer = self.output_process(answer)
        return answer
