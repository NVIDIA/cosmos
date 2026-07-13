import copy
import logging
import string
import warnings
from typing import Dict, Optional, Union

import pandas as pd
import torch
import transformers
from PIL import Image

from .base import BaseModel


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",  # noqa E501
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")

    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    input_ids = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id = []

        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except Exception:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            if role == 'assistant':
                input_id += tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n")
            else:
                conv = [{"role": role, "content": content}]
                encode_id = tokenizer.apply_chat_template(conv)[1:]
                input_id += encode_id

        for idx, encode_id in enumerate(input_id):
            if encode_id == image_token_index:
                input_id[idx] = -200
        input_ids.append(input_id)
    input_ids = torch.tensor(input_ids, dtype=torch.long)

    return input_ids


class InsightV(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(
        self,
        pretrained_reason: str = "THUdyh/Insight-V-Reason-LLaMA3",
        pretrained_summary: str = "THUdyh/Insight-V-Summary-LLaMA3",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_flash_attention_2=True,
        device_map="auto",
        conv_template="llava_llama_3",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        **kwargs,
    ) -> None:
        super().__init__()
        try:
            from llava.mm_utils import get_model_name_from_path
            from llava.model.builder import load_pretrained_model
        except Exception as err:
            logging.critical('Please install Insight-V before using Insight-V')
            logging.critical('Please install Insight-V from https://github.com/dongyh20/Insight-V')
            logging.critical('Please install VLMEvalKit after installing Insight-V')
            raise err
        # Do not use kwargs for now
        self._tokenizer, self.reason_model, self._image_processor, self._max_length = load_pretrained_model(
            pretrained_reason, None, get_model_name_from_path(pretrained_reason),
            device_map=device_map, use_flash_attention_2=use_flash_attention_2,
        )
        _, self.summary_model, _, _ = load_pretrained_model(
            pretrained_summary, None, get_model_name_from_path(pretrained_summary),
            device_map=device_map, use_flash_attention_2=use_flash_attention_2,
        )
        self._config = self.reason_model.config
        self.reason_model.eval()
        self.summary_model.eval()

        self.device = device
        self.dtype = dtype
        self.reason_model = self.reason_model.to(device)
        self.summary_model = self.summary_model.to(device)

        self.conv_mode = "llava_llama_3"

        kwargs_default = dict(
            do_sample=True, reason_temperature=0.7, summary_temperature=0.2,
            reason_max_new_tokens=16384, summary_max_new_tokens=512, top_p=0.95, num_beams=1, use_cache=True,
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        return True

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

        reasoning_prompt = prompt + "\n\nPerform step-by-step reasoning of the problem. Only provide the reasoning process."  # noqa E501

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=reasoning_prompt))
        return message

    def generate_inner(self, message, dataset=None):
        try:
            from llava.constants import DEFAULT_IM_END_TOKEN  # noqa: F401
            from llava.constants import IMAGE_TOKEN_INDEX  # noqa: F401
            from llava.constants import (DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN,  # noqa: F401
                                         IGNORE_INDEX)
            from llava.conversation import SeparatorStyle, conv_templates  # noqa: F401
            from llava.mm_utils import process_images, tokenizer_image_token  # noqa: F401
        except Exception as err:
            logging.critical('Please install Insight-V before using Insight-V')
            logging.critical('Please install Insight-V from https://github.com/dongyh20/Insight-V')
            logging.critical('Please install VLMEvalKit after installing Insight-V')
            raise err
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], 'PLACEHOLDER')
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        content, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
                reasoning_question = msg['value']
            elif msg['type'] == 'image':
                if self.reason_model.config.mm_use_im_start_end:
                    content += DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
                else:
                    content += DEFAULT_IMAGE_TOKEN + '\n'
                images.append(msg['value'])

        images = [Image.open(s).convert('RGB') for s in images]
        image_sizes = [img.size for img in images]
        image_tensor = process_images(images, self._image_processor, self._config)
        if type(image_tensor) is list:
            image_tensor = [_image.to(dtype=torch.bfloat16, device='cuda') for _image in image_tensor]
        else:
            image_tensor = image_tensor.to(dtype=torch.bfloat16, device='cuda')

        prompt = prompt.replace('PLACEHOLDER', content)

        input_ids = preprocess_llama3([[{'from': 'human', 'value': prompt},{'from': 'gpt','value': None}]], self._tokenizer, has_image=True).cuda()  # noqa E501
        pad_token_ids = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else self._tokenizer.eos_token_id  # noqa E501
        attention_masks = input_ids.ne(pad_token_ids).to('cuda')

        # reasoning content
        reasoning_cont = self.reason_model.generate(
            input_ids,
            attention_mask=attention_masks,
            pad_token_id=pad_token_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True if self.kwargs["reason_temperature"] > 0 else False,
            temperature=self.kwargs["reason_temperature"],
            top_p=self.kwargs["top_p"],
            num_beams=self.kwargs["num_beams"],
            max_new_tokens=self.kwargs["reason_max_new_tokens"],
            use_cache=True,
        )
        reasoning_outputs = self._tokenizer.batch_decode(reasoning_cont, skip_special_tokens=False)
        print("Reasoning Output:", reasoning_outputs)

        reason_chain = reasoning_outputs[0].split("<thoughts>")[-1].split("</thoughts>")[0]
        reason_chain = "<thoughts>\n" + reason_chain + "</thoughts>\n"

        original_question = reasoning_question.replace("\n\nPerform step-by-step reasoning of the problem. Only provide the reasoning process.","").strip()  # noqa E501
        summary_question = "I will give you a reasoning process of the question. You should determine whether the reasoning process is correct about the question. If it is correct, please summarize the answer based on the reasoning process. If it is incorrect, answer the question and ignore the reasoning process. You shold directly give the summarization or answer as you are directly answer the QUESTION  without saying your judgement about the reasoning process.\n\nQUESTION: " + original_question + f"\n\nREASON PROCESS: {reason_chain}"  # noqa E501

        input_ids_summary = preprocess_llama3([[{'from': 'human', 'value': summary_question},{'from': 'gpt','value': None}]], self._tokenizer, has_image=True).cuda()  # noqa E501
        attention_masks_summary = input_ids_summary.ne(pad_token_ids).to('cuda')

        # summary content
        summary_cont = self.summary_model.generate(
            input_ids_summary,
            attention_mask=attention_masks_summary,
            pad_token_id=pad_token_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True if self.kwargs["summary_temperature"] > 0 else False,
            temperature=self.kwargs["summary_temperature"],
            top_p=self.kwargs["top_p"],
            num_beams=self.kwargs["num_beams"],
            max_new_tokens=self.kwargs["summary_max_new_tokens"],
            use_cache=True,
        )

        summary_outputs = self._tokenizer.batch_decode(summary_cont, skip_special_tokens=True)[0].strip()
        print("Summary Output:", summary_outputs)
        return summary_outputs
