import os
import string
import sys
import warnings

import pandas as pd
import torch
from PIL import Image

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string, listinstr
from .base import BaseModel


class VITAQwen2(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = '<image>'
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path='VITA/vita', root=None, **kwargs):
        if root is None:
            root = os.environ.get('VITA_ROOT', '')
        sys.path.append(root)
        assert model_path is not None
        try:
            from vita.conversation import conv_templates
            from vita.model.builder import load_pretrained_model
            from vita.util.mm_utils import get_model_name_from_path, tokenizer_image_token
        except Exception:
            warnings.warn('Please install vita first.')

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name, model_type='qwen2p5_instruct', device_map='cuda'
        )
        model = model.eval()
        # model.tie_weights()

        audio_encoder = model.get_audio_encoder()
        # audio_encoder.to(device="cuda", dtype=torch.float16)
        audio_encoder.to(dtype=torch.float16)
        # audio_processor not used, consider removing
        _ = audio_encoder.audio_processor

        conv_mode = 'qwen2p5_instruct'
        self.stop_str = '<|im_end|>'
        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.image_size = 448

    def use_custom_prompt(self, dataset):
        return True

    def build_multi_choice_prompt(self, line, dataset=None):
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
            prompt += '\n请直接回答选项字母。' if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        return prompt

    def build_mme_rw_prompt(self, line, dataset_name):
        SYS = {
            'MME-RealWorld': (
                'Select the best answer to the above multiple-choice question based on the image. '
                'Respond with only the letter (A, B, C, D, or E) of the correct option. \n'
                'The best answer is:'
            ),
            'MME-RealWorld-CN': (
                '根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n'
                '最佳答案为：'
            ),
        }
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        choice_prompt = line['multi-choice options'] + '\n'
        question += ' ' + choice_prompt + SYS[dataset_name]

        prompt = question

        prompt += '\n请直接回答选项字母。' if cn_string(
            prompt) else "\nAnswer with the option's letter from the given choices directly."

        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and listinstr(['MME'], dataset) and dataset != 'MME-RealWorld':
            question = line['question']
            prompt = question + ' Answer the question using a single word or phrase.'
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question']
            prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ' and dataset != 'MME-RealWorld':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset == 'MME-RealWorld':
            prompt = self.build_mme_rw_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            if 'MathVista' in dataset:
                prompt = line['question']
                # prompt = 'According to the question shown in the image, please first conduct reasoning...'
            elif listinstr(['LLaVABench'], dataset):
                question = line['question']
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['MMVet'], dataset):
                prompt = line['question']
            else:
                question = line['question']
                prompt = question + '\nAnswer the question using a single word or phrase.'
        elif DATASET_TYPE(dataset) == 'CreationVQA':
            question = line['question']
            prompt = question + '\nAnswer this question in detail.'
        else:
            prompt = line['question']

        message = [dict(type='image', value=s) for s in tgt_path]
        message.extend([dict(type='text', value=prompt)])
        return message

    def set_max_num(self, dataset):
        if dataset is not None and listinstr(['ChartQA_TEST', 'MMMU_DEV_VAL'], dataset):
            self.max_num = 12
        elif dataset is not None and listinstr(['DocVQA_VAL', 'DocVQA_TEST'], dataset):
            self.max_num = 18
        elif dataset is not None and listinstr(['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench'], dataset):
            self.max_num = 24
        else:
            self.max_num = 6

    def generate_inner(self, message, dataset=None):
        from vita.util.data_utils_video_audio_patch import dynamic_preprocess
        from vita.util.mm_utils import KeywordsStoppingCriteria
        self.set_max_num(dataset)
        content, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            else:
                # 这里分patch，同时计算patch数量
                image = Image.open(msg['value']).convert('RGB')
                image, p_num = dynamic_preprocess(
                    image, min_num=1, max_num=self.max_num, image_size=self.image_size, use_thumbnail=True
                )
                assert len(p_num) == 1
                images += image
                content += (self.DEFAULT_IMAGE_TOKEN * p_num[0] + '\n')

        preprocess = self.image_processor.preprocess
        image_tokenizer = self.tokenizer_image_token
        image_tensor = [
            preprocess(f, return_tensors='pt')['pixel_values'][0].half().cuda() for f in images
        ]
        image_tensor = torch.stack(image_tensor)

        conv = self.conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        if self.DEFAULT_IMAGE_TOKEN in content:
            modality = 'image'
        else:
            modality = 'lang'
        prompt_question = conv.get_prompt(modality)
        print(prompt_question)

        input_ids = image_tokenizer(prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda()

        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        audio = torch.zeros(400, 80)
        audio_length = audio.shape[0]
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        audios = dict()
        audios['audios'] = audio.half().cuda()
        audios['lengths'] = audio_length.half().cuda()
        audio_for_llm_lens = 60
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
        audios["lengths_for_llm"] = audio_for_llm_lens.cuda()

        sf_masks = torch.tensor([0] * len(image_tensor)).cuda()
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            audios=audios,
            sf_masks=sf_masks,
            do_sample=False,
            temperature=0.01,
            max_new_tokens=2048,
            stopping_criteria=[stopping_criteria],
            shared_v_pid_stride=None,  # 2 #16 #8 #4 #1 #None,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        if '☞' in text_outputs or '☜' in text_outputs or '☟' in text_outputs:
            return text_outputs[1:]
        else:
            return text_outputs


class VITA(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = '<image>'
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path='VITA/vita', root=None, **kwargs):
        sys.path.append(root)
        assert model_path is not None
        try:
            from vita.conversation import conv_templates
            from vita.model.builder import load_pretrained_model
            from vita.util.mm_utils import get_model_name_from_path, tokenizer_image_token
        except Exception:
            warnings.warn('Please install vita first.')

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name, model_type='mixtral-8x7b', device_map="auto"
        )

        audio_encoder = model.get_audio_encoder()
        audio_encoder.to(dtype=torch.float16)
        # audio_processor = audio_encoder.audio_processor

        conv_mode = 'mixtral_two'
        self.stop_str = '</s>'
        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.image_size = 448

    def use_custom_prompt(self, dataset):
        return True

    def build_multi_choice_prompt(self, line, dataset=None):
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
            prompt += '\n请直接回答选项字母。' if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        return prompt

    def build_mme_rw_prompt(self, line, dataset_name):
        SYS = {
            'MME-RealWorld': (
                'Select the best answer to the above multiple-choice question based on the image. '
                'Respond with only the letter (A, B, C, D, or E) of the correct option. \n'
                'The best answer is:'
            ),
            'MME-RealWorld-CN': (
                '根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n'
                '最佳答案为：'
            ),
        }
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        choice_prompt = line['multi-choice options'] + '\n'
        question += ' ' + choice_prompt + SYS[dataset_name]

        prompt = question

        prompt += '\n请直接回答选项字母。' if cn_string(
            prompt) else "\nAnswer with the option's letter from the given choices directly."

        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and listinstr(['MME'], dataset) and dataset != 'MME-RealWorld':
            question = line['question']
            prompt = question + ' Answer the question using a single word or phrase.'
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question']
            prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ' and dataset != 'MME-RealWorld':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset == 'MME-RealWorld':
            prompt = self.build_mme_rw_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            if 'MathVista' in dataset:
                prompt = line['question']
            elif listinstr(['LLaVABench'], dataset):
                question = line['question']
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['MMVet'], dataset):
                prompt = line['question']
            else:
                question = line['question']
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']
        message = [dict(type='image', value=s) for s in tgt_path]
        message.extend([dict(type='text', value=prompt)])
        return message

    def set_max_num(self, dataset):
        if dataset is not None and listinstr(['ChartQA_TEST', 'MMMU_DEV_VAL'], dataset):
            self.max_num = 12
        elif dataset is not None and listinstr(['DocVQA_VAL', 'DocVQA_TEST'], dataset):
            self.max_num = 18
        elif dataset is not None and listinstr(['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench'], dataset):
            self.max_num = 24
        else:
            self.max_num = 6

    def generate_inner(self, message, dataset=None):
        from vita.util.data_utils_video_audio_patch import dynamic_preprocess
        from vita.util.mm_utils import KeywordsStoppingCriteria
        self.set_max_num(dataset)
        content, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            else:
                # 这里分patch，同时计算patch数量
                image = Image.open(msg['value']).convert('RGB')
                image, p_num = dynamic_preprocess(
                    image, min_num=1, max_num=self.max_num, image_size=self.image_size, use_thumbnail=True
                )
                assert len(p_num) == 1
                images += image
                content += (self.DEFAULT_IMAGE_TOKEN * p_num[0] + '\n')

        preprocess = self.image_processor.preprocess
        image_tokenizer = self.tokenizer_image_token
        image_tensor = [
            preprocess(f, return_tensors='pt')['pixel_values'][0].half().cuda() for f in images
        ]
        image_tensor = torch.stack(image_tensor)

        conv = self.conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        if self.DEFAULT_IMAGE_TOKEN in content:
            modality = 'image'
        else:
            modality = 'lang'
        prompt_question = conv.get_prompt(modality)
        print(prompt_question)

        input_ids = image_tokenizer(prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda()

        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        audio = torch.zeros(400, 80)
        audio_length = audio.shape[0]
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        audios = dict()
        audios['audios'] = audio.half().cuda()
        audios['lengths'] = audio_length.half().cuda()
        audio_for_llm_lens = 60
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
        audios["lengths_for_llm"] = audio_for_llm_lens.cuda()

        sf_masks = torch.tensor([0] * len(image_tensor)).cuda()
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            audios=audios,
            sf_masks=sf_masks,
            do_sample=False,
            temperature=0.01,
            max_new_tokens=2048,
            stopping_criteria=[stopping_criteria],
        )
        input_token_len = input_ids.shape[1]
        cont = cont[:, input_token_len:]
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        if '<1>' in text_outputs or '<2>' in text_outputs or '<3>' in text_outputs:
            return text_outputs[1:]
        else:
            return text_outputs
