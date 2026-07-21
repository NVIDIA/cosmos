import torch
from transformers import AutoModelForCausalLM

from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE
from vlmeval.smp import cn_string, listinstr
from ..base import BaseModel


class Ovis(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='AIDC-AI/Ovis1.5-Llama3-8B', **kwargs):
        assert model_path is not None
        # Recommend to install `transformers==4.43.2` and `torch==2.1.2`.
        self.model_path = model_path
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            multimodal_max_length=8192,
            trust_remote_code=True
        )
        self.model = self.model.eval().to(device=self.device)
        self.eos_token_id = self.model.generation_config.eos_token_id
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.pad_token_id = self.text_tokenizer.pad_token_id
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.conversation_formatter = self.model.get_conversation_formatter()
        self.image_placeholder = '<image>'
        self.gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True
        )
        self.gen_kwargs.update(kwargs)

    def use_custom_prompt(self, dataset):
        if DATASET_TYPE(dataset) == 'Y/N' or DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if DATASET_TYPE(dataset) == 'Y/N':
            prompt = self.build_yorn_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        else:
            raise RuntimeError(f'Invalid dataset type: {DATASET_TYPE(dataset)}')
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        # interleave dataset
        if dataset.startswith('MMMU_'):
            from vlmeval.dataset import MMMUDataset
            message = MMMUDataset.split_MMMU(message)

        return message

    def build_yorn_prompt(self, line, dataset=None):
        prompt = line['question']
        if listinstr(['HallusionBench'], dataset):
            prompt += ' Please answer yes or no.'
        prompt += '\n请用单个词或短语回答问题。' if cn_string(
            prompt) else '\nAnswer the question using a single word or phrase.'
        return prompt

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

    def generate_inner(self, message, dataset=None):
        prompt, input_ids, attention_mask, pixel_values = self.prepare_inputs(message)
        output_ids = self.model.generate(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **self.gen_kwargs
        )
        response = self.text_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        return response

    def prepare_inputs(self, message):
        # build query
        images = [x['value'] for x in message if x['type'] == 'image']
        texts = [x['value'] for x in message if x['type'] == 'text']
        if len(images) == 0:
            query = '\n'.join(texts)
        elif len(images) == 1 and len(texts) == 1:
            query = self.image_placeholder + '\n' + texts[0]
        else:  # interleave sample
            chunks = [x['value'] if x['type'] == 'text' else self.image_placeholder for x in message]
            query = '\n'.join(chunks)

        # format conversation
        prompt, input_ids = self.conversation_formatter.format_query(query)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.device)

        # preprocess images
        if len(images) == 0:
            pixel_values = [None]
        else:
            preprocessed_images = [self.visual_tokenizer.preprocess_image(Image.open(image)) for image in images]
            pixel_values = [torch.cat(preprocessed_images, dim=0).to(device=self.device, dtype=self.dtype)]

        return prompt, input_ids, attention_mask, pixel_values


class Ovis1_6(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='AIDC-AI/Ovis1.6-Gemma2-9B', **kwargs):
        assert model_path is not None
        # Recommend to install `python=3.10`, `transformers==4.44.2`, `torch==2.2.0`, and `numpy==1.24.3`
        self.model_path = model_path
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            multimodal_max_length=8192,
            trust_remote_code=True
        )
        self.model = self.model.eval().to(device=self.device)
        self.eos_token_id = self.model.generation_config.eos_token_id
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.pad_token_id = self.text_tokenizer.pad_token_id
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.max_partition = 9
        self.image_placeholder = '<image>'
        self.gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True
        )
        self.gen_kwargs.update(kwargs)

    def use_custom_prompt(self, dataset):
        if DATASET_TYPE(dataset) == 'Y/N' or DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_yorn_prompt(self, line, dataset=None):
        prompt = line['question'] + '\nAnswer the question using a single word or phrase.'
        return prompt

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
            prompt += "\nAnswer with the option's letter from the given choices directly."

        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if DATASET_TYPE(dataset) == 'Y/N':
            prompt = self.build_yorn_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        else:
            raise RuntimeError(f'Invalid dataset type: {DATASET_TYPE(dataset)}')
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        # interleave dataset
        if dataset.startswith('MMMU_'):
            from vlmeval.dataset import MMMUDataset
            message = MMMUDataset.split_MMMU(message)

        return message

    def generate_inner(self, message, dataset=None):
        prompt, input_ids, attention_mask, pixel_values = self.prepare_inputs(message)
        output_ids = self.model.generate(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **self.gen_kwargs
        )
        response = self.text_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return response

    def prepare_inputs(self, message):
        # build query
        images = [x['value'] for x in message if x['type'] == 'image']
        texts = [x['value'] for x in message if x['type'] == 'text']
        if len(images) == 0:
            query = '\n'.join(texts)
        elif len(images) == 1 and len(texts) == 1:
            query = self.image_placeholder + '\n' + texts[0]
        else:  # interleaved sample
            chunks = [x['value'] if x['type'] == 'text' else self.image_placeholder for x in message]
            query = '\n'.join(chunks)

        # preprocess inputs
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, [Image.open(image) for image in images], max_partition=self.max_partition
        )

        # move to self.device
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.device)
        pixel_values = [
            pixel_values.to(device=self.device, dtype=self.dtype) if pixel_values is not None else None
        ]

        return prompt, input_ids, attention_mask, pixel_values


class Ovis1_6_Plus(Ovis1_6):
    # Recommend to install `python=3.10`, `transformers==4.46.2`, `torch==2.4.0`, and `numpy==1.25.0`

    def build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        import string

        import pandas as pd

        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
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
            prompt += 'Please select the correct answer from the options above.'
        prompt = prompt.rstrip()
        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset.startswith('MMMU_'):
            prompt = self.build_mmmu_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'Y/N':
            prompt = self.build_yorn_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        else:
            raise RuntimeError(f'Invalid dataset type: {DATASET_TYPE(dataset)}')

        message = [dict(type='image', value=s) for s in tgt_path] + [dict(type='text', value=prompt)]

        return message


class Ovis2(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    SIZE_DICT = {
        (24, 896): '1B',  # (num_hidden_layers, hidden_size)
        (28, 1536): '2B',
        (36, 2048): '4B',
        (28, 3584): '8B',
        (48, 5120): '16B',
        (64, 5120): '34B'
    }

    def __init__(self, model_path='AIDC-AI/Ovis2-8B', **kwargs):
        assert model_path is not None
        # Recommend to install `python=3.10`, `transformers==4.46.2`, `torch==2.4.0`, and `numpy==1.25.0`
        self.model_path = model_path
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            multimodal_max_length=32768,
            trust_remote_code=True
        )
        self.size = self.SIZE_DICT[
            (self.model.config.llm_config.num_hidden_layers, self.model.config.llm_config.hidden_size)]
        self.model = self.model.eval().to(device=self.device)
        self.eos_token_id = self.model.generation_config.eos_token_id
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.pad_token_id = self.text_tokenizer.pad_token_id
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.image_placeholder = '<image>'
        self.gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True
        )
        self.use_cot = {
            '1B': {'MathVerse', 'MathVision'},
            '2B': {'MMVet', 'MMStar', 'MathVerse', 'MathVision'},
            '4B': {'MathVerse', 'MathVision'},
            '8B': {'MMVet', 'MMStar', 'MMMU', 'MathVista', 'MathVerse', 'MathVision'},
            '16B': {'MMVet', 'MMStar', 'MMMU', 'MathVista', 'MathVerse', 'MathVision'},
            '34B':  {'MMVet', 'MMStar', 'MMMU', 'MathVista', 'MathVerse', 'MathVision'}
        }
        self.frame_selector = None
        if kwargs.pop("frame_selection", False):
            from .utils.mdp3 import MDP3
            self.frame_selector = MDP3(
                n_selection=int(kwargs.pop("n_frames", 32)),
                visual_encoder_name_or_path=kwargs.pop("frame_selection_vlm", "google/siglip-so400m-patch14-384"),
                device=f"cuda:{self.device}"
            )
        self.gen_kwargs.update(kwargs)

    def use_custom_prompt(self, dataset):
        if any(dataset.startswith(prefix) for prefix in ['MMVet', 'MathVista', 'MathVerse', 'MathVision']):
            return True
        if DATASET_TYPE(dataset) == 'Y/N' or DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_yorn_prompt(self, line, dataset=None):
        prompt = line['question']
        if listinstr(['HallusionBench'], dataset) and self.size == '34B':
            prompt += ' Please answer yes or no.'
        prompt += '\nAnswer the question using a single word or phrase.'
        return prompt

    def build_multi_choice_prompt(self, line, dataset=None, use_cot=False):
        prompt = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            prompt = hint + '\n' + prompt

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            prompt += f'\n{key}. {item}'

        if len(options):
            if use_cot:
                prompt += "\nProvide a step-by-step solution to the problem, and conclude with 'the answer is' followed by the final solution."
            else:
                prompt += "\nAnswer with the option's letter from the given choices directly."

        return prompt

    def build_mmvet_prompt(self, line, dataset=None, use_cot=False):
        prompt = line['question']
        if use_cot:
            prompt += "\nProvide a step-by-step solution to the problem carefully."
        return prompt

    def build_math_prompt(self, line, dataset=None, use_cot=False):
        prompt = line['question']
        if use_cot:
            prompt += "\nProvide a step-by-step solution to the problem, and conclude with 'the answer is' followed by the final solution."
        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        use_cot = any(dataset.startswith(prefix) for prefix in self.use_cot[self.size])

        if dataset == 'MMVet':
            prompt = self.build_mmvet_prompt(line, dataset, use_cot)
        elif any(dataset.startswith(prefix) for prefix in ('MathVista', 'MathVerse', 'MathVision')):
            prompt = self.build_math_prompt(line, dataset, use_cot)
        elif DATASET_TYPE(dataset) == 'Y/N':
            prompt = self.build_yorn_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset, use_cot)
        else:
            raise RuntimeError(f'Invalid dataset type: {DATASET_TYPE(dataset)}')

        message = [dict(type='image', value=s) for s in tgt_path] + [dict(type='text', value=prompt)]

        # interleave dataset
        if dataset.startswith('MMMU_'):
            from vlmeval.dataset import MMMUDataset
            message = MMMUDataset.split_MMMU(message)

        return message

    def generate_inner(self, message, dataset=None):
        def _extract_answer(text):
            answer_index = text.lower().find('the answer is')
            if answer_index != -1:
                answer_index += len('the answer is')
                answer = text[answer_index:].lstrip(':').strip()
            else:
                answer = text
            return answer

        # DynaMath
        if dataset == 'DynaMath' and self.size == '34B':
            message[-1]['value'] += "\nProvide a step-by-step solution to the problem, and conclude with 'the answer is' followed by the final solution."

        prompt, input_ids, attention_mask, pixel_values, max_partition = self.prepare_inputs(message, dataset)
        output_ids = self.model.generate(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **self.gen_kwargs
        )
        response = self.text_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if "conclude with 'the answer is' followed by the final solution." in prompt:
            response = _extract_answer(response)

        return response

    def prepare_inputs(self, message, dataset=None):
        # build query
        images = [x['value'] for x in message if x['type'] == 'image']
        texts = [x['value'] for x in message if x['type'] == 'text']
        if DATASET_MODALITY(dataset) == 'VIDEO': # video inputs
            chunks = [self.image_placeholder for x in message if x['type'] != 'text']
            chunks += [x['value'].strip() for x in message if x['type'] == 'text' and x['value'] != '']
            query = '\n'.join(chunks)
        elif len(images) == 0: # text-only inputs
            query = '\n'.join(texts)
        elif len(images) == 1 and len(texts) == 1: # single-image inputs
            query = self.image_placeholder + '\n' + texts[0]
        else:  # interleaved inputs
            chunks = [x['value'].strip() if x['type'] == 'text' else self.image_placeholder for x in message]
            query = '\n'.join(chunks)

        # preprocess inputs
        if DATASET_MODALITY(dataset) == 'VIDEO':
            max_partition = 1
        elif (dataset != None) and any(
            dataset.startswith(prefix) for prefix in
            ('HallusionBench', 'TextVQA', 'ChartQA', 'OCRBench', 'InfoVQA', 'DocVQA', 'MTVQA')):
            max_partition = 12
        elif len(images) > 1:
            max_partition = max(1, 12 // len(images))
        else:
            max_partition = 9

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, [Image.open(image) for image in images], max_partition=max_partition, frame_selector=self.frame_selector
        )

        # move to self.device
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.device)
        pixel_values = [
            pixel_values.to(device=self.device, dtype=self.dtype) if pixel_values is not None else None
        ]

        return prompt, input_ids, attention_mask, pixel_values, max_partition


class OvisU1(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='AIDC-AI/Ovis-U1-3B', **kwargs):
        assert model_path is not None
        # Recommend to install `transformers==4.51.3`, `torch==2.4.0`, and `numpy==1.24.3`
        self.model_path = model_path
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            multimodal_max_length=32768,
            trust_remote_code=True
        )
        self.model = self.model.eval().to(device=self.device)
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.pad_token_id = self.text_tokenizer.pad_token_id
        self.eos_token_id = self.text_tokenizer.eos_token_id
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.image_placeholder = '<image>'
        self.gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True
        )
        self.min_pixels = 200704  # 448*448
        self.max_pixels = 2408448  # 1344*1792
        self.frame_selector = None
        if kwargs.pop("frame_selection", False):
            from .utils.mdp3 import MDP3
            self.frame_selector = MDP3(
                n_selection=int(kwargs.pop("n_frames", 32)),
                visual_encoder_name_or_path=kwargs.pop("frame_selection_vlm", "google/siglip-so400m-patch14-384"),
                device=f"cuda:{self.device}"
            )
        self.gen_kwargs.update(kwargs)
        self.use_cot = {'MMMU'}

    def use_custom_prompt(self, dataset):
        if any(dataset.startswith(prefix) for prefix in ['MMVet', 'MathVista', 'MathVerse', 'MathVision']):
            return True
        if DATASET_TYPE(dataset) == 'Y/N' or DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_yorn_prompt(self, line, dataset=None):
        prompt = line['question']
        if listinstr(['HallusionBench'], dataset):  # and self.size == '34B':
            prompt += ' Please answer yes or no.'
        prompt += '\nAnswer the question using a single word or phrase.'
        return prompt

    def build_multi_choice_prompt(self, line, dataset=None, use_cot=False):
        prompt = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            prompt = hint + '\n' + prompt

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            prompt += f'\n{key}. {item}'

        if len(options):
            if use_cot:
                prompt += "\nProvide a step-by-step solution to the problem, and conclude with 'the answer is' followed by the final solution."
            else:
                prompt += "\nAnswer with the option's letter from the given choices directly."

        return prompt

    def build_mmvet_prompt(self, line, dataset=None, use_cot=False):
        prompt = line['question']
        if use_cot:
            prompt += "\nProvide a step-by-step solution to the problem carefully."
        return prompt

    def build_math_prompt(self, line, dataset=None, use_cot=False):
        prompt = line['question']
        if use_cot:
            prompt += "\nProvide a step-by-step solution to the problem, and conclude with 'the answer is' followed by the final solution."
        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        use_cot = any(dataset.startswith(prefix) for prefix in self.use_cot)

        if dataset == 'MMVet':
            prompt = self.build_mmvet_prompt(line, dataset, use_cot)
        elif any(dataset.startswith(prefix) for prefix in ('MathVista', 'MathVerse', 'MathVision')):
            prompt = self.build_math_prompt(line, dataset, use_cot)
        elif DATASET_TYPE(dataset) == 'Y/N':
            prompt = self.build_yorn_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset, use_cot)
        else:
            raise RuntimeError(f'Invalid dataset type: {DATASET_TYPE(dataset)}')

        message = [dict(type='image', value=s) for s in tgt_path] + [dict(type='text', value=prompt)]

        # interleave dataset
        if dataset.startswith('MMMU_'):
            from vlmeval.dataset import MMMUDataset
            message = MMMUDataset.split_MMMU(message)

        return message

    def generate_inner(self, message, dataset=None):
        def _extract_answer(text):
            answer_index = text.lower().find('the answer is')
            if answer_index != -1:
                answer_index += len('the answer is')
                answer = text[answer_index:].lstrip(':').strip()
            else:
                answer = text
            return answer

        # DynaMath
        if dataset == 'DynaMath':
            message[-1][
                'value'] += "\nProvide a step-by-step solution to the problem, and conclude with 'the answer is' followed by the final solution."

        prompt, input_ids, attention_mask, pixel_values, grid_thws = self.prepare_inputs(message, dataset)
        output_ids = self.model.generate(
            input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            attention_mask=attention_mask,
            **self.gen_kwargs
        )
        response = self.text_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print('\n========================************========================')
        print(f'prompt: {prompt}<<<\n')
        print(f'output: {response}\n')

        think_end = response.rfind('</think>')
        if think_end != -1:
            think_end += len('</think>')
            response = response[think_end:].strip()
            print(f'extract answer: {response}\n')

        if "conclude with 'the answer is' followed by the final solution." in prompt:
            response = _extract_answer(response)
            print(f'extract answer: {response}\n')

        print('------------------------------------------------------------\n', flush=True)

        return response

    def prepare_inputs(self, message, dataset=None):
        # build query
        images = [x['value'] for x in message if x['type'] == 'image']
        texts = [x['value'] for x in message if x['type'] == 'text']
        # print(f"=============={DATASET_MODALITY(dataset)}============")
        if DATASET_MODALITY(dataset) == 'VIDEO':  # video inputs
            chunks = [self.image_placeholder for x in message if x['type'] != 'text']
            chunks += [x['value'].strip() for x in message if x['type'] == 'text' and x['value'] != '']
            query = '\n'.join(chunks)
            # print(query, chunks)
        elif len(images) == 0:  # text-only inputs
            query = '\n'.join(texts)
        elif len(images) == 1 and len(texts) == 1:  # single-image inputs
            query = self.image_placeholder + '\n' + texts[0]
        else:  # interleaved inputs
            chunks = [x['value'].strip() if x['type'] == 'text' else self.image_placeholder for x in message]
            query = '\n'.join(chunks)

        # preprocess inputs
        min_pixels = self.min_pixels
        max_pixels = self.max_pixels
        enable_thinking = os.getenv("OvisThink") == 'True'
        prompt, input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            query, [Image.open(image) for image in images],
            frame_selector=self.frame_selector,
            enable_thinking=enable_thinking,
            min_pixels=min_pixels,
            max_pixels=max_pixels,  # 2000*2000,
        )

        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.device)
        pixel_values = torch.cat([
            pixel_values.to(device=self.device, dtype=self.dtype) if pixel_values is not None else None
        ], dim=0)
        grid_thws = torch.cat([
            grid_thws.to(device=self.device) if grid_thws is not None else None
        ], dim=0)

        return prompt, input_ids, attention_mask, pixel_values, grid_thws


class Ovis2_5(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    SIZE_DICT = {
        (28, 2048): '2B',  # (num_hidden_layers, hidden_size)
        (36, 4096): '9B'
    }

    def __init__(self, model_path='AIDC-AI/Ovis2.5-9B', **kwargs):
        assert model_path is not None
        # Recommend to install dependencies as follows:
        # `pip install vllm==0.10.2 --extra-index-url https://wheels.vllm.ai/0.10.2/`

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        from vllm import LLM
        self.model_path = model_path
        self.dtype = torch.bfloat16
        dist_kwargs = dict()
        prev_local_rank = os.getenv("LOCAL_RANK")
        if prev_local_rank is not None:
            os.environ["LOCAL_RANK"] = "0"
            torch.cuda.set_device(0)
            dist_kwargs["distributed_executor_backend"] = "external_launcher"
        self.model = LLM(
            model=self.model_path,
            dtype=self.dtype,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
            **dist_kwargs
        )
        size_key = (
            self.model.llm_engine.model_config.hf_config.llm_config.num_hidden_layers,
            self.model.llm_engine.model_config.hf_config.llm_config.hidden_size
        )
        self.size = self.SIZE_DICT[size_key]
        if prev_local_rank is not None:
            os.environ["LOCAL_RANK"] = prev_local_rank
        self.tokenizer = self.model.get_tokenizer()
        self.image_placeholder = '<image>'
        self.video_placeholder = '<video>'
        self.benchmark_with_thinking_map = {
            '2B': ('HallusionBench', 'MMStar', 'MMMU', 'MathVista', 'MathVerse', 'MathVision', 'LogicVista', 'WeMath', 'DynaMath'),
            '9B': ('MMBench', 'HallusionBench', 'MMStar', 'MMMU', 'MathVista', 'MathVerse', 'MathVision', 'LogicVista', 'WeMath', 'DynaMath')
        }
        self.objective_prompt_suffix = "End your response with 'Final answer: '."

    def use_custom_prompt(self, dataset):
        if any(dataset.startswith(prefix) for prefix in
               ['MathVista', 'MathVerse', 'MathVision', 'LogicVista']):
            return True
        if DATASET_TYPE(dataset) == 'Y/N' or DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_yorn_prompt(self, line, dataset=None):
        prompt = line['question']
        if listinstr(['HallusionBench'], dataset):
            prompt += ' Please answer yes or no.'
        prompt += f'\n{self.objective_prompt_suffix}'
        return prompt

    def build_multi_choice_prompt(self, line, dataset=None):
        prompt = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            prompt = hint + '\n' + prompt

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            prompt += f'\n{key}. {item}'

        if any(dataset.startswith(prefix) for prefix in ['AI2D']):
            prompt += "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += f'\n{self.objective_prompt_suffix}'

        return prompt

    def build_math_prompt(self, line, dataset=None):
        prompt = line['question']
        prompt += f'\n{self.objective_prompt_suffix}'
        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if any(dataset.startswith(prefix) for prefix in ['MathVista', 'MathVerse', 'MathVision', 'LogicVista']):
            prompt = self.build_math_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'Y/N':
            prompt = self.build_yorn_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        else:
            raise RuntimeError(f'Invalid dataset type: {DATASET_TYPE(dataset)}')

        message = [dict(type='image', value=s) for s in tgt_path] + [dict(type='text', value=prompt)]

        # interleave dataset
        if dataset.startswith('MMMU_'):
            from vlmeval.dataset import MMMUDataset
            message = MMMUDataset.split_MMMU(message)

        return message

    def vllm_infer(self, messages, min_pixels=1024 * 1024, max_pixels=1792 * 1792, enable_thinking=False, max_new_tokens=8192, thinking_budget=6144):
        from vllm import SamplingParams
        if enable_thinking:
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                repetition_penalty=1.05,
                temperature=0.6,
                top_p=0.95,
                top_k=20
            )
        else:
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0
            )

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        vllm_input = {"prompt": prompt}
        images = []
        for message in messages:
            images.extend([x["image"] for x in message["content"] if x["type"] == "image"])
        if images:
            vllm_input["multi_modal_data"] = {"image": images}
            vllm_input["mm_processor_kwargs"] = {"images_kwargs": {"min_pixels": min_pixels, "max_pixels": max_pixels}}
        vllm_inputs = [vllm_input]
        use_thinking_budget = enable_thinking and thinking_budget is not None and thinking_budget > 0
        if not use_thinking_budget:
            vllm_outputs = self.model.generate(vllm_inputs, sampling_params, use_tqdm=False)
        else:
            phase1_params = sampling_params.clone()
            phase1_params.max_tokens = thinking_budget
            phase1_outputs = self.model.generate(vllm_inputs, phase1_params, use_tqdm=False)

            phase2_inputs = []
            vllm_outputs = [None] * len(vllm_inputs)  # 用于存放最终结果的列表

            for i, p1_output in enumerate(phase1_outputs):
                p1_text = p1_output.outputs[0].text
                p1_finish_reason = p1_output.outputs[0].finish_reason

                # 分支 1: 模型在预算内自然停止 (生成了EOS)
                if p1_finish_reason == 'stop':
                    # 生成已完成，直接使用这个结果，无需进入Phase 2
                    vllm_outputs[i] = p1_output
                    continue

                # 分支 2 & 3: 模型因达到长度限制而停止
                elif p1_finish_reason == 'length':
                    p2_input = vllm_inputs[i].copy()

                    # 分支 2: 输出中包含 </think>
                    if '</think>' in p1_text:
                        p2_input["prompt"] = vllm_inputs[i]["prompt"] + p1_text

                    # 分支 3: 输出中不包含 </think>
                    else:
                        forced_stop_text = (
                            "\n\nConsidering the limited time by the user, I have to give the solution "
                            "based on the thinking directly now.\n</think>\n\n"
                        )
                        p1_text += forced_stop_text
                        p2_input["prompt"] = vllm_inputs[i]["prompt"] + p1_text

                    # 记录中间文本，并准备第二阶段的输入
                    p2_input["intermediate_text"] = p1_text  # 暂存第一阶段文本
                    p2_input["original_index"] = i  # 记录原始索引，以便回填结果
                    phase2_inputs.append(p2_input)

            # 执行阶段 2 (如果需要)
            if phase2_inputs:
                phase2_params = sampling_params.clone()
                phase2_params.max_tokens = max_new_tokens - thinking_budget

                phase2_outputs = self.model.generate(phase2_inputs, phase2_params, use_tqdm=False)

                # 组合两阶段的结果
                for i, p2_output in enumerate(phase2_outputs):
                    original_index = phase2_inputs[i]["original_index"]
                    intermediate_text = phase2_inputs[i]["intermediate_text"]

                    # 创建一个模拟的 RequestOutput 对象来存储合并后的结果
                    # 这样做是为了与后处理流程保持一致
                    final_text = intermediate_text + p2_output.outputs[0].text

                    # 从p1复制一个输出对象来修改
                    final_output = phase1_outputs[original_index]
                    final_output.outputs[0].text = final_text
                    vllm_outputs[original_index] = final_output
        outputs = []
        for i, vllm_output in enumerate(vllm_outputs):
            raw_output_text = vllm_output.outputs[0].text
            if '</think>' in raw_output_text:
                thinking, response = raw_output_text.split('</think>', 1)
                thinking = thinking.split('<think>', 1)[-1].strip()
            else:
                thinking, response = "", raw_output_text
            response = response.strip()
            outputs.append({"response": response, "thinking": thinking})
        return outputs[0], vllm_inputs[0]["prompt"]

    def generate_inner(self, message, dataset=None):
        def _extract_final_answer(response):
            import re
            """
            从字符串中抽取"the final answer is" 或 "final answer" 后面的内容。

            Args:
              response: 输入的字符串。

            Returns:
              抽取并清理后的内容字符串，如果找不到则返回response。
            """
            match = re.search(r'.*(?:the\s+final\s+answer\s+is|final\s+answer\s*:)\s*[:\s]*(.*)',
                              response,
                              re.IGNORECASE | re.DOTALL)

            if match:
                return match.group(1)
            else:
                return response

        messages = self.prepare_inputs(message, dataset)
        enable_thinking = any(dataset.startswith(prefix) for prefix in self.benchmark_with_thinking_map[self.size])
        min_pixels = 448 * 448 if any(dataset.startswith(prefix) for prefix in ['OCRBench']) else 1024 * 1024
        max_pixels = 1792 * 1792
        vllm_output, prompt = self.vllm_infer(messages, min_pixels, max_pixels, enable_thinking)
        response = vllm_output["response"]

        if self.objective_prompt_suffix in prompt:
            response = _extract_final_answer(response)

        return response

    def prepare_inputs(self, message, dataset=None):
        content = []
        for item in message:
            if item['type'] == 'image':
                image = Image.open(item['value'])
                content.append(dict(type='image', image=image))
            elif item['type'] == 'text':
                text = item['value']
                content.append(dict(type='text', text=text))
        return [dict(role="user", content=content)]
