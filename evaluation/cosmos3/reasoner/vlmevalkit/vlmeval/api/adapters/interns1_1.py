import io
import os
import string

import pandas as pd
from PIL import Image

from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE
from vlmeval.smp import LMUDataRoot, get_logger, listinstr
from .base import ModelAdapter, register_adapter

logger = get_logger(__name__)


@register_adapter('interns1_1_no_think')
class InternS1_1NoThinkAdapter(ModelAdapter):

    def __init__(self):
        self.cot_prompt = None
        self.screen_parse = True
        self.split_think = True

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset, system_prompt=None):
        if dataset in [
            'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
            'optics_dataset', 'quantum_dataset', 'statistics_dataset',
            'Physics', 'SFE', 'SFE-zh', 'IPhO_2025', 'XLRS-Bench-lite',
            'OmniEarth-Bench',
        ]:
            return False
        elif listinstr([
            'MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT',
            'MMAlignBench', 'ScreenSpot', 'ChartQAPro', 'MMMU',
        ], dataset):
            return False
        elif DATASET_MODALITY(dataset) == 'VIDEO':
            return False
        elif DATASET_TYPE(dataset) in ['Y/N', 'MCQ', 'VQA', 'GUI']:
            return True
        return False

    def build_prompt(self, line, dataset=None):
        from pathlib import Path

        import yaml

        from vlmeval.dataset import build_dataset, infer_dataset_basename
        from vlmeval.vlm.internvl.utils import (build_mcq_cot_prompt, build_multi_choice_prompt,
                                                build_qa_cot_prompt, format_nav_prompt,
                                                pile_action_history)

        assert self.use_custom_prompt(dataset)

        if listinstr(['ChartMimic'], dataset):
            input_figure_path_rel = line['input_figure']
            ROOT = LMUDataRoot()
            img_root = os.path.join(ROOT, 'images', 'ChartMimic')
            tgt_path = [os.path.join(img_root, input_figure_path_rel)]
        else:
            tgt_path = self.dump_image(line, dataset)

        if DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath', 'QSpatial',
                            'WeMath', 'LogicVista', 'MM-IFEval', 'ChartMimic'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        elif DATASET_TYPE(dataset) == 'GUI':
            vlmeval_root = Path(__file__).parent.parent.parent
            tmpl_path = os.path.join(vlmeval_root, 'vlm/internvl/gui_template.yaml')
            with open(tmpl_path, 'r') as f:
                GUI_TEMPLATE = yaml.load(f, Loader=yaml.FullLoader)
            ds_basename = infer_dataset_basename(dataset)
            ds = build_dataset(dataset, skeleton=True)
            action_space = ds.get_action_space()
            traj_dict = ds.get_trajectory(line)
            prompt_config = GUI_TEMPLATE[ds_basename]
            if 'history' in prompt_config['placeholders']:
                traj_dict['history'] = pile_action_history(traj_dict['history'])
            prompt = format_nav_prompt(
                (
                    'Please provide the bounding box coordinate of the region this sentence describes: <ref>{task}</ref>'  # noqa: E501
                    if self.screen_parse
                    else prompt_config['template']
                ),
                prompt_config['placeholders'],
                action_space=action_space,
                **traj_dict,
            )
        else:
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def process_inputs(self, inputs, dataset=None):
        from vlmeval.vlm.internvl.utils import build_video_prompt, reorganize_prompt

        image_items = [x.copy() for x in inputs if x['type'] == 'image']
        image_num = len(image_items)
        prompt = reorganize_prompt(inputs, image_num, dataset=dataset)

        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = build_video_prompt(prompt, dataset)

        if listinstr(['MMMU'], dataset) and len(image_items) > 0:
            image = Image.open(image_items[0]['value'])
            image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
            tmp_image = io.BytesIO()
            image.save(tmp_image, format='PNG')
            image_items[0]['value'] = tmp_image

        prompt = prompt.replace('<image>', '<IMAGE_TOKEN>')
        return [*image_items, dict(type='text', value=prompt)]

    def postprocess(self, response, dataset=None):
        if self.split_think and '<think>' in response and '</think>' in response:
            thinking, _, answer = response.partition('<think>')[-1].partition('</think>')
            logger.info('-----------Thinking-----------\n'
                        f'{thinking}\n'
                        '------------------------------')
            return answer
        elif self.split_think and '</think>' in response:
            thinking, _, answer = response.partition('</think>')
            logger.info('-----------Thinking-----------\n'
                        f'{thinking}\n'
                        '------------------------------')
            return answer
        return response


@register_adapter('interns1_1_think')
class InternS1_1ThinkAdapter(ModelAdapter):

    def __init__(self):
        self.cot_prompt = None
        self.screen_parse = True
        self.split_think = True

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset, system_prompt=None):
        if dataset in ['SFE', 'SFE-zh', 'IPhO_2025']:
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        if listinstr(['SFE'], dataset):
            return self._build_sfe_prompt(line, dataset)
        elif listinstr(['IPhO_2025'], dataset):
            return self._build_hipho_prompt(line, dataset)
        else:
            raise ValueError(f'Custom prompt not supported for dataset: {dataset}')

    def _build_sfe_prompt(self, line, dataset):
        MCQ_PROMPT = 'You are an expert in {discipline} and need to solve the following question.'
        EXACT_MATCH_PROMPT = 'You are an expert in {discipline} and need to solve the following question.'
        OPEN_QUESTION_PROMPT = 'You are an expert in {discipline} and need to solve the following question.'

        tgt_path = self.dump_image(line, dataset)
        question_type = line['question_type']
        field = line['category']
        question = line['question']

        if question_type == 'exact_match':
            prompt = EXACT_MATCH_PROMPT.format(discipline=field)
            question = prompt + ' ' + question
        elif question_type == 'mcq':
            prompt = MCQ_PROMPT.format(discipline=field)
            question = prompt + ' ' + question
            if not pd.isna(line['A']):
                question += '\nChoices are:\n'
                for ch in string.ascii_uppercase[:15]:
                    if not pd.isna(line[ch]):
                        question += f'{ch}. {line[ch]}\n'
                    else:
                        break
        elif question_type == 'open_ended':
            prompt = OPEN_QUESTION_PROMPT.format(discipline=field)
            question = prompt + ' ' + question

        prompt_segs = question.split('<image>')
        assert len(prompt_segs) == len(tgt_path) + 1
        msgs = []
        for i in range(len(tgt_path)):
            text = prompt_segs[i].strip()
            if text:
                msgs.append(dict(type='text', value=text))
            msgs.append(dict(type='image', value=tgt_path[i]))
        text = prompt_segs[-1].strip()
        if text:
            msgs.append(dict(type='text', value=text))
        return msgs

    def _build_hipho_prompt(self, line, dataset):
        def safe_str(val):
            return '' if pd.isna(val) or val == '' else str(val)

        context = safe_str(line.get('context', ''))
        question = safe_str(line['question'])
        information = safe_str(line.get('information', ''))

        SYSTEM_PROMPTS_EN = (
            'Please answer the problem adhering to the following rules:\n'
            '1. Please use LaTeX format to represent the variables and formulas '
            'used in the solution process and results.\n'
            '2. Please put the final answer(s) in \\boxed{}, note that the unit of '
            'the answer should not be included in \\boxed{}.\n'
            '3. If the problem requires multiple answers, list them in order, each in a separate \\boxed{}.\n'
            'Problem: Information:{information}\n'
            'Context:{context}\n'
            'Question: {problem}'
        )
        formatted_prompt = (
            SYSTEM_PROMPTS_EN
            .replace('{context}', context)
            .replace('{problem}', question)
            .replace('{information}', information)
        )

        msgs = []
        image_val = str(line.get('image', '')).strip()
        if image_val and not image_val.startswith('NO_IMAGE_PLACEHOLDER_'):
            tgt_path = self.dump_image(line, dataset)
            if tgt_path and tgt_path != ['']:
                if isinstance(tgt_path, list):
                    msgs.extend([dict(type='image', value=p) for p in tgt_path])
                else:
                    msgs.append(dict(type='image', value=tgt_path))

        msgs.append(dict(type='text', value=formatted_prompt))
        return msgs

    def process_inputs(self, inputs, dataset=None):
        from vlmeval.vlm.internvl.utils import build_video_prompt, reorganize_prompt

        image_items = [x.copy() for x in inputs if x['type'] == 'image']
        image_num = len(image_items)
        prompt = reorganize_prompt(inputs, image_num, dataset=dataset)

        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = build_video_prompt(prompt, dataset)

        if listinstr(['MMMU'], dataset) and len(image_items) > 0:
            image = Image.open(image_items[0]['value'])
            image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
            tmp_image = io.BytesIO()
            image.save(tmp_image, format='PNG')
            image_items[0]['value'] = tmp_image

        prompt = prompt.replace('<image>', '<IMAGE_TOKEN>')
        return [*image_items, dict(type='text', value=prompt)]

    def postprocess(self, response, dataset=None):
        if self.split_think and '<think>' in response and '</think>' in response:
            thinking, _, answer = response.partition('<think>')[-1].partition('</think>')
            logger.info('-----------Thinking-----------\n'
                        f'{thinking}\n'
                        '------------------------------')
            return answer
        elif self.split_think and '</think>' in response:
            thinking, _, answer = response.partition('</think>')
            logger.info('-----------Thinking-----------\n'
                        f'{thinking}\n'
                        '------------------------------')
            return answer
        return response
