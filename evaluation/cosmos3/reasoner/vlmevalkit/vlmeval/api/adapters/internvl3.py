import copy
import io
import os

from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE
from vlmeval.smp import listinstr
from .base import ModelAdapter, register_adapter

_GUI_TEMPLATE = None


def _get_gui_template():
    global _GUI_TEMPLATE
    if _GUI_TEMPLATE is None:
        from pathlib import Path

        import yaml
        vlmeval_root = Path(__file__).parent.parent.parent
        tmpl_path = os.path.join(vlmeval_root, 'vlm/internvl/gui_template.yaml')
        with open(tmpl_path, 'r') as f:
            _GUI_TEMPLATE = yaml.load(f, Loader=yaml.FullLoader)
    return _GUI_TEMPLATE


@register_adapter('internvl3')
class InternVL3Adapter(ModelAdapter):

    def __init__(self):
        self.cot_prompt = None
        self.screen_parse = True
        self.split_think = True

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def override_model_args(self, dataset):
        think_ds = [
            'MMMU', 'MathVista', 'SFE', 'Physics', 'MathVision',
            'OlympiadBench', 'IPhO_2025', 'MaCBench',
        ]
        think_system = (
            'You are an expert reasoner with extensive experience in all '
            'areas. You approach problems through systematic thinking and '
            'rigorous reasoning. Your response should reflect deep '
            'understanding and precise logical thinking, making your '
            'solution path and reasoning clear to others. Please put your '
            'thinking process within <think>...</think> tags.'
        )
        if listinstr(think_ds, dataset):
            return dict(system_prompt=think_system)
        else:
            return dict(temperature=0.)

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
        from vlmeval.dataset import build_dataset, infer_dataset_basename
        from vlmeval.smp import LMUDataRoot
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
            GUI_TEMPLATE = _get_gui_template()
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
        from PIL import Image

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

    def get_max_num(self, dataset):
        if dataset is None:
            return None
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'VCR_EN', 'VCR_ZH', 'OCRVQA']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if DATASET_MODALITY(dataset) == 'VIDEO':
            return 1
        elif listinstr(res_12_datasets, dataset):
            return 12
        elif listinstr(res_18_datasets, dataset):
            return 18
        elif listinstr(res_24_datasets, dataset):
            return 24
        elif DATASET_TYPE(dataset) == 'GUI':
            return 12
        return None

    def process_payload(self, payload, dataset=None):
        max_num = self.get_max_num(dataset)
        if max_num is not None:
            payload = copy.deepcopy(payload)
            for msg in payload['messages']:
                content = msg['content']
                if isinstance(content, dict) and content.get('type') == 'image_url':
                    content['image_url']['max_dynamic_patch'] = max_num
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'image_url':
                            item['image_url']['max_dynamic_patch'] = max_num
        return payload

    def postprocess(self, response, dataset=None):
        if self.split_think and '<think>' in response and '</think>' in response:
            _, _, answer = response.partition('<think>')[-1].partition('</think>')
            return answer
        return response
