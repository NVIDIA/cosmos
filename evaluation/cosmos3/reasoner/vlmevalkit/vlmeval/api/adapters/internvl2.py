import copy
from functools import partial

from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE
from vlmeval.smp import LMUDataRoot, listinstr
from .base import ModelAdapter, register_adapter


@register_adapter('internvl2')
class InternVL2Adapter(ModelAdapter):

    def __init__(self, use_mpo_prompt=False):
        self.use_mpo_prompt = use_mpo_prompt
        self.cot_prompt = None
        self.screen_parse = False

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset, system_prompt=None):
        assert dataset is not None
        if dataset in [
            'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
            'optics_dataset', 'quantum_dataset', 'statistics_dataset',
            'Physics', 'SFE', 'SFE-zh',
            'XLRS-Bench-lite', 'OmniEarth-Bench',
        ]:
            return False
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT', 'MMAlignBench'], dataset):
            return False
        if system_prompt is not None and '<think>' in system_prompt and listinstr(
            ['MicroVQA', 'MSEarthMCQ', 'MMSci_DEV_MCQ', 'MMMU', 'VisuLogic'], dataset
        ):
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            return False
        return True

    def build_prompt(self, line, dataset=None):
        import os
        from pathlib import Path

        import yaml

        from vlmeval.dataset import build_dataset, infer_dataset_basename
        from vlmeval.vlm.internvl.utils import (build_mcq_cot_prompt, build_mpo_prompt,
                                                build_multi_choice_prompt, build_qa_cot_prompt,
                                                format_nav_prompt, pile_action_history)

        use_mpo_prompt = self.use_mpo_prompt and (
            getattr(self, 'use_cot', False)
            or dataset in ['MMStar', 'HallusionBench', 'OCRBench']
        )

        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)

        if not listinstr(['ChartMimic'], dataset):
            tgt_path = self.dump_image(line, dataset)
        else:
            input_figure_path_rel = line['input_figure']
            ROOT = LMUDataRoot()
            img_root = os.path.join(ROOT, 'images', 'ChartMimic')
            tgt_path = [os.path.join(img_root, input_figure_path_rel)]

        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
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
        elif dataset is not None and DATASET_TYPE(dataset) == 'GUI':
            vlmeval_root = Path(__file__).parent.parent.parent
            with open(os.path.join(vlmeval_root, 'vlm/internvl/gui_template.yaml'), 'r') as f:
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

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        if use_mpo_prompt:
            message = build_mpo_prompt(message, line, dataset)
        return message

    def get_max_num(self, dataset):
        if dataset is None:
            return 6
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
        return 6

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
        if self.use_mpo_prompt:
            from vlmeval.vlm.internvl.utils import mpo_post_processing
            return mpo_post_processing(response, dataset)
        return response


register_adapter('internvl2-mpo-cot', partial(InternVL2Adapter, use_mpo_prompt=True))
