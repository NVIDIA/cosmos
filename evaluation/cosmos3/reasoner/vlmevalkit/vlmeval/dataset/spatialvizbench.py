import string
from collections import OrderedDict

import pandas as pd

from vlmeval.smp import load, toliststr
from .image_mcq import ImageMCQDataset


class SpatialVizBench(ImageMCQDataset):
    """
    SpatialViz-Bench.

    Reference:
      SpatialViz-Bench: An MLLM Benchmark for Spatial Visualization
      https://arxiv.org/abs/2507.07610
    """

    TYPE = 'MCQ'

    SPATIALVIZ_TSV_URL = 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SpatialVizBench.tsv'  # noqa: E501
    SPATIALVIZ_TSV_MD5 = '5ed4cb6463bfed0a92e52ffc4e8b6257'

    VARIANTS = ['SpatialVizBench', 'SpatialVizBench_CoT']

    DATASET_URL = {}
    DATASET_MD5 = {}

    for name in VARIANTS:
        DATASET_URL[name] = SPATIALVIZ_TSV_URL
        DATASET_MD5[name] = SPATIALVIZ_TSV_MD5

    _CATEGORY_TASK_ORDER = None

    def __init__(self, dataset, skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

        self._CATEGORY_TASK_ORDER = None
        self.use_cot = self.parse_dataset_name(dataset)
        print(f'Evaluate {dataset} with CoT = {self.use_cot}')

    @staticmethod
    def parse_dataset_name(name: str) -> bool:
        if not isinstance(name, str):
            return False

        lower = name.lower()
        return lower.endswith('_cot')

    @classmethod
    def category_task_order(cls) -> OrderedDict:
        if cls._CATEGORY_TASK_ORDER is None:
            cls._CATEGORY_TASK_ORDER = OrderedDict([
                ('MentalRotation', ['2DRotation', '3DRotation', '3ViewProjection']),
                ('MentalFolding', ['PaperFolding', 'CubeUnfolding', 'CubeReconstruction']),
                ('VisualPenetration', ['CrossSection', 'CubeCounting', 'CubeAssembly']),
                ('MentalAnimation', ['ArrowMoving', 'BlockMoving', 'MechanicalSystem']),
            ])
        return cls._CATEGORY_TASK_ORDER

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }

        options_text = ''
        for key, item in options.items():
            options_text += f'{key}. {item}\n'

        # prompt format follow SpatialViz paper
        if not self.use_cot:
            pre_prompt = (
                'Answer with a single option letter (A, B, C, or D), enclosed within the <answer></answer> tag. '
                'For example: <answer>A</answer>. Ensure that your output contains only the final answer, '
                'without any intermediate reasoning or additional content.'
            )
        else:
            pre_prompt = (
                'You should first provide a reasoning process, '
                'then provide a single option (A, B, C or D) as the final answer. '
                'The reasoning process and the answer are enclosed within '
                '<think></think> and <answer></answer> tags, respectively, '
                'i.e., <think>reasoning process</think>, <answer>answer</answer>.'
            )

        prompt = pre_prompt + '\n' + 'Question:' + question + '\n' + options_text

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import build_mcq_score_fn, eval_mcq_score

        # Select MCQ scoring function (rule-based or LLM-based) according to judge_kwargs['model'].
        score_fn = build_mcq_score_fn(**judge_kwargs)

        category_task_order = self.category_task_order()
        raw = eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col=['category', 'task'],
            order={
                'category': list(category_task_order.keys()),
                'task': sum(category_task_order.values(), []),
            },
            dataset_name=getattr(self, 'dataset_name', 'SpatialVizBench'),
        )

        pretty = OrderedDict()
        pretty['overall'] = raw['overall']

        for cat, tasks in category_task_order.items():
            for t in tasks:
                raw_key = f'task.{t}_accuracy'
                if raw_key in raw:
                    pretty[f'{t}_accuracy'] = raw[raw_key]

            cat_raw_key = f'category.{cat}_accuracy'
            if cat_raw_key in raw:
                pretty[f'{cat}_accuracy'] = raw[cat_raw_key]

        keys_str = ', '.join(pretty.keys())
        vals_str = ', '.join(f'{v:.3f}' for v in pretty.values())
        pretty['tabulated_keys'] = keys_str
        pretty['tabulated_results'] = vals_str

        return pretty
