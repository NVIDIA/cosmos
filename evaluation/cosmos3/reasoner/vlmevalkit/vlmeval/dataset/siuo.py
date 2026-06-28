import copy as cp

import pandas as pd

from ..smp import d2df, dump, load
from ..smp.file import get_intermediate_file_path
from .image_base import ImageBaseDataset
from .siuo_gen import SIUOGenDataset
from .siuo_mcq import SIUOMCQDataset


class SIUODataset(ImageBaseDataset):
    """Unified SIUO dataset.

    SIUO contains two subsets:
    - SIUO_GEN: open-ended generation, key metric `overall_avg_combined`
    - SIUO_MCQ: multi-choice QA, key metric `Overall`

    Final SIUO score is an equal-weight average of these two subset scores.
    """

    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    SUB_DATASETS = ['SIUO_GEN', 'SIUO_MCQ']

    @classmethod
    def supported_datasets(cls):
        return ['SIUO']

    def __init__(self, dataset='SIUO', **kwargs):
        assert dataset == 'SIUO'
        self.dataset_name = dataset
        self.dataset_map = {
            'SIUO_GEN': SIUOGenDataset('SIUO_GEN', **kwargs),
            'SIUO_MCQ': SIUOMCQDataset('SIUO_MCQ', **kwargs),
        }

        data_all = []
        for dname in self.SUB_DATASETS:
            data = self.dataset_map[dname].data.copy()
            data['SUB_DATASET'] = [dname] * len(data)
            data_all.append(data)

        data = pd.concat(data_all, ignore_index=True)
        data['original_index'] = data.pop('index')
        data['index'] = list(range(len(data)))
        self.data = data
        self.meta_only = False

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        dname = line['SUB_DATASET']
        idx = line['original_index']
        org_data = self.dataset_map[dname].data
        org_line = cp.deepcopy(org_data[org_data['index'] == idx]).iloc[0]
        return self.dataset_map[dname].build_prompt(org_line)

    def dump_image(self, line):
        dname = line['SUB_DATASET']
        idx = line['original_index']
        org_data = self.dataset_map[dname].data
        org_line = cp.deepcopy(org_data[org_data['index'] == idx]).iloc[0]
        return self.dataset_map[dname].dump_image(org_line)

    def _extract_single_metric(self, score_obj, key):
        if isinstance(score_obj, pd.DataFrame):
            if key in score_obj.columns and len(score_obj):
                return float(score_obj[key].iloc[0])
            return None
        if isinstance(score_obj, dict):
            v = score_obj.get(key, None)
            return float(v) if v is not None else None
        return None

    def evaluate(self, eval_file, **judge_kwargs):
        data_all = load(eval_file)

        # Split unified prediction file into two subset prediction files.
        for dname in self.SUB_DATASETS:
            tgt = eval_file.replace(self.dataset_name, dname)
            sub = data_all[data_all['SUB_DATASET'] == dname].copy()
            sub.pop('index')
            sub['index'] = sub.pop('original_index')
            sub.pop('SUB_DATASET')
            dump(sub, tgt)

        # Evaluate subsets with their native logic.
        gen_file = eval_file.replace(self.dataset_name, 'SIUO_GEN')
        mcq_file = eval_file.replace(self.dataset_name, 'SIUO_MCQ')

        gen_score_obj = self.dataset_map['SIUO_GEN'].evaluate(gen_file, **judge_kwargs)
        mcq_score_obj = self.dataset_map['SIUO_MCQ'].evaluate(mcq_file, **judge_kwargs)

        gen_score = self._extract_single_metric(gen_score_obj, 'overall_avg_combined')
        if gen_score is None:
            gen_score = self._extract_single_metric(gen_score_obj, 'overall_avg_safety_gpt')
        mcq_score = self._extract_single_metric(mcq_score_obj, 'Overall')

        # Equal-weight fusion to output one single SIUO score.
        valid_scores = [x for x in [gen_score, mcq_score] if x is not None]
        siuo_score = sum(valid_scores) / len(valid_scores) if len(valid_scores) else 0.0

        ret = {
            'SIUO_GEN': round(gen_score, 2) if gen_score is not None else None,
            'SIUO_MCQ': round(mcq_score, 2) if mcq_score is not None else None,
            'SIUO': round(siuo_score, 2),
        }

        score = d2df(ret)
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(score, score_file)
        return score
