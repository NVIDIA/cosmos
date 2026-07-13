from functools import partial

import numpy as np

from vlmeval.smp import d2df, dump, get_intermediate_file_path, load, mp
from .image_base import ImageBaseDataset


class PlotQA(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {'PlotQA': 'https://opencompass.openxlab.space/utils/VLMEval/PlotQA.tsv'}
    DATASET_MD5 = {'PlotQA': '473ade193d72b01d7b1c9fc615b1b49a'}

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        msgs = super().build_prompt(line)
        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        from .utils.vqa_eval import process_line

        print("Evaluating PlotQA results...")
        data = load(eval_file)

        # Given it shares similarities with ChartQA, we use 'relaxed_accuracy'.
        # Ensure predictions are strings
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        lt = len(data)
        pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]

        # Use relaxed_accuracy as used in ChartQA
        res = pool.map(partial(process_line, method='relaxed_accuracy'), lines)

        data['eval_gt'] = [r['gt'] for r in res]
        data['eval_pred'] = [r['pred'] for r in res]
        data['eval_match'] = [r['match'] for r in res]

        dump(data, get_intermediate_file_path(eval_file, '_results'))

        # Calculate Overall Score (using ChartQA logic: max match per item if multiple answers, else just match)
        hits = [np.max(r['match']) for r in res]
        score = np.mean(hits) * 100

        ret = {'Overall': score}
        ret = d2df(ret)
        ret.round(2)

        dump(ret, get_intermediate_file_path(eval_file, '_acc'))
        return ret
