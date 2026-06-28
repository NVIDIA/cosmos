import os.path as osp
from collections import defaultdict

import numpy as np

from vlmeval.smp import d2df, dump, load
from vlmeval.smp.file import get_intermediate_file_path
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge
from .utils.chartx_eval import ChartX_auxeval, chartx_scrm_eval


class ChartX(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'ChartX': 'https://opencompass.openxlab.space/utils/VLMEval/ChartX.tsv'}
    DATASET_MD5 = {'ChartX': 'ffeb5bc765c9a7a78ef326410903d04d'}

    def __init__(self, dataset='ChartX', **kwargs):
        super().__init__(dataset=dataset, **kwargs)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        msgs = super().build_prompt(line)
        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        print("Evaluating ChartX results...")
        data = load(eval_file)

        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        # --- 1. Structure Extraction (SCRM) ---
        scores = {}
        if 'category' in data.columns:
            se_data = data[data['category'] == 'structure']
            if len(se_data) > 0:
                print(f"Evaluating Structure Extraction ({len(se_data)} samples)...")
                preds = se_data['prediction'].tolist()
                gts = se_data['answer'].tolist()
                scrm_res = chartx_scrm_eval(preds, gts)
                scores['SCRM'] = scrm_res['SCRM']
                scores['AP50_Strict'] = scrm_res['AP50_Strict']

        # --- 2. GPT Evaluation ---
        # Mandatory GPT evaluation for QA, Desc, Sum, Redraw
        try:
            model = build_judge(max_tokens=128, **judge_kwargs)
            judge_working = model.working()
        except BaseException:
            judge_working = False

        if judge_working:
            print("Running GPT-based evaluation for non-structure tasks...")
            target_indices = []
            if 'category' in data.columns:
                target_indices = data[data['category']
                                      != 'structure'].index.tolist()
            else:
                target_indices = data.index.tolist()

            # Prepare for auxiliary eval
            tmp_gpt = get_intermediate_file_path(
                eval_file, '_gpt_eval', 'pkl')

            ans = {}
            if osp.exists(tmp_gpt):
                ans = load(tmp_gpt)

            # Identify pending items
            pending_indices = [i for i in target_indices if i not in ans]
            if pending_indices:
                tups = [(model, data.iloc[i]) for i in pending_indices]

                track_progress_rich(
                    ChartX_auxeval,
                    tups,
                    nproc=judge_kwargs.get('nproc', 4),
                    chunksize=judge_kwargs.get('nproc', 4),
                    keys=pending_indices,
                    save=tmp_gpt
                )
                ans = load(tmp_gpt)

            # Aggregate scores
            gpt_scores_list = []
            for idx in target_indices:
                if idx in ans:
                    gpt_scores_list.append(ans[idx]['score'])
                else:
                    gpt_scores_list.append(0)

            scores['GPT_Overall'] = np.mean(
                gpt_scores_list) if gpt_scores_list else 0

            # Breakdown
            if 'category' in data.columns:
                cat_scores = defaultdict(list)
                for idx in target_indices:
                    cat = data.iloc[idx]['category']
                    if idx in ans:
                        cat_scores[cat].append(ans[idx]['score'])

                for cat, val_list in cat_scores.items():
                    scores[f'{cat}_GPT'] = np.mean(val_list) if val_list else 0
        else:
            print(
                "Warning: OpenAI API not working or Key missing. Skipping GPT-based evaluation.")
            print(DEBUG_MESSAGE)
            scores['GPT_Overall'] = 0.0
            if 'category' in data.columns:
                for cat in data['category'].unique():
                    if cat != 'structure':
                        scores[f'{cat}_GPT'] = 0.0

        # Save score table
        ret = d2df(scores)
        ret.round(2)
        dump(ret, get_intermediate_file_path(eval_file, '_acc'))

        return ret
