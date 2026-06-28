import re

import pandas as pd

from vlmeval.smp import d2df, dump, get_intermediate_file_path, load, toliststr
from .image_base import ImageBaseDataset


def _extract_option(pred):
    s = str(pred or "")
    boxed = re.findall(r"\\boxed\{\s*([A-Z])\s*\}", s)
    if len(boxed):
        return boxed[-1]
    m = re.findall(r"\(([A-Z])\)", s)
    if len(m):
        return m[-1]
    m = re.findall(r"\b([A-G])\b", s.upper())
    if len(m):
        return m[-1]
    return ""


class SIUOMCQDataset(ImageBaseDataset):
    TYPE = 'MCQ'
    MODALITY = 'IMAGE'
    DATASET_URL = {'SIUO_MCQ': 'https://opencompass.openxlab.space/utils/VLMEval/SIUO_MCQ.tsv'}
    DATASET_MD5 = {'SIUO_MCQ': 'a72a8a5789cbe8c83126890bb3a9a6e9'}

    @classmethod
    def supported_datasets(cls):
        return ['SIUO_MCQ']

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line) if not self.meta_only else toliststr(line['image_path'])
        options = {k: line[k] for k in 'ABCDEFG' if k in line and not pd.isna(line[k])}
        prompt = f"Question: {line['question']}\n"
        if len(options):
            prompt += 'Options:\n'
            for k, v in options.items():
                prompt += f'{k}. {v}\n'
            prompt += 'Please select one option letter only.'
        msgs = [dict(type='image', value=p) for p in tgt_path]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'prediction' in data and 'answer' in data
        data['pred_option'] = [_extract_option(x) for x in data['prediction']]
        data['answer'] = [str(x).strip().upper() for x in data['answer']]
        data['correct'] = [int(p == a) for p, a in zip(data['pred_option'], data['answer'])]

        ret = {'Overall': round(data['correct'].mean() * 100 if len(data) else 0, 2)}
        if 'category' in data:
            for c in sorted(set(data['category'])):
                sub = data[data['category'] == c]
                ret[c] = round(sub['correct'].mean() * 100 if len(sub) else 0, 2)

        detailed_file = get_intermediate_file_path(eval_file, '_detailed', 'xlsx')
        dump(data, detailed_file)
        score = d2df(ret)
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(score, score_file)
        return score
