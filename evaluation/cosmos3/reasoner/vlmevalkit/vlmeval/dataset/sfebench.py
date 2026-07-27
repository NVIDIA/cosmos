import os.path as osp
import string
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from vlmeval.smp import dump, load
from vlmeval.smp.file import get_intermediate_file_path
from vlmeval.utils import track_progress_rich
from .image_vqa import ImageVQADataset
from .utils.judge_util import build_judge

EVAL_TEMPLATE = """
You are a strict evaluator assessing answer correctness. You must score the model's prediction on a scale from 0 to 9.
0 represents an entirely incorrect answer and 9 indicates a highly correct answer.

# Input
Question
{question}
Ground Truth Answer
{answer}
Model Prediction
{prediction}

# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer
from it.
- For multiple-choice questions: Assign a higher score if the predicted answer matches the
ground truth, either by option letters or content. Include partial credit for answers that are
close in content.
- For exact match and open-ended questions:
    - Assign a high score if the prediction matches the answer semantically, considering variations in format.
    - Deduct points for partially correct answers or those with incorrect additional information.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Scoring Guide
Provide a single integer from 0 to 9 to reflect your judgment of the answer's correctness.
# Strict Output format example
4
"""


def report_score(df):
    # assert group in [None, 'category']
    res = defaultdict(list)

    if 'split' in df:
        splits = list(set(df['split']))
        res['split'] = splits
    else:
        df['split'] = ['none'] * len(df)
        res['split'] = ['none']

    for group in [None, 'category']:
        if group is None:
            res['Overall'] = [np.mean(df[df['split'] == sp]['score']) / 9 * 100 for sp in res['split']]
        elif group not in df:
            continue
        else:
            abilities = list(set(df[group]))
            abilities.sort()
            for ab in abilities:
                sub_df = df[df[group] == ab]
                res[ab] = [np.mean(sub_df[sub_df['split'] == sp]['score']) / 9 * 100 for sp in res['split']]
    return pd.DataFrame(res)


def make_prompt(line):
    question = line['question']
    answer = line['answer']
    tmpl = EVAL_TEMPLATE
    prompt = tmpl.format(
        question=question,
        answer=answer,
        prediction=line['prediction']
    )
    return prompt


def SFE_auxeval(model, data):
    if isinstance(data, pd.DataFrame) and len(data) > 1:
        lt = len(data)
        for i in range(lt):
            total_score = 0
            item = data.iloc[i]
            prompt = make_prompt(item)
            retry = 3
            for j in range(retry):
                output = model.generate(prompt, temperature=0.5 * j)
                if output.isdigit() and 0 <= int(output) <= 9:
                    total_score += int(output)
                    break
        avg_score = total_score / lt
        return dict(score=avg_score, log='Success to Judge')
    else:
        item = data.iloc[0] if isinstance(data, pd.DataFrame) else data
        prompt = make_prompt(item)
        retry = 3
        for i in range(retry):
            output = model.generate(prompt, temperature=0.5 * i)
            if output.isdigit() and 0 <= int(output) <= 9:
                return dict(score=int(output), log='Success to Judge')
        return dict(score=0, log='Fail to Judge')


class SFE(ImageVQADataset):

    DATASET_URL = {
        'SFE': 'https://opencompass.openxlab.space/utils/VLMEval/SFE.tsv',
        'SFE-zh': 'https://opencompass.openxlab.space/utils/VLMEval/SFE-zh.tsv'
    }

    DATASET_MD5 = {
        'SFE': 'd4601425e7c9a62446b63a1faee17da5',
        'SFE-zh': '3e0250b7f30da55bf8f7b95eace66d82'
    }

    MCQ_PROMPT = (
        "You are an expert in {discipline} and need to solve the following question. "
        + "The question is a multiple-choice question. "
        + "Answer with the option letter from the given choices."
    )

    EXACT_MATCH_PROMPT = (
        "You are an expert in {discipline} and need to solve the following question. "
        + "The question is an exact match question. Answer the question using a single word or phrase."
    )

    OPEN_QUESTION_PROMPT = (
        "You are an expert in {discipline} and need to solve the following question. "
        + "The question is an open-ended question. Answer the question using a phrase."
    )

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)

        question_type = line['question_type']
        field = line['category']
        question = line['question']

        if question_type == 'exact_match':
            prompt = self.EXACT_MATCH_PROMPT.format(discipline=field)
            question = prompt + " " + question
        elif question_type == 'mcq':
            prompt = self.MCQ_PROMPT.format(discipline=field)
            question = prompt + " " + question
            if not pd.isna(line['A']):
                question += '\nChoices are:\n'
                for ch in string.ascii_uppercase[:15]:
                    if not pd.isna(line[ch]):
                        question += f'{ch}. {line[ch]}\n'
                    else:
                        break
        elif question_type == 'open_ended':
            prompt = self.OPEN_QUESTION_PROMPT.format(discipline=field)
            question = prompt + " " + question

        prompt_segs = question.split('<image>')
        assert len(prompt_segs) == len(tgt_path) + 1
        msgs = []
        for i in range(len(tgt_path)):
            text = prompt_segs[i].strip()
            if text != '':
                msgs.append(dict(type='text', value=text))
            msgs.append(dict(type='image', value=tgt_path[i]))
        text = prompt_segs[-1].strip()
        if text != '':
            msgs.append(dict(type='text', value=text))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        _ = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        storage = get_intermediate_file_path(eval_file, '_judge')
        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        if not osp.exists(storage):
            ans_map = {} if not osp.exists(tmp_file) else load(tmp_file)

            model = judge_kwargs.pop('model', 'gpt-4o-1120')
            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(model=model, **judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    model = None

            if model is not None:
                if 'g_index' not in data:
                    lines = [data.iloc[i] for i in range(len(data))]
                    indices = [x['index'] for x in lines if x['index'] not in ans_map]
                    lines = [x for x in lines if x['index'] not in ans_map]
                    tups = [(model, line) for line in lines]
                else:
                    main_data = data[[x == y for x, y in zip(data['index'], data['g_index'])]]
                    lines = [data[data['g_index'] == x] for x in main_data['index']]
                    indices = [x.iloc[0]['g_index'] for x in lines if x.iloc[0]['g_index'] not in ans_map]
                    lines = [x for x in lines if x.iloc[0]['g_index'] not in ans_map]
                    tups = [(model, x) for x in lines]
                    data = main_data

                if len(lines):
                    res = track_progress_rich(
                        SFE_auxeval, tups, nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file)
                    for k, v in zip(indices, res):
                        ans_map[k] = v

            judge_results = [ans_map[x] for x in data['index']]
            data['score'] = [x['score'] for x in judge_results]
            dump(data, storage)
        data = load(storage)
        score = report_score(data)

        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(score, score_file)
        return score
