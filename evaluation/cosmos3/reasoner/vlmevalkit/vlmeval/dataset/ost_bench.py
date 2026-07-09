import os.path as osp

import numpy as np
import pandas as pd
from tqdm import tqdm

from vlmeval.smp import toliststr
from .image_base import ImageBaseDataset

OST_INSTRUCTION = """
# OST-Bench evaluation on VLMEvalkit

## Data Preparation

1. Download the images of OST-Bench dataset from [kaggle](https://www.kaggle.com/datasets/jinglilin/ostbench/data) \
or [huggingface](https://huggingface.co/datasets/rbler/OST-Bench) and put them under `LMUDATA`. Below shows the \
expected folder structure.

```
LMUDATA/
├──OST.tsv
├──images/
├────OST/
├──────1mp3d_0004_region0
├──────1mp3d_0004_region10
├──────...
```

## Model Config
When evaluating the performance of models `llava/qwenvl/InternVL` series, set `max_new_tokens` to 4096 to \
ensure complete reproducibility of the results.  Additionally, when using the LLaVA_OneVision series of models, \
set `self.model.config.image_aspect_ratio` = 'pt'  (under `vlmeval/vlm/llava/llava.py`).
"""

num_mapping = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}  # noqa: E501
skip_type = ['None']


def name_mapping(raw_name):
    if raw_name == 'A_room-size(float)':
        return 'None'
    new_name = raw_name.replace('float', 'Estimation')

    if 'quantity' not in new_name:
        new_name = new_name.replace('int', 'Temporal-loc')
    else:
        new_name = new_name.replace('int', 'Counting')

    new_name = new_name.replace('option', 'Judgement')
    new_name = new_name.replace('A_object-', 'Agent_visible_info-')
    new_name = new_name.replace('AO_', 'Agent_object_spatial-')
    new_name = new_name.replace('A_', 'Agent_state-')
    return new_name


def Judgement_evalution(pred, gt, options):
    """Evaluation for Judgement Questions.
    Args:
        pred (str): Output of the model.
        gt (str): _description_
        options (list of str): All possible options.
    """
    gt = gt.replace('\"', '')
    assert gt in options

    def longest_common_subsequence(str1, str2):
        m, n = len(str1), len(str2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
    pred = str(pred)
    option_idx = np.argmax([
        longest_common_subsequence(pred.strip().lower(), option.strip().lower()) for option in options
    ])
    return float(gt == options[option_idx])


def Estimation_evaluation(pred, gt):
    """Evaluation for Estimation Questions (follow VSI)
    Args:
        pred (_type_): _description_
        gt (_type_): _description_
        anchor (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    try:
        pred = float(pred)
        gt = float(gt)
    except Exception:
        return 0.0
    delta_ratio = abs(gt - pred) / abs(gt)
    citerion_list = [0.5 + 0.05 * i for i in range(10)]
    metric = 0.1 * sum([int(delta_ratio < 1 - citerion) for citerion in citerion_list])
    return metric


def Enumeration_evalution(pred, gt):
    """Evaluation for Temporal-Loc and Counting.
    Args:
        pred (_type_): _description_
        gt (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        if isinstance(pred, str):
            for word in num_mapping:
                if word in pred.lower():
                    pred = num_mapping[word]
                    break
        pred = int(pred)
        gt = int(gt)
    except Exception:
        return 0.0
    return float(pred == gt)


class OST_evaluator:

    def __init__(self):
        pass

    def evaluation(self, sample):

        if sample['type'] in skip_type or 'pred' not in sample.keys():
            return sample

        if 'Estimation' in sample['type']:
            eval_function = Estimation_evaluation
        elif 'Judgement' in sample['type']:
            eval_function = Judgement_evalution
        elif 'Counting' in sample['type'] or 'Temporal-loc' in sample['type']:
            eval_function = Enumeration_evalution
        else:
            return sample

        if 'Judgement' in sample['type']:
            sample['metric'] = eval_function(sample['pred'], sample['answer'], sample['option'])
        else:
            sample['metric'] = eval_function(sample['pred'], sample['answer'])

        return sample


def process_answer(raw_answer):
    if isinstance(raw_answer, list):
        raw_answer = raw_answer[0]

    raw_answer = raw_answer.replace('\'', '\"')
    try:
        if 'reason' not in raw_answer or 'answer' not in raw_answer:
            answer_text = raw_answer
        elif '\"answer\":' in raw_answer:
            answer_text = raw_answer.split('\"answer\":')[1].split('\"reason\"')[0].split('\n')[0].strip().strip(',').strip()  # noqa
        elif 'answer:' in raw_answer:
            answer_text = raw_answer.split('answer:')[1].split('reason')[0].split('\n')[0].strip().strip(',').strip()
        else:
            answer_text = ''
        answer_text = answer_text.strip('\"')

    except Exception:
        print('answer format error:', raw_answer)
    return answer_text


def collect_results(static_results):
    full_dict = {}
    full_dict['Agent_object_spatial-direction(Judgement)'] = sum([static_results[f'Agent_object_spatial-direction(Judgement{i})'] for i in range(1, 4)]) / 3.0  # noqa
    full_dict['Agent_object_spatial-distance(Judgement)'] = sum([static_results[f'Agent_object_spatial-distance(Judgement{i})'] for i in range(1, 4)]) / 3.0  # noqa
    full_dict['Agent_visible_info-existence(Temporal-loc)'] = (static_results['Agent_visible_info-existence(Temporal-loc1)'] + static_results['Agent_visible_info-existence(Temporal-loc2)']) / 2.0  # noqa
    for type_name in static_results:
        if 'Agent_object_spatial-direction(Judgement' in type_name or 'Agent_object_spatial-distance(Judgement' in type_name or 'Agent_visible_info-existence(Temporal-loc' in type_name:  # noqa
            continue
        full_dict[type_name] = static_results[type_name]

    overall_dict = {}
    overall_dict['A_state(Judge)'] = (full_dict['Agent_state-orientation(Judgement)'] + full_dict['Agent_state-position(Judgement)']) / 2.0  # noqa
    overall_dict['A_state(Esti)'] = (full_dict['Agent_state-orientation(Estimation)'] + full_dict['Agent_state-position(Estimation)']) / 2.0  # noqa

    overall_dict['A_info(Judge)'] = (full_dict['Agent_visible_info-existence(Judgement)'] + full_dict['Agent_visible_info-order(Judgement)'] + full_dict['Agent_visible_info-diversity(Judgement)']) / 3.0  # noqa
    overall_dict['A_info(Temp)'] = full_dict['Agent_visible_info-existence(Temporal-loc)']
    overall_dict['A_info(Count)'] = full_dict['Agent_visible_info-quantity(Counting)']

    overall_dict['AO(Esti)'] = (full_dict['Agent_object_spatial-direction(Estimation)'] + full_dict['Agent_object_spatial-distance(Estimation)']) / 2.0  # noqa
    overall_dict['AO(Temp)'] = (full_dict['Agent_object_spatial-direction(Temporal-loc)'] + full_dict['Agent_object_spatial-distance(Temporal-loc)']) / 2.0  # noqa
    overall_dict['AO(Judge)'] = (full_dict['Agent_object_spatial-direction(Judgement)'] + full_dict['Agent_object_spatial-distance(Judgement)']) / 2.0  # noqa

    overall_dict['Judgement'] = (full_dict['Agent_object_spatial-direction(Judgement)'] + full_dict['Agent_object_spatial-distance(Judgement)'] + full_dict['Agent_visible_info-existence(Judgement)'] + full_dict['Agent_visible_info-order(Judgement)'] + full_dict['Agent_visible_info-diversity(Judgement)'] + full_dict['Agent_state-orientation(Judgement)'] + full_dict['Agent_state-position(Judgement)']) / 7.0  # noqa
    overall_dict['Estimation'] = (full_dict['Agent_object_spatial-direction(Estimation)'] + full_dict['Agent_object_spatial-distance(Estimation)'] + full_dict['Agent_state-orientation(Estimation)'] + full_dict['Agent_state-position(Estimation)']) / 4.0  # noqa
    overall_dict['Temporal-loc'] = (full_dict['Agent_visible_info-existence(Temporal-loc)'] + full_dict['Agent_object_spatial-direction(Temporal-loc)'] + full_dict['Agent_object_spatial-distance(Temporal-loc)']) / 3.0  # noqa
    overall_dict['Counting'] = full_dict['Agent_visible_info-quantity(Counting)']

    overall_dict['Overall'] = (overall_dict['A_state(Judge)'] + overall_dict['A_state(Esti)'] + overall_dict['A_info(Judge)'] + overall_dict['A_info(Temp)'] + overall_dict['A_info(Count)'] + overall_dict['AO(Judge)'] + overall_dict['AO(Esti)'] + overall_dict['AO(Temp)']) / 8.0  # noqa

    return full_dict, overall_dict


class ImageILDataset(ImageBaseDataset):

    DATASET_URL = {'OST': 'https://opencompass.openxlab.space/utils/VLMEval/OST.tsv'}
    DATASET_MD5 = {'OST': 'd5d528680379cf2795a47723ce0906e2'}

    TYPE = 'IL'

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
            tgt_path_w_root = [osp.join(self.img_root, im) for im in tgt_path]
            if all([osp.exists(pth) for pth in tgt_path_w_root]):
                tgt_path = tgt_path_w_root
        else:
            tgt_path = self.dump_image(line)
        for im_path in tgt_path:
            assert osp.exists(im_path), OST_INSTRUCTION

        q = line['question']

        pics_number = 0
        if '<ImageHere>' in q:
            content = []
            tag_number = q.count('<ImageHere>')
            images = tgt_path[pics_number: pics_number + tag_number]
            pics_number += tag_number
            q_split = q.split('<ImageHere>')
            for i in range(tag_number):
                qsp, im = q_split[i], images[i]
                if qsp != '':
                    content.append(dict(type='text', value=qsp))
                content.append(dict(type='image', value=im))
            if q_split[-1] != '':
                content.append(dict(type='text', value=q_split[-1]))
        else:
            content = [dict(type='text', value=q)]

        return content


class OSTDataset(ImageILDataset):

    def evaluate(self, eval_file, **judge_kwargs):
        sum_ = 0
        df = pd.read_excel(eval_file, sheet_name=0)

        total_cnt = {}
        correct_cnt = {}

        st_eval = OST_evaluator()

        for index in tqdm(df.index):
            sample = {}
            type = name_mapping(df.at[index, 'type'])
            sample['type'] = type

            if type == 'None':
                continue
            df.at[index, 'prediction'] = str(df.at[index, 'prediction'])
            if 'Failed' in df.at[index, 'prediction'] or len(df.at[index, 'prediction']) == 0:
                continue
            sample['pred'] = process_answer(df.at[index, 'prediction'])
            sample['answer'] = df.at[index, 'answer']
            sample['option'] = eval(df.at[index, 'option'])

            sample = st_eval.evaluation(sample)

            if 'metric' not in sample.keys():
                continue
            if type not in total_cnt:
                total_cnt[type] = 0
                correct_cnt[type] = 0
            total_cnt[type] += 1
            sum_ += 1
            correct_cnt[type] += sample['metric']
        static_results = {k: correct_cnt[k] / total_cnt[k] for k in total_cnt.keys()}
        full_dict, overall_dict = collect_results(static_results)

        print('-------------------------- Evaluation Result--------------------------')
        print('Total Samples:', sum_)
        print('Overall Accuracy:', overall_dict['Overall'])
        print('-----------------------------------------------------------------------')
        print('Judgement Accuracy:', overall_dict['Judgement'])
        print('Estimation Accuracy:', overall_dict['Estimation'])
        print('Temporal-loc Accuracy:', overall_dict['Temporal-loc'])
        print('Counting Accuracy:', overall_dict['Counting'])
        print('-----------------------------------------------------------------------')
        print('Agent State(Judgement) Accuracy:', overall_dict['A_state(Judge)'])
        print('Agent State(Estimation) Accuracy:', overall_dict['A_state(Esti)'])
        print('Agent Visible Info(Judgement) Accuracy:', overall_dict['A_info(Judge)'])
        print('Agent Visible Info(Temporal-loc) Accuracy:', overall_dict['A_info(Temp)'])
        print('Agent Visible Info(Counting) Accuracy:', overall_dict['A_info(Count)'])
        print('Agent Object Spatial(Judgement) Accuracy:', overall_dict['AO(Judge)'])
        print('Agent Object Spatial(Estimation) Accuracy:', overall_dict['AO(Esti)'])
        print('Agent Object Spatial(Temporal-loc) Accuracy:', overall_dict['AO(Temp)'])
        print('-----------------------------------------------------------------------')

        return static_results
