import json
import os.path as osp
import re

import pandas as pd
from tqdm import tqdm

from vlmeval import dump, load
from vlmeval.dataset.utils.bmmr_grade import math_equal


def extract_boxed_content(text):
    result = []
    i = 0
    pattern = r'\boxed{'
    len_pattern = len(pattern)

    while i < len(text):
        # 搜索模式 \boxed{
        if text[i:i + len_pattern] == pattern:
            start = i + len_pattern
            brace_level = 1
            content = []
            i = start

            # 逐字符遍历并跟踪括号层级
            while i < len(text) and brace_level > 0:
                if text[i] == '{':
                    brace_level += 1
                elif text[i] == '}':
                    brace_level -= 1
                if brace_level > 0:  # 最后一个}不加入内容
                    content.append(text[i])
                i += 1

            # 如果找到闭合括号则保存结果
            if brace_level == 0:
                result.append(''.join(content))
        else:
            i += 1
    if len(result) == 0:
        return ['No Answer']
    return result


def extract_text(input_string):
    # 使用正则表达式提取 \text{} 中的文本
    pattern = r'\\text{(.*?)}'
    matches = re.findall(pattern, input_string)
    return matches


def extract_uppercase(s):
    # 使用列表推导式来提取大写字母
    uppercase_letters = [char for char in s if char.isupper()]
    # 将列表转换为字符串
    return uppercase_letters


SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''), ('\\%', '%'),
    (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
    ('\\text{and}', ','), ('\\text{m}', '\\text{}')
]
REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
    'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
    '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}',
    r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots'
]


def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = str(final_answer).split('=')[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(
        r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(
        r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer


def open_end_verify(ref, cand):
    gt_ans = ref
    if type(gt_ans) is list:
        gt_ans = gt_ans[0]
    # gt_ans = extract_answer(gt_ans)
    gt_ans = normalize_final_answer(gt_ans)
    if len(gt_ans) == 0:
        return {'acc': 0}

    ans = extract_boxed_content(cand)[-1]
    ans = normalize_final_answer(ans)
    # raw_judge = check_is_correct(ans, gt_ans)

    raw_judge = False
    # raw_judge = gt_ans.lower() in ans.lower()
    if not raw_judge:
        # ans = extract_boxed_content(raw_ans.split('Answer###')[-1])[0]

        raw_judge = math_equal(gt_ans, ans)

    return {'acc': raw_judge}


def multichoice_verify(ref, cand):
    correct_cnt = 0
    correct_ness = []
    gt_ans = ref
    if len(gt_ans) == 0:
        # correct_ness = [False] * len(data['model_answer_answer']) # data['model_answer_answer'] is the rollout answers
        return {'acc': 0}

    ans = extract_uppercase(extract_boxed_content(cand.split('Answer###')[-1])[0])
    choice_correct_cnt = 0
    if len(gt_ans) == 1 and gt_ans[0].startswith('[') and gt_ans[0].endswith(']'):
        gt_ans = gt_ans[0]
        gt_ans = gt_ans.replace("'", "\"")
        gt_ans = json.loads(gt_ans)
    if len(ans) == len(gt_ans):
        for c in ans:
            if c in gt_ans:
                choice_correct_cnt += 1
        correct_cnt += choice_correct_cnt / len(gt_ans)
    if choice_correct_cnt / len(gt_ans) == 1:
        correct_ness.append(True)
    else:
        correct_ness.append(False)

    return {'acc': correct_ness[0]}


def get_acc_for_reference_based_metrics(
    references, candidates, image_id_list, task_types, reference_based_metrics_file
):
    """
    Get the accuracy for the reference-based metrics.
    """
    existing_data = load(reference_based_metrics_file) if osp.exists(reference_based_metrics_file) else {}
    idx = 1
    print(f"Calculating metrics for {len(references)} samples")
    assert len(references) == len(candidates) == len(image_id_list)
    for ref, cand, image_id, task_type in tqdm(zip(references, candidates, image_id_list, task_types)):
        if not cand.strip():
            print(cand)
            continue
        default_acc_score = {'acc': 0.0}
        if image_id not in existing_data:
            existing_data[image_id] = {}
        acc_score = existing_data.get(image_id, {}).get('acc_score', default_acc_score)
        if acc_score == default_acc_score:
            if task_type is None:
                task_type = 'open_end'
            if task_type == "open_end":
                acc_score = open_end_verify(ref, cand)
            elif task_type == "mc":
                acc_score = multichoice_verify(ref, cand)
            else:
                raise ValueError(f"Task type {task_type} not supported")
            existing_data[image_id]['acc_score'] = acc_score

        if idx % 50 == 0:
            print(f"Saving 50 samples to {reference_based_metrics_file}")
            dump(existing_data, reference_based_metrics_file)

        idx += 1
    dump(existing_data, reference_based_metrics_file)
    print(f"Saved all samples to {reference_based_metrics_file}")

    return existing_data


def merge_rating(refer_based_metrics_output_file_name):
    refer_based_metrics_output_file = load(refer_based_metrics_output_file_name)

    refer_based_metrics_output_file['acc_score'] = None  # 初始化列
    for idx, item in refer_based_metrics_output_file.iterrows():
        ref_based_metrics = eval(item['reference_based_metrics'])
        refer_based_metrics_output_file.at[idx, 'acc_score'] = ref_based_metrics['acc_score']['acc']

    df = refer_based_metrics_output_file
    metrics = ['acc_score']
    # 计算cot为True的结果
    cot_true_df = df[df['cot']]
    cot_true_metrics = {
        'acc_score': [cot_true_df[metrics].mean().values[0]]
    }

    cot_false_df = df[~df['cot']]
    cot_false_metrics = {
        'acc_score': [cot_false_df[metrics].mean().values[0]]
    }

    # 计算cot为True时不同language的结果
    cot_lang_df = df[df['cot']].groupby('language')[metrics].mean()
    cot_lang_metrics = {
        'acc_score': cot_lang_df['acc_score'].values
    }

    df['category_id'] = df['category_id'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df['category_id'] = df['category_id'].apply(lambda x: [item[:2] for item in x])

    # 只计算cot=True的数据
    cot_df = df[df['cot']]

    # 为每个数据行创建多行，每个category_id一行
    expanded_rows = []
    for idx, row in cot_df.iterrows():
        for cat_id in row['category_id']:
            new_row = row.copy()
            new_row['category_id'] = cat_id
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)
    category_id_df = expanded_df.groupby('category_id')[metrics].mean()
    category_id_metrics = {
        'acc_score': category_id_df['acc_score'].values
    }

    # 合并所有结果
    result_dict = {
        'CoT': cot_true_metrics['acc_score'],
        'no_CoT': cot_false_metrics['acc_score'],
        'En': [cot_lang_metrics['acc_score'][0]],
        'Zh': [cot_lang_metrics['acc_score'][1]]
    }
    id2name = {"02": "Arts",
               "03": "Soc. Sci.",
               "04": "Bus.",
               "05": "Nat. Sci.",
               "06": "ICTs",
               "07": "Eng.",
               "08": "Agri.",
               "09": "Health",
               "11": "UnClassified"}
    # 添加不同category_id的COT结果
    for cat_id, score in zip(category_id_df.index, category_id_metrics['acc_score']):
        if cat_id != "11":  # 跳过id为11的结果
            result_dict[f'{id2name[cat_id]}'] = [score]
    result_df = pd.DataFrame(result_dict)

    return result_df
