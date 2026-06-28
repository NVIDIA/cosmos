import re
from collections import defaultdict

import pandas as pd

from vlmeval.smp.file import load
from vlmeval.utils import can_infer

FAIL_MSG = 'Failed to obtain answer via API.'

judge_prompts = '''Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer_1}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]". Again, you must output a score by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".''' # noqa e501


def get_gpt4_ICE():
    example_1 = """
Question: 2023年品牌二的三收入比17年品牌一的收入多多少？\n
Model response: 2023年品牌二的收入为420亿元人民币，2017年品牌一的收入为820亿元人民币。\n420 - 820 = -400亿元人民币\n
所以，2023年品牌二的收入比2017年品牌一的收入少了400亿元人民币。\n
Extracted answer: 400亿元人民币
"""

    example_2 = """
Question: What is the total price of all dishes with chicken?\n
Model response: The total price of all dishes with chicken is $103.00.\n
Extracted answer: $103.00
"""

    example_3 = """
Question: 如果2021年的全年营业收入和全年归母净利润的YOY和2022年一样，那么2020年全年归母净利润占全年营业收入的多少？\n
Model response: 2021年的全年营业收入和全年归母净利润的YOY和2022年一样，那么2020年全年归母净利润占全年营业收入的百分比为：\n0.52亿 / 1.25亿 * 100% ≈ 41.60%\n
Extracted answer: 41.60%
"""

    example_4 = """
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_5 = """
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]


def build_ocrr_gpt4_prompt(line):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt


def list_to_dict(lst):
    return {chr(65 + i): val for i, val in enumerate(lst)}


def post_check(line, prefetch=False):
    res = None
    ans = line['answer']
    response = line['prediction'] if prefetch else line['res']
    try:
        if line['question_type'] == 'multi_choice':
            ans = line['answer_option']
            choices = list_to_dict(eval(line['choices']))
            res = can_infer(response, choices)
            if prefetch:
                return res
        else:
            if line['answer_type'] == 'integer':
                res = int(response)
                ans = int(line['answer'])
            elif line['answer_type'] == 'float':
                res = float(response)
                ans = float(line['answer'])
            else:
                res = str(response).replace(' ', '')
                ans = str(ans).replace(' ', '')
    except ValueError:
        pass
    if res == ans:
        return res if prefetch else True
    else:
        return False


def OcrR_auxeval(model, line):
    prompt = build_ocrr_gpt4_prompt(line)
    log = ''
    retry = 5

    reason_prompt = judge_prompts.format(question=line['question'], ref_answer_1=line['reasoning'], answer=line['prediction']) # noqa e501
    for i in range(6):
        reason_score = model.generate(reason_prompt, temperature=i * 0.3)
        match = re.search(r'\[\[(\d+)\]\]', reason_score)
        if match is not None:
            break
    reason_score = int(match.group(1)) / 10

    if post_check(line, prefetch=True):
        res = post_check(line, prefetch=True)
        return dict(log='Prefetch succeed', res=res, reason_score=reason_score)

    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)
        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log=log, res=res, reason_score=reason_score)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='', reason_score=0.0)


def OcrR_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    fetch = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    tot_rp = defaultdict(lambda: 0)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        cate = item['task']
        tot['Overall'] += 1
        tot[cate] += 1
        if item['log'] == 'Prefetch succeed':
            fetch['Overall'] += 1
            fetch[cate] += 1
        if post_check(item, prefetch=False):
            hit['Overall'] += 1
            hit[cate] += 1

    for i in range(lt):
        item = data.iloc[i]
        cate = item['task']
        tot_rp['Overall_RP'] += item['reason_score']
        tot_rp[cate + '_RP'] += item['reason_score']

    res = defaultdict(list)
    for k in tot.keys():
        res['Task'].append(k)
        res['tot'].append(tot[k])
        res['prefetch'].append(fetch[k])
        res['hit'].append(hit[k])
        res['prefetch_rate'].append(fetch[k] / tot[k] * 100)
        res['acc'].append(hit[k] / tot[k] * 100)

    for k in tot_rp.keys():
        res['Task'].append(k)
        res['tot'].append(tot[k.replace('_RP', '')])
        res['prefetch'].append(0)
        res['hit'].append(0)
        res['prefetch_rate'].append(0)
        res['acc'].append(tot_rp[k] / tot[k.replace('_RP', '')] * 100)

    res = pd.DataFrame(res)
    return res
