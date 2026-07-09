import ast
import re

import pandas as pd

from vlmeval.smp import load

FAIL_MSG = 'Failed to obtain answer via API.'

# ************** Answer Evaluation ****************


def get_ICE():
    example_1 = """
Ground truth answer: 502 \n
Predicted answer: The mass of block (B) is:
[
\\boxed{ 50 \\sqrt{101} }
] \n
Judegement: 1
"""

    example_2 = """
Ground truth answer: 46.3 kN \n
Predicted answer: The tension ( T_B ) in the cable is approximately:
[
\\boxed{46300 }
] \n
Judegement: 1
"""

    example_3 = """
Ground truth answer: 12 m/s \n
Predicted answer: The speed of the box after 2.00 seconds is:
[
\\boxed{11.3, \\text{m/s}}
] \n
Judegement: 0
"""

    example_4 = """
Ground truth answer: 36.00 kg \n
Predicted answer: The mass of the hanging block ( m_2 ) must be approximately:
[
\\boxed{36.1, \\text\\{kg\\}}
] \n
Judegement: 1
"""

    example_5 = """
Ground truth answer: 3.2 m \n
Predicted answer: The stuntman and villain slide approximately \\frac\\{10\\}{3.1415} meters**.
Judegement: 1
"""

    return [example_1, example_2, example_3, example_4, example_5]


def get_ICE_MC():
    example_1 = """
Ground truth answer: A \n
Predicted answer: A \n
Judegement: 1
"""

    example_2 = """
Ground truth answer: B \n
Predicted answer: A \n
Judegement: 0
"""

    example_3 = """
Ground truth answer: C \n
Predicted answer: ### Step 1: Calculate ( l_1 )
The lightbulb is ( 2.50, \\text\\{m\\}) above the floor, and the bottom of the mirror is (0.50, \\text\\{m\\}) \
above the floor. The vertical distance from the lightbulb to the bottom of the mirror is:
[
\\Delta y_1 = 2.50, \\text\\{m\\} - 0.50, \\text\\{m\\} = 2.00, \\text\\{m\\}.
] \n
Judegement: 0
"""

    example_4 = """
Ground truth answer: D \n
Predicted answer: The correct option is D. \n
Judegement: 1
"""

    return [example_1, example_2, example_3, example_4]


def build_phyx_gpt4_prompt(line, pred):
    task_description = """
Please read the following example. Given predicted answer and ground truth answer,
compare the these two answers, then ONLY output judegement 1/0 for matched/unmatched at the end of the prompt.
If the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If the given predicted mentions "approximately", then allow the Approximation Error, \
such as 0.49 and approximately 0.5, 0.81 and approximately 0.8. \n
"""
    gt_answer = line['answer']
    prompt = task_description
    examples = get_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += 'Ground truth answer: {} \n'.format(gt_answer)
    prompt += 'Predicted answer: {} \n'.format(pred)
    prompt += 'Judegement:'
    return prompt


def build_phyx_gpt4_prompt_MC(line, pred):
    task_description = """
Please read the following example. Given predicted answer and ground truth answer for Multi-Choice question.
The ground truth answer would be A/B/C/D. The predicted answer would be some words containing A/B/C/D.
Please compare the these two answers, then ONLY output judegement 1/0 for matched/unmatched at the end of the prompt. \n
"""
    gt_answer = line['answer']
    prompt = task_description
    examples = get_ICE_MC()
    for example in examples:
        prompt += example + '\n'
    prompt += 'Ground truth answer: {} \n'.format(gt_answer)
    prompt += 'Predicted answer: {} \n'.format(pred)
    prompt += 'Judegement:'
    return prompt


def mapping_str(input):
    d = {r"\dfrac": r"\frac", r"\pi": "3.14"}
    output = input
    for k, v in d.items():
        try:
            output = output.replace(k, v)
        except Exception:
            pass
    return output


def safe_literal_eval(s):
    s = s.strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    if not s.startswith("{"):
        s = "{" + s
    if not s.endswith("}"):
        s = s + "}"
    s = re.sub(r'([{,]\s*)([^"\{\}\:\,\s]+)\s*:', r'\1"\2":', s)
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def extract_boxed_content(s):
    start = s.find(r'\boxed{')
    if start == -1:
        return None
    content_start = start + len(r'\boxed{')
    rest = s[content_start:]
    depth = 0
    for i, ch in enumerate(rest):
        if ch == '{':
            depth += 1
        elif ch == '}':
            if depth == 0:
                return rest[:i]
            else:
                depth -= 1
    return None


def PhyX_auxeval(model, line):
    log = ''
    retry = 5

    gt_answer = str(line['answer'])
    prediction = line['prediction']

    # try extract final answer using re rules
    tmp = PhyX_process_line(line)

    if tmp["extracted"] == "Fail to Call API":
        log += "Fail to Call API"
        prediction = "Fail to Call API"
        return dict(log=log, res=0, extracted=prediction)

    if tmp["extracted"] != "SAME as predict":
        prediction = tmp["extracted"]

    # judge via LLM
    if gt_answer.strip().lower() == prediction.strip().lower():
        return dict(log="Matched at string level", res=1, extracted=prediction)

    prompt = build_phyx_gpt4_prompt(line, prediction)
    for i in range(retry):
        res = model.generate(prompt, temperature=i * 0.5)
        if FAIL_MSG in res:
            log += f'Try {i}: answer and prediction are {gt_answer} and {prediction}, failed to compare.\n'
        else:
            log += 'Compared at semantic level. '
            # print(res)
            if "1" in res or 1 == res:
                log += "Semantic equal via LLM."
                return dict(log=log, res=1, extracted=prediction)
            elif "0" in res or 0 == res:
                log += "LLM judgement {}".format(res)
                return dict(log=log, res=0, extracted=prediction)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res=0, extracted=prediction)


def PhyX_auxeval_MC(model, line):
    log = ''
    retry = 5

    gt_answer = str(line['answer'])
    prediction = line['prediction']

    tmp = PhyX_process_line_MC(line)

    if tmp["extracted"] == "Fail to Call API":
        log += "Fail to Call API"
        prediction = "Fail to Call API"
        return dict(log=log, res=0, extracted=prediction)

    if tmp["extracted"] != "SAME as predict":
        prediction = tmp["extracted"]

    # match at string level
    if gt_answer.strip().lower() == prediction.strip().lower():
        return dict(log="Matched at string level", res=1, extracted=prediction)
    else:
        # prediction is A/B/C/D, then labeled as unmatch
        if prediction.strip() in ["A", "B", "C", "D"]:
            return dict(log="Unmatched at string level", res=0, extracted=prediction)

    prompt = build_phyx_gpt4_prompt_MC(line, prediction)
    for i in range(retry):
        res = model.generate(prompt, temperature=i * 0.5)
        if FAIL_MSG in res:
            log += f'Try {i}: answer and prediction are {gt_answer} and {prediction}, failed to compare.\n'
        else:
            log += 'Compared at semantic level. '
            if "1" in res or 1 == res:
                log += "Semantic equal via LLM."
                return dict(log=log, res=1, extracted=prediction)
            elif "0" in res or 0 == res:
                log += "LLM judgement {}".format(res)
                return dict(log=log, res=0, extracted=prediction)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res=0, extracted=prediction)


def PhyX_acc(result_file):
    data = load(result_file)
    lt = len(data)
    res = {}
    hit = 0
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        if cate in res.keys():
            res[cate].append(item['res'])
        else:
            res[cate] = [item['res']]
        hit += item['res']

    final_res = {}
    final_res["Overall Acc"] = hit / lt
    for k, v in res.items():
        final_res[k] = sum(v) / len(v)
    df = pd.DataFrame(final_res, index=[0])
    return df


def PhyX_process_line(line):
    ret = {}

    answers = str(line['answer'])

    ret["index"] = line["index"]
    ret['gt'] = answers

    # with reasoning, extract content part
    prediction_str = line['prediction']
    with_reasoning = False
    try:
        pred_dict = safe_literal_eval(prediction_str)
        if isinstance(pred_dict, dict) and 'content' in pred_dict and pred_dict['content'] != "":
            ret['pred'] = pred_dict['content'].strip()
            with_reasoning = True
    except Exception:
        pass

    if not with_reasoning:
        ret['pred'] = prediction_str.strip()

    if ret['pred'] == FAIL_MSG:
        ret['match'] = 0
        ret["extracted"] = "Fail to Call API"
        return ret

    boxed_answer = extract_boxed_content(ret['pred'])
    if boxed_answer is not None:
        boxed_answer = mapping_str(boxed_answer)
        ret["extracted"] = boxed_answer
    else:
        pattern = r'\b(?:final\s+answer|correct\s+answer)\b[^:：]*[:：]\s*(.*?)(?=\n\n\n|\Z)'
        flags = re.IGNORECASE | re.DOTALL
        match = re.search(pattern, ret['pred'], flags=flags)
        if match:
            extracted_answer = match.group(1)
            extracted_answer = mapping_str(extracted_answer)
            ret["extracted"] = extracted_answer
        else:
            ret["extracted"] = "SAME as predict"

    if (
        ret['gt'].strip().lower() == ret["extracted"].strip().lower()
        or ret['gt'].strip().lower() == ret["pred"].strip().lower()
        or ret['gt'] in ret['pred']
    ):
        ret['match'] = 1
        return ret

    # unmatch at string level
    ret['match'] = 0
    return ret


def PhyX_process_line_MC(line):
    ret = {}

    answers = str(line['answer'])

    ret["index"] = line["index"]
    ret['gt'] = answers
    ret['pred'] = line['prediction'].strip()

    pattern = r'\b(?:correct|answer|option|Answer|Option|Correct)\b[\s\S]*?([A-D])'
    match = re.search(pattern, ret['pred'])

    if ret['pred'] == FAIL_MSG:
        ret['match'] = 0
        ret["extracted"] = "Fail to Call API"
        return ret

    if match:
        extracted_answer = match.group(1)
        # compare string
        ret["extracted"] = extracted_answer
        if ret['gt'].strip().lower() == extracted_answer.strip().lower():
            ret['match'] = 1
            return ret
    else:
        # try another match strategy
        matches = re.findall(r'([ABCD]):', ret['pred'])
        if matches:
            extracted_answer = matches[-1]
            ret["extracted"] = extracted_answer
            if ret['gt'].strip().lower() == extracted_answer.strip().lower():
                ret['match'] = 1
                return ret
        else:
            ret["extracted"] = "SAME as predict"

    if ret['gt'] + ":" in ret['pred'] or ret['gt'] + "**" in ret['pred'] or "**" + ret['gt'] in ret['pred']:
        ret['match'] = 1
    else:
        ret['match'] = 0

    return ret
