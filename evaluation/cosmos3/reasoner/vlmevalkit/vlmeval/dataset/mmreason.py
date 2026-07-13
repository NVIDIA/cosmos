import pandas as pd

from vlmeval.smp.file import load

FAIL_MSG = 'Failed to obtain answer via API.'


def get_gpt4_extract_ICE():
    example_1 = """
1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)
""" # noqa

    example_2 = """
2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D
""" # noqa

    example_3 = """
3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)
""" # noqa

    example_4 = """
4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null
""" # noqa

    example_5 = """
5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3
""" # noqa

    example_6 = """
6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]


def get_gpt4_score_ICE():
    example_1 = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent.
Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 70°
    Expression 2: 120

No

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: d
    Expression 2: b

No

    Expression 1: 1.881 yr
    Expression 2: 1.92

No

    Expression 1: 44.1 mm
    Expression 2: 0.032 m

No

    Expression 1: 1,4,3,2
    Expression 2: 1,2,3,4

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes

    Expression 1: R28
    Expression 2: 28
Yes
(ignore currency symbols if the numeric value is the same)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()

    return example_1


def build_MMReason_gpt4_extract_prompt(line):
    task_description = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
""" # noqa
    prediction = str(line['prediction'])
    demo_prompt = task_description
    examples = get_gpt4_extract_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"Model response: '{prediction}'\nExtracted Answer: "
    full_prompt = f'{demo_prompt}7.\n{test_prompt}'

    return full_prompt


def build_MMReason_gpt4_score_prompt(line):
    full_prompt = get_gpt4_score_ICE()
    full_prompt = full_prompt % {"expression1": line['extract'], "expression2": line['answer']}

    return full_prompt


def post_check_score(line, prefetch=False):
    ans = str(line['answer']).strip()
    response = str(line['extract']).strip()

    if response == ans:
        return response if prefetch else True
    else:
        return False


def MMReason_auxeval_extract(model, line):
    prompt = build_MMReason_gpt4_extract_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log_extract=log, extract=res)
    log += 'All 5 retries failed.\n'
    return dict(log_extract=log, extract='')


def MMReason_auxeval_score(model, line):
    prompt = build_MMReason_gpt4_score_prompt(line)
    log = ''
    retry = 5
    if post_check_score(line, prefetch=True):
        res = post_check_score(line, prefetch=True)
        return dict(log_score='Prefetch succeed', score=True)
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res or res.strip() not in ['Yes', 'No']:
            log += f'Try {i}: output is {prediction}, res is {res}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log_score=log, score=res == 'Yes' or res.strip() == 'Yes')
    log += 'All 5 retries failed.\n'
    return dict(log_score=log, score=False)


def get_acc_with_condition(res_pd, key, value):
    """
    Calculate the accuracy of predictions with a specific condition
    """
    total_pd = res_pd[res_pd[key] == value]
    correct_pd = total_pd[total_pd['score']]
    acc = '{:.2f}'.format(len(correct_pd) / len(total_pd) * 100) if len(total_pd) > 0 else '0.00'
    return len(correct_pd), len(total_pd), acc


def MMReason_acc(result_file):
    df = load(result_file)
    total = len(df)
    correct = sum(1 for _, row in df.iterrows() if row['score'])
    accuracy = round(correct / total * 100, 2)
    scores = {'average': {'discipline': 'Overall', 'accuracy': accuracy, 'correct': correct, 'total': total}}

    discipline_total = dict()
    discipline_correct = dict()
    for _, row in df.iterrows():
        if row['discipline'] not in discipline_total:
            discipline_total[row['discipline']] = 1
        elif row['discipline'] in discipline_total:
            discipline_total[row['discipline']] += 1

        if row['discipline'] not in discipline_correct:
            discipline_correct[row['discipline']] = 0

        if row['score']:
            discipline_correct[row['discipline']] += 1

    for key in discipline_total.keys():
        scores[key] = {
            'discipline': key, 'accuracy': discipline_correct[key] / discipline_total[key],
            'correct': discipline_correct[key], 'total': discipline_total[key]
        }

    return pd.DataFrame.from_dict(scores, orient='index')
