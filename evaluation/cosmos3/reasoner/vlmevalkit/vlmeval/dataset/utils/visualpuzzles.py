import re
from collections import defaultdict

from vlmeval.smp.file import load

Option_list = ['A', 'B', 'C', 'D']


def extract_last_boxed_content(text):
    stack = []
    last_boxed_content = None
    text = str(text)
    if len(text) < 3:
        return text

    pattern = re.finditer(r'\\boxed\{|[^\\]\}', text)

    try:
        for match in pattern:
            if match.group().endswith(r'\boxed{'):
                stack.append(match.end())
            elif match.group().endswith('}') and stack:
                start = stack.pop()
                if not stack:
                    last_boxed_content = text[start:match.start() + 1]

        if last_boxed_content:
            latex_commands = [r'\text{', r'\rm{', r'\mathbf{', '$']
            for cmd in latex_commands:
                last_boxed_content = last_boxed_content.replace(cmd, '')
            last_boxed_content = last_boxed_content.replace('}', '')

        if (
            "LETTER".lower() in last_boxed_content.lower()
            or "or" in last_boxed_content
            or len(last_boxed_content) > 2
        ):
            last_boxed_content = text

    except Exception:
        last_boxed_content = text

    return 'N' if last_boxed_content is None else last_boxed_content


def extract_lang_content(ans):
    ans = str(ans)
    ans = ans.replace("<|endoftext|>", "")
    for c in Option_list:
        if (
            ans.endswith(f" {c}.")
            or ans.endswith(f" ({c}).")
            or ans.startswith(f"{c}\n")
            or ans.startswith(f"({c})\n")
            or ans.startswith(f"({c}) {c}\n")
        ):
            return c

    lower_ans = ans.lower()
    for flag in [
        "answer:",
        'the final answer is:',
        'the answer is option:',
        'the answer is:',
        'the correct answer is option:',
        'the correct answer is:',
        'the answer should be:',
        'the final answer is',
        'the answer is option',
        'the answer is',
        'the correct answer is option',
        'the correct answer is',
        'the answer should be'
    ]:
        if flag in lower_ans:
            lower_ans = lower_ans.split(flag)[-1].strip()
            lower_ans = lower_ans.split('\n')[0].split('.')[0]
            upper_ans = lower_ans.upper()
            if upper_ans in Option_list:
                return upper_ans

    return ans


def extract_answer(ans):
    matches = re.findall(r"\banswer\s*:\s*([A-Z])\b", ans, re.IGNORECASE)
    if matches:
        ans = matches[-1]
        return ans
    else:
        return "Z"


def VisulPuzzles_acc(result_file):
    categories = [
        'overall',
        'spatial',
        'algorithmic',
        'analogical',
        'inductive',
        'deductive',
    ]
    difficulties = [
        'easy',
        'medium',
        'hard',
        'overall'
    ]

    data = load(result_file)
    lt = len(data)
    # print(data[0])
    hit = defaultdict(lambda: 0)
    tot = defaultdict(lambda: 0)
    from tqdm import tqdm
    for i in tqdm(range(lt)):
        item = data.iloc[i]
        cate = item['category']
        tot['overall'] += 1
        tot[cate] += 1

        if extract_answer(item['prediction']).lower() == item['answer'].lower():
            hit['overall'] += 1
            hit[cate] += 1

    res = defaultdict(list)

    for k in categories:
        res['category'].append(k)
        res['tot'].append(tot[k])
        res['hit'].append(hit[k])
        res['acc'].append(hit[k] / tot[k] * 100)

    hit_level = defaultdict(lambda: 0)
    tot_level = defaultdict(lambda: 0)
    res_level = defaultdict(list)
    for i in tqdm(range(lt)):
        item = data.iloc[i]
        level = item['difficulty']
        tot_level['overall'] += 1
        tot_level[level] += 1

        if extract_answer(item['prediction']).lower() == item['answer'].lower():
            hit_level['overall'] += 1
            hit_level[level] += 1

    for k in difficulties:
        res_level['level'].append(k)
        res_level['tot'].append(tot_level[k])
        res_level['hit'].append(hit_level[k])
        res_level['acc'].append(hit_level[k] / tot_level[k] * 100)

    return res, res_level
