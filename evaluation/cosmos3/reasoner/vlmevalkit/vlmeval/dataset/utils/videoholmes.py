import re

from vlmeval.smp import load

FAIL_MSG = 'Failed to obtain answer via API.'

TASK_CATEGORIES = [
    'SR', 'IMC', 'TCI', 'TA', 'MHR', 'PAR', 'CTI',
]


def get_dimension_rating(data_path, score_col='score', type_col='question_type'):
    data = load(data_path)
    acc_by_type = {}
    for qtype, group in data.groupby(type_col):
        correct = (group[score_col] == 1).sum()
        total = len(group)
        acc = correct / total if total > 0 else 0
        acc_by_type[qtype] = {
            'correct': int(correct),
            'total': int(total),
            'acc': acc
        }

    total_correct = (data[score_col] == 1).sum()
    total_count = len(data)
    total_acc = total_correct / total_count if total_count > 0 else 0

    result = {
        'acc_by_type': acc_by_type,
        'total': {
            'correct': int(total_correct),
            'total': int(total_count),
            'acc': total_acc
        }
    }

    return result


def extract_option(pred):

    pattern = r'<answer>\s*(.*?)\s*</answer>'
    try:
        matches = re.findall(pattern, pred, re.DOTALL)
    except Exception:
        matches = []

    if matches:
        choise = matches[-1].strip()
        if 'A ' in choise or 'A:' in choise or '[A' in choise:
            predicted_answer = 'A'
        elif 'B ' in choise or 'B:' in choise or '[B' in choise:
            predicted_answer = 'B'
        elif 'C ' in choise or 'C:' in choise or '[C' in choise:
            predicted_answer = 'C'
        elif 'D ' in choise or 'D:' in choise or '[D' in choise:
            predicted_answer = 'D'
        elif 'E ' in choise or 'E:' in choise or '[E' in choise:
            predicted_answer = 'E'
        elif 'F ' in choise or 'F:' in choise or '[F' in choise:
            predicted_answer = 'F'
        elif 'A' in choise:
            predicted_answer = 'A'
        elif 'B' in choise:
            predicted_answer = 'B'
        elif 'C' in choise:
            predicted_answer = 'C'
        elif 'D' in choise:
            predicted_answer = 'D'
        elif 'E' in choise:
            predicted_answer = 'E'
        elif 'F' in choise:
            predicted_answer = 'F'
        else:
            predicted_answer = 'WRONG'
    else:
        predicted_answer = 'WRONG'
    return predicted_answer
