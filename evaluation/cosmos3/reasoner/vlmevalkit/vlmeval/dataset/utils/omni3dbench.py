from collections import defaultdict

import pandas as pd

OMNI3DBENCH_PROMPT = """
I will ask you a question based on an image. Answer with either true/false, one word or number, \
place your answer between <ans></ans> tags. Only include your answer. Question:
"""


def extract_answer(prediction):
    if '<ans>' in prediction:
        return prediction.split('<ans>')[1].split('</ans>')[0]
    else:
        return prediction


def Omni3DBench_acc(data):
    mra_thresholds = [0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    correct_at_threshold = {key: 0 for key in mra_thresholds}
    yn_correct = 0
    yn_n = 0
    num_ct_n = 0
    num_ct_correct = 0
    multi_correct = 0
    multi_n = 0
    num_other_n = 0

    for i in range(len(data)):
        row = data.iloc[i]
        ans_type = row['answer_type']
        gt = row['answer']
        pred = extract_answer(row['prediction'])

        # Numeric (count)
        if ans_type == "int":
            num_ct_n += 1
            try:
                pred = int(pred)
            except Exception:
                continue
            gt = int(gt)
            if gt == pred:
                num_ct_correct += 1
        elif ans_type == "str":
            # Yes/No
            if gt in ["yes", "no"]:
                yn_n += 1
                try:
                    if gt in pred.lower():
                        yn_correct += 1
                    elif gt == "yes" and "true" in pred.lower():
                        yn_correct += 1
                    elif gt == "no" and "false" in pred.lower():
                        yn_correct += 1
                except Exception:
                    continue
            # Multi-choice
            else:
                multi_n += 1
                try:
                    if gt == pred.lower():
                        multi_correct += 1
                except Exception:
                    continue
        elif ans_type == "float":
            # Numeric (other)
            # Calculated Mean Relative Accuracy (MRA) introduced in VSI-Bench (https://arxiv.org/abs/2412.14171)
            num_other_n += 1
            for threshold in mra_thresholds:
                try:
                    pred = float(pred)
                except Exception:
                    continue
                gt = float(gt)
                if abs(gt - pred) / gt < threshold:
                    correct_at_threshold[threshold] += 1

    # Compute averages
    yn_acc = yn_correct / yn_n if yn_n != 0 else None
    multi_acc = multi_correct / multi_n if multi_n != 0 else None
    num_ct_acc = num_ct_correct / num_ct_n if num_ct_n != 0 else None
    num_other_mra = 0

    if num_other_n != 0:
        for threshold in mra_thresholds:
            correct_at_threshold[threshold] /= num_other_n
            num_other_mra += correct_at_threshold[threshold]

        num_other_mra = num_other_mra / len(mra_thresholds)
    else:
        num_other_mra = None

    res = defaultdict(list)
    res['Yes/No Accuracy'].append(yn_acc)
    res['Multiple Choice Accuracy'].append(multi_acc)
    res['Numeric (count) Accuracy'].append(num_ct_acc)
    res['Numeric (other) Mean Relative Accuracy'].append(num_other_mra)
    res = pd.DataFrame(res)
    return res
