import numpy as np

from vlmeval.smp import load


def get_dimension_rating(data_path):
    P_SUBTASKS = [
        'Attributes',
        'Material',
        'Physical State',
        'Object Retrieval',
        'OCR',
    ]

    R_SUBTASKS = [
        'Perspective Transform',
        'Ordering',
        'Contact and Occlusion',
        'Spatial Containment',
        'Comparison',
    ]

    data = load(data_path)
    results = {}
    results['Overall'] = {}
    results['Perception'] = {}
    for subtask in P_SUBTASKS:
        results['Perception'][subtask] = {'true': 0, 'false': 0}
    results['Reasoning'] = {}
    for subtask in R_SUBTASKS:
        results['Reasoning'][subtask] = {'true': 0, 'false': 0}

    all_iou = []

    for i in range(len(data)):
        question = data.iloc[i]
        Task = question['category'].split('/')[0]
        Subtask = question['category'].split('/')[1]
        if question['score'] >= 0:
            cnt = question['score']
            results[Task][Subtask]['true'] += cnt
            results[Task][Subtask]['false'] += 1 - cnt
        all_iou.append(question['iou'])

    sum_all, succ_all = 0, 0
    for task, tasks_values in results.items():
        cnt_task, sum_task = 0, 0
        for substask, subtask_value in tasks_values.items():
            cnt_subtask, sum_subtask = 0, 0
            cnt_subtask += subtask_value['true']
            sum_subtask += subtask_value['false'] + subtask_value['true']
            if (subtask_value['false'] + subtask_value['true']) > 0:
                acc = subtask_value['true'] / (subtask_value['false'] + subtask_value['true'])
            else:
                acc = 0
            results[task][substask] = acc

            cnt_task += cnt_subtask
            sum_task += sum_subtask
        succ_all += cnt_task
        sum_all += sum_task
        # results[task]['Overall'] = acc_task
    results['Overall'] = succ_all / sum_all
    results['mIoU'] = np.mean(all_iou)
    return results
