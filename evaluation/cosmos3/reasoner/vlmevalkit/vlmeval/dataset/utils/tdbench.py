import os
import re
import warnings
from functools import reduce

import pandas as pd

rotations_all = ['rot0', 'rot90', 'rot180', 'rot270']


def extract_bbox_from_string(bbox_str):
    bbox_str = bbox_str.replace('\n', '')
    parsed = re.findall(r'(\d?(?:\.\d+)?),\s?(\d?(?:\.\d+)?),\s?(\d?(?:\.\d+)?),\s?(\d?(?:\.\d+)?)', bbox_str)
    if len(parsed) == 1:
        return float(parsed[0][0]), float(parsed[0][1]), float(parsed[0][2]), float(parsed[0][3])
    else:
        raise RuntimeError(f'Invalid VLM output: {bbox_str}. '
                           f'Correct coordinate should be [a, b, c, d], where each number is 0, 0.*, or 1. ')


def calculate_bbox_iou(pred_bbox, gt_bbox):
    pred_x_min, pred_y_min, pred_x_max, pred_y_max = pred_bbox
    gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_bbox

    x_min_intersect = max(pred_x_min, gt_x_min)
    y_min_intersect = max(pred_y_min, gt_y_min)
    x_max_intersect = min(pred_x_max, gt_x_max)
    y_max_intersect = min(pred_y_max, gt_y_max)

    if x_max_intersect < x_min_intersect or y_max_intersect < y_min_intersect:
        return 0.0

    intersection_area = (x_max_intersect - x_min_intersect) * (y_max_intersect - y_min_intersect)

    pred_area = (pred_x_max - pred_x_min) * (pred_y_max - pred_y_min)
    gt_area = (gt_x_max - gt_x_min) * (gt_y_max - gt_y_min)

    union_area = pred_area + gt_area - intersection_area

    iou = intersection_area / union_area

    return iou


def calculate_centroid_containment(pred_bbox, gt_bbox):
    label_x0, label_y0, label_x1, label_y1 = gt_bbox
    pred_x0, pred_y0, pred_x1, pred_y1 = pred_bbox
    pred_center_x = (pred_x0 + pred_x1) / 2
    pred_center_y = (pred_y0 + pred_y1) / 2
    if (label_x0 <= pred_center_x <= label_x1) and (label_y0 <= pred_center_y <= label_y1):
        return 1
    else:
        return 0


def evaluate_bbox(pred_bbox, gt_bbox, method):
    if method == 'centroid':
        return calculate_centroid_containment(pred_bbox, gt_bbox)
    elif method == 'iou':
        return calculate_bbox_iou(pred_bbox, gt_bbox)


def rotational_eval(eval_file):
    rotations = set(rotations_all)
    match = re.search(r'(rot\d+)', eval_file)
    if not match:
        warnings.warn(f"[RotationalEval] Invalid file name format: {eval_file}."
                      f"Expected format includes rotation like 'rot0', 'rot90', etc.")
        return False
    current_rotation = match.group(1)

    # Collect all existing rotation's result files
    data = []
    rotation_files = {rot: eval_file.replace(current_rotation, rot) for rot in rotations}
    for rot, path in rotation_files.items():
        filename = os.path.basename(path)
        if os.path.exists(path):
            df_rot = pd.read_excel(path)[['index', 'category', 'hit']]
            df_rot.rename(columns={'hit': f'hit_{rot}'}, inplace=True)
            data.append(df_rot)
        else:
            rotations.remove(rot)
            warnings.warn(f"[RotationEval] Skipped rotation {rot} because {filename} does not exist.")

    # Merge dataframes from different rotations
    df_all = reduce(lambda df1, df2: pd.merge(df1, df2, on=['index', 'category'], how='inner'), data)
    hit_columns = [col for col in df_all.columns if col.startswith('hit_')]
    # Find all correct and all wrong
    df_all['hit_all'] = df_all[hit_columns].eq(1).all(axis=1)
    df_all['miss_all'] = df_all[hit_columns].eq(0).all(axis=1)

    # Count for each category
    acc = df_all.groupby('category').agg(
        total_questions=('index', 'count'),
        hit_all=('hit_all', lambda x: x.astype(float).sum()),
        miss_all=('miss_all', lambda x: x.astype(float).sum()),
        **{rot: (f'hit_{rot}', lambda x: x.astype(float).sum()) for rot in rotations},
    )

    cols = acc.columns != 'total_questions'
    acc.loc[:, cols] = acc.loc[:, cols].div(acc['total_questions'], axis=0)
    acc.loc['Average'] = acc.mean().round(3)

    # Clean up output
    acc['total_questions'] = acc['total_questions'].astype(int)
    for rot in rotations_all:
        if rot not in acc.columns:
            acc[rot] = 'No Data'
    return acc[['total_questions', 'hit_all', 'miss_all', *rotations_all]]
