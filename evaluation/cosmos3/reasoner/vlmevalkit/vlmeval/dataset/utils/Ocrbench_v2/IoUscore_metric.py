# flake8: noqa
import ast
import os
import re

import ipdb

from vlmeval.dataset.utils.Ocrbench_v2.vqa_metric import vqa_evaluation


def calculate_iou(box1, box2):

    try:
        box1 = [int(coordinate) for coordinate in box1]
        box2 = [int(coordinate) for coordinate in box2]
    except:
        return 0

    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou


def vqa_with_position_evaluation(predict, img_metas):

    score_content, score_bbox = .0, .0
    if "answer" in predict.keys():
        score_content = vqa_evaluation(predict["answer"], img_metas["answers"])
    if "bbox" in predict.keys():
        gt_bbox = img_metas["bbox"]
        try:
            predict_bbox_list = ast.literal_eval(predict["bbox"])
            score_bbox = calculate_iou(predict_bbox_list, gt_bbox)
        except:
            score_bbox = 0
    return 0.5 * score_content + 0.5 * score_bbox


def extract_coordinates(text):
    # Regex pattern to match coordinates in either (x1, y1, x2, y2) or [x1, y1, x2, y2] format

    pattern = r'[\(\[]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\)\]]'

    matches = list(re.finditer(pattern, text))
    coords_list = []
    coords_set = set()
    for match in matches:

        x1, y1, x2, y2 = map(int, match.groups())

        if all(0 <= n <= 1000 for n in [x1, y1, x2, y2]):
            coords = (x1, y1, x2, y2)

            if coords in coords_set:
                coords_list = [c for c in coords_list if c != coords]

            coords_list.append(coords)
            coords_set.add(coords)
    if coords_list:
        last_coords = coords_list[-1]
        return list(last_coords)
    else:
        return None


if __name__ == "__main__":

    print("Example for Text Grounding task.")
    box1 = [50, 50, 150, 150]
    box2 = [60, 60, 140, 140]
    iou_score = calculate_iou(box1, box2)
    print(f"IoU score: {iou_score}")

    print("Example for VQA with position task.")
    pred = {"content": "The content is Hello Buddies", "bbox": box1}
    gt = {"content": "Hello Buddies", "bbox": box2}

    vqa_score = vqa_evaluation(pred["content"], gt["content"])
    iou_score = calculate_iou(pred["bbox"], gt["bbox"])

    print(f"VQA score: {vqa_score}")
    print(f"IoU score: {iou_score}")
