import json
import os
import re
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import dump, get_intermediate_file_path, load, misc, toliststr


def weighted_row_sum(data, third_rows, weight_col=1, start_col=2):

    data = np.array(data)
    m, n = data.shape
    rows = slice(m - third_rows, m)
    cols = slice(start_col, None)
    weighted_sum = np.sum(data[rows, cols].astype(float) * data[rows, weight_col].astype(float)[:, np.newaxis], axis=0) / np.sum(data[rows, weight_col].astype(float))  # noqa: E501
    weighted_sum = ['Mean', np.sum(data[rows, weight_col].astype(float))] + weighted_sum.tolist()
    temp = data.tolist()
    temp.append(weighted_sum)
    return temp


def weighted_total(data, weight_col=1, start_col=2):
    data = np.array(data)
    m, n = data.shape
    rows = slice(0, m)
    cols = slice(start_col, None)
    weighted_sum = np.sum(data[rows, cols].astype(float) * data[rows, weight_col].astype(float)[:, np.newaxis], axis=0) / np.sum(data[rows, weight_col].astype(float))  # noqa: E501
    weighted_sum = ['Total', np.sum(data[rows, weight_col].astype(float))] + weighted_sum.tolist()
    return weighted_sum


def box_iou(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    xB = min(boxA[2], boxB[2])
    yA = max(boxA[1], boxB[1])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def clean_string(s):
    while s and (s[0] in ":[]()' ."):
        s = s[1:]
    while s and (s[-1] in ":[]()' ."):
        s = s[:-1]
    return s


def convert_if_number(answer):
    if isinstance(answer, (int, float)):
        return str(answer)
    return answer


def remove_symbols(input_string):
    input_string = str(input_string)
    if 'correct answer is:' in input_string:
        input_string = input_string.split('correct answer is:')[-1]
    cleaned_string = re.sub(r'[\*\n\""]', '', input_string)
    return cleaned_string


def extract_options(text):

    pattern = re.compile(r"\[([^\]]+)\]")
    matches = pattern.findall(text)

    if matches:
        option_string = matches[-1]
        if "'" not in option_string:
            option_list = option_string.split(", ")
        else:
            option_list = [item.strip().strip("'") for item in option_string.split("', '")]
        return option_list
    return []


def compare_and_count(array_a, array_b):
    count = 0
    for a, b in zip(array_a, array_b):
        if a == 1 and b == 1:
            count += 1
        if a > b:
            count += 1
    return count


def isfile(path):
    return os.path.isfile(path)


def load_json_data(path):
    with open(path, 'r', encoding='utf-8') as json_f:
        task_data = json.load(json_f)
        return task_data


def save_json_data(path, data):
    with open(path, 'w', encoding='utf-8') as json_f:
        json.dump(data, json_f, ensure_ascii=False, indent=4)


def Geneal_criterion_QA(third_task_data, MODEL=None):
    ques_total_num = 0
    right_num = 0
    obey_insytruction = 0
    for d_ind, sample in enumerate(third_task_data):
        reference = sample['reference']
        prediction = sample['prediction']
        for q_ind, pred in enumerate(prediction):
            ques_nopath = sample['questions'][q_ind].lower()
            tips = extract_options(ques_nopath)
            if len(tips) == 0:
                pass
            pred = remove_symbols(pred)
            ques_total_num += 1
            clean_pred = clean_string(pred).lower()
            options_nums = clean_pred.split("', '")
            reference_q_ind = convert_if_number(reference[q_ind]).lower()
            if len(options_nums) == 1:
                if clean_pred in ques_nopath:
                    obey_insytruction += 1
                if clean_pred == reference_q_ind:
                    right_num += 1
                elif reference_q_ind in clean_pred:
                    # filter
                    if reference_q_ind in tips:
                        tips.remove(reference_q_ind)
                        if not any(tip in clean_pred for tip in tips):
                            right_num += 1
    return ques_total_num, right_num / ques_total_num, obey_insytruction / ques_total_num, 0


def Grounding_criterion_QA(third_task_data, MODEL=None):
    resize_model_lists = ["qwen", "internvl", "gemini", "DriveMM", 'ivl', 'seed']
    ques_total_num = 0
    right_num = 0
    loc_union = []
    obey_insytruction = 0
    PATTERN = re.compile(
        r'\[\s*([^\],]*\d+[^\],]*)\s*,\s*([^\],]*\d+[^\],]*)\s*,\s*([^\],]*\d+[^\],]*)\s*,\s*([^\],]*\d+[^\],]*)\s*\]')
    box_num = 0
    for d_ind, sample in enumerate(third_task_data):
        reference = sample['reference']
        prediction = sample['prediction']
        for q_ind, pred in enumerate(prediction):
            ques_total_num += 1
            ques_nopath = sample['questions'][q_ind].lower()
            if 'located in the image?' in ques_nopath:
                matches = PATTERN.findall(pred)
                cleaned_matches = [[float(re.sub(r'[^0-9.]', '', part)) for part in match] for match in matches]
                if len(matches) == 1:
                    box_num += 1
                    obey_insytruction += 1
                    predict_bbox = cleaned_matches[0]
                else:
                    predict_bbox = [0.0, 0.0, 0.0, 0.0]

                if sum(predict_bbox) < 4:
                    predict_bbox = [x * 1000 for x in predict_bbox]
                # By default we use [x1, y1, x2, y2] normalized in [0, 1000]
                if MODEL is None or any(mn.lower() in MODEL.lower() for mn in resize_model_lists):
                    bbox_gt = list(map(int, misc.toliststr(sample['reference'][q_ind])))
                    width, height = sample['dimension'][q_ind]
                    width, height = float(width), float(height)
                    bbox_gt = [int(1000 * bbox_gt[0] / width), int(1000 * bbox_gt[1] / height),
                               int(1000 * bbox_gt[2] / width), int(1000 * bbox_gt[3] / height)]
                elif MODEL == "gemini":
                    bbox_gt = [bbox_gt[1], bbox_gt[0], bbox_gt[3], bbox_gt[2]]
                else:
                    bbox_gt = sample['reference'][q_ind]
                iou = box_iou(predict_bbox, bbox_gt)
                if iou > 0.5:
                    right_num += 1
                loc_union.append(iou)
            else:
                tips = extract_options(ques_nopath)
                pred = remove_symbols(pred)
                clean_pred = clean_string(pred).lower()
                options_nums = clean_pred.split("', '")
                reference_q_ind = convert_if_number(reference[q_ind]).lower()
                if len(options_nums) == 1:
                    if clean_pred in ques_nopath:
                        obey_insytruction += 1
                    if clean_pred == reference_q_ind:
                        right_num += 1

                    elif reference_q_ind in clean_pred:
                        # filter
                        if reference_q_ind in tips:
                            tips.remove(reference_q_ind)
                            if not any(tip in clean_pred for tip in tips):
                                right_num += 1

    mean_iou = sum(loc_union) / len(loc_union)
    return ques_total_num, right_num / ques_total_num, obey_insytruction / ques_total_num, mean_iou


def Relation_criterion_QA(third_task_data, MODEL=None):
    ques_total_num = 0
    total_score = 0
    obey_insytruction = 0
    totol_improve_score = 0
    for d_ind, sample in enumerate(third_task_data):
        reference = sample['reference']
        prediction = sample['prediction']
        scores_list = []
        for q_ind, pred in enumerate(prediction):
            ques_total_num += 1
            if 'corresponds to' in pred:
                pattern = r'corresponds to No.([+-]?\d+|[+-]?\d+/\d+)'
                match = re.search(pattern, pred)
                if match:
                    pred_num = match.group(1).split('/')
                else:
                    pred_num = []
            elif 'corresponding to' in pred:
                pattern = r"corresponding to.*is\s+(-?\d+(?:/\d+)*)"
                match = re.search(pattern, pred)
                if match:
                    pred_num = match.group(1).split("/")
                else:
                    pred_num = []
            else:
                pattern = r"(-?\d+(?:/\d+)*)"
                match = re.findall(pattern, pred)
                if match:
                    obey_insytruction += 1
                    pred_num = match[-1].split("/")
                else:
                    pred_num = []

            ref_num = reference[q_ind].split('/')
            if any(p_num not in ref_num for p_num in pred_num):
                scores_list.append(0)
                continue
            else:
                temp = 0
                for p_num in pred_num:
                    if p_num in ref_num:
                        temp += 1 / len(ref_num)
                        total_score += 1 / len(ref_num)
                scores_list.append(temp)
        scores_list = np.array(scores_list)
        scores = compare_and_count(scores_list[len(scores_list) // 2:], scores_list[:len(scores_list) // 2])
        totol_improve_score += scores
    return ques_total_num, total_score / ques_total_num, obey_insytruction / ques_total_num, totol_improve_score * 2 / ques_total_num  # noqa: E501


def RoadChange_criterion_QA(third_task_data, MODEL=None):
    ques_total_num = 0
    right_num = 0
    obey_insytruction = 0
    totol_improve_score = 0
    for d_ind, sample in enumerate(third_task_data):
        reference = sample['reference']
        prediction = sample['prediction']
        scores_list = []
        for q_ind, pred in enumerate(prediction):
            ques_nopath = sample['questions'][q_ind].lower()
            tips = extract_options(ques_nopath)
            pred = remove_symbols(pred)
            ques_total_num += 1
            clean_pred = clean_string(pred).lower()
            options_nums = clean_pred.split("', '")
            reference_q_ind = convert_if_number(reference[q_ind]).lower()
            if len(options_nums) == 1:
                if clean_pred in ques_nopath:
                    obey_insytruction += 1
                if clean_pred == reference_q_ind:
                    right_num += 1
                    scores_list.append(1)
                elif reference_q_ind in clean_pred:
                    # filter
                    if reference_q_ind in tips:
                        tips.remove(reference_q_ind)
                        if not any(tip in clean_pred for tip in tips):
                            right_num += 1
                            scores_list.append(1)
                        else:
                            scores_list.append(0)
                    else:
                        scores_list.append(0)
                else:
                    scores_list.append(0)
        scores_list = np.array(scores_list)
        scores = compare_and_count(scores_list[len(scores_list) // 2:], scores_list[:len(scores_list) // 2])
        totol_improve_score += scores

    return ques_total_num, right_num / ques_total_num, obey_insytruction / ques_total_num, totol_improve_score * 2 / ques_total_num  # noqa: E501


def RoadSpeed_criterion_QA(third_task_data, MODEL=None):
    ques_total_num = 0
    right_num = 0
    obey_insytruction = 0
    totol_improve_score = 0
    for d_ind, sample in enumerate(third_task_data):
        reference = sample['reference']
        prediction = sample['prediction']
        scores_list = []
        for q_ind, pred in enumerate(prediction):
            ques_total_num += 1
            pattern = r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
            matches = re.findall(pattern, pred)

            matches_gt = re.findall(pattern, reference[q_ind])
            ref_gt = [matches_gt[0][0], matches_gt[0][1]]
            temp = 0
            if len(matches) == 1:
                pred_limit = [matches[0][0], matches[0][1]]
                obey_insytruction += 1
                for a, b in zip(ref_gt, pred_limit):
                    if a == b:
                        temp += 0.5
            right_num += temp
            scores_list.append(temp)

        scores_list = np.array(scores_list)
        scores = compare_and_count(scores_list[len(scores_list) // 2:], scores_list[:len(scores_list) // 2])
        totol_improve_score += scores

    return ques_total_num, right_num / ques_total_num, obey_insytruction / ques_total_num, totol_improve_score * 2 / ques_total_num  # noqa: E501


def Judge_criterion_QA(third_task_data, MODEL=None):
    des_ques_total_num = 0
    judge_ques_total_num = 0
    des_right_num = 0
    judge_right_num = 0
    obey_insytruction = 0
    for d_ind, sample in enumerate(third_task_data):
        reference = sample['reference']
        prediction = sample['prediction']
        for q_ind, pred in enumerate(prediction):
            ques_nopath = sample['questions'][q_ind].lower()
            tips = extract_options(ques_nopath)
            if len(tips) == 0:
                pass
            pred = remove_symbols(pred)
            clean_pred = clean_string(pred).lower()
            options_nums = clean_pred.split("', '")
            reference_q_ind = clean_string(convert_if_number(reference[q_ind])).lower()
            if 'yes' == reference_q_ind or 'no' == reference_q_ind:
                judge_ques_total_num += 1
            else:
                des_ques_total_num += 1
            if len(options_nums) == 1:
                # if clean_pred in ques_nopath:
                if ''.join(clean_pred.split(';')) in ques_nopath:
                    obey_insytruction += 1
                if clean_pred == reference_q_ind:
                    if 'yes' == reference_q_ind or 'no' == reference_q_ind:
                        judge_right_num += 1
                    else:
                        des_right_num += 1
                elif reference_q_ind in clean_pred:
                    # filter
                    if reference_q_ind in tips:
                        tips.remove(reference_q_ind)
                        if not any(tip in clean_pred for tip in tips):
                            if 'yes' == reference_q_ind or 'no' == reference_q_ind:
                                judge_right_num += 1
                            else:
                                des_right_num += 1
                else:
                    pass
    if des_ques_total_num == 0:
        return (judge_ques_total_num + des_ques_total_num), des_right_num, obey_insytruction / (judge_ques_total_num + des_ques_total_num), judge_right_num / judge_ques_total_num  # noqa: E501
    else:
        return (judge_ques_total_num + des_ques_total_num), des_right_num / des_ques_total_num, obey_insytruction / (judge_ques_total_num + des_ques_total_num), judge_right_num / judge_ques_total_num  # noqa: E501


func_mapping = {
    'Pavement_Marking': Geneal_criterion_QA,
    'Traffic_Sign': Geneal_criterion_QA,
    'Traffic_Light': Geneal_criterion_QA,
    'Right_Of_Way': Geneal_criterion_QA,
    'Light': Geneal_criterion_QA,
    'Weather': Geneal_criterion_QA,
    'Lane_Recognition': Geneal_criterion_QA,
    'Vehicle_Status': Geneal_criterion_QA,
    'Vehicle_Recognition': Grounding_criterion_QA,
    'VRU_Recognition': Grounding_criterion_QA,
    'Obstruction_Recognition': Grounding_criterion_QA,
    'Light_Lane_Relation': Relation_criterion_QA,
    'Sign_Sign_Relation': Relation_criterion_QA,
    'Sign_Lane_Relation': Relation_criterion_QA,
    'Lane_Change_Relation': RoadChange_criterion_QA,
    'Lane_Speed_Relation': RoadSpeed_criterion_QA,
    'VRU_Cutin': Judge_criterion_QA,
    'Vehicle_Cutin': Judge_criterion_QA,
    'VRU_Cross': Judge_criterion_QA,
    'Long_Short_Parking': Geneal_criterion_QA,
    'Vehicle_Bahavior': Geneal_criterion_QA,
    'VRU_Bahavior': Geneal_criterion_QA,
    'Key_Obsturction_Detection': Judge_criterion_QA,
    'Spatial_Temporal_Reasoning': Judge_criterion_QA,
    'Risk_Prediction': Judge_criterion_QA,
    'Drive_Efficiency': Geneal_criterion_QA,
    'Longitudinal': Geneal_criterion_QA,
    'Lateral': Geneal_criterion_QA
}


all_tasks = {
    "Traffic_Knowledge_Understanding": {
        "Road_Traffic_Signals": [
            "Traffic_Light",
            "Pavement_Marking",
            "Traffic_Sign"
        ],
        "Road_Passage_Provisions": [
            "Right_Of_Way"
        ]
    },
    "General_Element_Recognition": {
        "Foreground": [
            "VRU_Recognition",
            "Vehicle_Recognition",
            "Vehicle_Status",
            "Lane_Recognition",
            "Obstruction_Recognition"
        ],
        "Background": [
            "Light",
            "Weather"
        ]
    },
    "Traffic_Graph_Generation": {
        "Signal_Element_Relation": [
            "Sign_Sign_Relation",
            "Sign_Lane_Relation",
            "Light_Lane_Relation"
        ],
        "Lane_Element_Relation": [
            "Lane_Speed_Relation",
            "Lane_Change_Relation"
        ]
    },
    "Target_Attribute_Comprehension": {
        "Intention_Judgment": [
            "VRU_Cutin",
            "Vehicle_Cutin",
            "VRU_Cross",
            "Long_Short_Parking"
        ],
        "Behavior_Understanding": [
            "Vehicle_Bahavior",
            "VRU_Bahavior"
        ]
    },
    "Ego_Decision_Planning": {
        "Ego_Action_Reasoning": [
            "Key_Obsturction_Detection",
            "Spatial_Temporal_Reasoning",
            "Risk_Prediction",
            "Drive_Efficiency"
        ],
        "Meta_Action_Decision": [
            "Longitudinal",
            "Lateral"
        ],
        "Ego_trajectory_Planning": [
            "Trajectory"
        ]
    }
}

weights = {
    'Vehicle_Recognition': [0.3, 0.5, 0.2],
    'VRU_Recognition': [0.3, 0.5, 0.2],
    'Obstruction_Recognition': [0.3, 0.5, 0.2],
    'Sign_Sign_Relation': [0.3, 0.5, 0.2],
    'Sign_Lane_Relation': [0.3, 0.5, 0.2],
    'Light_Lane_Relation': [0.3, 0.5, 0.2],
    'Lane_Speed_Relation': [0.3, 0.5, 0.2],
    'Lane_Change_Relation': [0.3, 0.5, 0.2],
    'VRU_Cutin': [0.7, 0.1, 0.2],
    'Vehicle_Cutin': [0.7, 0.1, 0.2],
    'VRU_Cross': [0.7, 0.1, 0.2],
    'Key_Obsturction_Detection': [0.8, 0, 0.2],
    'Risk_Prediction': [0.7, 0.1, 0.2],
    'Spatial_Temporal_Reasoning': [0.4, 0.4, 0.2]
}


def get_vladbench_image_dir(image_dir=None):
    """
    Resolve VLADBench image directory.
    Priority:
    1. User provided local directory
    2. Auto-download from HuggingFace
    """
    if image_dir is not None:
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"VLADBench image dir not found: {image_dir}")
        return image_dir

    return download_vladbench_from_hf()


def download_vladbench_from_hf():
    local_root = os.path.expanduser("~/.cache/vlmeval/vladbench/")

    if os.path.exists(local_root):
        return local_root

    print("VLADBench not found locally, downloading from HuggingFace...")

    snapshot_download(
        repo_id="depth2world/VLADBench",
        repo_type="dataset",
        local_dir=local_root,
        local_dir_use_symlinks=False,
    )
    return local_root


class VLADBench(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {
        "VLADBench": "https://huggingface.co/datasets/depth2world/VLADBench/blob/main/VLADBench.tsv",
    }
    DATASET_MD5 = {"VLADBench": "53c01aa8f9afe2d084728fc8eb21caea"}

    IMAGE_DIR = None  # e.g., your_localdir/VLADBench/
    """
    Resolve VLADBench image directory.
    Priority:
    1. User provided local directory
    2. Auto-download from HuggingFace
    """

    def __init__(self, *args, **kwargs):
        if self.IMAGE_DIR is None:
            self.IMAGE_DIR = get_vladbench_image_dir(self.IMAGE_DIR)
        super().__init__(*args, **kwargs)

    def build_prompt(self, line: Union[int, pd.Series]) -> List[Dict[str, str]]:
        """
        Build a prompt for the model from a data line.

        Args:
            line: Either an index into the dataset or a pandas Series

        Returns:
            List of message dictionaries containing the image and question
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = toliststr(line["image"])
        question = line['question']
        # form messages
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=os.path.join(self.IMAGE_DIR, p)) for p in tgt_path])

        else:
            msgs = [dict(type='image', value=os.path.join(self.IMAGE_DIR, tgt_path))]
        msgs.append(dict(type='text', value=question))

        return msgs

    def get_scores(self, result_file: str) -> pd.DataFrame:
        data = load(result_file)
        model_name = os.path.basename(result_file).split('_')[0]

        all_results = []
        total_results = []
        for fir_ind, fir_task in enumerate(all_tasks):
            sec_tasks = all_tasks[fir_task]
            for sec_ind, sec_task in enumerate(sec_tasks):
                if sec_task == 'Ego_trajectory_Planning':
                    continue
                third_tasks = sec_tasks[sec_task]
                third_rows = 0
                for third_ind, third_task in enumerate(third_tasks):
                    # filter samples of third task
                    filter_data = data[data['category3'] == third_task]
                    # prepare data structure for evaluation: list(dict(list))
                    third_task_data = []
                    same_vision_qas = {'reference': [], 'prediction': [], 'questions': [], 'dimension': []}
                    dindex = 0
                    for index, row in filter_data.iterrows():
                        if dindex != row['dindex']:
                            third_task_data.append(same_vision_qas)
                            same_vision_qas = {'reference': [], 'prediction': [], 'questions': [], 'dimension': []}
                        same_vision_qas['reference'].append(row['answer'])
                        same_vision_qas['prediction'].append(row['prediction'])
                        same_vision_qas['questions'].append(row['question'])
                        same_vision_qas['dimension'].append(toliststr(row['dimension']))
                        dindex = row['dindex']
                    third_task_data.append(same_vision_qas)

                    # compute score
                    third_rows += 1
                    model_scores = [third_task]
                    ques_total_num, right_num, obey_instruction, others = func_mapping[third_task](
                        third_task_data, model_name)

                    # weighted sum score
                    if third_task in weights:
                        weight = weights[third_task]
                    else:
                        weight = [0, 0.8, 0.2]
                    temp_score = 100 * others * weight[0] + 100 * right_num * \
                        weight[1] + 100 * obey_instruction * weight[2]

                    model_scores.append(temp_score)
                    model_scores.insert(1, ques_total_num)
                    all_results.append(model_scores)
                    total_results.append(model_scores)

                all_results = weighted_row_sum(all_results, third_rows)

        total_ = weighted_total(total_results)
        all_results.append(total_)
        df = pd.DataFrame(all_results, columns=['Task', 'num', model_name])
        return df

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> pd.DataFrame:
        """
        Evaluate model predictions on the ChartQAPro dataset.

        Args:
            eval_file: Path to the file containing model predictions
            **judge_kwargs: Additional arguments for the judge model

        Returns:
            DataFrame with evaluation scores by category
        """
        score = self.get_scores(eval_file)
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(score, score_file)
        return score
