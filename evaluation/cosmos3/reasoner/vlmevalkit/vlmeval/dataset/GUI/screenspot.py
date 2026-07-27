import ast
import itertools
import json
import os
import os.path as osp
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, get_logger, load, toliststr

logger = get_logger(__name__)

"""
{
    "img_filename": "web_3b0ad239-da6b-4f6f-8f12-f674dc90ff33.png",
    "bbox": [42, 1102, 197, 70],
    "question": "view the details of the item",
    "data_type": "text",
    "data_source": "shop"
},
{
    "img_filename": "web_3b0ad239-da6b-4f6f-8f12-f674dc90ff33.png",
    "bbox": [93, 74, 86, 132],
    "question": "view the previous photo",
    "data_type": "icon",
    "data_source": "shop"
}
"""

SYSTEM_PROMPT = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform pyautogui click/moveTo action to complete the task. The answer format is `pyautogui.click(x=?, y=?), x and y is necessary`"""  # noqa: E501

USER_INSTRUCTION = """Please complete the following tasks by clicking using `pyautogui.click`:\n{instruction}"""

SYSTEM_PROMPT_V2 = """You are a GUI agent. You are given a screenshot of the screen and the description of a target element. You need to click the target element using `pyautogui.click`. The answer format is `pyautogui.click(x=?, y=?), x and y is necessary`"""  # noqa: E501
USER_INSTRUCTION_V2 = """Please click the following target element using `pyautogui.click`:\n{description}"""


def parse_bbox_aguvis(response):
    match = re.search(r"x=([\d.]+), y=([\d.]+)", response)
    if match:
        click_point = [float(match.group(1)), float(match.group(2))]
    else:
        click_point = [0.0, 0.0]
    return click_point


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area

    # Compute the Intersection over Union
    iou = intersection_area / union_area

    return iou


def compute_accuracy(box1, box2, threshold=0.5):
    """
    Compute the accuracy of two bounding boxes based on a specified threshold.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - threshold (float): Threshold for the IoU to consider the prediction correct.

    Returns:
    - float: Accuracy of the prediction based on the IoU threshold.
    """
    iou = compute_iou(box1, box2)
    return iou >= threshold


def compute_center_accuracy(box1, box2):
    """
    Compute if the center point of box 2 is within box 1.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - bool: True if the center point of box 2 is within box 1, False otherwise.
    """
    # Compute the center point of box 2
    center_x = (box2[0] + box2[2]) / 2
    center_y = (box2[1] + box2[3]) / 2

    # Check if the center point is within box 1
    return box1[0] <= center_x <= box1[2] and box1[1] <= center_y <= box1[3]


def convert_bbox(bbox, image_path):
    new_bbox = bbox if isinstance(bbox, list) else ast.literal_eval(bbox)
    new_bbox = [
        new_bbox[0],
        new_bbox[1],
        new_bbox[0] + new_bbox[2],
        new_bbox[1] + new_bbox[3],
    ]
    image = Image.open(image_path)
    img_size = image.size
    new_bbox = [
        new_bbox[0] / img_size[0],
        new_bbox[1] / img_size[1],
        new_bbox[2] / img_size[0],
        new_bbox[3] / img_size[1],
    ]
    return new_bbox


class ScreenSpot(ImageBaseDataset):
    MODALITY = "IMAGE"
    TYPE = "GUI"
    DATASET_URL = {
        "ScreenSpot_Mobile": "https://opencompass.openxlab.space/utils/benchmarks/GUI/ScreenSpot/ScreenSpot_Mobile.tsv",  # noqa
        "ScreenSpot_Desktop": "https://opencompass.openxlab.space/utils/benchmarks/GUI/ScreenSpot/ScreenSpot_Desktop.tsv",  # noqa
        "ScreenSpot_Web": "https://opencompass.openxlab.space/utils/benchmarks/GUI/ScreenSpot/ScreenSpot_Web.tsv",  # noqa
        "ScreenSpot_v2_Mobile": "https://opencompass.openxlab.space/utils/benchmarks/GUI/ScreenSpot_v2/ScreenSpot_v2_Mobile.tsv",  # noqa
        "ScreenSpot_v2_Desktop": "https://opencompass.openxlab.space/utils/benchmarks/GUI/ScreenSpot_v2/ScreenSpot_v2_Desktop.tsv",  # noqa
        "ScreenSpot_v2_Web": "https://opencompass.openxlab.space/utils/benchmarks/GUI/ScreenSpot_v2/ScreenSpot_v2_Web.tsv",  # noqa
    }  # path
    DATASET_URL_V2 = {
        "ScreenSpot_Mobile": "$WORK_DIR/screenspot_mobile_ug.json",
        "ScreenSpot_Desktop": "$WORK_DIR/screenspot_desktop_ug.json",
        "ScreenSpot_Web": "$WORK_DIR/screenspot_web_ug.json",
    }  # path
    DATASET_MD5 = {
        "ScreenSpot_Mobile": "a5b5299843a75c9b9574c47bc13b2c53",
        "ScreenSpot_Desktop": "e6e7bac21b6b2475276404fce2458132",
        "ScreenSpot_Web": "e51d168c14b8582427cf3107d236cfc5",
        "ScreenSpot_v2_Mobile": "234c858ab4f0e787e8388a73df65a4b7",
        "ScreenSpot_v2_Desktop": "5f2aa2a497327bd33b2512a0c75cf994",
        "ScreenSpot_v2_Web": "01cd0877ee1b735a6d5190b053ba9482",
    }
    EVAL_TYPE = "point"  # point or rectangle
    RE_TYPE = "functional"  # type of referring expressions: functional or composite

    def __init__(
        self,
        dataset="ScreenSpot_Mobile",
        skip_noimg=True,
        skeleton=False,
        re_type="functional",
    ):
        # st()
        ROOT = LMUDataRoot()
        # You can override this variable to save image files to a different directory
        self.dataset_name = dataset
        self.img_root = osp.join(ROOT, "images", self.dataset_name)
        self.RE_TYPE = re_type
        if skeleton:
            return

        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and "image" in data:
            data = data[~pd.isna(data["image"])]

        self.meta_only = True
        self.parse_response_func = parse_bbox_aguvis  # TODO: parse function can be specified through kwargs when initializing the dataset # noqa: E501

        # The image field can store the base64 encoded image or another question index (for saving space)
        if "image" in data:
            data["image"] = [str(x) for x in data["image"]]
            image_map = {x: y for x, y in zip(data["index"], data["image"])}
            for k in image_map:
                if len(image_map[k]) <= 64:
                    idx = image_map[k]
                    assert idx in image_map and len(image_map[idx]) > 64
                    image_map[k] = image_map[idx]

            images = [toliststr(image_map[k]) for k in data["index"]]
            data["image"] = [x[0] if len(x) == 1 else x for x in images]
            self.meta_only = False

        self.data = data

    def prepare_tsv(self, url, file_md5=None):
        # st()
        if self.RE_TYPE == "functional":
            return super().prepare_tsv(url=url, file_md5=file_md5)
        else:
            data_path = self.DATASET_URL_V2[self.dataset_name]
        return pd.DataFrame(load(data_path))

    @classmethod
    def get_action_space(self):
        return ""

    @classmethod
    def get_trajectory(self, line):
        traj_dict = {}
        if self.RE_TYPE == "functional":
            traj_dict["task"] = line["question"]
        else:
            traj_dict["task"] = line["description"]
        return traj_dict

    def build_prompt(self, line):
        # st()
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)

        if self.RE_TYPE == "functional":
            user_instruction = USER_INSTRUCTION.format(instruction=line["question"])
        else:
            user_instruction = USER_INSTRUCTION_V2.format(
                description=line["description"]
            )

        msgs = []
        # add system prompt
        if self.RE_TYPE == "functional":
            msgs.append(dict(role="system", type="text", value=SYSTEM_PROMPT))
        else:
            msgs.append(dict(role="system", type="text", value=SYSTEM_PROMPT_V2))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=user_instruction))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        # st()
        if self.EVAL_TYPE == "point":
            return self.evaluate_point(eval_file, **judge_kwargs)

        elif self.EVAL_TYPE == "rectangle":
            return self.evaluate_rectangle(eval_file, **judge_kwargs)

    def evaluate_rectangle(self, eval_file, **judge_kwargs):
        scorers = {
            "IoU": compute_iou,
            "ACC@0.1": lambda x, y: compute_accuracy(x, y, 0.1),
            "ACC@0.3": lambda x, y: compute_accuracy(x, y, 0.3),
            "ACC@0.5": lambda x, y: compute_accuracy(x, y, 0.5),
            "ACC@0.7": lambda x, y: compute_accuracy(x, y, 0.7),
            "ACC@0.9": lambda x, y: compute_accuracy(x, y, 0.9),
            "Center_ACC": compute_center_accuracy,
        }
        results_dict = {}
        for key in scorers.keys():
            results_dict.update(
                {
                    key: [],
                    key + "_text": [],
                    key + "_icon": [],
                }
            )

        result = []
        data = load(eval_file)

        assert "bbox" in data and "prediction" in data
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            bbox = convert_bbox(
                line["bbox"], os.path.join(self.img_root, line["image_path"])
            )
            prediction = str(line["prediction"])
            try:
                click_point = parse_bbox_aguvis(prediction)

                match = {}
                for score_key, score_value in scorers.items():
                    score = score_value(bbox, click_point)
                    if score_key != "IoU":
                        match[score_key.replace("ACC", "match")] = score
                    results_dict[score_key].append(score)
                    if line["data_type"] == "text":
                        results_dict[score_key + "_text"].append(score)
                    else:
                        results_dict[score_key + "_icon"].append(score)
            except Exception:
                click_point = None
                match = {score_key: False for score_key in scorers.keys() if score_key != "IoU"}
            result.append(
                {
                    "img_path": os.path.join(self.img_root, line["image_path"]),
                    "text": line["question"],
                    "bbox": line["bbox"],
                    "parsed_bbox": bbox,
                    "type": line["data_type"],
                    "source": line["data_source"],
                    "pred": click_point,
                    "num_matched": sum(match.values()),
                    **match,
                }
            )
        for key in results_dict:
            if len(results_dict[key]) == 0:
                results_dict[key] = str(0)
            else:
                results_dict[key] = str(sum(results_dict[key]) / len(results_dict[key]))
        score_pth = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(results_dict, score_pth)

        failure_cases_path = os.environ.get("FAILURE_CASES_PATH", None)
        if failure_cases_path is not None:
            failure_cases = [res for res in result if not res["match"] and res["is_wrong_format"]]
            failure_cases.sort(key=lambda r: r["num_matched"], reverse=True)

            with open(failure_cases_path, "w") as f:
                json.dump(failure_cases, f, indent=4, ensure_ascii=False)
        return results_dict

    def evaluate_point(self, eval_file, **judge_kwargs):
        # -1: format_err, 0: wrong, 1: correct
        stats = defaultdict(list)
        # Will include instance-level results
        result = []

        data = load(eval_file)
        assert "bbox" in data and "prediction" in data
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            bbox = (
                line["bbox"]
                if isinstance(line["bbox"], list)
                else ast.literal_eval(line["bbox"])
            )
            # The format of bbox is (x1, y1, w, h)
            x1, y1, w, h = bbox
            bbox = (x1, y1, x1 + w - 1, y1 + h - 1)

            image = Image.open(os.path.join(self.img_root, line["image_path"]))
            img_size = image.size

            def make_safe(value):
                if value == -1:
                    # we can tolerate -1 as a special value and nomalize it to 0
                    return 0
                else:
                    return value

            bbox = [
                make_safe(bbox[0]) / img_size[0],
                make_safe(bbox[1]) / img_size[1],
                make_safe(bbox[2]) / img_size[0],
                make_safe(bbox[3]) / img_size[1],
            ]

            if any([x < 0 or x > 1 for x in bbox]):
                raise ValueError(f"bbox out of range: {bbox} | {line['bbox']} | {img_size}")

            key = line['data_type'] if 'category' not in line else line['category'] + ":" + line['data_type']
            prediction = str(line["prediction"])
            try:
                click_point = parse_bbox_aguvis(prediction)
                # Do Normalization By Default
                if click_point[0] > 1 or click_point[1] > 1:
                    click_point = (click_point[0] / img_size[0], click_point[1] / img_size[1])

                match = (bbox[0] <= click_point[0] <= bbox[2]) and \
                    (bbox[1] <= click_point[1] <= bbox[3])

                if match:
                    stats[key].append(1)
                else:
                    stats[key].append(0)
                is_wrong_format = False

            except Exception as e:
                logger.warning(f"exception in screenspot eval:{e}")
                stats[key].append(-1)
                match, is_wrong_format, click_point = False, True, None

            result.append(
                {
                    "img_path": os.path.join(self.img_root, line["image_path"]),
                    "text": line["question"],
                    "bbox": line["bbox"],
                    "parsed_bbox": bbox,
                    "type": line["data_type"],
                    "source": line["data_source"],
                    "match": match,
                    "is_wrong_format": is_wrong_format,
                    "pred": click_point,
                }
            )

        final_score_dict = {}
        # Record the number of each category
        final_score_dict.update({k + ':cnt': len(stats[k]) for k in stats})
        # Calculate the Overall stats
        full_stats = []
        for v in stats.values():
            full_stats.extend(v)
        final_score_dict['Overall_Accuracy'] = np.mean([x > 0 for x in full_stats]) * 100
        final_score_dict['Format_Err_Rate'] = np.mean([x < 0 for x in full_stats]) * 100
        # Calculate the Accuracy of Text / Icon
        text_stats = [v for k, v in stats.items() if k.endswith('text') for x in v]
        text_stats = itertools.chain(*text_stats)
        final_score_dict['Text_Accuracy'] = np.mean([x > 0 for x in text_stats]) * 100
        icon_stats = [v for k, v in stats.items() if k.endswith('icon') for x in v]
        icon_stats = itertools.chain(*icon_stats)
        final_score_dict['Icon_Accuracy'] = np.mean([x > 0 for x in icon_stats]) * 100
        # Calculate the Accuracy of Each Category
        if 'category' in data:
            cates = list(set(data['category']))
            for c in cates:
                sub_stats = [v for k, v in stats.items() if k.split(":")[0] == c for x in v]
                sub_stats = itertools.chain(*sub_stats)
                final_score_dict[c + '_Accuracy'] = np.mean([x > 0 for x in sub_stats]) * 100

        score_pth = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(final_score_dict, score_pth)

        failure_cases_path = os.environ.get("FAILURE_CASES_PATH", None)
        if failure_cases_path is not None:
            def click_distance(bbox, click_point):
                x, y = click_point
                x1, y1, x2, y2 = bbox
                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                abs_shift_to_center = [abs(x - xc), abs(y - yc)]  # noqa: E501
                width_outside, height_outside = [max(0, abs_shift_to_center[0] - w / 2), max(0, abs_shift_to_center[1] - h / 2)]  # noqa: E501
                return (width_outside ** 2 + height_outside ** 2) ** 0.5  # noqa: E501

            wrong_format_result = [res for res in result if res["is_wrong_format"]]
            missed_result = [res for res in result if not res["match"] and not res["is_wrong_format"]]
            missed_result.sort(key=lambda r: click_distance(r["parsed_bbox"], r["pred"]), reverse=True)
            failure_cases = wrong_format_result + missed_result

            with open(failure_cases_path, "w") as f:
                json.dump(failure_cases, f, indent=4, ensure_ascii=False)
        return final_score_dict
