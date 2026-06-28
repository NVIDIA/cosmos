import ast
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

SYSTEM_PROMPT = "You are a GUI agent. You are given a task and a screenshot of the screen. " \
    "You need to perform pyautogui click/moveTo action to complete the task. " \
    "The answer format is `pyautogui.click(x=?, y=?), x and y is necessary`"

USER_INSTRUCTION = "Please complete the following tasks by clicking using `pyautogui.click`:\n{instruction}"


def parse_bbox_aguvis(response):
    match = re.search(r"x=([\d.]+), y=([\d.]+)", response)
    if match:
        click_point = [float(match.group(1)), float(match.group(2))]
    else:
        click_point = [0.0, 0.0]
    return click_point


class VenusBench_GD(ImageBaseDataset):
    MODALITY = "IMAGE"
    TYPE = "GUI"
    DATASET_URL = {
        "VenusBench-GD": "https://huggingface.co/datasets/Zery/VBGD_Dataset/resolve/main/VenusBench.tsv",
    }
    DATASET_MD5 = {
        'VenusBench-GD': '6a2fe92d3ecf5a3b6503a1fe4891c5ea'
    }

    def __init__(
        self,
        dataset="VenusBench-GD",
        skip_noimg=True,
        skeleton=False,
    ):
        ROOT = LMUDataRoot()
        self.dataset_name = dataset
        self.img_root = osp.join(ROOT, "images", self.dataset_name)

        if skeleton:
            return

        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and "image" in data:
            data = data[~pd.isna(data["image"])]

        # Verify we have index properly
        if "index" not in data:
            data["index"] = [str(idx + 1) for idx in range(len(data))]

        self.meta_only = True
        self.parse_response_func = parse_bbox_aguvis

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

    @classmethod
    def get_action_space(self):
        return ""

    @classmethod
    def get_trajectory(self, line):
        traj_dict = {}
        traj_dict["task"] = line["question"]
        return traj_dict

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        user_instruction = USER_INSTRUCTION.format(instruction=line["question"])
        msgs = []
        msgs.append(dict(role="system", type="text", value=SYSTEM_PROMPT))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=user_instruction))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        stats = defaultdict(list)
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
            # The format of bbox in VenusBench-GD is (x_min, y_min, x_max, y_max)
            image = Image.open(os.path.join(self.img_root, line["image_path"]))
            img_size = image.size

            # Absolute to relative
            bbox = [
                bbox[0] / img_size[0],
                bbox[1] / img_size[1],
                bbox[2] / img_size[0],
                bbox[3] / img_size[1],
            ]

            key = line["category"] + ":" + line['ui_type']
            prediction = str(line["prediction"])
            try:
                click_point = self.parse_response_func(prediction)
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
                logger.warning(f"exception in venusbench eval:{e}")
                stats[key].append(-1)
                match, is_wrong_format, click_point = False, True, None

            result.append(
                {
                    "img_path": os.path.join(self.img_root, line["image_path"]),
                    "text": line["question"],
                    "bbox": line["bbox"],
                    "parsed_bbox": bbox,
                    "type": line["ui_type"],
                    "category": line["category"],
                    "match": match,
                    "is_wrong_format": is_wrong_format,
                    "pred": click_point,
                }
            )

        final_score_dict = {}
        final_score_dict.update({k + ':cnt': len(stats[k]) for k in stats})

        full_stats = []
        for v in stats.values():
            full_stats.extend(v)
        final_score_dict['Overall_Accuracy'] = np.mean([x > 0 for x in full_stats]) * 100
        final_score_dict['Format_Err_Rate'] = np.mean([x < 0 for x in full_stats]) * 100

        cates = list(set([line["category"] for line in lines]))
        for c in cates:
            sub_stats = [v for k, v in stats.items() if k.split(":")[0] == c for x in v]
            if len(sub_stats) > 0:
                final_score_dict[c + '_Accuracy'] = np.mean([x[0] > 0 for x in [sub_stats]]) * 100

        score_pth = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(final_score_dict, score_pth)
        return final_score_dict
