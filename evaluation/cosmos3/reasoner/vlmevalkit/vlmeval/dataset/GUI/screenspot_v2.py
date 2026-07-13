import ast
import os.path as osp
import re

import pandas as pd
from PIL import Image

from vlmeval.smp import LMUDataRoot, get_logger, load, toliststr
from .screenspot import ScreenSpot

logger = get_logger(__name__)

"""
{
    "img_filename": "web_3b0ad239-da6b-4f6f-8f12-f674dc90ff33.png",
    "bbox": [42, 1102, 197, 70],
    "instruction": "view the details of the item",
    "data_type": "text",
    "data_source": "shop"
},
{
    "img_filename": "web_3b0ad239-da6b-4f6f-8f12-f674dc90ff33.png",
    "bbox": [93, 74, 86, 132],
    "instruction": "view the previous photo",
    "data_type": "icon",
    "data_source": "shop"
}
"""

SYSTEM_PROMPT = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform pyautogui click/moveTo action to complete the task. The answer format is `pyautogui.click(x=?, y=?), x and y is necessary`"""  # noqa: E501

USER_INSTRUCTION = """Please complete the following tasks by clicking using `pyautogui.click`:\n{instruction}"""  # noqa: E501

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


class ScreenSpotV2(ScreenSpot):
    MODALITY = "IMAGE"
    TYPE = "GUI"
    DATASET_URL = {
        "ScreenSpot_v2_Mobile": "ScreenSpot_v2_Mobile.tsv",
        "ScreenSpot_v2_Desktop": "ScreenSpot_v2_Desktop.tsv",
        "ScreenSpot_v2_Web": "ScreenSpot_v2_Web.tsv",
    }  # path
    DATASET_MD5 = {}
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
        self.img_root = osp.join(ROOT, "ScreenSpot_v2", "screenspotv2_image")
        self.RE_TYPE = re_type
        if skeleton:
            return

        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and "image" in data:
            data = data[~pd.isna(data["image"])]

        data["index"] = [str(idx + 1) for idx, x in enumerate(data["bbox"])]

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

        if "img_filename" in data:
            paths = [toliststr(x) for x in data["img_filename"]]
            data["image_path"] = [x[0] if len(x) == 1 else x for x in paths]

        # if np.all([istype(x, int) for x in data["index"]]):
        #     data["index"] = [int(x) for x in data["index"]]

        self.data = data
        self.post_build(dataset)

    def prepare_tsv(self, url, file_md5=None):
        # st()
        if self.RE_TYPE == "functional":
            data_root = LMUDataRoot()
            data_path = osp.join(data_root, "ScreenSpot_v2", url)
        else:
            data_path = self.DATASET_URL_V2[self.dataset_name]
        return pd.DataFrame(load(data_path))
