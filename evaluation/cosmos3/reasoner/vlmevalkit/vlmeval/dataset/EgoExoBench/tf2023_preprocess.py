import json
import os

import cv2
import numpy as np

# replace the path with your actual path
ann_file = 'EgoExoBench/MCQ/Ego-Exo-View-Transition/ego_wearer_identification.json'


def add_bbox(bbox_img_path):

    bbox_dir = os.path.dirname(bbox_img_path)
    os.makedirs(bbox_dir, exist_ok=True)
    vid, frame_idx, person_id = bbox_img_path.split('/')[-4], bbox_img_path.split('/')[-2], bbox_img_path.split('/')[-1].split('.')[0]  # noqa: E501
    import os.path as osp
    json_file = os.path.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(bbox_img_path)))), vid, 'Segmentation/T', frame_idx + '.json')  # noqa: E501
    ori_img_path = json_file.replace('.json', '.jpg')

    with open(json_file, mode='r', encoding="utf-8") as f:
        configs = json.load(f)
    shapes = configs["shapes"]

    mask = np.zeros((configs["imageHeight"], configs["imageWidth"], 1), np.uint8)

    if not os.path.exists(ori_img_path):
        ori_img_path = ori_img_path.replace('T/', '')

    if not os.path.exists(ori_img_path):
        ori_img_path = ori_img_path.replace('Segmentation/', 'frame/T/')

    original_image = cv2.imread(ori_img_path)

    for shape in shapes:
        if shape['label'] != person_id:
            continue

        cv2.fillPoly(mask, [np.array(shape["points"], np.int32)], 1)

        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        stats = stats[stats[:, 4].argsort()]
        bboxs = stats[:-1]

        for b in bboxs:
            x0, y0 = b[0], b[1]
            x1 = b[0] + b[2]
            y1 = b[1] + b[3]

            start_point, end_point = (x0, y0), (x1, y1)
            color = (0, 0, 255)
            thickness = 2
            mask_bboxs = cv2.rectangle(original_image, start_point, end_point, color, thickness)
            mask_bboxs = cv2.resize(mask_bboxs, (540, 360))
            cv2.imwrite(bbox_img_path, mask_bboxs)
        return


def rescale_img(img_path, width, height):
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, (width, height))
    cv2.imwrite(img_path, resized_img)


with open(ann_file, 'r') as f:
    ann_data = json.load(f)
    for aitem in ann_data.values():
        image_paths = []
        for oitem in aitem['options']:
            add_bbox(oitem['image_paths'][0])

        for img_path in aitem['query']['image_paths']:
            rescale_img(img_path, 960, 540)
