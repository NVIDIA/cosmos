import json
import os
import xml.etree.ElementTree as ET

import cv2

# replace with your actual path
ann_file = 'EgoExoBench/MCQ/Ego-Exo-Relation/person_relation.json'


def add_bbox(bbox_img_path):
    bbox_dir = os.path.dirname(bbox_img_path)
    os.makedirs(bbox_dir, exist_ok=True)
    ori_img_dir = os.path.dirname(bbox_img_path).replace('bbox', 'frame_sel')
    frame_idx, person_id = os.path.basename(bbox_img_path).split('.')[0].split('_')
    ori_img_path = os.path.join(ori_img_dir, frame_idx + '.jpg')
    xml_file = ori_img_path.replace('data', 'GT_xml').replace('frame_sel/', '').replace('.jpg', '.xml')

    tree = ET.parse(xml_file)
    root = tree.getroot()
    im = cv2.imread(ori_img_path)
    for object in root.findall('object'):
        object_name = object.find('name').text
        if object_name != person_id:
            continue
        im_copy = im.copy()
        Xmin = int(object.find('rectangle').find('xmin').text)
        Ymin = int(object.find('rectangle').find('ymin').text)
        Xmax = int(object.find('rectangle').find('xmax').text)
        Ymax = int(object.find('rectangle').find('ymax').text)
        color = (255, 0, 0)
        cv2.rectangle(im_copy, (Xmin, Ymin), (Xmax, Ymax), color, 3)
        cv2.imwrite(bbox_img_path, im_copy)
        return


with open(ann_file, 'r') as f:
    ann_data = json.load(f)
    for aitem in ann_data.values():
        image_paths = []
        image_paths.extend(aitem['query']['image_paths'])
        for oitem in aitem['options']:
            image_paths.extend(oitem['image_paths'])

        for image_path in image_paths:
            add_bbox(image_path)
