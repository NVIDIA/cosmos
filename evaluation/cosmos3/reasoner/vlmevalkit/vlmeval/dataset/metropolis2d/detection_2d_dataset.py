"""
2D Detection Dataset for images in KITTI format.

Following the structure of omni3D and omni3d_dataset.py, this dataset evaluates
2D object detection using mAP and AP50 metrics.

The validation data should be in KITTI format with:
- images/ directory containing image files
- labels/ directory containing label files (KITTI format)

The question for each image:
'Locate every instance that belongs to the following categories: 'sedan, SUV, bus, truck'.
For each instance of the class, report bbox coordinates in JSON format.
Do not group instances and report only individual instances.'

For evaluation, predicted labels and groundtruth labels are mapped to "car"
('sedan, SUV, bus, truck' --> "car") and evaluated using mAP and AP50 metrics.
"""

import os

import numpy as np
import pandas as pd
import PIL.Image
from PIL import ImageOps

from ...smp import get_logger, load
from ..image_base import ImageBaseDataset
from .utils import (compute_2d_iou, compute_ap_from_matches, load_dataset_config,
                    parse_bbox_2d_from_text, parse_kitti_label, scale_bbox)

# Categories that map to "car" for evaluation
CAR_CATEGORIES = {'sedan', 'suv', 'bus', 'truck', 'car'}

# Default prompt for detection
DETECTION_PROMPT = (
    "Locate every instance that belongs to the following categories: 'sedan, SUV, bus, truck'. "
    "For each instance of the class, report bbox coordinates in JSON format. "
    "Do not group instances and report only individual instances."
)


def map_label_to_car(label):
    """
    Map various vehicle labels to 'car'.

    Args:
        label: Original label string

    Returns:
        'car' if label is a vehicle type, otherwise original label
    """
    if label.lower() in CAR_CATEGORIES:
        return 'car'
    return label.lower()


class Metropolis2DDetectionDataset(ImageBaseDataset):
    """Dataset class for 2D object detection evaluation in KITTI format."""

    TYPE = 'VQA'  # Use VQA type so predictions are treated as text
    MODALITY = 'IMAGE'

    @classmethod
    def supported_datasets(cls):
        return ['Metropolis2D', 'Metropolis2D_val']

    def __init__(self, dataset='Metropolis2D', data_root=None, **kwargs):
        """
        Args:
            dataset: Dataset name (used to look up config in datasets.yaml)
            data_root: Root directory containing 'images' and 'labels' subdirectories.
                       If None, will be loaded from datasets.yaml based on dataset name.
        """
        self.dataset_name = dataset
        self.data_root = data_root

        if data_root is None:
            # Try to load from datasets.yaml
            dataset_cfg = load_dataset_config(dataset)
            if dataset_cfg and 'data_root' in dataset_cfg:
                self.data_root = dataset_cfg['data_root']

        if self.data_root is None:
            raise ValueError(
                f"data_root must be specified or configured in datasets.yaml for dataset '{dataset}'"
            )

        self.img_root = self.data_root
        self.images_dir = os.path.join(self.data_root, 'images')
        self.labels_dir = os.path.join(self.data_root, 'labels')

        # Cache for ground truth
        self._gt_cache = {}

        # Build data structure
        self.data = self._build_data_structure()

        # Call post build hook for compatibility
        try:
            self.post_build(self.dataset_name)
        except Exception:
            pass

    def _get_image_files(self):
        """Get list of image files from the images directory."""
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []

        for filename in sorted(os.listdir(self.images_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append(filename)

        return image_files

    def _load_ground_truth(self, image_filename):
        """
        Load ground truth labels for an image.

        Args:
            image_filename: Name of the image file

        Returns:
            List of ground truth objects with 'label' and 'bbox'
        """
        if image_filename in self._gt_cache:
            return self._gt_cache[image_filename]

        # Construct label path (same name as image, but .txt extension)
        base_name = os.path.splitext(image_filename)[0]
        label_path = os.path.join(self.labels_dir, base_name + '.txt')

        gt_objects = parse_kitti_label(label_path)

        # Map labels to 'car'
        for obj in gt_objects:
            obj['original_label'] = obj['label']
            obj['label'] = map_label_to_car(obj['label'])

        self._gt_cache[image_filename] = gt_objects
        return gt_objects

    def _build_data_structure(self):
        """Build the data structure for VLMEvalKit format."""
        logger = get_logger('Metropolis2D')

        image_files = self._get_image_files()
        logger.info(f"Found {len(image_files)} images in {self.images_dir}")

        data_list = []
        for idx, image_filename in enumerate(image_files):
            image_path = os.path.join(self.images_dir, image_filename)

            # Load ground truth to check if there are objects
            gt_objects = self._load_ground_truth(image_filename)

            row = {
                'index': str(idx),
                'image_path': image_path,
                'image_filename': image_filename,
                'question': DETECTION_PROMPT,
                'num_gt_objects': len(gt_objects),
            }
            data_list.append(row)

        logger.info(f"Built dataset with {len(data_list)} samples")
        return pd.DataFrame(data_list)

    def build_prompt(self, line):
        """Build prompt for 2D detection."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_path = line['image_path']

        question = line['question']

        msgs = [
            dict(type='image', value=image_path),
            dict(type='text', value=question)
        ]

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate predictions using COCO-style mAP and AP50 metrics.

        All predictions and ground truth labels are mapped to 'car' category
        before evaluation.
        """
        logger = get_logger('Metropolis2D')

        # Try different file extensions if the specified one doesn't exist
        if not os.path.exists(eval_file):
            base = os.path.splitext(eval_file)[0]
            for ext in ['.xlsx', '.tsv', '.json', '.pkl']:
                candidate = base + ext
                if os.path.exists(candidate):
                    eval_file = candidate
                    logger.info(f'Using alternate file format: {eval_file}')
                    break
            else:
                logger.error(f'No prediction file found with base: {base}')
                return {
                    'mAP': 0.0,
                    'AP50': 0.0,
                    'total_predictions': 0,
                    'valid_bbox_predictions': 0,
                    'error': 'Prediction file not found'
                }

        try:
            data = load(eval_file)
            logger.info(f'Loaded {len(data)} predictions from {eval_file}')
        except Exception as e:
            logger.error(f'Failed to load predictions: {e}')
            return {
                'mAP': 0.0,
                'AP50': 0.0,
                'total_predictions': 0,
                'valid_bbox_predictions': 0,
                'error': str(e)
            }

        # IoU thresholds for COCO-style mAP
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        # Collect all detections globally: {iou_thresh: [(confidence, tp_flag), ...]}
        all_detections = {t: [] for t in iou_thresholds}
        total_gt = 0
        total_pred = 0
        valid_count = 0

        # Process each image
        for idx, row in data.iterrows():
            # Parse prediction
            pred_text = str(row.get('prediction', ''))
            pred_boxes_raw = parse_bbox_2d_from_text(pred_text)

            image_path = row.get('image_path', '')
            # Load image
            try:
                pil_image = PIL.Image.open(image_path)
                pil_image = pil_image.convert('RGB')
                pil_image = ImageOps.exif_transpose(pil_image)
            except Exception as e:
                logger.error(f"Failed to load image {image_path}: {e}")
                pil_image = PIL.Image.new('RGB', (640, 480), color='black')

            # Normalize prediction format
            pred_boxes = []
            for pred in pred_boxes_raw:
                if isinstance(pred, dict):
                    # Handle different bbox key names
                    bbox = None
                    for key in ['bbox_2d']:
                        if key in pred:
                            bbox = pred[key]
                            break

                    if bbox is not None:
                        bbox = scale_bbox(bbox, pil_image.height, pil_image.width, scale_factor=1000)

                    if bbox is not None and len(bbox) >= 4:
                        pred_boxes.append({
                            'bbox': bbox[:4],
                            'label': map_label_to_car(pred.get('label', 'car')),
                            'score': pred.get('score', pred.get('confidence', 1.0))
                        })

            if len(pred_boxes) > 0:
                valid_count += 1

            # Get ground truth
            image_filename = row.get('image_filename', '')
            if not image_filename:
                image_path = row.get('image_path', '')
                image_filename = os.path.basename(image_path)

            gt_boxes = self._load_ground_truth(image_filename)

            # Filter to only 'car' category
            gt_boxes_car = [gt for gt in gt_boxes if gt['label'] == 'car']
            pred_boxes_car = [p for p in pred_boxes if p['label'] == 'car']

            total_gt += len(gt_boxes_car)
            total_pred += len(pred_boxes_car)

            # Sort predictions by confidence (descending)
            pred_boxes_car = sorted(pred_boxes_car, key=lambda x: x.get('score', 1.0), reverse=True)

            # Match predictions to ground truth at each IoU threshold
            for iou_thresh in iou_thresholds:
                gt_matched = [False] * len(gt_boxes_car)

                for pred in pred_boxes_car:
                    pred_bbox = pred['bbox']
                    confidence = pred.get('score', 1.0)
                    best_iou = 0.0
                    best_gt_idx = -1

                    for gt_idx, gt in enumerate(gt_boxes_car):
                        if gt_matched[gt_idx]:
                            continue
                        iou = compute_2d_iou(pred_bbox, gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_iou >= iou_thresh and best_gt_idx >= 0:
                        # True positive
                        all_detections[iou_thresh].append((confidence, 1))
                        gt_matched[best_gt_idx] = True
                    else:
                        # False positive
                        all_detections[iou_thresh].append((confidence, 0))

        # Compute AP at each IoU threshold using COCO-style calculation
        ap_per_threshold = {}
        for iou_thresh in iou_thresholds:
            detections = all_detections[iou_thresh]
            if len(detections) == 0:
                ap_per_threshold[iou_thresh] = 0.0
                continue

            confidences = [d[0] for d in detections]
            tp_flags = [d[1] for d in detections]

            ap, _, _ = compute_ap_from_matches(tp_flags, total_gt, confidences)
            ap_per_threshold[iou_thresh] = ap

        # AP50 is AP at IoU=0.5
        ap50 = ap_per_threshold.get(0.5, 0.0)

        # mAP is mean AP over IoU thresholds 0.5:0.05:0.95
        mAP = np.mean([ap_per_threshold[t] for t in iou_thresholds])

        # Also compute precision/recall at IoU=0.5 for reference
        detections_50 = all_detections[0.5]
        if len(detections_50) > 0:
            tp_50 = sum(d[1] for d in detections_50)
            fp_50 = len(detections_50) - tp_50
            precision_50 = tp_50 / (tp_50 + fp_50) if (tp_50 + fp_50) > 0 else 0.0
            recall_50 = tp_50 / total_gt if total_gt > 0 else 0.0
        else:
            precision_50 = 0.0
            recall_50 = 0.0

        result = {
            'mAP': float(mAP * 100),
            'AP50': float(ap50 * 100),
            'precision_50': float(precision_50 * 100),
            'recall_50': float(recall_50 * 100),
            'total_predictions': len(data),
            'valid_bbox_predictions': valid_count,
            'valid_rate': valid_count / len(data) if len(data) > 0 else 0,
            'total_gt_objects': total_gt,
            'total_pred_objects': total_pred,
        }

        logger.info(f"mAP: {result['mAP']:.2f}%")
        logger.info(f"AP50: {result['AP50']:.2f}%")
        logger.info(f"Precision@50: {result['precision_50']:.2f}%")
        logger.info(f"Recall@50: {result['recall_50']:.2f}%")
        logger.info(f"Valid predictions: {valid_count}/{len(data)} ({result['valid_rate']:.2%})")

        return result
